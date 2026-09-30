"""Frozen PottsMPNN + trainable HBondHead as one DDP-wrappable module.

This lets the linear H-bond / salt-bridge head be trained with the same Fabric/DDP
harness as PottsMPNN (multi-GPU, bf16), via ``HBondHeadTrainer``.

EVERYTHING in the pretrained PottsMPNN is frozen — the graph featurization, the token
embedding ``W_s``, the encoder layers, AND the decoder layers / output heads (``W_out``,
``etab_out``). The ONLY trainable parameters are ``self.head`` (the linear readout).
``self.potts`` is always in ``eval()`` (no dropout) and run under ``no_grad``; because
the encoder is sequence-agnostic, the head sees the edge embedding ``h_E`` plus the
swapped-in token embeddings ``W_s(S_i)`` / ``W_s(S_j)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mpnn.model.pottsmpnn import PottsMPNN
from mpnn.model.layers.graph_embeddings import PottsProteinFeatures
from mpnn.hbond_head import (
    HBondHead, build_edge_features, reverse_edge_index, HBOND, DONOR, ACCEPTOR, SALTBRIDGE,
    AMBIGUOUS_TOKENS,
)


# Bond chemistry (formal charge + donor/acceptor/salt-bridge capability per token) lives in the
# lightweight, torch-free module ``mpnn.chemistry`` so the annotation transforms can reuse it without
# importing this model. Re-exported here for backward compatibility.
from mpnn.chemistry import (  # noqa: E402
    BondCapability,
    DEFAULT_CAPABILITY,
    BOND_CHEMISTRY,
    TITRATABLE_STATES,
    capability as _capability,
)


def _charge_by_idx(idx_to_token, device=None) -> torch.Tensor:
    """``[V]`` float tensor of formal charge per token id, read from :data:`BOND_CHEMISTRY`.

    ``NaN`` for any token NOT in ``BOND_CHEMISTRY`` (UNK, bare HIS/ASP/GLU) or in ``AMBIGUOUS_TOKENS``
    (``*-A``), so residues with unresolved charge DROP OUT of clash scoring (rather than being treated
    as neutral). This is the single charge source used by :meth:`HBondModel.protonation_switch_bonds`.
    """
    V = len(idx_to_token)
    q = torch.full((V,), float("nan"), dtype=torch.float32)
    for i in range(V):
        name = str(idx_to_token[i])
        if name in BOND_CHEMISTRY and name not in AMBIGUOUS_TOKENS:
            q[i] = float(BOND_CHEMISTRY[name].charge)
    return q.to(device) if device is not None else q


class HBondModel(nn.Module):
    """``forward(network_input) -> {"logits": [B,L,K,4], "E_idx": [B,L,K]}``.

    Args:
        extended_vocab: build the 32-token Potts model (required for protonation
            microstates). Must match the checkpoint loaded by the trainer.
        n_hidden: HBondHead hidden width (0 => single linear layer).
        n_layers: number of ReLU hidden layers in the head when ``n_hidden > 0``.
        use_node: include the encoder node embeddings in the head input (5H); ``False``
            reproduces the legacy 3H token+edge+token representation. Must match the
            checkpoint being loaded — use :meth:`load_from_checkpoint` to auto-detect.
        potts: optional pre-built PottsMPNN (else one is constructed here).
    """

    def __init__(self, *, extended_vocab: str | bool = "v4", n_hidden: int = 0, n_layers: int = 2,
                 use_node: bool = True, potts: PottsMPNN | None = None):
        super().__init__()
        self.extended_vocab = extended_vocab
        if potts is None:
            if isinstance(extended_vocab, str):
                # a vocab NAME ("v6") sizes W_s/W_out/etab off its token_encoding (v6 = 30 tokens); a bare
                # truthy bool keeps the default 32-token vocab (v3/v4). Mirrors trainers/hbond_head.py.
                from mpnn.transforms.extended_vocab import get_vocab
                potts = PottsMPNN(graph_featurization_module=PottsProteinFeatures(
                    token_encoding=get_vocab(extended_vocab)["token_encoding"]))
            elif extended_vocab:
                potts = PottsMPNN(graph_featurization_module=PottsProteinFeatures())
            else:
                potts = PottsMPNN()
        self.potts = potts                                    # FROZEN pretrained model
        self.use_node = use_node
        # Head input = concat([h_V[i],] W_s(S_i), h_E, W_s(S_j), [h_V[j]]) => 5 (or 3) H-blocks.
        self.head = HBondHead(hidden_dim=potts.hidden_dim, n_hidden=n_hidden,
                              in_mult=(5 if use_node else 3), n_layers=n_layers)  # ONLY trainable part
        self.head.apply(PottsMPNN.init_weights)
        self.freeze_pretrained()

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, *, extended_vocab: str | bool = "v4", map_location="cpu"):
        """Build an HBondModel whose head architecture MATCHES the checkpoint, then load it.

        Infers ``use_node`` (5H vs legacy 3H), ``n_hidden`` and ``n_layers`` from the saved
        ``head.net.*`` shapes, so callers don't have to hand-match config. Works for old and
        new checkpoints. Returns ``(model, ckpt_dict)``.
        """
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        hidden_dim = 128  # PottsMPNN hidden dim
        # Collect the head Linear layers (keys "head.net.<idx>.weight" or "head.net.weight").
        lin = {k: v for k, v in sd.items()
               if k.startswith("head.net") and k.endswith("weight") and v.dim() == 2}
        if not lin:
            raise ValueError(f"No head Linear weights found in checkpoint {ckpt_path}.")
        first = min(lin, key=lambda k: (len(k), k))          # first layer = smallest index / plain
        in_dim = lin[first].shape[1]
        if in_dim % hidden_dim != 0:
            raise ValueError(f"Head input dim {in_dim} is not a multiple of hidden_dim {hidden_dim}.")
        in_mult = in_dim // hidden_dim
        n_lin = len(lin)
        if n_lin == 1:                                       # single Linear => pure-linear head
            n_hidden, n_layers = 0, 0
        else:
            n_hidden = lin[first].shape[0]
            n_layers = n_lin - 1                             # hidden ReLU blocks (last Linear = output)
        model = cls(extended_vocab=extended_vocab, n_hidden=n_hidden, n_layers=n_layers,
                    use_node=(in_mult == 5))
        model.load_state_dict(sd, strict=True)
        return model, ckpt

    def freeze_pretrained(self) -> None:
        """Freeze the entire PottsMPNN (encoder + W_s token embedding + decoder + heads).

        Leaves only ``self.head`` trainable, and asserts exactly that so nothing leaks.
        """
        for p in self.potts.parameters():
            p.requires_grad_(False)
        self.potts.eval()
        for p in self.head.parameters():
            p.requires_grad_(True)

        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        head_params = {f"head.{n}" for n, _ in self.head.named_parameters()}
        assert trainable == head_params, (
            f"Only the head must be trainable. Unexpected trainable params: "
            f"{sorted(trainable - head_params)}; missing: {sorted(head_params - trainable)}"
        )

    def train(self, mode: bool = True):
        # Frozen feature extractor: never put the PottsMPNN in train mode (no dropout),
        # even when the trainer calls model.train(). Only the head toggles.
        super().train(mode)
        self.potts.eval()
        return self

    @torch.no_grad()
    def gather_hbond_salt_bridge_graph(self, network_input: dict) -> dict:
        """The directed k-NN interaction graph with its per-edge head-input features.

        Runs ONLY the frozen, sequence-agnostic encoder + the token-embedding swap, so this is the
        reusable substrate for any sequence / protonation state. For directed edge ``i -> j =
        E_idx[i, k]`` the feature is ``concat(h_V[i], W_s(S_i), h_E[i,k], W_s(S_j), h_V[j])``. Returns::

            E_idx         [B, L, K]      neighbour indices (the directed graph; k=0 is the self-edge)
            edge_features [B, L, K, 5H]  the head's per-edge input
            h_E           [B, L, K, H]   sequence-agnostic edge embedding (reuse across sequences)
            h_V           [B, L, H]      sequence-agnostic node embedding (reuse across sequences)
            S             [B, L]         the token sequence the features were built for

        Mutates ``network_input['input_features']`` in place (adds the standard masks), like forward.
        """
        inp = network_input["input_features"]
        self.potts.sample_and_construct_masks(inp)
        graph_features = self.potts.graph_featurization(inp)
        encoder_features = self.potts.encode(inp, graph_features)
        h_E = encoder_features["h_E"]          # [B, L, K, H]
        h_V = encoder_features["h_V"]          # [B, L, H]
        E_idx = graph_features["E_idx"]        # [B, L, K]
        edge_features = build_edge_features(self.potts.W_s, inp["S"], h_E, E_idx, h_V,
                                            use_node=self.use_node)  # [B, L, K, 5H or 3H]
        return {"E_idx": E_idx, "edge_features": edge_features, "h_E": h_E, "h_V": h_V, "S": inp["S"]}

    @torch.no_grad()
    def predict_hbond_graph(self, network_input: dict) -> dict:
        """Predict every directed-edge interaction in the graph up front, with confidence scores.

        One pass over the whole k-NN graph. Per directed edge ``i -> j = E_idx[i, k]``::

            E_idx         [B, L, K]     the graph
            p_hbond       [B, L, K]     P(i, j form an H-bond)        (sigmoid)
            p_donor       [B, L, K]     P(i donates to j | H-bond)    (direction softmax: +acceptor=1)
            p_acceptor    [B, L, K]     P(i accepts from j | H-bond)
            p_saltbridge  [B, L, K]     P(i, j form a salt bridge)    (sigmoid)
            logits        [B, L, K, 4]  raw head logits

        Confidence scores are the probabilities; interpret donor/acceptor only where ``p_hbond`` is
        high. Cheap to re-run for a different sequence/protonation via the same h_E (see
        :meth:`gather_hbond_salt_bridge_graph`).
        """
        graph = self.gather_hbond_salt_bridge_graph(network_input)
        logits = self.head(graph["edge_features"])                    # [B, L, K, 4]
        p_dir = torch.softmax(logits[..., DONOR:ACCEPTOR + 1], dim=-1)
        return {
            "E_idx": graph["E_idx"],
            "p_hbond": torch.sigmoid(logits[..., HBOND]),
            "p_donor": p_dir[..., 0],
            "p_acceptor": p_dir[..., 1],
            "p_saltbridge": torch.sigmoid(logits[..., SALTBRIDGE]),
            "logits": logits,
        }

    @torch.no_grad()
    def predict_hbonds(self, h_V, h_E, E_idx, S, hbond_threshold: float = 0.5,
                       reduce: str = "mean") -> list[list]:
        """Reciprocal-combined H-bond list with donor/acceptor roles, from precomputed encoder outputs.

        For every RECIPROCAL k-NN pair (both ``i->j`` and ``j->i`` in the graph) it combines the two
        directions: presence = ``reduce(p_hbond(i->j), p_hbond(j->i))`` (default mean — the best
        undirected predictor), and the role is decided from BOTH directions' softmax
        (``P(i is donor) = mean(p_donor(i->j), p_acceptor(j->i))``). Each undirected pair is returned
        once (``i < j``). ``h_V``/``h_E``/``E_idx`` are the frozen, sequence-agnostic encoder outputs
        (get them from :meth:`gather_hbond_salt_bridge_graph` or ``hbond_head.encode_edges``), so you
        can swap ``S`` (e.g. a protonation microstate) and re-call this cheaply without re-encoding.

        Args:
            h_V:   ``[L, H]`` or ``[1, L, H]`` encoder node embeddings.
            h_E:   ``[L, K, H]`` or ``[1, L, K, H]`` encoder edge embeddings.
            E_idx: ``[L, K]`` or ``[1, L, K]`` neighbour indices.
            S:     ``[L]`` or ``[1, L]`` token sequence (swap for a different sequence/protonation).
            hbond_threshold: keep pairs with combined ``p_hbond >= hbond_threshold``.
            reduce: how to combine the two directions — ``"mean"`` | ``"max"`` | ``"min"`` | ``"noisy_or"``.

        Returns:
            ``list[[i, j, role_i, role_j]]`` — ``i < j`` (python ints into the token order of ``S``);
            roles are ``"donor"`` / ``"acceptor"`` (always opposite).
        """
        if h_V.dim() == 2:
            h_V = h_V.unsqueeze(0)
        if h_E.dim() == 3:
            h_E = h_E.unsqueeze(0)
        if E_idx.dim() == 2:
            E_idx = E_idx.unsqueeze(0)
        if S.dim() == 1:
            S = S.unsqueeze(0)

        feats = build_edge_features(self.potts.W_s, S, h_E, E_idx, h_V, use_node=self.use_node)
        logits = self.head(feats)[0]                                   # [L, K, 4]
        p_hb = torch.sigmoid(logits[..., HBOND])                       # [L, K]
        p_dir = torch.softmax(logits[..., DONOR:ACCEPTOR + 1], dim=-1)
        p_dn, p_ac = p_dir[..., 0], p_dir[..., 1]                      # [L, K]

        E = E_idx[0]
        L, K = E.shape
        rev_k, has_rev = reverse_edge_index(E)                         # [L, K]
        rows = torch.arange(L, device=E.device)[:, None].expand(L, K)
        keep = has_rev & (E != rows) & (rows < E)                     # each reciprocal pair once, i<j
        ii, kk = torch.where(keep)
        jj, rk = E[ii, kk], rev_k[ii, kk]

        p_ij, p_ji = p_hb[ii, kk], p_hb[jj, rk]
        combine = {
            "mean": lambda a, b: 0.5 * (a + b),
            "max": torch.maximum,
            "min": torch.minimum,
            "noisy_or": lambda a, b: 1.0 - (1.0 - a) * (1.0 - b),
        }[reduce]
        p_comb = combine(p_ij, p_ji)
        # P(i is donor) from both edges: p_donor(i->j) and p_acceptor(j->i) both estimate it.
        p_i_donor = 0.5 * (p_dn[ii, kk] + p_ac[jj, rk])

        sel = p_comb >= hbond_threshold
        out = []
        for i, j, pid in zip(ii[sel].tolist(), jj[sel].tolist(), p_i_donor[sel].tolist()):
            if pid >= 0.5:
                out.append([i, j, "donor", "acceptor"])
            else:
                out.append([i, j, "acceptor", "donor"])
        return out

    @staticmethod
    @torch.no_grad()
    def count_ph_bonds_from_pred(pred: dict, S, residue_mask, idx_to_token, *,
                                 threshold: float = 0.5, chemistry: str = "annotate",
                                 reduce: str = "mean") -> dict:
        """Count predicted H-bonds + salt-bridges for ONE structure and group them by protonation state.

        Pure function over a single example's ``predict_hbond_graph`` output — no model/forward needed
        (so it is directly unit-testable on a hand-built ``pred``). It reciprocal-combines the directed
        per-edge probabilities into physical bonds exactly like :meth:`predict_hbonds`
        (``presence = reduce(p_ij, p_ji)``, ``P(i donor) = mean(p_donor(i→j), p_acceptor(j→i))``), maps
        each bond's endpoints to their protonation-state token names, and tags each bond ``chem_ok``
        against ``BOND_CHEMISTRY``:
          * H-bond  → donor endpoint must ``can_donate`` AND acceptor endpoint must ``can_accept``.
          * salt-bridge → the two endpoints must have OPPOSITE formal charge (``charge_i*charge_j < 0``).

        ``chemistry="annotate"`` (default) keeps every predicted bond and only flags validity;
        ``chemistry="filter"`` drops chemically-impossible ones.

        Args:
            pred: one example's ``{p_hbond, p_donor, p_acceptor, p_saltbridge, E_idx}`` — each ``[L,K]``
                (a leading singleton batch dim is accepted and squeezed).
            S: ``[L]`` token sequence the prediction was made for (protonation microstates).
            residue_mask: ``[L]`` bool/0-1 — padded/invalid residues are excluded (both endpoints gated).
            idx_to_token: array mapping token idx → name (``encoding.idx_to_token``).
            threshold: keep a bond when its reciprocal-combined probability ``>= threshold``.
            reduce: how to combine the two directions — ``"mean"``|``"max"``|``"min"``|``"noisy_or"``.

        Returns:
            ``{"counts": {state: {hbond_as_donor, hbond_as_acceptor, saltbridge, chem_impossible}},
               "bonds":  [{i, j, state_i, state_j, type, donor, acceptor, chem_ok}]}``
            where ``state`` ranges over :data:`TITRATABLE_STATES`. Counts are per participating endpoint
            (a bond increments both of its titratable endpoints). NOTE: side-chain↔side-chain only
            (the head is trained without backbone bonds); non-reciprocal directed edges (~16%) are
            dropped, like :meth:`predict_hbonds`.
        """
        assert chemistry in ("filter", "annotate")

        # squeeze an optional leading batch dim on every tensor
        p_hbond = pred["p_hbond"]; p_donor = pred["p_donor"]
        p_acceptor = pred["p_acceptor"]; p_salt = pred["p_saltbridge"]; E = pred["E_idx"]
        if E.dim() == 3:
            p_hbond, p_donor, p_acceptor, p_salt, E = (t[0] for t in (p_hbond, p_donor, p_acceptor, p_salt, E))
        if S.dim() == 2:
            S = S[0]
        rm = (residue_mask[0] if residue_mask.dim() == 2 else residue_mask).bool()

        L, K = E.shape
        rev_k, has_rev = reverse_edge_index(E)                         # [L, K]
        rows = torch.arange(L, device=E.device)[:, None].expand(L, K)
        # each reciprocal pair once (i<j), drop self-edge (E==rows) and padded residues (both endpoints)
        keep = has_rev & (E != rows) & (rows < E) & rm[rows] & rm[E]
        ii, kk = torch.where(keep)
        jj, rk = E[ii, kk], rev_k[ii, kk]

        combine = {
            "mean": lambda a, b: 0.5 * (a + b),
            "max": torch.maximum,
            "min": torch.minimum,
            "noisy_or": lambda a, b: 1.0 - (1.0 - a) * (1.0 - b),
        }[reduce]
        p_hb = combine(p_hbond[ii, kk], p_hbond[jj, rk])
        p_sb = combine(p_salt[ii, kk], p_salt[jj, rk])
        p_i_donor = 0.5 * (p_donor[ii, kk] + p_acceptor[jj, rk])       # P(i is the donor)

        is_hb = (p_hb >= threshold).tolist()
        is_sb = (p_sb >= threshold).tolist()
        i_is_donor = (p_i_donor >= 0.5).tolist()
        ii_l, jj_l, S_l = ii.tolist(), jj.tolist(), S.tolist()

        counts = {s: {"hbond_as_donor": 0, "hbond_as_acceptor": 0, "saltbridge": 0, "chem_impossible": 0}
                  for s in TITRATABLE_STATES}
        bonds: list[dict] = []

        def _bump(name, key, ok):
            if name in counts:
                counts[name][key] += 1
                if not ok:
                    counts[name]["chem_impossible"] += 1

        for idx in range(len(ii_l)):
            i, j = ii_l[idx], jj_l[idx]
            name_i, name_j = str(idx_to_token[S_l[i]]), str(idx_to_token[S_l[j]])
            if is_hb[idx]:
                if i_is_donor[idx]:
                    dn, ac, dn_name, ac_name = i, j, name_i, name_j
                else:
                    dn, ac, dn_name, ac_name = j, i, name_j, name_i
                ok = _capability(dn_name).can_donate and _capability(ac_name).can_accept
                if not (chemistry == "filter" and not ok):
                    bonds.append(dict(i=i, j=j, state_i=name_i, state_j=name_j, type="hbond",
                                      donor=dn, acceptor=ac, chem_ok=ok))
                    _bump(dn_name, "hbond_as_donor", ok)
                    _bump(ac_name, "hbond_as_acceptor", ok)
            if is_sb[idx]:
                ok = _capability(name_i).charge * _capability(name_j).charge < 0
                if not (chemistry == "filter" and not ok):
                    bonds.append(dict(i=i, j=j, state_i=name_i, state_j=name_j, type="saltbridge",
                                      donor=None, acceptor=None, chem_ok=ok))
                    _bump(name_i, "saltbridge", ok)
                    _bump(name_j, "saltbridge", ok)
        return {"counts": counts, "bonds": bonds}

    @torch.no_grad()
    def count_ph_bonds(self, network_input: dict, threshold: float = 0.5,
                       chemistry: str = "annotate", reduce: str = "mean") -> list[dict]:
        """Run the H-bond head and count predicted pH-sensitive H-bonds + salt-bridges per protonation
        state, for every structure in the batch.

        Thin wrapper: one :meth:`predict_hbond_graph` pass (frozen encoder + head, teacher-forced on the
        given protonation sequence ``S``), then :meth:`count_ph_bonds_from_pred` per example. See that
        method for the returned ``{"counts", "bonds"}`` schema and the ``chemistry`` / ``reduce`` knobs.

        Returns a list of length ``B`` (one dict per structure; a single dict for the usual B=1).
        """
        assert chemistry in ("filter", "annotate")
        pred = self.predict_hbond_graph(network_input)                # {E_idx, p_hbond, p_donor, ...}
        inp = network_input["input_features"]
        S = inp["S"]
        rmask = inp["residue_mask"] if "residue_mask" in inp else torch.ones_like(S)
        idx_to_token = self.potts.graph_featurization_module.TOKEN_ENCODING.idx_to_token
        B = pred["E_idx"].shape[0]
        out = []
        for b in range(B):
            pred_b = {k: pred[k][b] for k in ("p_hbond", "p_donor", "p_acceptor", "p_saltbridge", "E_idx")}
            out.append(self.count_ph_bonds_from_pred(
                pred_b, S[b], rmask[b], idx_to_token,
                threshold=threshold, chemistry=chemistry, reduce=reduce))
        return out

    @torch.no_grad()
    def protonation_switch_bonds(self, h_V, h_E, E_idx, S, hbond_threshold: float = 0.5,
                                 salt_threshold: float = 0.5, reduce: str = "mean") -> list[dict]:
        """Classify how each titratable residue's predicted bonds respond to its protonation switch.

        The switch-based **clash** metric, built on the same chemistry source as :meth:`count_ph_bonds`
        (:data:`BOND_CHEMISTRY`). For every titratable residue ``r`` (HIS / ASP / GLU) the head's
        predicted H-bonds + salt bridges involving ``r`` are read out in BOTH its neutral state
        (HID / ASP-P / GLU-P) and its charged state (HIS-P / ASP-D / GLU-D) — a pure token swap, no
        re-encode. A partner ``j`` is "bonded" in a state if the reciprocal-combined
        ``p_hbond >= hbond_threshold`` OR ``p_saltbridge >= salt_threshold`` there. The partner's charge
        ``q_j`` is its native microstate charge (from ``BOND_CHEMISTRY``); the sign of
        ``q_charged(r) * q_j`` says what ``r``'s charged state does to that contact::

            effect = "clash"      if q_charged*q_j > 0  -> like charges: the switch ACTIVELY REPELS
                     "saltbridge" if q_charged*q_j < 0  -> opposite charges: attraction lost on neutralisation
                     "neutral"    if q_j == 0           -> partner uncharged: only an H-bond toggles

        ``protonation_dependent`` = the bond is present in exactly one of the two states. The count of
        ``effect == "clash"`` protonation-dependent bonds is the "salt-bridge clash if switch" score.
        Proximity comes ONLY from predicted bonds, so a clash is counted only where a favourable bond is
        predicted in ``r``'s neutral state (the head is trained on favourable bonds — a like-charged pair
        is never predicted directly). Args mirror :meth:`predict_hbonds`; returns one row (dict) per
        (titratable residue, bonded partner).
        """
        if h_V.dim() == 3: h_V = h_V[0]
        if h_E.dim() == 4: h_E = h_E[0]
        if E_idx.dim() == 3: E_idx = E_idx[0]
        if S.dim() == 2: S = S[0]
        W = self.potts.W_s
        idx_to_token = self.potts.graph_featurization_module.TOKEN_ENCODING.idx_to_token
        E = E_idx; L, K = E.shape
        rev_k, has_rev = reverse_edge_index(E)
        charge = _charge_by_idx(idx_to_token, E.device)                # [V]; NaN = ambiguous/unknown
        q_native = charge[S]                                           # [L]
        combine = {"mean": lambda a, b: 0.5 * (a + b), "max": torch.maximum,
                   "min": torch.minimum, "noisy_or": lambda a, b: 1 - (1 - a) * (1 - b)}[reduce]

        # Per-family charged / neutral token indices from THIS model's vocab (not a hardcoded 32-token
        # table): the vocab's protonated/deprotonated maps give the two microstates, and BOND_CHEMISTRY
        # charge disambiguates charged (nonzero) vs neutral (0). First-match preserves the v4 order
        # (neutral HIS = HID); for v6 the neutral His is HIS-S. Reproduces the old module globals for v4.
        from mpnn.transforms.extended_vocab import get_vocab
        _name_to_idx = {str(t): i for i, t in enumerate(idx_to_token)}
        _vocab = get_vocab(self.extended_vocab if isinstance(self.extended_vocab, str) else "v4")
        family_by_idx, charged_idx, neutral_idx = {}, {}, {}
        for fam in ("HIS", "ASP", "GLU"):
            states = list(_vocab["aa_protonated"][fam]) + list(_vocab["aa_deprotonated"][fam])
            for nm in states + [fam]:                                  # + the bare parent token
                if nm in _name_to_idx and nm not in AMBIGUOUS_TOKENS:
                    family_by_idx[_name_to_idx[nm]] = fam
            for nm in states:
                if nm in _name_to_idx and nm in BOND_CHEMISTRY:
                    (charged_idx if BOND_CHEMISTRY[nm].charge != 0 else neutral_idx).setdefault(
                        fam, _name_to_idx[nm])

        def bonds_for_state(r: int, tok: int):
            """Reciprocal p_hbond, p_salt over r's K neighbours with r's token set to `tok`."""
            nbrs = E[r]                                                # [K]
            emb_r = W(torch.tensor(tok, device=E.device))             # [H]
            emb_nb = W(S[nbrs]); hv_nb = h_V[nbrs]; hv_r = h_V[r]      # [K,H],[K,H],[H]
            exp = lambda v: v.unsqueeze(0).expand(K, v.shape[-1])
            if self.use_node:
                f_out = torch.cat([exp(hv_r), exp(emb_r), h_E[r], emb_nb, hv_nb], -1)
                f_in = torch.cat([hv_nb, emb_nb, h_E[nbrs, rev_k[r]], exp(emb_r), exp(hv_r)], -1)
            else:
                f_out = torch.cat([exp(emb_r), h_E[r], emb_nb], -1)
                f_in = torch.cat([emb_nb, h_E[nbrs, rev_k[r]], exp(emb_r)], -1)
            lo, li = self.head(f_out), self.head(f_in)                # [K,4] each
            hv = has_rev[r]
            p_hb = torch.where(hv, combine(torch.sigmoid(lo[:, HBOND]), torch.sigmoid(li[:, HBOND])),
                               torch.sigmoid(lo[:, HBOND]))
            p_sb = torch.where(hv, combine(torch.sigmoid(lo[:, SALTBRIDGE]), torch.sigmoid(li[:, SALTBRIDGE])),
                               torch.sigmoid(lo[:, SALTBRIDGE]))
            return nbrs, p_hb, p_sb

        out: list[dict] = []
        for r in range(L):
            fam = family_by_idx.get(int(S[r]))
            if fam is None or fam not in charged_idx or fam not in neutral_idx:
                continue                                              # not titratable (or state missing)
            q_charged = float(charge[charged_idx[fam]])
            nbrs, pnh, pns = bonds_for_state(r, neutral_idx[fam])
            _, pch, pcs = bonds_for_state(r, charged_idx[fam])
            for k in range(K):
                j = int(nbrs[k])
                if j == r:
                    continue
                qj = float(q_native[j])
                if qj != qj:                                          # NaN -> ambiguous partner
                    continue
                pres_neu = bool(pnh[k] >= hbond_threshold or pns[k] >= salt_threshold)
                pres_chg = bool(pch[k] >= hbond_threshold or pcs[k] >= salt_threshold)
                if not (pres_neu or pres_chg):
                    continue
                pc = q_charged * qj
                out.append({
                    "res": r, "family": fam, "partner": j, "q_partner": qj,
                    "q_charged": q_charged, "charge_product_charged": pc,
                    "effect": "clash" if pc > 0 else ("saltbridge" if pc < 0 else "neutral"),
                    "present_neutral": pres_neu, "present_charged": pres_chg,
                    "protonation_dependent": pres_neu != pres_chg,
                    "p_hbond_neutral": float(pnh[k]), "p_salt_neutral": float(pns[k]),
                    "p_hbond_charged": float(pch[k]), "p_salt_charged": float(pcs[k]),
                })
        return out

    def forward(self, network_input: dict) -> dict:
        # Training path: same graph + features as gather_hbond_salt_bridge_graph (encoder is frozen /
        # no-grad), but gradients flow through the trainable head into the raw logits for the loss.
        graph = self.gather_hbond_salt_bridge_graph(network_input)
        logits = self.head(graph["edge_features"])   # [B, L, K, 4]
        return {"logits": logits, "E_idx": graph["E_idx"]}
