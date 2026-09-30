"""Linear H-bond / salt-bridge readout over PottsMPNN edge embeddings.

The PottsMPNN encoder is *sequence-agnostic*: ``h_V`` is initialised to zeros and
``h_E = W_e(E)`` is built only from backbone geometry (see
``mpnn.model.mpnn.ProteinMPNN.encode``). So for a fixed structure the edge
embedding ``h_E [L, K, H]`` and neighbour index ``E_idx [L, K]`` are constant, and
the only sequence-dependent inputs are the per-token embeddings ``W_s(S_i)``.

This module trains a small head over the ``<TOKEN | edge_embedding | TOKEN>``
representation::

    x_ik = concat( W_s(S_i), h_E[i, k], W_s(S_j) )     with j = E_idx[i, k]

predicting, for each directed k-NN edge ``i -> j``:

    [ hbond, donor, acceptor, saltbridge ]

where ``hbond`` / ``saltbridge`` are independent presence logits (sigmoid) and
``donor`` / ``acceptor`` are a 2-way direction softmax *from focal i's view*
(i donates to j vs. i accepts from j). Because the head reuses the model's own
``W_s`` embedding, predictions for any sequence — including protonation
microstates (HIS-P vs HID/HIE, ASP-P vs ASP-D, ...) — are obtained by swapping the
token embedding, *without re-running the encoder*. That is what makes counting
"protonation-dependent" H-bonds cheap.

Trained / consumed by ``scripts/11_train_hbond_head.py`` and usable from the
``05_*`` redesign scripts to assess designed-interface chemistry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpnn.model.layers.message_passing import gather_nodes
from mpnn.transforms.feature_aggregation.token_encodings import POTTS_MPNN_TOKEN_ENCODING

# Output channel order of the head logits.
HBOND, DONOR, ACCEPTOR, SALTBRIDGE = 0, 1, 2, 3
CHANNELS = ("hbond", "donor", "acceptor", "saltbridge")

# Tokens whose H-bond donor/acceptor role / protonation is unresolved — edges touching
# these are masked from the H-bond + direction loss (salt-bridge presence is geometry-
# only and keeps them). UNK is added at use-time from the encoding's unknown tokens.
AMBIGUOUS_TOKENS = ("HIS-A", "ASP-A", "GLU-A")

# ── Titratable protonation-switch mapping ────────────────────────────────────────────────────────
# The per-family neutral<->charged microstate indices used to test whether flipping a residue's
# protonation turns a predicted bond into a like-charge clash are derived FROM THE MODEL'S OWN VOCAB,
# inside ``HBondModel.protonation_switch_bonds`` (model/hbond_model.py) — the vocab's
# aa_protonated / aa_deprotonated maps give the two microstates and ``BOND_CHEMISTRY`` charge
# disambiguates charged from neutral.
#
# There used to be module-level ``TITRATABLE_SWITCH`` / ``FAMILY_BY_IDX`` / ``NEUTRAL_IDX`` /
# ``CHARGED_IDX`` globals here, built at IMPORT time from the 32-token POTTS_MPNN_TOKEN_ENCODING and
# listing neutral His as ("HID","HIE","HIS") with no HIS-S. They were imported by hbond_model.py and
# never read (the function shadows them with correct per-vocab locals), but under a v6 model every
# index in them pointed at the wrong token — two of the 32-token indices don't even exist in the
# 30-token set. Removed rather than made vocab-aware: the vocabulary is not known at import time, so
# there is no correct value for a module-level constant to hold.


class HBondHead(nn.Module):
    """Per-edge H-bond / salt-bridge head over the edge feature (see build_edge_features).

    With ``n_hidden > 0`` the head is an MLP with ``n_layers`` ReLU-activated hidden layers::

        input -> [Linear -> ReLU] x n_layers -> Linear(-> 4)

    e.g. ``n_layers=2`` gives ``Linear -> ReLU -> Linear -> ReLU -> Linear``. ``n_hidden=0``
    collapses to a single ``nn.Linear`` (pure-linear head).

    Args:
        hidden_dim: PottsMPNN hidden dimension H. Input dim is ``in_mult * hidden_dim``.
        n_hidden: hidden-layer width (0 => pure linear head).
        dropout: dropout after each hidden ReLU (0 => none).
        in_mult: number of concatenated H-blocks in the edge feature (5 for the
            node+token+edge+token+node representation).
        n_layers: number of hidden ReLU-activated Linear blocks when ``n_hidden > 0``.
    """

    def __init__(self, hidden_dim: int = 128, n_hidden: int = 0, dropout: float = 0.0,
                 in_mult: int = 5, n_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.in_mult = in_mult          # #concatenated H-blocks in the edge feature (see build_edge_features)
        self.n_layers = n_layers
        in_dim = in_mult * hidden_dim
        if n_hidden and n_hidden > 0:
            layers: list[nn.Module] = []
            d = in_dim
            for _ in range(max(1, n_layers)):
                layers.append(nn.Linear(d, n_hidden))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                d = n_hidden
            layers.append(nn.Linear(d, len(CHANNELS)))
            self.net: nn.Module = nn.Sequential(*layers)
        else:
            self.net = nn.Linear(in_dim, len(CHANNELS))

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """``edge_feats [..., in_mult*H]`` -> ``logits [..., 4]``."""
        return self.net(edge_feats)


def build_edge_features(
    W_s: nn.Embedding,
    S: torch.Tensor,
    h_E: torch.Tensor,
    E_idx: torch.Tensor,
    h_V: torch.Tensor | None = None,
    use_node: bool = True,
) -> torch.Tensor:
    """Build the per-directed-edge head inputs.

    ``use_node=True`` (default, 5H) — the ``<node | TOKEN | edge | TOKEN | node>`` feature::

        concat( h_V[i], W_s(S_i), h_E[i,k], W_s(S_j), h_V[j] )      # [B, L, K, 5H]

    ``use_node=False`` (legacy 3H) — ``concat(W_s(S_i), h_E[i,k], W_s(S_j))``.

    Both ``h_V`` and ``h_E`` come from the frozen, sequence-agnostic encoder (message passing
    over the backbone graph — no ``S``), so they are fixed per structure; only the ``W_s`` token
    blocks change with the sequence. ``h_V`` adds each residue's aggregated neighbourhood context.

    Args:
        W_s:   token embedding ``nn.Embedding(V, H)`` (``model.W_s``).
        S:     ``[L]`` / ``[B, L]`` integer token sequence.
        h_E:   ``[B, L, K, H]`` encoder edge embeddings.
        E_idx: ``[B, L, K]`` neighbour indices.
        h_V:   ``[B, L, H]`` encoder node embeddings (required when ``use_node``).
        use_node: include the node-embedding blocks (5H) vs. token+edge+token only (3H).
    """
    if S.dim() == 1:
        S = S.unsqueeze(0)
    B, L, K, H = h_E.shape
    emb = W_s(S)                                       # [B, L, H]
    emb_i = emb.unsqueeze(2).expand(B, L, K, H)        # token, focal i
    emb_j = gather_nodes(emb, E_idx)                   # token, neighbour j
    if not use_node:
        return torch.cat([emb_i, h_E, emb_j], dim=-1)  # [B, L, K, 3H] (legacy)
    if h_V is None:
        raise ValueError("build_edge_features(use_node=True) requires h_V.")
    hV_i = h_V.unsqueeze(2).expand(B, L, K, H)         # node context, focal i
    hV_j = gather_nodes(h_V, E_idx)                    # node context, neighbour j
    return torch.cat([hV_i, emb_i, h_E, emb_j, hV_j], dim=-1)   # [B, L, K, 5H]


@torch.no_grad()
def encode_edges(model, network_input: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encoder-only pass returning ``(h_E [1,L,K,H], E_idx [1,L,K], h_V [1,L,H])``.

    Mirrors ``PottsMPNN.run_potts_encoder`` but returns the raw edge embedding ``h_E`` and
    node embedding ``h_V`` (not the Potts ``etab``). Both are sequence-agnostic, so compute
    them once per structure and reuse across sequences / protonation states. Mutates
    ``network_input["input_features"]`` in place (adds the standard masks).
    """
    inp = network_input["input_features"]
    model.sample_and_construct_masks(inp)
    graph_features = model.graph_featurization(inp)
    encoder_features = model.encode(inp, graph_features)
    return encoder_features["h_E"], graph_features["E_idx"], encoder_features["h_V"]


@torch.no_grad()
def predict_bonds(
    head: HBondHead,
    W_s: nn.Embedding,
    S: torch.Tensor,
    h_E: torch.Tensor,
    E_idx: torch.Tensor,
    h_V: torch.Tensor,
) -> dict:
    """Per-directed-edge bond probabilities for sequence ``S``.

    Returns a dict of tensors shaped like ``E_idx`` (batch dim preserved):
        ``p_hbond``, ``p_donor``, ``p_acceptor``, ``p_saltbridge``, ``logits``.
    ``p_donor + p_acceptor == 1`` (direction softmax); interpret them only where
    ``p_hbond`` is high.
    """
    feats = build_edge_features(W_s, S, h_E, E_idx, h_V)
    logits = head(feats)                               # [B, L, K, 4]
    p_dir = torch.softmax(logits[..., DONOR:ACCEPTOR + 1], dim=-1)
    return {
        "p_hbond": torch.sigmoid(logits[..., HBOND]),
        "p_donor": p_dir[..., 0],
        "p_acceptor": p_dir[..., 1],
        "p_saltbridge": torch.sigmoid(logits[..., SALTBRIDGE]),
        "logits": logits,
    }


def bond_labels_from_partners(
    E_idx: torch.Tensor,
    donates_to: torch.Tensor,
    accepts_from: torch.Tensor,
    salt_partners: torch.Tensor,
) -> dict:
    """Expand the stored token-pair partner lists into per-edge labels aligned to
    the model's ``E_idx`` — one label per neighbour slot ``k`` (easy to slice into).

    The partner lists are produced by
    ``mpnn.transforms.bond_annotation.BuildBondEdgeLabels`` (pad value ``-1``). For
    every directed edge ``i -> j = E_idx[i,k]``::

        donor[i,k]    = j in donates_to[i]
        acceptor[i,k] = j in accepts_from[i]
        hbond[i,k]    = donor | acceptor
        salt[i,k]     = j in salt_partners[i]

    Args:
        E_idx:         ``[L, K]`` or ``[B, L, K]`` neighbour indices.
        donates_to:    ``[L, P]`` / ``[B, L, P]`` donor partner token idxs (pad -1).
        accepts_from:  ``[L, P]`` / ``[B, L, P]`` acceptor partner token idxs (pad -1).
        salt_partners: ``[L, Q]`` / ``[B, L, Q]`` salt partner token idxs (pad -1).

    Returns:
        dict of bool tensors ``donor``, ``acceptor``, ``hbond``, ``salt`` shaped like
        ``E_idx``. Padding ``-1`` never matches a valid neighbour index (``>= 0``).
    """
    squeeze = E_idx.dim() == 2
    if squeeze:
        E_idx = E_idx.unsqueeze(0)
        donates_to = donates_to.unsqueeze(0)
        accepts_from = accepts_from.unsqueeze(0)
        salt_partners = salt_partners.unsqueeze(0)

    j = E_idx.unsqueeze(-1)                                  # [B, L, K, 1]

    def _isin(partners):                                    # partners [B, L, P]
        return (j == partners.unsqueeze(2)).any(dim=-1)     # [B, L, K]

    donor = _isin(donates_to)
    acceptor = _isin(accepts_from)
    out = {
        "donor": donor,
        "acceptor": acceptor,
        "hbond": donor | acceptor,
        "salt": _isin(salt_partners),
    }
    if squeeze:
        out = {k: v[0] for k, v in out.items()}
    return out


def ambiguous_token_mask(S: torch.Tensor, encoding=POTTS_MPNN_TOKEN_ENCODING) -> torch.Tensor:
    """Per-token bool mask of ambiguous-protonation (``*-A``) and unknown tokens.

    These tokens have an unresolved donor/acceptor role, so edges that touch them are
    excluded from the H-bond and direction losses (see :func:`edge_masks`).
    """
    amb_tokens = list(AMBIGUOUS_TOKENS) + list(encoding.unknown_tokens)
    amb_idx = torch.tensor(
        [encoding.token_to_idx[t] for t in amb_tokens if t in encoding.token_to_idx],
        device=S.device, dtype=S.dtype,
    )
    return torch.isin(S, amb_idx)


def edge_masks(
    E_idx: torch.Tensor,
    residue_mask: torch.Tensor,
    amb_node: torch.Tensor,
    labels: dict,
) -> dict:
    """Per-edge training masks aligned to ``E_idx`` (shapes match ``E_idx``).

    - ``valid``    : non-self edge with both endpoints valid residues.
    - ``hbond``    : ``valid`` and neither endpoint ambiguous (``*-A`` / UNK).
    - ``salt``     : ``valid`` (salt-bridge presence is geometry-only — ``*-A`` His/Asp/Glu
                      are the actual salt-bridge participants, so they are kept).
    - ``dir``      : ``hbond`` mask ∧ edge is an H-bond ∧ exactly one of donor/acceptor
                      (drop the rare "both" case whose direction is undefined).

    All inputs may be ``[L, ...]`` or ``[B, L, ...]``.
    """
    squeeze = E_idx.dim() == 2
    if squeeze:
        E_idx = E_idx.unsqueeze(0)
        residue_mask = residue_mask.unsqueeze(0)
        amb_node = amb_node.unsqueeze(0)
        labels = {k: v.unsqueeze(0) for k, v in labels.items()}

    B, L, K = E_idx.shape
    rows = torch.arange(L, device=E_idx.device).view(1, L, 1).expand(B, L, K)
    j = E_idx
    rm = residue_mask.bool()
    valid = (j != rows) & rm.unsqueeze(-1) & gather_nodes(rm.unsqueeze(-1).float(), E_idx).squeeze(-1).bool()
    amb = amb_node.bool()
    amb_edge = amb.unsqueeze(-1) | gather_nodes(amb.unsqueeze(-1).float(), E_idx).squeeze(-1).bool()
    mask_hb = valid & ~amb_edge
    mask_salt = valid
    mask_dir = mask_hb & labels["hbond"] & (labels["donor"] ^ labels["acceptor"])
    out = {"valid": valid, "hbond": mask_hb, "salt": mask_salt, "dir": mask_dir}
    if squeeze:
        out = {k: v[0] for k, v in out.items()}
    return out


def _masked_bce(logit, target, mask, neg_per_pos=None):
    """Masked BCE-with-logits; optionally subsample negatives to ``neg_per_pos`` x positives."""
    m = mask.reshape(-1)
    z = logit.reshape(-1)[m]
    y = target.reshape(-1).float()[m]
    if z.numel() == 0:
        return None
    if neg_per_pos is not None:
        pos = y > 0.5
        n_pos = int(pos.sum())
        n_neg = int((~pos).sum())
        if 0 < n_pos < n_neg:
            neg_idx = torch.where(~pos)[0]
            keep = neg_idx[torch.randperm(n_neg, device=neg_idx.device)[: neg_per_pos * n_pos]]
            sel = torch.cat([torch.where(pos)[0], keep])
            z, y = z[sel], y[sel]
    return F.binary_cross_entropy_with_logits(z, y)


def hbond_head_loss(logits, labels, masks, neg_per_pos=None) -> tuple:
    """Joint head loss: BCE(hbond) + BCE(salt) + CE(donor/acceptor on H-bond edges).

    Args:
        logits: ``[..., 4]`` head output for the same edges as ``labels``/``masks``.
        labels: dict from :func:`bond_labels_from_partners`.
        masks:  dict from :func:`edge_masks`.
        neg_per_pos: negative:positive subsample ratio for the BCE terms (None = no
            subsampling — use for validation; an int e.g. 10 for training).

    Returns:
        ``(total_loss, parts)`` where ``parts`` is a dict of the scalar sub-losses
        (missing terms — e.g. no H-bond edges in the batch — are omitted).
    """
    parts = {}
    l_hb = _masked_bce(logits[..., HBOND], labels["hbond"], masks["hbond"], neg_per_pos)
    if l_hb is not None:
        parts["hbond"] = l_hb
    l_sb = _masked_bce(logits[..., SALTBRIDGE], labels["salt"], masks["salt"], neg_per_pos)
    if l_sb is not None:
        parts["salt"] = l_sb
    md = masks["dir"].reshape(-1)
    if md.any():
        dl = logits[..., DONOR:ACCEPTOR + 1].reshape(-1, 2)[md]
        dt = labels["acceptor"].reshape(-1).long()[md]    # 0 = donor, 1 = acceptor
        parts["dir"] = F.cross_entropy(dl, dt)
    total = sum(parts.values()) if parts else logits.sum() * 0.0
    return total, parts


def aggregate_undirected(
    p_edge: torch.Tensor,
    E_idx: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    reduce: str = "max",
) -> dict[tuple[int, int], float]:
    """Collapse a directed per-edge score ``[L, K]`` to unique undirected pairs.

    For each undirected pair ``{i, j}`` the up-to-two directed scores (``i->j`` and
    ``j->i``) are combined with ``reduce`` (``"max"`` or ``"mean"``). Self-edges
    (``k=0``), ``j==i`` and masked residues are skipped.

    Returns ``{(i, j): score}`` with ``i < j``.
    """
    p = p_edge.detach().cpu()
    eidx = E_idx.detach().cpu()
    if p.dim() == 3:
        p = p[0]
    if eidx.dim() == 3:
        eidx = eidx[0]
    L, K = p.shape
    rm = None if residue_mask is None else residue_mask.detach().cpu().bool().view(-1)

    acc: dict[tuple[int, int], list[float]] = {}
    for i in range(L):
        if rm is not None and not bool(rm[i]):
            continue
        for k in range(1, K):
            j = int(eidx[i, k])
            if j == i:
                continue
            if rm is not None and not bool(rm[j]):
                continue
            key = (i, j) if i < j else (j, i)
            acc.setdefault(key, []).append(float(p[i, k]))

    if reduce == "mean":
        return {k: float(sum(v) / len(v)) for k, v in acc.items()}
    return {k: float(max(v)) for k, v in acc.items()}


def count_predicted_hbonds(
    p_hbond: torch.Tensor,
    E_idx: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
    reduce: str = "max",
) -> dict:
    """Expected and thresholded H-bond counts over unique undirected pairs.

    Returns ``{"expected": float, "thresholded": int, "pair_probs": {...}}`` where
    ``expected`` sums the combined per-pair probability and ``thresholded`` counts
    pairs above ``threshold``.
    """
    pair_probs = aggregate_undirected(p_hbond, E_idx, residue_mask, reduce=reduce)
    vals = list(pair_probs.values())
    return {
        "expected": float(sum(vals)),
        "thresholded": int(sum(v >= threshold for v in vals)),
        "pair_probs": pair_probs,
    }


def reverse_edge_index(E_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each directed edge ``i -> j = E_idx[i,k]`` find the slot in row ``j``
    that points back to ``i`` (the reciprocal edge ``j -> i``).

    Same construction the Potts head uses in
    ``PottsMPNN.compute_potts_context`` to symmetrise the energy table. The k-NN
    graph is asymmetric (~16% of edges are non-reciprocal — see project memory),
    so ``has_reverse`` flags which edges actually have a partner.

    Args:
        E_idx: ``[L, K]`` or ``[1, L, K]`` neighbour indices.

    Returns:
        ``(reverse_k [L, K], has_reverse [L, K])`` — ``reverse_k[i,k]`` is the slot
        ``k'`` with ``E_idx[j, k'] == i``; ``has_reverse[i,k]`` is False when no
        such slot exists (and ``reverse_k`` is then meaningless).
    """
    eidx = E_idx[0] if E_idx.dim() == 3 else E_idx                 # [L, K]
    L, K = eidx.shape
    device = eidx.device
    src = torch.arange(L, device=device)[:, None].expand(L, K)     # i
    neighbor_lists = eidx[eidx]                                    # [L, K, K]: E_idx[j, :]
    reverse_matches = neighbor_lists == src.unsqueeze(-1)          # [L, K, K]
    has_reverse = reverse_matches.any(dim=-1)                      # [L, K]
    reverse_k = reverse_matches.long().argmax(dim=-1)             # [L, K]
    return reverse_k, has_reverse


def directional_agreement(
    pred: dict,
    E_idx: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
) -> dict:
    """Self-consistency of the head across reciprocal edges ``i->j`` / ``j->i``.

    A directionally coherent head should satisfy, for every reciprocal pair:
        ``p_donor(i->j) ≈ p_acceptor(j->i)``     (if i donates, j accepts)
        ``p_hbond(i->j) ≈ p_hbond(j->i)``        (presence is symmetric)
        ``p_saltbridge(i->j) ≈ p_saltbridge(j->i)``

    This compares the two directions WITHOUT using any labels, so it is a useful
    diagnostic on designed / unlabelled structures too.

    Args:
        pred: output of :func:`predict_bonds` (tensors shaped like ``E_idx``).
        E_idx: ``[L, K]`` or ``[1, L, K]`` neighbour indices.
        residue_mask: optional ``[L]`` validity mask.

    Returns:
        dict with the paired per-edge arrays (``donor_ij``, ``acceptor_ji``,
        ``hbond_ij``, ``hbond_ji``, ``salt_ij``, ``salt_ji``; each ``[M]`` over the
        ``M`` reciprocal directed edges) and summary scalars: ``r_donor_acceptor``
        (Pearson corr of ``donor(i->j)`` vs ``acceptor(j->i)``), ``r_hbond``,
        ``r_saltbridge``, and ``direction_consistency`` (fraction of mutually
        called H-bonds where ``donor(i->j) > 0.5`` matches ``acceptor(j->i) > 0.5``).
    """
    def _sq(t):
        return (t[0] if t.dim() == 3 else t).detach().cpu()

    p_donor = _sq(pred["p_donor"])
    p_acc = _sq(pred["p_acceptor"])
    p_hb = _sq(pred["p_hbond"])
    p_sb = _sq(pred["p_saltbridge"])
    eidx = (E_idx[0] if E_idx.dim() == 3 else E_idx).detach().cpu()
    L, K = eidx.shape

    reverse_k, has_reverse = reverse_edge_index(eidx)
    rows = torch.arange(L)[:, None].expand(L, K)
    j = eidx                                                       # [L, K]

    # Keep non-self reciprocal edges; drop the self-edge (k=0) and i==j.
    keep = has_reverse.clone()
    keep[:, 0] = False
    keep &= j != rows
    if residue_mask is not None:
        rm = residue_mask.detach().cpu().bool().view(-1)
        keep &= rm[rows] & rm[j]
    # Avoid double counting each undirected pair: keep the i<j orientation only.
    keep &= rows < j

    ii, kk = torch.where(keep)
    jj = j[ii, kk]
    rk = reverse_k[ii, kk]                                         # slot in row j -> i

    out = {
        "donor_ij": p_donor[ii, kk].numpy(),
        "acceptor_ji": p_acc[jj, rk].numpy(),
        "hbond_ij": p_hb[ii, kk].numpy(),
        "hbond_ji": p_hb[jj, rk].numpy(),
        "salt_ij": p_sb[ii, kk].numpy(),
        "salt_ji": p_sb[jj, rk].numpy(),
    }

    def _corr(a, b):
        if len(a) < 2 or a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))

    both_hb = (out["hbond_ij"] >= 0.5) & (out["hbond_ji"] >= 0.5)
    if both_hb.sum() > 0:
        consistency = float(
            ((out["donor_ij"][both_hb] > 0.5) == (out["acceptor_ji"][both_hb] > 0.5)).mean()
        )
    else:
        consistency = float("nan")

    out.update(
        n_reciprocal_pairs=int(len(ii)),
        r_donor_acceptor=_corr(out["donor_ij"], out["acceptor_ji"]),
        r_hbond=_corr(out["hbond_ij"], out["hbond_ji"]),
        r_saltbridge=_corr(out["salt_ij"], out["salt_ji"]),
        direction_consistency=consistency,
    )
    return out
