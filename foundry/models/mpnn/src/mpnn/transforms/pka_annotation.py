import contextlib
import io
import json
import pickle
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
import numpy as np
from atomworks.ml.transforms.base import Transform

NEIGHBOR_AA_ORDER = (
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
)

class RemoveHeteroAtoms(Transform):
    def forward(self, data):
        aa = data["atom_array"]
        data["atom_array"] = aa[~aa.hetero]
        return data


class CalculatePackingDensity(Transform):
    """
    Cell-list packing density for all residues.

    For each residue, queries every heavy atom against a cell list and counts
    all non-self heavy atoms within cell_size_A Å. Stored as per-atom
    annotation ``packing_density``.

    Optionally stores local amino-acid neighbour counts for each residue using
    residue anchor points: ``CB`` for standard residues and ``CA`` for glycine.
    These are saved as per-atom annotations ``neighbor_count_<aa>`` in a fixed
    order.
    """

    def __init__(
        self,
        cell_size_A: float = 8.0,
        save_neighbor_counts: bool = False,
        neighbor_residue_order: tuple[str, ...] = NEIGHBOR_AA_ORDER,
    ):
        self.cell_size_A = cell_size_A
        self.save_neighbor_counts = save_neighbor_counts
        self.neighbor_residue_order = tuple(neighbor_residue_order)

    def forward(self, data):
        aa = data["atom_array"]
        n_atoms = len(aa)
        if n_atoms == 0:
            raise ValueError("Coordinates must not be empty")
        packing = np.zeros(n_atoms, dtype=float)
        neighbor_count_arrays = None
        neighbor_residue_to_idx = None
        residue_anchor_coords = None
        residue_anchor_names = None
        residue_anchor_valid_idx = None
        residue_anchor_cell_list = None
        if self.save_neighbor_counts:
            neighbor_count_arrays = {
                res_name: np.zeros(n_atoms, dtype=np.int16)
                for res_name in self.neighbor_residue_order
            }
            neighbor_residue_to_idx = {
                res_name: idx for idx, res_name in enumerate(self.neighbor_residue_order)
            }

        cell_list = struc.CellList(aa.coord, cell_size=self.cell_size_A)
        res_starts = struc.get_residue_starts(aa)

        if self.save_neighbor_counts:
            residue_anchor_coords = np.full((len(res_starts), 3), np.nan, dtype=float)
            residue_anchor_names = np.empty(len(res_starts), dtype=object)

            for k, start in enumerate(res_starts):
                end = res_starts[k + 1] if k + 1 < len(res_starts) else n_atoms
                atoms = aa[start:end]
                res_name = str(atoms.res_name[0])
                residue_anchor_names[k] = res_name

                if res_name == "GLY":
                    anchor_mask = atoms.atom_name == "CA"
                else:
                    anchor_mask = atoms.atom_name == "CB"
                    if not np.any(anchor_mask):
                        anchor_mask = atoms.atom_name == "CA"

                if np.any(anchor_mask):
                    residue_anchor_coords[k] = atoms.coord[np.flatnonzero(anchor_mask)[0]]

            residue_anchor_valid_idx = np.flatnonzero(np.isfinite(residue_anchor_coords).all(axis=1))
            if len(residue_anchor_valid_idx) > 0:
                residue_anchor_cell_list = struc.CellList(
                    residue_anchor_coords[residue_anchor_valid_idx],
                    cell_size=self.cell_size_A,
                )

        for k, start in enumerate(res_starts):
            end = res_starts[k + 1] if k + 1 < len(res_starts) else n_atoms
            chain = aa.chain_id[start]
            resid = int(aa.res_id[start])

            nearby: set = set()
            for coord in aa.coord[start:end]:
                hits = cell_list.get_atoms(coord, radius=self.cell_size_A)
                nearby.update(hits[hits >= 0].tolist())

            hits = np.array(list(nearby), dtype=int)
            non_self_mask = ~((aa.res_id[hits] == resid) & (aa.chain_id[hits] == chain))
            count = int(np.sum(non_self_mask))
            packing[start:end] = float(count)

            if self.save_neighbor_counts:
                residue_neighbor_counts = np.zeros(len(self.neighbor_residue_order), dtype=np.int16)
                if residue_anchor_cell_list is not None and np.isfinite(residue_anchor_coords[k]).all():
                    neighbor_hits = residue_anchor_cell_list.get_atoms(
                        residue_anchor_coords[k], radius=self.cell_size_A
                    )
                    neighbor_hits = neighbor_hits[neighbor_hits >= 0]
                    neighbor_residue_indices = residue_anchor_valid_idx[neighbor_hits]
                    neighbor_residue_indices = neighbor_residue_indices[neighbor_residue_indices != k]

                    for neighbor_res_idx in np.unique(neighbor_residue_indices):
                        neighbor_res_name = residue_anchor_names[neighbor_res_idx]
                        if neighbor_res_name not in neighbor_residue_to_idx:
                            continue
                        residue_neighbor_counts[neighbor_residue_to_idx[neighbor_res_name]] += 1

                for neighbor_res_name, values in neighbor_count_arrays.items():
                    values[start:end] = residue_neighbor_counts[
                        neighbor_residue_to_idx[neighbor_res_name]
                    ]

        aa.set_annotation("packing_density", packing)
        if self.save_neighbor_counts:
            for neighbor_res_name, values in neighbor_count_arrays.items():
                aa.set_annotation(f"neighbor_count_{neighbor_res_name.lower()}", values)
            data.setdefault("log_dict", {})
            data["log_dict"]["neighbor_count_order"] = list(self.neighbor_residue_order)
            data["log_dict"]["neighbor_count_anchor"] = "CB, CA for GLY"
        data["atom_array"] = aa
        return data


class AnnotatePKA(Transform):
    """
    Run PROPKA on the atom array and attach per-atom ``pka`` annotation.

    Optionally applies a power-law sidechain-burial correction:
        pka_corrected = pka + his_sc_intercept
                        + his_sc_rasa_weight * (1 - mean_rasa_sc)^sc_exponent
    Set his_sc_intercept=0.0 and his_sc_rasa_weight=0.0 (default) to return
    raw PROPKA pKa in pka_corrected.
    """

    _BB = frozenset({"N", "CA", "C", "O", "OXT"})

    def __init__(
        self,
        his_sc_intercept: float = 0.0,
        his_sc_rasa_weight: float = 0.0,
        sc_exponent: float = 3.0,
    ):
        self.his_sc_intercept = his_sc_intercept
        self.his_sc_rasa_weight = his_sc_rasa_weight
        self.sc_exponent = sc_exponent

    def _correct_HIS(self, aa, pka_values: np.ndarray) -> np.ndarray:
        pka_corr = pka_values.copy()
        intercept = float(self.his_sc_intercept) if np.isfinite(self.his_sc_intercept) else 0.0
        weight = float(self.his_sc_rasa_weight) if np.isfinite(self.his_sc_rasa_weight) else 0.0
        if intercept == 0.0 and weight == 0.0:
            return pka_corr
        res_starts = struc.get_residue_starts(aa)
        for idx, start in enumerate(res_starts):
            end = res_starts[idx + 1] if idx + 1 < len(res_starts) else len(aa)
            if aa.res_name[start] != "HIS":
                continue
            atoms = aa[start:end]
            sc_mask = ~np.isin(atoms.atom_name, list(self._BB))
            valid = atoms.rasa[sc_mask & ~np.isnan(atoms.rasa)]
            shift = intercept
            if len(valid) > 0:
                shift += weight * (1.0 - float(np.mean(valid))) ** self.sc_exponent
            orig = pka_values[start:end]
            pka_corr[start:end] = np.where(~np.isnan(orig), orig + shift, orig)
        return pka_corr

    def forward(self, data):
        import propka.run  # imported here: only the pKa transforms need it, not the design path

        aa = data["atom_array"]
        pka_values = np.full(len(aa), np.nan, dtype=np.float32)

        try:
            buf = io.StringIO()
            pdb_file = pdb_io.PDBFile()
            pdb_io.set_structure(pdb_file, aa)
            pdb_file.write(buf)
            buf.seek(0)

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                mol = propka.run.single("structure.pdb", stream=buf, write_pka=False)

            pka_map = {}
            for conf_name, conf in mol.conformations.items():
                if conf_name == "AVR":
                    continue
                for group in conf.get_titratable_groups():
                    chain = group.atom.chain_id.strip()
                    res_num = int(group.atom.res_num)
                    pka_map[(chain, res_num)] = float(group.pka_value)

            pka_values = np.array(
                [pka_map.get((ch.strip(), int(rid)), np.nan) for ch, rid in zip(aa.chain_id, aa.res_id)],
                dtype=np.float32,
            )
        except (ZeroDivisionError, Exception):
            pass  # leave pka_values as NaN; downstream consumers treat NaN as missing

        aa.set_annotation("pka", pka_values)
        aa.set_annotation("pka_corrected", self._correct_HIS(aa, pka_values))
        data["atom_array"] = aa
        return data


class _ModelArtifactMixin:
    @staticmethod
    def _load_payload(model_path: str) -> dict:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found: {p}")
        with p.open("rb") as f:
            payload = pickle.load(f)
        if "feature_names" not in payload:
            raise ValueError(f"Invalid model artifact: missing 'feature_names' in {p}")

        has_legacy = ("scaler" in payload and "model" in payload)
        has_avg = ("coef_raw_mean" in payload and "intercept_raw_mean" in payload)
        if not (has_legacy or has_avg):
            raise ValueError(
                "Invalid model artifact: expected either legacy keys "
                "('scaler', 'model') or averaged keys "
                "('coef_raw_mean', 'intercept_raw_mean')."
            )
        return payload

    @staticmethod
    def _load_sidecar_meta(model_path: str) -> dict | None:
        p = Path(model_path)
        sidecar = p.with_name(f"{p.stem}_meta.json")
        if not sidecar.exists():
            return None
        try:
            with sidecar.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _coerce_feature_stats(stats_obj) -> dict[str, dict[str, float]]:
        if not isinstance(stats_obj, dict):
            return {}
        cleaned: dict[str, dict[str, float]] = {}
        for feat, dct in stats_obj.items():
            if not isinstance(feat, str) or not isinstance(dct, dict):
                continue
            item = {}
            for key in ("mean", "std", "min", "max", "p01", "p99"):
                val = dct.get(key, np.nan)
                try:
                    item[key] = float(val)
                except Exception:
                    item[key] = np.nan
            cleaned[feat] = item
        return cleaned

    def _load_feature_stats(self, payload: dict, sidecar_meta: dict | None) -> dict[str, dict[str, float]]:
        stats = self._coerce_feature_stats(payload.get("feature_stats", {}))
        if stats:
            return stats
        if isinstance(sidecar_meta, dict):
            return self._coerce_feature_stats(sidecar_meta.get("feature_stats", {}))
        return {}

    def _is_in_distribution(self, x: np.ndarray, feature_names: list[str]) -> bool:
        if not self.feature_stats:
            return True
        for val, feat in zip(x, feature_names):
            stats = self.feature_stats.get(feat)
            if stats is None:
                continue
            if not np.isfinite(val):
                return False
            mean = float(stats.get("mean", np.nan))
            std = float(stats.get("std", np.nan))
            lo = float(stats.get("p01", stats.get("min", np.nan)))
            hi = float(stats.get("p99", stats.get("max", np.nan)))

            if np.isfinite(lo) and val < lo:
                return False
            if np.isfinite(hi) and val > hi:
                return False
            if np.isfinite(mean) and np.isfinite(std) and std > 1e-12:
                z = abs((float(val) - mean) / std)
                if z > self.ood_z_thresh:
                    return False
        return True


class AnnotateCorrectedPKA(_ModelArtifactMixin, Transform):
    """
    Apply a saved sparse linear model to annotate ML-corrected HIS pKa.

    Expected artifact formats (pickle):

    1) Legacy sklearn payload:
      {
        "feature_names": [...],
        "scaler": fitted StandardScaler,
        "model": fitted Lasso/ElasticNet model,
        ...
      }

    2) Split-averaged linear payload:
      {
        "feature_names": [...],
        "coef_raw_mean": [...],
        "intercept_raw_mean": float,
        ...
      }

    Output per-atom annotations:
      - ``pka_corrected_ml``: corrected pKa from the ML model
    """

    _BB = frozenset({"N", "CA", "C", "O", "OXT"})
    _RING = frozenset({"ND1", "NE2"})

    def __init__(
        self,
        model_path: str,
        reject_ood: bool = False,
        ood_z_thresh: float = 4.0,
    ):
        self.model_path = str(model_path)
        payload = self._load_payload(self.model_path)
        sidecar_meta = self._load_sidecar_meta(self.model_path)
        self.feature_names = list(payload["feature_names"])
        self.mode = "legacy_sklearn"
        self.scaler = None
        self.model = None
        self.coef_raw = None
        self.intercept_raw = None
        self.feature_stats = self._load_feature_stats(payload, sidecar_meta)
        self.reject_ood = bool(reject_ood)
        try:
            z = float(ood_z_thresh)
        except Exception:
            z = np.nan
        self.ood_z_thresh = z if np.isfinite(z) and z > 0 else 4.0

        if "coef_raw_mean" in payload and "intercept_raw_mean" in payload:
            self.mode = "averaged_linear"
            self.coef_raw = np.asarray(payload["coef_raw_mean"], dtype=float)
            self.intercept_raw = float(payload["intercept_raw_mean"])
            if len(self.coef_raw) != len(self.feature_names):
                raise ValueError(
                    "Invalid averaged artifact: len(coef_raw_mean) does not match "
                    "len(feature_names)."
                )
        else:
            self.scaler = payload["scaler"]
            self.model = payload["model"]

        max_abs_delta = payload.get("max_abs_delta", None)
        if max_abs_delta is None:
            self.max_abs_delta = None
        else:
            try:
                v = float(max_abs_delta)
            except Exception:
                v = np.nan
            self.max_abs_delta = v if np.isfinite(v) and v > 0 else None

    @staticmethod
    def _safe_mean(arr: np.ndarray) -> float:
        valid = arr[np.isfinite(arr)]
        return float(np.mean(valid)) if len(valid) > 0 else np.nan

    @staticmethod
    def _safe_first(arr: np.ndarray) -> float:
        valid = arr[np.isfinite(arr)]
        return float(valid[0]) if len(valid) > 0 else np.nan

    @staticmethod
    def _count_positive(arr: np.ndarray | None, mask: np.ndarray | None = None):
        if arr is None:
            return np.nan
        vals = arr[mask] if mask is not None else arr
        return int(np.sum(vals > 0))

    def _extract_feature_map(self, aa, start: int, end: int) -> dict[str, float]:
        his = aa[start:end]
        cats = set(aa.get_annotation_categories())

        rasa = his.rasa if "rasa" in cats else np.full(len(his), np.nan, dtype=float)
        depth = (
            his.surface_depth
            if "surface_depth" in cats
            else np.full(len(his), np.nan, dtype=float)
        )
        active_donor = (
            his.active_donor
            if "active_donor" in cats
            else None
        )
        active_acceptor = (
            his.active_acceptor
            if "active_acceptor" in cats
            else None
        )
        active_positive = (
            his.active_positive
            if "active_positive" in cats
            else None
        )
        active_negative = (
            his.active_negative
            if "active_negative" in cats
            else None
        )
        packing = (
            his.packing_density
            if "packing_density" in cats
            else np.full(len(his), np.nan, dtype=float)
        )
        pka = (
            his.pka
            if "pka" in cats
            else np.full(len(his), np.nan, dtype=float)
        )

        atom_names = his.atom_name
        sc_mask = ~np.isin(atom_names, list(self._BB))
        ring_mask = np.isin(atom_names, list(self._RING))
        bb_mask = np.isin(atom_names, list(self._BB))

        rasa_mean = self._safe_mean(rasa)
        rasa_max = float(np.nanmax(rasa)) if np.isfinite(rasa).any() else np.nan
        rasa_sidechain = self._safe_mean(rasa[sc_mask])
        ring_rasa_mean = self._safe_mean(rasa[ring_mask])
        backbone_rasa_mean = self._safe_mean(rasa[bb_mask])

        depth_mean = self._safe_mean(depth)
        depth_sidechain = self._safe_mean(depth[sc_mask])
        ring_depth_mean = self._safe_mean(depth[ring_mask])
        backbone_depth_mean = self._safe_mean(depth[bb_mask])

        hbond_donor = self._count_positive(active_donor)
        hbond_acceptor = self._count_positive(active_acceptor)
        hbond_total = (
            hbond_donor + hbond_acceptor
            if np.isfinite(hbond_donor) and np.isfinite(hbond_acceptor)
            else np.nan
        )
        ring_n_hbond_donor = self._count_positive(active_donor, ring_mask)
        ring_n_hbond_acceptor = self._count_positive(active_acceptor, ring_mask)

        charged_pos = self._count_positive(active_positive)
        charged_neg = self._count_positive(active_negative)
        charged_total = (
            charged_pos + charged_neg
            if np.isfinite(charged_pos) and np.isfinite(charged_neg)
            else np.nan
        )
        ring_n_charged = self._count_positive(active_positive, ring_mask)

        packing_density_8A = self._safe_first(packing)
        pipeline_pka = self._safe_first(pka)

        return {
            "pipeline_pka": pipeline_pka,
            "rasa_mean": rasa_mean,
            "rasa_max": rasa_max,
            "rasa_sidechain": rasa_sidechain,
            "ring_rasa_mean": ring_rasa_mean,
            "backbone_rasa_mean": backbone_rasa_mean,
            "depth_mean": depth_mean,
            "depth_sidechain": depth_sidechain,
            "ring_depth_mean": ring_depth_mean,
            "backbone_depth_mean": backbone_depth_mean,
            "hbond_donor": hbond_donor,
            "hbond_acceptor": hbond_acceptor,
            "hbond_total": hbond_total,
            "ring_n_hbond_donor": ring_n_hbond_donor,
            "ring_n_hbond_acceptor": ring_n_hbond_acceptor,
            "ring_n_charged": ring_n_charged,
            "charged_pos": charged_pos,
            "charged_total": charged_total,
            "packing_density_8A": packing_density_8A,
        }

    def forward(self, data: dict) -> dict:
        aa = data["atom_array"]
        n_atoms = len(aa)

        pka_corr_ml = np.full(n_atoms, np.nan, dtype=float)

        res_starts = struc.get_residue_starts(aa)
        for idx, start in enumerate(res_starts):
            end = res_starts[idx + 1] if idx + 1 < len(res_starts) else n_atoms
            if aa.res_name[start] != "HIS":
                continue

            fmap = self._extract_feature_map(aa, start, end)
            x = np.array([fmap.get(name, np.nan) for name in self.feature_names], dtype=float)
            pka_raw = fmap.get("pipeline_pka", np.nan)

            if np.isfinite(x).all():
                in_domain = self._is_in_distribution(x, self.feature_names)
                if self.reject_ood and not in_domain:
                    pka_hat = pka_raw
                else:
                    if self.mode == "averaged_linear":
                        delta_hat = float(np.dot(self.coef_raw, x) + self.intercept_raw)
                    else:
                        x_scaled = self.scaler.transform(x.reshape(1, -1))
                        delta_hat = float(self.model.predict(x_scaled)[0])
                    if self.max_abs_delta is not None:
                        delta_hat = float(np.clip(delta_hat, -self.max_abs_delta, self.max_abs_delta))
                    pka_hat = pka_raw - delta_hat if np.isfinite(pka_raw) else pka_raw
            else:
                pka_hat = pka_raw
            pka_corr_ml[start:end] = pka_hat

        aa.set_annotation("pka_corrected_ml", pka_corr_ml)
        data["atom_array"] = aa
        return data
