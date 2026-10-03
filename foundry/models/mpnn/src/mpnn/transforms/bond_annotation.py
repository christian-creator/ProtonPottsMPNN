import os
import string
import subprocess
import tempfile
from datetime import datetime
from typing import Any, Tuple

import numpy as np
import biotite.structure as struc
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

import ipdb
# ---------------------------------------------------------------------------
# Salt-bridge detection (PLIP-compatible, no hydrogens required)
# ---------------------------------------------------------------------------

SALTBRIDGE_DIST_MAX = 5.5
MIN_DIST = 0.5

PLIP_POSITIVE_ATOMS = {
    "ARG": {"NE", "NH1", "NH2"},
    "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"},
}
PLIP_NEGATIVE_ATOMS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}

# Backbone (main-chain) H-bonding atoms: the amide N is the only backbone DONOR and the carbonyl O
# (plus the C-terminal OXT) the only backbone ACCEPTOR. An H-bond is "side-chain" on a given side iff
# that side's atom is NOT one of these. BuildBondEdgeLabels(sidechain_only=True) keeps only bonds that
# are side-chain on BOTH endpoints, dropping backbone and side-chain<->backbone H-bonds so the head
# learns side-chain / protonation-dependent chemistry only (consistent with PottsMPNN protonation).
BACKBONE_DONOR_ATOMS = {"N"}
BACKBONE_ACCEPTOR_ATOMS = {"O", "OXT"}


def annotate_salt_bridges(
    atom_array: AtomArray,
    dist_max: float = SALTBRIDGE_DIST_MAX,
    min_dist: float = MIN_DIST,
) -> Tuple[AtomArray, list, int]:
    """
    Find salt bridges using PLIP distance criteria and charge-centre atoms.

    Returns (atom_array, pairs, n_pairs).
    Does NOT set annotations — that is done in the Transform.forward().
    """
    pos_mask = np.zeros(len(atom_array), dtype=bool)
    neg_mask = np.zeros(len(atom_array), dtype=bool)

    for i in range(len(atom_array)):
        res  = atom_array.res_name[i]
        name = atom_array.atom_name[i]
        if res in PLIP_POSITIVE_ATOMS and name in PLIP_POSITIVE_ATOMS[res]:
            pos_mask[i] = True
        elif res in PLIP_NEGATIVE_ATOMS and name in PLIP_NEGATIVE_ATOMS[res]:
            neg_mask[i] = True

    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]

    pairs = []
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        cell_list = struc.CellList(atom_array.coord[neg_mask], cell_size=dist_max)
        for i in pos_idx:
            neighbours = cell_list.get_atoms(atom_array.coord[i], radius=dist_max)
            for j in neighbours[neighbours >= 0]:
                dist = np.linalg.norm(atom_array.coord[i] - atom_array.coord[neg_idx[j]])
                if dist >= min_dist:
                    pairs.append((int(i), int(neg_idx[j])))

    return atom_array, pairs, len(pairs)


class AnnotateSaltBridges(Transform):
    """
    Transform that detects salt bridges using PLIP charge-centre criteria.

    Uses SALTBRIDGE_DIST_MAX = 5.5 Å and MIN_DIST = 0.5 Å, matching PLIP's
    detector. Sets two per-atom annotations:
        active_positive  — 1 if the atom acts as the positively charged partner
        active_negative  — 1 if the atom acts as the negatively charged partner

    Logs ``salt_bridge_count`` in data["log_dict"].
    """

    def __init__(self, dist_max: float = SALTBRIDGE_DIST_MAX, min_dist: float = MIN_DIST):
        self.dist_max = dist_max
        self.min_dist = min_dist

    def forward(self, data: dict) -> dict:
        atom_array: AtomArray = data["atom_array"]

        atom_array, pairs, n_pairs = annotate_salt_bridges(
            atom_array, dist_max=self.dist_max, min_dist=self.min_dist
        )

        active_positive = np.zeros(len(atom_array), dtype=float)
        active_negative = np.zeros(len(atom_array), dtype=float)
        for pos_i, neg_i in pairs:
            active_positive[pos_i] = 1.0
            active_negative[neg_i] = 1.0

        atom_array.set_annotation("active_positive", active_positive)
        atom_array.set_annotation("active_negative", active_negative)

        # Persist residue-level salt-bridge pairs (keyed by chain_iid + res_id, which
        # are stable through atomization / occupancy filtering) so BuildBondEdgeLabels
        # can build E_idx-alignable token-pair labels. Stored symmetric below.
        data["salt_pairs"] = [
            (
                (atom_array.chain_iid[pi], int(atom_array.res_id[pi])),
                (atom_array.chain_iid[nj], int(atom_array.res_id[nj])),
            )
            for pi, nj in pairs
        ]

        data.setdefault("log_dict", {})
        data["log_dict"]["salt_bridge_count"] = n_pairs
        data["atom_array"] = atom_array

        return data




def save_atomarray_to_pdb(atom_array, output_path):
    def _handle_nan_coords(atom_array, noise_level=1e-3):
        coords = atom_array.coord
        nan_mask = np.isnan(coords)
        coords[nan_mask] = np.random.uniform(
            -noise_level, noise_level, size=nan_mask.sum()
        )
        atom_array.coord = coords
        return atom_array, nan_mask

    atom_array, nan_mask = _handle_nan_coords(atom_array)

    chain_iids = np.unique(atom_array.chain_iid)
    if len(chain_iids) > 52:
        raise ValueError(
            "Too many chain_iids, cannot convert to PDB", "skipping HBPLUS"
        )

    all_possible_chainIDS = string.ascii_letters
    chain_map = {}
    for item in chain_iids:
        if len(item) == 1:
            chain_map[item] = item
            all_possible_chainIDS = all_possible_chainIDS.replace(item, "")
    for item in chain_iids:
        if len(item) > 1:
            chain_map[item] = all_possible_chainIDS[0]
            all_possible_chainIDS = all_possible_chainIDS.replace(chain_map[item], "")

    new_chain_ids = [chain_map[i] for i in atom_array.chain_iid]
    inverted_chain_map = {v: k for k, v in chain_map.items()}
    atom_array.chain_id = new_chain_ids
    atom_array.b_factor = np.zeros(len(atom_array))

    pdb = PDBFile()
    pdb.set_structure(atom_array)
    pdb.write(output_path)

    return atom_array, nan_mask, inverted_chain_map


def check_atom_array_has_hydrogen(data: dict[str, Any]):
    if not np.any(data["atom_array"].element == "H"):
        raise ValueError("Key `atom_array` in data has no hydrogens.")


def calculate_hbonds(
    atom_array: AtomArray,
    # HBPLUS's own defaults are D-A 3.9, H-A 2.5, DHA >= 90 deg. We tighten D-A to 3.5 but must NOT
    # loosen H-A: at -h 3.0 the heavy atoms are forced close while the hydrogen is free to point
    # sideways, which admits ~18% more "bonds" whose median DHA angle is 116 deg (vs 158 deg for the
    # ones -h 2.5 keeps; 62% of them fall below 120 deg). Those near-perpendicular contacts are not
    # H-bonds, and they fed the protonation classifier fake donor/acceptor roles -- e.g. Arg B234
    # NH2 -> Glu B136 OE2 at 3.11 A / 95.5 deg in Epha2_pH_4, which pinned a GLU-D on no real bond.
    cutoff_HA_dist: float = 2.5,
    cutoff_DA_distance: float = 3.5,
    motif_interface_only: bool = True,
    od1_oe1: bool = False,
    od2_oe2: bool = False,
    filter_capability: bool = False,
) -> Tuple[np.ndarray, np.ndarray, AtomArray]:
    def _run_hbplus_cmd(hbplus_cmd, pdb_path, chain_map, tmpdir, mode_name):
        subprocess.run(
            hbplus_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=tmpdir,
            check=True,
        )

        hb2_path = pdb_path.replace(".pdb", ".hb2")
        with open(hb2_path, "r") as hb_file:
            hb_lines = hb_file.readlines()

        hbonds = []
        for i in range(8, len(hb_lines)):
            d_chain = hb_lines[i][0]
            d_resi = str(int(hb_lines[i][1:5].strip()))
            d_resn = hb_lines[i][6:9].strip()
            d_ins = hb_lines[i][5].replace("-", " ")
            d_atom = hb_lines[i][9:13].strip()
            a_chain = hb_lines[i][14]
            a_resi = str(int(hb_lines[i][15:19].strip()))
            a_ins = hb_lines[i][19].replace("-", " ")
            a_resn = hb_lines[i][20:23].strip()
            a_atom = hb_lines[i][23:27].strip()
            # .hb2 is fixed-width FORTRAN; parse by column, NOT .split(): 1-char
            # nucleic-acid resnames (" G"/" C") leave spaces inside the donor/
            # acceptor id, so .split() mis-tokenises ~3% of lines (all nucleotide/
            # ligand contacts) and shifts every numeric index. Columns (0-based):
            #   [27:32] = donor->acceptor distance     [46:51] = D-H..A angle (deg)
            #   [52:57] = H..acceptor distance         (the sharpest geometric term)
            dist = float(hb_lines[i][27:32].strip())
            try:
                dha_angle = float(hb_lines[i][46:51].strip())
                if dha_angle < 0:           # HBPLUS uses -1.0 for "angle undefined"
                    dha_angle = np.nan
            except (ValueError, IndexError):
                dha_angle = np.nan
            # H..A distance. Additive: v3/v4 read the annotations, not this list, so carrying an extra
            # key is inert for them; downstream feature models (EV6) aggregate ha_min per functional atom.
            try:
                ha_dist = float(hb_lines[i][52:57].strip())
                if ha_dist < 0:             # HBPLUS uses -1.0 for "undefined"
                    ha_dist = np.nan
            except (ValueError, IndexError):
                ha_dist = np.nan

            hbonds.append(
                {
                    "d_chain": chain_map[d_chain],
                    "d_resi": d_resi,
                    "d_resn": d_resn,
                    "d_ins": d_ins,
                    "d_atom": d_atom,
                    "a_chain": chain_map[a_chain],
                    "a_resi": a_resi,
                    "a_resn": a_resn,
                    "a_ins": a_ins,
                    "a_atom": a_atom,
                    "dist": dist,
                    "dha_angle": dha_angle,
                    "ha_dist": ha_dist,
                    "hbplus_modes": mode_name,
                }
            )
        return hbonds

    def _merge_hbonds(*hbonds_lists):
        merged = {}
        for hbonds in hbonds_lists:
            for item in hbonds:
                key = (
                    item["d_chain"],
                    item["d_resi"],
                    item["d_resn"],
                    item["d_ins"],
                    item["d_atom"],
                    item["a_chain"],
                    item["a_resi"],
                    item["a_resn"],
                    item["a_ins"],
                    item["a_atom"],
                )
                if key in merged:
                    modes = set(merged[key]["hbplus_modes"].split(","))
                    modes.update(item["hbplus_modes"].split(","))
                    merged[key]["hbplus_modes"] = ",".join(sorted(modes))
                else:
                    merged[key] = dict(item)
        return list(merged.values())

    hbplus_exe = os.environ.get("HBPLUS_PATH")

    if not hbplus_exe:
        raise ValueError(
            "HBPLUS_PATH environment variable not set. "
            "Please set it to the path of the hbplus executable in order to calculate hydrogen bonds."
        )

    hbplus_exe = os.path.abspath(hbplus_exe)
    if not os.path.isfile(hbplus_exe):
        raise ValueError(
            f"HBPLUS_PATH is set to {hbplus_exe!r}, which is not a file. "
            "Please point it at the hbplus executable."
        )


    # save_atomarray_to_pdb remaps chain_id based on chain_iid (needed to write
    # a valid PDB file for hbplus), but we must not let that mutation leak into
    # the atom_array that the rest of the pipeline sees.
    original_chain_ids = atom_array.chain_id.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        dtstr = datetime.now().strftime("%Y%m%d%H%M%S")
        pdb_filename = f"{dtstr}_{np.random.randint(10000)}.pdb"
        pdb_path = os.path.join(tmpdir, pdb_filename)
        atom_array, _, chain_map = save_atomarray_to_pdb(atom_array, pdb_path)

        hbplus_cmd = [
            hbplus_exe,
            "-h",
            str(cutoff_HA_dist),
            "-d",
            str(cutoff_DA_distance),
        ]
        if od1_oe1:
            hbplus_cmd.extend(
                ["-E", "ASP", " OD1", "1", "-E", "GLU", " OE1", "1"]
            )
        if od2_oe2:
            hbplus_cmd.extend(
                ["-E", "ASP", " OD2", "1", "-E", "GLU", " OE2", "1"]
            )
        hbplus_cmd.extend([pdb_path, pdb_path])

        if od1_oe1 or od2_oe2:
            default_hbplus_cmd = [
                hbplus_exe,
                "-h",
                str(cutoff_HA_dist),
                "-d",
                str(cutoff_DA_distance),
                pdb_path,
                pdb_path,
            ]
            hbonds_default = _run_hbplus_cmd(
                default_hbplus_cmd,
                pdb_path,
                chain_map,
                tmpdir,
                "default",
            )
            hbonds_override = _run_hbplus_cmd(
                hbplus_cmd,
                pdb_path,
                chain_map,
                tmpdir,
                "override",
            )
            hbonds = _merge_hbonds(hbonds_default, hbonds_override)
        else:
            hbonds = _run_hbplus_cmd(
                hbplus_cmd,
                pdb_path,
                chain_map,
                tmpdir,
                "default",
            )

    # Restore the original chain_id — hbond matching below uses chain_iid, so
    # this is safe and prevents the remapped IDs from leaking downstream.
    atom_array.chain_id = original_chain_ids

    if filter_capability:
        # Drop a-priori-impossible directed bonds (acceptor can't accept / donor can't donate),
        # e.g. the carboxyl-as-donor override's "carboxyl O -> Lys NZ" (Lys is donor-only). Lazy
        # import avoids a bond_annotation <-> charge_network import cycle.
        from mpnn.transforms.charge_network import capability_ok
        hbonds = [h for h in hbonds if capability_ok(h)]

    donor_array = np.zeros(len(atom_array))
    acceptor_array = np.zeros(len(atom_array))
    donor_mask = np.bool_(donor_array)
    acceptor_mask = np.bool_(acceptor_array)

    # Per-atom geometry of the best (shortest) H-bond in each role. NaN = the atom
    # plays no such role (so isnan() == "not a donor/acceptor"). We store ALL the
    # evidence from BOTH HBPLUS passes and make NO protonation decision here — that
    # lives in AnnotateProtonationStates. In particular a His that "accepts" only
    # because of the carboxyl-as-donor override pass IS recorded; vocab decides
    # whether to treat that as a salt bridge (-> HIS-P) using the active_positive
    # flag. (Override masking a real Ser/Thr/Tyr acceptor on the same N was measured
    # at ~0.2% of His-N and 0 Ser/Thr/Tyr cases, so a single best-acceptor field is
    # fine.) active_donor/active_acceptor booleans below stay merged across passes.
    n_atoms = len(atom_array)
    active_donor_dist = np.full(n_atoms, np.nan)
    active_donor_angle = np.full(n_atoms, np.nan)
    active_donor_ha = np.full(n_atoms, np.nan)
    active_acceptor_dist = np.full(n_atoms, np.nan)
    active_acceptor_angle = np.full(n_atoms, np.nan)
    active_acceptor_ha = np.full(n_atoms, np.nan)
    # ALL H-bond partners this atom DONATES to, as "RESN:ATOM" tokens (raw facts; only
    # the BEST donor bond's geometry is kept above). The full list lets vocab look up,
    # e.g., whether a carboxyl donates to ANOTHER carboxylate (ASP/GLU OD/OE) -> shared-
    # proton pair where THIS (donor) carboxyl carries the proton -> protonated. The
    # acceptor end is NOT forced protonated: a carboxylate accepting an H-bond is the
    # dyad case (COOH..COO-, e.g. HIV protease / pepsin), so it stays COO-/ambiguous.
    # No chemical categorisation / label decision is made here.
    donor_partners: dict = {}
    # ALL H-bond partners that DONATE INTO this atom, as "RESN:ATOM" tokens. Mirror of
    # donor_partners. Lets vocab apply the capability rule: if the group donating into a
    # carboxyl O cannot itself accept (Lys NZ, Arg NE/NH*, per BOND_CHEMISTRY), the bond
    # direction is forced, so the O is unambiguously the acceptor -- and a *cation* donor
    # additionally means the carboxyl is COO- (a salt-bridging H-bond). This is stronger
    # than a bare distance-sphere salt-bridge test, which can fire on an unrelated cation.
    acceptor_partners: dict = {}

    selected_hbonds = []
    for item in hbonds:
        current_donor_mask = (
            (atom_array.chain_iid == item["d_chain"])
            & (atom_array.res_id == float(item["d_resi"]))
            & (atom_array.atom_name == item["d_atom"])
        )
        current_acceptor_mask = (
            (atom_array.chain_iid == item["a_chain"])
            & (atom_array.res_id == float(item["a_resi"]))
            & (atom_array.atom_name == item["a_atom"])
        )

        # Ensure that we can uniquely identify the donor and acceptor atoms
        if current_donor_mask.sum() != 1:
            raise ValueError(
                f"Unable to uniquely identify a donor atom with chain_iid={item['d_chain']}, res_id={item['d_resi']}, atom_name={item['d_atom']}."
            )
        if current_acceptor_mask.sum() != 1:
            raise ValueError(
                f"Unable to uniquely identify an acceptor atom with chain_iid={item['a_chain']}, res_id={item['a_resi']}, atom_name={item['a_atom']}."
            )

        if motif_interface_only:
            current_donor_is_motif = atom_array.is_motif_atom[current_donor_mask][0]
            current_acceptor_is_motif = atom_array.is_motif_atom[
                current_acceptor_mask
            ][0]
            keep_hbond = current_donor_is_motif != current_acceptor_is_motif
        else:
            keep_hbond = True

        if keep_hbond:
            selected_hbonds.append(item)
            donor_mask |= current_donor_mask
            acceptor_mask |= current_acceptor_mask

            d_idx = int(np.argmax(current_donor_mask))
            a_idx = int(np.argmax(current_acceptor_mask))
            d, ang, ha = item["dist"], item["dha_angle"], item.get("ha_dist", np.nan)
            # geometry from ALL passes (default + carboxyl-as-donor override), keeping
            # the shortest bond per role. Nothing is filtered out here; the acid->HIS-P
            # call is made later in AnnotateProtonationStates.
            if np.isnan(active_donor_dist[d_idx]) or d < active_donor_dist[d_idx]:
                active_donor_dist[d_idx] = d
                active_donor_angle[d_idx] = ang
                active_donor_ha[d_idx] = ha
            donor_partners.setdefault(d_idx, []).append(f"{item['a_resn']}:{item['a_atom']}")
            if np.isnan(active_acceptor_dist[a_idx]) or d < active_acceptor_dist[a_idx]:
                active_acceptor_dist[a_idx] = d
                active_acceptor_angle[a_idx] = ang
                active_acceptor_ha[a_idx] = ha
            acceptor_partners.setdefault(a_idx, []).append(f"{item['d_resn']}:{item['d_atom']}")

    donor_array[donor_mask] = 1
    acceptor_array[acceptor_mask] = 1

    active_donor_partner_resns = np.full(n_atoms, "", dtype=object)
    for idx, partners in donor_partners.items():
        active_donor_partner_resns[idx] = ";".join(partners)

    active_acceptor_partner_resns = np.full(n_atoms, "", dtype=object)
    for idx, partners in acceptor_partners.items():
        active_acceptor_partner_resns[idx] = ";".join(partners)

    atom_array.set_annotation("active_donor", donor_array)
    atom_array.set_annotation("active_acceptor", acceptor_array)
    atom_array.set_annotation("active_donor_dist", active_donor_dist)
    atom_array.set_annotation("active_donor_angle", active_donor_angle)
    atom_array.set_annotation("active_donor_ha", active_donor_ha)
    atom_array.set_annotation("active_donor_partner_resns", active_donor_partner_resns)
    atom_array.set_annotation("active_acceptor_dist", active_acceptor_dist)
    atom_array.set_annotation("active_acceptor_angle", active_acceptor_angle)
    atom_array.set_annotation("active_acceptor_ha", active_acceptor_ha)
    atom_array.set_annotation(
        "active_acceptor_partner_resns", active_acceptor_partner_resns
    )

    return atom_array, selected_hbonds, len(selected_hbonds)


class CalculateHbondsPlus(Transform):
    """Transform for calculating Hbonds. HBPLUS places its own hydrogens, so run this on an
    H-stripped array: deposited hydrogens would otherwise decide the donor/acceptor roles."""

    def __init__(
        self,
        cutoff_HA_dist: float = 2.5,   # HBPLUS default; see calculate_hbonds
        cutoff_DA_distance: float = 3.5,
        motif_interface_only: bool = True,
        od1_oe1: bool = False,
        od2_oe2: bool = False,
        filter_capability: bool = False,
    ):
        self.cutoff_HA_dist = cutoff_HA_dist
        self.cutoff_DA_distance = cutoff_DA_distance
        self.motif_interface_only = motif_interface_only
        self.od1_oe1 = od1_oe1
        self.od2_oe2 = od2_oe2
        self.filter_capability = filter_capability

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name"])
        # check_atom_array_has_hydrogen(data)

    def forward(self, data: dict) -> dict:
        atom_array: AtomArray = data["atom_array"]

        atom_array, hbonds, _ = calculate_hbonds(
            atom_array,
            cutoff_HA_dist=self.cutoff_HA_dist,
            cutoff_DA_distance=self.cutoff_DA_distance,
            motif_interface_only=self.motif_interface_only,
            od1_oe1=self.od1_oe1,
            od2_oe2=self.od2_oe2,
            filter_capability=self.filter_capability,
        )

        data.setdefault("log_dict", {})
        log_dict = data["log_dict"]

        hbond_types = np.vstack((atom_array.active_donor, atom_array.active_acceptor)).T

        final_hbond_types = hbond_types
        if self.motif_interface_only:
            final_hbond_types[:, 0] *= np.array(atom_array.is_motif_atom)
            final_hbond_types[:, 1] *= np.array(atom_array.is_motif_atom)
        log_dict["hbond_total_count"] = np.sum(final_hbond_types)

        atom_array.set_annotation("active_donor", final_hbond_types[:, 0])
        atom_array.set_annotation("active_acceptor", final_hbond_types[:, 1])
        log_dict["hbond_subsample_atoms"] = np.sum(final_hbond_types)

        # Persist the merged (default + carboxyl-as-donor override) residue-level
        # donor->acceptor pairs. The per-atom active_donor/active_acceptor flags above
        # lose WHICH residue bonds which; BuildBondEdgeLabels needs the pairs to build
        # E_idx-alignable token-pair labels. Keyed by chain_iid + res_id (stable through
        # atomization / occupancy filtering).
        # Each pair also carries whether the bond is side-chain-mediated on the DONOR side (donor atom
        # is not a backbone N) and on the ACCEPTOR side (acceptor atom is not a backbone O/OXT).
        # BuildBondEdgeLabels keeps a bond only when BOTH are side-chain (side-chain<->side-chain).
        data["hbond_pairs"] = [
            (hb["d_chain"], int(hb["d_resi"]), hb["a_chain"], int(hb["a_resi"]),
             hb["d_atom"] not in BACKBONE_DONOR_ATOMS, hb["a_atom"] not in BACKBONE_ACCEPTOR_ATOMS)
            for hb in hbonds
        ]

        # Persist the RAW merged bond records too, with the settings that produced them. `hbond_pairs`
        # above is lossy (residue-level, no geometry), but a feature-based labeller needs the per-bond
        # atom names and D-A / DHA / H..A geometry. Rather than run HBPLUS again, EV6 reads this pool
        # (extended_vocab_v6 -> ev6/features.py), which takes v6 from 5 HBPLUS subprocess calls per
        # structure to 2. Each record carries `hbplus_modes` ("default" / "override" / both), so a
        # consumer wanting only the default pass can filter -- _merge_hbonds keeps the DEFAULT pass's
        # geometry for bonds seen in both, so that filter reproduces a standalone default run exactly.
        # `params` travels with the pool because a consumer's features are only valid for the cutoffs
        # they were trained on: it lets EV6 ASSERT the pool is its own rather than trust the wiring.
        data["hbond_records"] = {
            "bonds": hbonds,
            "params": {
                "cutoff_HA_dist": self.cutoff_HA_dist,
                "cutoff_DA_distance": self.cutoff_DA_distance,
                "motif_interface_only": self.motif_interface_only,
                "od1_oe1": self.od1_oe1,
                "od2_oe2": self.od2_oe2,
                "filter_capability": self.filter_capability,
            },
        }

        data["log_dict"] = log_dict
        data["atom_array"] = atom_array

        return data


class BuildBondEdgeLabels(Transform):
    """Persist HBPLUS / salt-bridge labels as fixed-width token-pair partner lists.

    Reads the residue-level pairs stashed by ``CalculateHbondsPlus``
    (``data["hbond_pairs"]``) and ``AnnotateSaltBridges`` (``data["salt_pairs"]``),
    maps each endpoint to the FINAL non-atomized token index (the model's sequence
    order), and writes three integer features into ``input_features``:

        ``hbond_donates_to`` : ``[L, max_hbond_partners]`` token idxs that token ``i``
            DONATES to (``i`` is the H-bond donor); padded with ``-1``.
        ``hbond_accepts_from`` : ``[L, max_hbond_partners]`` token idxs that token ``i``
            ACCEPTS from (``i`` is the H-bond acceptor); padded with ``-1``.
        ``salt_partners``    : ``[L, max_salt_partners]`` token idxs that salt-bridge
            with token ``i`` (symmetric); padded with ``-1``.

    An H-bond is a directed donor->acceptor relation, so the acceptor list is the
    transpose of the donor list — we store both explicitly so each direction is a
    direct local lookup against the model's ``E_idx`` (no transpose/gather needed)
    and the stored ground-truth is symmetric and self-checking:
        donor    label for edge ``i -> j`` : ``j in hbond_donates_to[i]``
        acceptor label for edge ``i -> j`` : ``j in hbond_accepts_from[i]``
    Storing token-pair identity (not a per-edge ``[L, K]`` slice) keeps the labels
    robust to the training coordinate noise that perturbs the k-NN graph. Both widths
    are FIXED constants so the batch collator (which only pads the token axis) can
    stack them; overflowing partners are truncated (count logged in
    ``log_dict["bond_label_truncated_partners"]``).

    Must run AFTER tokenization (``EncodePottsMPNNNonAtomizedTokens`` /
    ``FeaturizeNonAtomizedTokens``) so the token order is final, and BEFORE
    ``ConvertToTorch`` so the arrays are converted with the other input features.
    """

    def __init__(self, max_hbond_partners: int = 16, max_salt_partners: int = 8,
                 hbond_scope: str = "sc_any", filter_capability: bool = False):
        self.max_hbond_partners = max_hbond_partners
        self.max_salt_partners = max_salt_partners
        # Drop a labeled bond a titratable side chain can't make in its FINAL protonation state
        # (e.g. a deprotonated carboxyl ASP-D/GLU-D cannot DONATE) — via mpnn.chemistry.BOND_CHEMISTRY.
        self.filter_capability = filter_capability
        # Which H-bonds to keep as labels, by whether each endpoint bonds via a SIDE-CHAIN atom
        # (donor atom not backbone N; acceptor atom not backbone O/OXT):
        #   "sc_sc"  : both endpoints side-chain (side-chain<->side-chain only) — original default.
        #   "sc_any" : at least one endpoint side-chain (adds side-chain<->backbone; still drops the
        #              ubiquitous, identity-agnostic backbone<->backbone secondary-structure N-H..O=C).
        #   "all"    : every H-bond incl. backbone<->backbone.
        # Excluding backbone<->backbone keeps each label a residue-identity-relevant, discrete
        # residue-pair interaction so pH-sensitive H-bonds can be COUNTED as whole bonds.
        if hbond_scope not in ("sc_sc", "sc_any", "all"):
            raise ValueError(f"hbond_scope must be 'sc_sc' | 'sc_any' | 'all', got {hbond_scope!r}")
        self.hbond_scope = hbond_scope

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "input_features"])
        check_atom_array_annotation(data, ["atomize", "chain_iid", "res_id"])

    def forward(self, data: dict) -> dict:
        atom_array: AtomArray = data["atom_array"]

        token_level = atom_array[~atom_array.atomize]
        token_level = token_level[get_token_starts(token_level)]
        L = len(token_level)
        pos_of = {
            (token_level.chain_iid[i], int(token_level.res_id[i])): i for i in range(L)
        }

        # Protonation-state capability: a titratable side chain in a can't-donate/can't-accept state
        # cannot make the bond (e.g. ASP-D donor). Only gates SIDE-CHAIN bond ends; backbone is
        # protonation-independent. Non-titratable tokens (label "") pass (BOND_CHEMISTRY.get -> None).
        tok_label = None
        if self.filter_capability and "protonation_label" in token_level.get_annotation_categories():
            from mpnn.chemistry import BOND_CHEMISTRY
            tok_label = [str(x) for x in token_level.protonation_label]

            def _capable(i, attr):
                cap = BOND_CHEMISTRY.get(tok_label[i])
                return getattr(cap, attr) if cap is not None else True

        donates_to: list[list[int]] = [[] for _ in range(L)]
        accepts_from: list[list[int]] = [[] for _ in range(L)]
        for d_chain, d_resi, a_chain, a_resi, donor_sc, acceptor_sc in data.get("hbond_pairs", []):
            d = pos_of.get((d_chain, d_resi))
            a = pos_of.get((a_chain, a_resi))
            if d is None or a is None or d == a:
                continue
            # Apply the side-chain scope (see __init__). donates_to/accepts_from stay exact
            # transposes regardless (both appended together below), so directional labels are
            # consistent for side-chain<->backbone bonds too.
            if self.hbond_scope == "sc_sc" and not (donor_sc and acceptor_sc):
                continue
            if self.hbond_scope == "sc_any" and not (donor_sc or acceptor_sc):
                continue
            if tok_label is not None and (
                (donor_sc and not _capable(d, "can_donate"))
                or (acceptor_sc and not _capable(a, "can_accept"))
            ):
                continue   # impossible in the resolved protonation state
            if a not in donates_to[d]:
                donates_to[d].append(a)   # d donates to a
            if d not in accepts_from[a]:
                accepts_from[a].append(d)  # a accepts from d

        salt: list[list[int]] = [[] for _ in range(L)]
        for (c1, r1), (c2, r2) in data.get("salt_pairs", []):
            p = pos_of.get((c1, r1))
            q = pos_of.get((c2, r2))
            if p is not None and q is not None and p != q:
                if q not in salt[p]:
                    salt[p].append(q)
                if p not in salt[q]:
                    salt[q].append(p)

        n_trunc = 0

        def _pack(lists: list[list[int]], width: int) -> np.ndarray:
            nonlocal n_trunc
            out = np.full((L, width), -1, dtype=np.int64)
            for i, partners in enumerate(lists):
                if len(partners) > width:
                    n_trunc += len(partners) - width
                    partners = partners[:width]
                out[i, : len(partners)] = partners
            return out

        data["input_features"].update(
            {
                "hbond_donates_to": _pack(donates_to, self.max_hbond_partners),
                "hbond_accepts_from": _pack(accepts_from, self.max_hbond_partners),
                "salt_partners": _pack(salt, self.max_salt_partners),
            }
        )
        data.setdefault("log_dict", {})["bond_label_truncated_partners"] = n_trunc
        return data


def subsample_one_hot_np(array, fraction):
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be in the range (0, 1].")

    array = array.copy()
    one_indices = np.argwhere(array == 1)
    num_ones = len(one_indices)
    keep_count = int(num_ones * fraction)

    np.random.shuffle(one_indices)
    keep_indices = one_indices[:keep_count]

    new_array = np.zeros_like(array)
    for i, j in keep_indices:
        new_array[i, j] = 1

    return new_array
 
