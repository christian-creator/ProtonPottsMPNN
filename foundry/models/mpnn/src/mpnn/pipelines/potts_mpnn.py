from atomworks.constants import AF3_EXCLUDED_LIGANDS, STANDARD_AA, UNKNOWN_AA
from atomworks.enums import ChainTypeInfo
from atomworks.ml.transforms.atom_array import AddWithinChainInstanceResIdx
from atomworks.ml.transforms.atomize import (
    AtomizeByCCDName,
    FlagNonPolymersForAtomization,
)
from atomworks.ml.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    SubsetToKeys,
)
from atomworks.ml.transforms.bfactor_conditioned_transforms import SetOccToZeroOnBfactor
from atomworks.ml.transforms.covalent_modifications import (
    FlagAndReassignCovalentModifications,
)
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskPolymerResiduesWithUnresolvedFrameAtoms,
    MaskResiduesWithSpecificUnresolvedAtoms,
)
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveUnresolvedLigandAtomsIfTooMany,
    RemoveUnresolvedPNUnits,
    RemoveUnresolvedTokens,
)
from mpnn.transforms.bond_annotation import AnnotateSaltBridges, BuildBondEdgeLabels, CalculateHbondsPlus
from mpnn.transforms.whole_chain_crop import CropChainsAroundSpecifiedPNUnits
from mpnn.transforms.feature_aggregation.mpnn import FeaturizeNonAtomizedTokens
from mpnn.transforms.feature_aggregation.potts_mpnn import EncodePottsMPNNNonAtomizedTokens
from mpnn.transforms.feature_aggregation.user_settings import FeaturizeUserSettings
import numpy as np
from atomworks.ml.transforms.base import Transform
from mpnn.transforms.pka_annotation import AnnotatePKA, CalculatePackingDensity, RemoveHeteroAtoms
from mpnn.transforms.extended_vocab import get_vocab
from mpnn.transforms.vocab_annotation import AnnotateProtonationStates, ApplyProtonationThreshold
from mpnn.transforms.precomputed import SaveAnnotationSnapshot


class RemoveAtomsWithNaNCoords(Transform):
    def forward(self, data: dict) -> dict:
        aa = data["atom_array"]
        data["atom_array"] = aa[~np.isnan(aa.coord).any(axis=-1)]
        return data


def TrainingRoute(transform):
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={True: Identity(), False: transform},
    )


def InferenceRoute(transform):
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={False: Identity(), True: transform},
    )


def ModelTypeRoute(transform, model_type: str):
    return ConditionalRoute(
        condition_func=lambda data: data["model_type"] == model_type,
        transform_map={True: transform, False: Identity()},
    )


def get_cleanup_transforms(
    b_factor_min: float | None,
    undesired_res_names: list[str],
    chain_crop_transform=None,
):
    return [
        RemoveHeteroAtoms(),
        RemoveAtomsWithNaNCoords(),
        RemoveUnresolvedPNUnits(),
        MaskPolymerResiduesWithUnresolvedFrameAtoms(),
        TrainingRoute(chain_crop_transform),
        TrainingRoute(
            HandleUndesiredResTokens(undesired_res_tokens=undesired_res_names),
        ),
        TrainingRoute(SetOccToZeroOnBfactor(b_factor_min, None)),
        RemoveUnresolvedPNUnits(),
        TrainingRoute(
            RemovePolymersWithTooFewResolvedResidues(min_residues=4),
        ),
        TrainingRoute(
            RemoveUnresolvedLigandAtomsIfTooMany(unresolved_ligand_atom_limit=5),
        ),
    ]


def get_pka_annotation_transforms(
    packing_density_cellsize: float,
):
    return [
        CalculatePackingDensity(
            cell_size_A=packing_density_cellsize,
            save_neighbor_counts=True,
        ),
        AnnotatePKA(),
    ]


def get_protonation_state_transforms(
    extended_vocab: str,
    od1_oe1: bool,
    od2_oe2: bool,
    saltbridge_dist_max: float,
    saltbridge_dist_min: float,
    use_salt_bridge: bool = True,
    deterministic: bool = True,
    protonation_seed: int | None = None,
    protonation_classifier=None,
    annotate_bonds: bool = True,
):
    """H-bond / salt-bridge annotation + protonation labelling for one named vocabulary.

    ``protonation_classifier`` replaces the vocabulary's labeller (see AnnotateProtonationStates).
    The H-bond and salt-bridge passes only feed the geometric labellers and the bond-label heads,
    so ``annotate_bonds=False`` leaves them out (and with them the need for HBPLUS).

    `cutoff_HA_dist`, `cutoff_DA_dist` and `filter_capability` are NOT free knobs: they decide which
    bonds ever reach the classifier, so they belong to the vocabulary and are taken from it
    (extended_vocab.get_vocab). v6 goes further and CONSUMES this pass's bonds directly (see
    CalculateHbondsPlus's `data["hbond_records"]`), so for it these cutoffs define the model's own
    feature pool -- changing them here silently changes what EV6's models see.
    """
    vocab = get_vocab(extended_vocab)
    labelling = AnnotateProtonationStates(
        extended_vocab=extended_vocab,
        use_salt_bridge=use_salt_bridge,
        # deterministic=False samples the ambiguous per-atom roles AND the symmetric
        # shared-proton dyads (one -P / one -D per pair) -- training augmentation / inspection.
        deterministic=deterministic,
        seed=protonation_seed,
        classify=protonation_classifier,
    )
    if not annotate_bonds:
        return [RemoveHydrogens(), labelling]
    return [
        # H off first, so HBPLUS always places its own: with deposited H (every neutron entry) the label
        # would be read off the observed answer. EV6's models are trained on H-stripped structures.
        RemoveHydrogens(),
        CalculateHbondsPlus(
            cutoff_HA_dist=vocab["cutoff_HA_dist"],
            cutoff_DA_distance=vocab["cutoff_DA_dist"],
            motif_interface_only=False,
            od1_oe1=od1_oe1,
            od2_oe2=od2_oe2,
            # A bond into an acceptor with no lone pair (Lys NZ, Arg NE/NH*, Trp NE1) is chemically
            # impossible however we label protonation, so it must never reach the classifier OR the
            # H-bond head's ground truth. Titratable side chains are state-unknown here (donate AND
            # accept) and are gated later, on the resolved label, by BuildBondEdgeLabels.
            # v3 predates BOND_CHEMISTRY and therefore runs with this OFF.
            filter_capability=vocab["filter_capability"],
        ),
        AnnotateSaltBridges(
            dist_max=saltbridge_dist_max,
            min_dist=saltbridge_dist_min,
        ),
        labelling,
    ]


def build_mpnn_transform_pipeline(
    *,
    model_type: str = None,
    occupancy_threshold_sidechain: float = 0.5,
    occupancy_threshold_backbone: float = 0.8,
    # Hydrogen bonds. NB cutoff_HA_dist, cutoff_DA_dist and filter_capability are NOT here: they decide
    # which bonds reach the protonation classifier, so they belong to the VOCABULARY and come from
    # extended_vocab (v6 feeds this pass's bonds straight into its models).
    od1_oe1: bool = True,
    od2_oe2: bool = True,
    # Salt bridges (defaults match PLIP criteria)
    saltbridge_dist_max: float = 5.5,
    saltbridge_dist_min: float = 0.5,

    # Packing density / pKa
    packing_density_cellsize: float = 8.0,
    # Pipeline
    is_inference: bool = False,
    minimal_return: bool = False,
    train_structure_noise_default: float = 0.1,
    protonation_label_rate: float = 0.9,
    # Bond-label supervision (H-bond / salt-bridge head training). Off by default so the
    # main PottsMPNN training pipeline is unchanged; turn on to emit token-pair partner
    # lists (hbond_donates_to / hbond_accepts_from / salt_partners) into input_features.
    build_bond_labels: bool = False,
    max_hbond_partners: int = 16,
    max_salt_partners: int = 8,
    hbond_scope: str = "sc_any",  # BuildBondEdgeLabels scope: sc_any (sc<->sc + sc<->backbone; DEFAULT,
                                  # matches the vocab classifier which uses all side-chain bonds) | sc_sc | all
    # WHICH protonation vocabulary to label with: "v3" (ev3's original salt-bridge labeller) or "v4"
    # (the rewritten atom-level labeller; what potts_sb_* trains on). See transforms/extended_vocab.py.
    # It carries its own cutoff_HA_dist / filter_capability -- those are part of the vocabulary.
    # MUST MATCH THE CHECKPOINT: the encoder sees these tokens in S.
    extended_vocab: str = "v4",
    use_salt_bridge: bool = True,       # PLIP salt-bridge proximity prior. Both vocabularies support it;
                                        # v3 consults it INSIDE the decision tree, v4 as a post-pass over
                                        # residues still *-A (a much smaller trigger set).
    deterministic: bool = True,         # False -> sample ambiguous roles + symmetric dyads (augmentation)
    protonation_seed: int | None = None,
    # A labeller to use instead of the vocabulary's (e.g. PropkaLabeller). Its tokens must belong to
    # extended_vocab. Unless build_bond_labels is set, the H-bond and salt-bridge passes then have no
    # consumer and are skipped, so HBPLUS is not run.
    protonation_classifier=None,
    # Train-time protonation OPERATING POINT (v6 only). When either is set, ApplyProtonationThreshold
    # re-derives `protonation_label` from the snapshot's persisted FLAML scores at this prob_thr, just
    # before S is built -- making the P-vs-D cut a per-run knob to sweep WITHOUT re-running FLAML. sd_cut
    # (the ambiguous -A gate) stays fixed at thresholds.json. None/None -> the baked label is used as-is
    # (today's behaviour). Requires snapshots carrying flaml_p (scripts/augment_snapshots_flaml.py); a
    # no-op on caches that lack it.
    his_prob_thr: float | None = None,
    acid_prob_thr: float | None = None,
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    # Complex training — ignored at inference
    complex_pair_probability: float = 0.0,
    complex_max_atoms: int = 5000,
    complex_min_atom_contacts: int = 10,
    complex_max_chains: int | None = None,
    # Precomputed annotations. When True, the cleaned+annotated FULL structure is supplied by the loader
    # (from a snapshot built by build_precompute_pipeline), so this pipeline SKIPS get_cleanup_transforms
    # and the whole HBPLUS/annotation block -- it runs only the chain crop + featurization tail. The label
    # is deterministic and crop-invariant except at pairing interfaces (full-context labels), so this is
    # equivalent up to that accepted difference. `precomputed_dir` is informational here (the loader owns
    # the lookup); kept for symmetry/assertions.
    precomputed: bool = False,
    precomputed_dir: str | None = None,
    device=None,
) -> Compose:
    if model_type not in ("potts_mpnn", "protein_mpnn"):
        raise ValueError(f"Unsupported model_type: {model_type}")

    transforms = [
        AddData({"model_type": model_type}),
        AddData({"is_inference": is_inference}),
    ]

    # Build the training-time chain crop. When complex_pair_probability > 0,
    # CropChainsAroundSpecifiedPNUnits replaces the monomer filter so that
    # contacting partner chains are included before HBPLUS runs (enabling
    # cross-chain H-bonds / salt bridges). The 52-chain and 99,999-atom limits
    # imposed by the PDB format are not exceeded because max_atoms bounds the crop.
    if complex_pair_probability > 0.0:
        # Use complex crop for both training and val (is_inference=False, complex_pair_probability>0).
        chain_crop_transform = CropChainsAroundSpecifiedPNUnits(
            complex_pair_probability=complex_pair_probability,
            max_atoms=complex_max_atoms,
            min_atom_contacts=complex_min_atom_contacts,
            max_chains=complex_max_chains,
        )
    elif is_inference:
        # True inference (deployment): no chain filtering — pass full structure through.
        chain_crop_transform = Identity()
    else:
        # Training with complex_pair_probability=0: monomer-only baseline.
        chain_crop_transform = FilterToSpecifiedPNUnits(
            extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"
        )

    if precomputed:
        # The snapshot (built by build_precompute_pipeline) already holds the cleaned, cropped, H-stripped,
        # ANNOTATED assembly at the FIXED training composition -- the p=1 multimer crop ran BEFORE
        # annotation, so a residue's label was computed with exactly the partner chains the encoder will
        # see (no interface leakage). Nothing upstream of the featurization tail remains: skip cleanup, the
        # crop, and the whole HBPLUS/annotation block. `chain_crop_transform` above is unused here.
        assert precomputed_dir is not None, "precomputed=True requires precomputed_dir"
    else:
        transforms += get_cleanup_transforms(
            b_factor_min=None,
            undesired_res_names=undesired_res_names,
            chain_crop_transform=chain_crop_transform,
        )

        # hbplus runs on the filtered single-chain structure, with hydrogens stripped.
        transforms += get_protonation_state_transforms(
            extended_vocab=extended_vocab,
            od1_oe1=od1_oe1,
            od2_oe2=od2_oe2,
            saltbridge_dist_max=saltbridge_dist_max,
            saltbridge_dist_min=saltbridge_dist_min,
            use_salt_bridge=use_salt_bridge,
            deterministic=deterministic,
            protonation_seed=protonation_seed,
            protonation_classifier=protonation_classifier,
            annotate_bonds=protonation_classifier is None or build_bond_labels,
        )

    # pKa annotation is training-only and expensive; not yet consumed by encoding.
    # transforms += get_pka_annotation_transforms(packing_density_cellsize)

    transforms += [
        # + --------- Atomization --------- +
        AddWithinChainInstanceResIdx(),
        FlagAndReassignCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + (UNKNOWN_AA,),
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),

        # + --------- Occupancy filtering --------- +
        MaskResiduesWithSpecificUnresolvedAtoms(
            chain_type_to_atom_names={
                ChainTypeInfo.PROTEINS: ["N", "CA", "C", "O"],
            },
            occupancy_threshold=occupancy_threshold_backbone,
        ),
        RemoveUnresolvedTokens(),

        # + --------- Encoding and featurization --------- +
        AddData({"input_features": dict()}),
        # Re-threshold the persisted FLAML scores to a per-run prob_thr just before S is built (v6). Added
        # only when a threshold override is given; itself a no-op when the atom_array lacks flaml_p (the
        # live inference/callback path, where the env-aware predictor already applied the threshold).
        *(
            [ApplyProtonationThreshold(his_prob_thr=his_prob_thr, acid_prob_thr=acid_prob_thr)]
            if (his_prob_thr is not None or acid_prob_thr is not None) else []
        ),
        EncodePottsMPNNNonAtomizedTokens(
            occupancy_threshold=occupancy_threshold_sidechain,
            protonation_label_rate=1.0 if is_inference else protonation_label_rate,
            # build S in the vocabulary's own token set (v6 -> 30-token; v3/v4/None -> 32-token default)
            encoding=get_vocab(extended_vocab)["token_encoding"] if extended_vocab else None,
        ),
        FeaturizeNonAtomizedTokens(),
        # Token-pair H-bond / salt-bridge labels (gated). Runs once the token order is
        # final and before ConvertToTorch so the arrays are converted with the rest.
        *(
            [BuildBondEdgeLabels(
                max_hbond_partners=max_hbond_partners,
                max_salt_partners=max_salt_partners,
                hbond_scope=hbond_scope,
                # same gate the vocabulary uses for its own bond set
                filter_capability=get_vocab(extended_vocab)["filter_capability"],
            )]
            if build_bond_labels else []
        ),
        FeaturizeUserSettings(
            is_inference=is_inference,
            minimal_return=minimal_return,
            train_structure_noise_default=train_structure_noise_default,
        ),
        ConvertToTorch(
            keys=["input_features"],
            **({"device": device} if device is not None else {}),
        ),
        SubsetToKeys(keys=["input_features", "atom_array"]),
    ]

    return Compose(transforms)


def build_precompute_pipeline(
    *,
    extended_vocab: str = "v4",
    precomputed_dir: str,
    od1_oe1: bool = True,
    od2_oe2: bool = True,
    saltbridge_dist_max: float = 5.5,
    saltbridge_dist_min: float = 0.5,
    use_salt_bridge: bool = True,
    deterministic: bool = True,
    protonation_seed: int | None = None,
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    complex_max_atoms: int = 5000,
    complex_min_atom_contacts: int = 10,
    complex_max_chains: int | None = None,
) -> Compose:
    """Offline pipeline that produces the annotation snapshots the training loader reads.

    It crops each structure to its FIXED multimer composition (``complex_pair_probability=1.0`` -- always
    keep contacting partner chains) BEFORE annotating, so a residue's protonation label is computed with
    exactly the chains the encoder will later see. It then runs the standard cleanup + HBPLUS/EV6 annotation
    and writes the resulting ``data`` (cropped, annotated, pre-atomization) to ``precomputed_dir`` keyed by
    ``example_id``. It STOPS there: atomization / encoding / noise are per-run and stay in the live tail.

    Run this over the whole manifest once (sharded); training then sets ``precomputed=True`` and only
    loads + featurizes. Deterministic composition is the whole point -- do NOT use a mixed
    ``0<complex_pair_probability<1`` here, or the frozen label would not match the encoder's chain set.
    """
    # p=1: always include contacting partners. is_inference=False below, so the TrainingRoute wrapper in
    # get_cleanup_transforms actually runs the crop (Identity at inference).
    chain_crop_transform = CropChainsAroundSpecifiedPNUnits(
        complex_pair_probability=1.0,
        max_atoms=complex_max_atoms,
        min_atom_contacts=complex_min_atom_contacts,
        max_chains=complex_max_chains,
    )
    transforms = [
        AddData({"model_type": "potts_mpnn"}),
        AddData({"is_inference": False}),
    ]
    transforms += get_cleanup_transforms(
        b_factor_min=None,
        undesired_res_names=undesired_res_names,
        chain_crop_transform=chain_crop_transform,
    )
    transforms += get_protonation_state_transforms(
        extended_vocab=extended_vocab,
        od1_oe1=od1_oe1,
        od2_oe2=od2_oe2,
        saltbridge_dist_max=saltbridge_dist_max,
        saltbridge_dist_min=saltbridge_dist_min,
        use_salt_bridge=use_salt_bridge,
        deterministic=deterministic,
        protonation_seed=protonation_seed,
    )
    transforms += [SaveAnnotationSnapshot(precomputed_dir)]
    return Compose(transforms)
