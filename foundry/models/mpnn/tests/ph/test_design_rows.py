"""Plain row output for designs."""

import pytest
from engine_fixtures import make_output


def test_design_rows_carry_sequence_centres_and_mutation_count(engine):
    design = make_output(
        engine,
        0,
        -12.5,
        ["ALA", "CYS", "ASP"],
        None,
        center_res_ids=[33, 93],
        center_protonation_types=["ASP-P", "HIS-P"],
        selective_energy=-3.0,
    )
    row = engine.PHDesignSet([design]).to_rows(reference_sequence="AAD")[0]
    assert row["canonical_sequence"] == "ACA"
    assert row["centers"] == ["33:ASP-P", "93:HIS-P"]
    assert row["n_mutations"] == 2
    assert (row["potts_energy"], row["selective_energy"]) == (-12.5, -3.0)
    assert engine.PHDesignSet([design]).to_rows()[0]["n_mutations"] is None
    with pytest.raises(ValueError, match="reference_sequence"):
        engine.PHDesignSet([design]).to_rows(reference_sequence="AA")
