"""run_ph_redesign returns the same designs whatever n_jobs is, and drops none silently."""

import time

from engine_fixtures import bare_engine, make_ctx, make_output


def test_run_ph_redesign_returns_the_same_designs_for_any_n_jobs(engine, monkeypatch):
    def fake_task(eng, ctx, criteria_list, base_seed, task):
        index, _ = task
        time.sleep(
            0.06 * (3 - index)
        )  # later tasks finish first: arrival != task order
        variant = ["ALA", "HIS", "ALA"] if index % 2 else ["ALA", "ALA", "GLY"]
        return engine.PHDesignSet(
            [
                make_output(
                    engine, 0, -1.0 - index, ["ALA"] * 3, index
                ),  # repeats exactly
                make_output(engine, 1, -2.0, variant, index),  # same id, two sequences
            ]
        )

    monkeypatch.setattr(engine, "_run_design_task", fake_task)
    runner = bare_engine(engine, make_ctx(engine))
    criteria = [
        engine.PHDesignCriteria(
            method="block_descent", seed_source="native", combined_lambda=lam
        )
        for lam in (0.0, 0.3, 0.6, 1.0)
    ]

    def run(n_jobs):
        designs = runner.run_ph_redesign(
            atom_array=None,
            binder_chain="A",
            criteria_list=criteria,
            seed=0,
            n_jobs=n_jobs,
        )
        return [
            (d.design_id(), d.seed_idx, tuple(d.extended_tokens), d.final_potts_energy)
            for d in designs
        ]

    serial, parallel = run(1), run(3)
    assert serial == parallel
    ids = [row[0] for row in serial]
    assert len(ids) == len(set(ids)) == 3  # exact repeats dropped, both sequences kept
    assert sorted(i for i in ids if "~" in i) == ["s_s1~2"]
    assert {row[1] for row in serial} == {0, 1}  # first task wins each repeat


def test_deduped_keeps_distinct_sequences_that_share_an_id(engine):
    a = make_output(engine, 0, -1.0, ["ALA", "GLY"], 0)
    b = make_output(engine, 0, -1.5, ["ALA", "HIS"], 0)
    again = make_output(engine, 0, -1.2, ["ALA", "GLY"], 0)
    kept = engine.PHDesignSet([a, b, again]).deduped()
    assert [d.design_id() for d in kept] == ["s_s0", "s_s0~2"]


def test_deduped_never_emits_an_id_twice_even_after_a_previous_dedupe(engine):
    """Results already holding x and x~2 plus another distinct x must not get a second x~2."""
    first = make_output(engine, 0, -1.0, ["ALA", "GLY"], 0)
    renamed = make_output(engine, 0, -1.5, ["ALA", "HIS"], 0, id_suffix="~2")
    third = make_output(engine, 0, -1.2, ["ALA", "TRP"], 0)
    kept = engine.PHDesignSet([first, renamed, third]).deduped()
    ids = [d.design_id() for d in kept]
    assert ids[:2] == ["s_s0", "s_s0~2"] and ids[2] == "s_s0~3"
    assert len(set(ids)) == 3
    assert [d.design_id() for d in kept.deduped()] == ids  # idempotent


def test_deduped_leaves_its_input_records_untouched(engine):
    a = make_output(engine, 0, -1.0, ["ALA", "GLY"], 0)
    b = make_output(engine, 0, -1.5, ["ALA", "HIS"], 0)
    kept = engine.PHDesignSet([a, b]).deduped()
    assert [d.design_id() for d in kept] == ["s_s0", "s_s0~2"]
    assert (a.id_suffix, b.id_suffix) == ("", "")
    assert b.design_id() == "s_s0"  # the input still has its own id
    assert kept[1] is not b


def test_reserved_ids_are_skipped_wherever_they_appear_in_the_input(engine):
    a = make_output(engine, 0, -1.0, ["ALA", "GLY"], 0)
    b = make_output(engine, 0, -1.1, ["ALA", "HIS"], 0)  # clashes with a
    later = make_output(engine, 0, -1.2, ["ALA", "TRP"], 0, id_suffix="~2")
    ids = [d.design_id() for d in engine.PHDesignSet([a, b, later]).deduped()]
    assert sorted(ids) == [
        "s_s0",
        "s_s0~2",
        "s_s0~3",
    ]  # b skips the ~2 that comes later
