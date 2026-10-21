"""Tests for block descent on the scalarised objective."""

import itertools

import pytest
import torch
from objective_fixtures import (
    BASE,
    DESIGN_BLOCK,
    NAMES_ACTIVE,
    make_context,
    make_objective,
)

from mpnn.ph.descent import block_descent, build_block

VALID = torch.tensor([True, True, False, False])  # ALA and bare HIS; no state tokens
DESIGNABLE = DESIGN_BLOCK


@pytest.fixture
def objective():
    return make_objective(make_context())


def run(objective, **kwargs):
    defaults = dict(
        tokens=BASE,
        designable=DESIGNABLE,
        names=NAMES_ACTIVE,
        weights=(0.3, 0.3, 0.4),
        valid_mask=VALID,
    )
    return block_descent(objective, **{**defaults, **kwargs})


@pytest.mark.parametrize("scalarisation", ["weighted_sum", "tchebycheff"])
@pytest.mark.parametrize("block_size", [1, 2])
def test_objective_never_increases(objective, scalarisation, block_size):
    result = run(objective, scalarisation=scalarisation, block_size=block_size)
    steps = list(zip(result.trace, result.trace[1:]))
    assert all(after <= before + 1e-12 for before, after in steps)
    assert result.value == pytest.approx(min(result.trace))


@pytest.mark.parametrize("scalarisation", ["weighted_sum", "tchebycheff"])
def test_a_block_covering_all_designable_positions_is_exact(objective, scalarisation):
    best = None
    for values in itertools.product([0, 1], repeat=len(DESIGNABLE)):
        tokens = BASE.clone()
        tokens[DESIGNABLE] = torch.tensor(values)
        value = objective.scalarised_value(
            tokens, NAMES_ACTIVE, (0.3, 0.3, 0.4), scalarisation
        )
        best = min(best or (value, values), (value, values))
    result = run(objective, scalarisation=scalarisation, block_size=2)
    assert result.value == pytest.approx(best[0])
    assert tuple(int(t) for t in result.tokens[DESIGNABLE]) == best[1]


def test_converges_and_is_idempotent(objective):
    first = run(objective, block_size=1)
    assert first.converged
    again = run(objective, block_size=1, tokens=first.tokens)
    assert again.n_changes == 0 and torch.equal(again.tokens, first.tokens)


def test_only_designable_positions_change_and_masked_tokens_never_appear(objective):
    result = run(objective, block_size=1)
    fixed = [p for p in range(len(BASE)) if p not in DESIGNABLE]
    assert torch.equal(result.tokens[fixed], BASE[fixed])
    assert all(VALID[int(t)] for t in result.tokens[DESIGNABLE])


def test_sampling_is_reproducible_and_returns_the_best_state_seen(objective):
    def sampled(seed):
        return run(
            objective,
            block_size=1,
            temperature=2.0,
            generator=torch.Generator().manual_seed(seed),
            max_rounds=4,
        )

    one, two = sampled(11), sampled(11)
    assert torch.equal(one.tokens, two.tokens) and one.trace == two.trace
    assert one.value == pytest.approx(min(one.trace))


def test_single_active_term_runs_single_objective(objective):
    result = run(objective, names=("potency",), weights=(1.0,), block_size=2)
    potency = objective.terms["potency"]
    reachable = []
    for values in itertools.product([0, 1], repeat=len(DESIGNABLE)):
        tokens = BASE.clone()
        tokens[DESIGNABLE] = torch.tensor(values)
        reachable.append(potency.value(objective.ctx, tokens))
    assert potency.value(objective.ctx, result.tokens) == pytest.approx(min(reachable))


def test_blocks_follow_the_partner_rank(objective):
    seen = []
    original = objective.best_assignment

    def spy(tokens, block, *args, **kwargs):
        seen.append(tuple(block))
        return original(tokens, block, *args, **kwargs)

    objective.best_assignment = spy
    run(objective, block_size=2, partner_rank=lambda p: [3, 2, 1, 0], max_rounds=1)
    assert set(seen) == {(1, 2)}  # non-designable partners are skipped


def test_build_block_defaults_wrap_around():
    assert build_block(5, [1, 3, 5], 2, None) == [1, 5]
    assert build_block(1, [1, 3, 5], 3, None) == [1, 3, 5]
    assert build_block(3, [1, 3, 5], 1, None) == [3]


def test_bad_arguments_fail_loudly(objective):
    with pytest.raises(ValueError, match="block_size"):
        run(objective, block_size=0)
    with pytest.raises(ValueError, match="sweep"):
        run(objective, sweep="random")


@pytest.mark.parametrize(
    "start", [(3, 3), (3, 2), (2, 3)]
)  # HIS-S / HIS-P: not allowed
@pytest.mark.parametrize("block_size", [1, 2])
@pytest.mark.parametrize("scalarisation", ["weighted_sum", "tchebycheff"])
def test_result_never_keeps_a_forbidden_token(
    objective, start, block_size, scalarisation
):
    """A forbidden native token must not survive, even when it scores best (here it does)."""
    tokens = BASE.clone()
    tokens[DESIGNABLE] = torch.tensor(start)
    result = run(
        objective, tokens=tokens, block_size=block_size, scalarisation=scalarisation
    )
    assert all(VALID[int(t)] for t in result.tokens[DESIGNABLE])
    # the returned value is the value of the returned (valid) sequence
    assert result.value == pytest.approx(
        objective.scalarised_value(
            result.tokens, NAMES_ACTIVE, (0.3, 0.3, 0.4), scalarisation
        )
    )


def test_a_forbidden_start_that_beats_every_allowed_design_is_replaced(objective):
    tokens = BASE.clone()
    tokens[DESIGNABLE] = torch.tensor([3, 3])
    start_value = objective.scalarised_value(tokens, NAMES_ACTIVE, (0.3, 0.3, 0.4))
    best_allowed = min(
        objective.scalarised_value(_with(values), NAMES_ACTIVE, (0.3, 0.3, 0.4))
        for values in itertools.product([0, 1], repeat=2)
    )
    assert start_value < best_allowed  # the setup the old code got wrong
    result = run(objective, tokens=tokens, block_size=2)
    assert result.value == pytest.approx(best_allowed)


def _with(values):
    tokens = BASE.clone()
    tokens[DESIGNABLE] = torch.tensor(values)
    return tokens
