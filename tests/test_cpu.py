import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from oracle import longest_run_scalar

from heat_risk import cpu
from heat_risk.spec import HAZARDS
from heat_risk.synthetic import generate
from heat_risk.thresholds import percentile_thresholds

NAN = np.nan


def f_to_c(f: float) -> float:
    return (f - 32.0) / 1.8


def toy(dtype: type) -> tuple[np.ndarray, ...]:
    """Three hand-computed locations, one scenario, five days.

    loc0: rh all NaN (heat index never counts); tx90 = 35, tx95 = 36.
    loc1: all-NaN baseline, so NaN thresholds; a NaN tmin breaks the hot-night run.
    loc2: NWS chart temperatures; tmax exactly equal to tx90 = 30 on the last day.
    """
    tmax = [
        [30, 35, 36, 20, 37],
        [20, 20, 20, 20, 20],
        [f_to_c(90), f_to_c(80), f_to_c(90), f_to_c(90), 30.0],
    ]
    tmin = [[22, 22, 10, 22, 22], [22, NAN, 22, 22, 22], [15] * 5]
    rh = [[NAN] * 5, [50] * 5, [70, 40, 70, 70, 90]]
    tx90 = [35, NAN, 30]
    tx95 = [36, NAN, 32]
    as_ = lambda x: np.array(x, dtype=dtype)  # noqa: E731
    return as_([tmax]), as_([tmin]), as_([rh]), as_(tx90), as_(tx95)


EXPECTED_COUNTS = [[3, 2, 4, 0], [0, 0, 4, 0], [4, 3, 0, 4]]
EXPECTED_RUNS = [[2, 1, 2, 0], [0, 0, 3, 0], [3, 2, 0, 3]]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_toy_hand_computed(dtype: type) -> None:
    r = cpu.evaluate(*toy(dtype))
    assert r.counts.dtype == np.int32 and r.longest_run.dtype == np.int32
    assert r.counts[0].tolist() == EXPECTED_COUNTS
    assert r.longest_run[0].tolist() == EXPECTED_RUNS


def test_zero_days() -> None:
    z = np.zeros((2, 3, 0))
    t = np.zeros(3)
    r = cpu.evaluate(z, z, z, t, t)
    assert r.counts.shape == (2, 3, len(HAZARDS))
    assert not r.counts.any() and not r.longest_run.any()


def test_rejects_mixed_dtype_thresholds() -> None:
    a = np.zeros((1, 2, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        cpu.evaluate(a, a, a, np.zeros(2), np.zeros(2))


@given(arrays(np.bool_, (3, 4, 17)))
def test_longest_runs_matches_scalar_oracle(mask: np.ndarray) -> None:
    got = cpu.longest_runs(mask, axis=2)
    want = [[longest_run_scalar(mask[i, j]) for j in range(4)] for i in range(3)]
    assert got.tolist() == want


inputs = st.tuples(
    st.integers(1, 3), st.integers(1, 6), st.integers(0, 40), st.integers(0, 2**32 - 1)
)


def _random(s: int, n_loc: int, d: int, seed: int) -> tuple[np.ndarray, ...]:
    data = generate(s, n_loc, d, seed=seed, baseline_days=60)
    tx90, tx95 = percentile_thresholds(data.baseline_tmax, np.float64)
    return data.tmax, data.tmin, data.rh, tx90, tx95


@settings(max_examples=50, deadline=None)
@given(inputs)
def test_counts_and_runs_bounded(shape: tuple[int, int, int, int]) -> None:
    tmax, tmin, rh, tx90, tx95 = _random(*shape)
    r = cpu.evaluate(tmax, tmin, rh, tx90, tx95)
    days = tmax.shape[2]
    assert (r.counts >= 0).all() and (r.counts <= days).all()
    assert (r.longest_run >= 0).all() and (r.longest_run <= r.counts).all()
    assert ((r.counts > 0) == (r.longest_run > 0)).all()


@settings(max_examples=50, deadline=None)
@given(inputs, st.floats(0.0, 5.0))
def test_monotone_in_threshold(shape: tuple[int, int, int, int], bump: float) -> None:
    tmax, tmin, rh, tx90, tx95 = _random(*shape)
    lo = cpu.evaluate(tmax, tmin, rh, tx90, tx95)
    hi = cpu.evaluate(tmax, tmin, rh, tx90 + bump, tx95 + bump)
    assert (hi.counts[..., :2] <= lo.counts[..., :2]).all()
    assert (hi.longest_run[..., :2] <= lo.longest_run[..., :2]).all()
    assert (hi.counts[..., 2:] == lo.counts[..., 2:]).all()


@settings(max_examples=50, deadline=None)
@given(inputs, st.randoms(use_true_random=False))
def test_location_permutation_invariant(shape: tuple[int, int, int, int], rnd) -> None:  # type: ignore[no-untyped-def]
    tmax, tmin, rh, tx90, tx95 = _random(*shape)
    perm = list(range(tmax.shape[1]))
    rnd.shuffle(perm)
    base = cpu.evaluate(tmax, tmin, rh, tx90, tx95)
    p = cpu.evaluate(tmax[:, perm], tmin[:, perm], rh[:, perm], tx90[perm], tx95[perm])
    np.testing.assert_array_equal(p.counts, base.counts[:, perm])
    np.testing.assert_array_equal(p.longest_run, base.longest_run[:, perm])
