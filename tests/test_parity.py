"""All implementations against the cpu reference and hand-built expectations.

Implementations that need a CUDA device are gpu-marked and skip without one; a skip is never a
pass. ``gpu_torch[cpu]`` runs the torch code on CPU tensors so it is exercised everywhere.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from oracle import longest_run_scalar

from heat_risk import cpu, gpu_cuda, gpu_torch
from heat_risk.spec import HAZARDS, HOT_NIGHT_C, HazardResult
from heat_risk.synthetic import avoid_threshold_band, generate
from heat_risk.thresholds import percentile_thresholds

Evaluate = Callable[..., HazardResult]

IMPLS = [
    pytest.param(cpu.evaluate, id="cpu"),
    pytest.param(lambda *a: gpu_torch.evaluate(*a, device="cpu"), id="gpu_torch[cpu]"),
    pytest.param(
        lambda *a: gpu_torch.evaluate(*a, device="cuda:0"),
        id="gpu_torch[cuda]",
        marks=pytest.mark.gpu,
    ),
    pytest.param(gpu_cuda.evaluate, id="gpu_cuda", marks=pytest.mark.gpu),
    pytest.param(
        lambda *a: gpu_cuda.evaluate(*a, layout="tiled"),
        id="gpu_cuda[tiled]",
        marks=pytest.mark.gpu,
    ),
]
DTYPES = [np.float32, np.float64]
HOT, TX90, TX95, HI = (HAZARDS.index(h) for h in ("hot_night", "tx90", "tx95", "heat_index"))


def _data(s: int, n_loc: int, days: int, dtype: type, seed: int = 0) -> tuple[np.ndarray, ...]:
    d = generate(s, n_loc, days, seed=seed, baseline_days=200, dtype=dtype)
    tx90, tx95 = percentile_thresholds(d.baseline_tmax, dtype)
    d = avoid_threshold_band(d, tx90, tx95)
    return d.tmax, d.tmin, d.rh, tx90, tx95


def _assert_same(got: HazardResult, want: HazardResult) -> None:
    assert got.counts.dtype == np.int32 and got.longest_run.dtype == np.int32
    np.testing.assert_array_equal(got.counts, want.counts)
    np.testing.assert_array_equal(got.longest_run, want.longest_run)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("s", "n_loc"),
    [(1, 1), (1, 127), (1, 128), (1, 129), (3, 85), (2, 257)],  # rows around the block size
)
def test_matches_cpu_on_random_inputs(
    impl: Evaluate, dtype: type, s: int, n_loc: int
) -> None:
    args = _data(s, n_loc, 403, dtype, seed=s * 1000 + n_loc)  # 403 days: not a tile multiple
    _assert_same(impl(*args), cpu.evaluate(*args))


@pytest.mark.parametrize("impl", IMPLS)
def test_float32_matches_float64_reference_away_from_cutoffs(impl: Evaluate) -> None:
    tmax, tmin, rh, tx90, tx95 = _data(3, 150, 365, np.float64, seed=5)
    want = cpu.evaluate(tmax, tmin, rh, tx90, tx95)  # float64 oracle
    f32 = [a.astype(np.float32) for a in (tmax, tmin, rh, tx90, tx95)]
    _assert_same(impl(*f32), want)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_days(impl: Evaluate, dtype: type) -> None:
    z = np.zeros((2, 3, 0), dtype=dtype)
    t = np.zeros(3, dtype=dtype)
    r = impl(z, z, z, t, t)
    assert r.counts.shape == (2, 3, len(HAZARDS))
    assert not r.counts.any() and not r.longest_run.any()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_all_nan_rows_and_all_nan_baseline(impl: Evaluate, dtype: type) -> None:
    tmax, tmin, rh, tx90, tx95 = (a.copy() for a in _data(2, 6, 100, dtype, seed=9))
    tmax[0, 1, :] = np.nan  # no tx90, tx95 or heat index days in this row
    tmin[1, 2, :] = np.nan  # no hot nights in this row
    rh[0, 3, :] = np.nan  # no heat index days in this row
    tx90[4] = tx95[4] = np.nan  # location 4: all-NaN baseline
    r = impl(tmax, tmin, rh, tx90, tx95)
    assert r.counts[0, 1, [TX90, TX95, HI]].tolist() == [0, 0, 0]
    assert r.counts[1, 2, HOT] == 0
    assert r.counts[0, 3, HI] == 0
    assert not r.counts[:, 4, [TX90, TX95]].any()
    _assert_same(r, cpu.evaluate(tmax, tmin, rh, tx90, tx95))


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_all_below_and_all_above(impl: Evaluate, dtype: type) -> None:
    days = 50
    shape = (1, 2, days)
    tmax = np.empty(shape, dtype=dtype)
    tmin = np.empty(shape, dtype=dtype)
    rh = np.full(shape, 50, dtype=dtype)
    tmax[0, 0], tmin[0, 0] = 10, 5  # far below everything (heat index ~ 50 F)
    tmax[0, 1], tmin[0, 1] = 45, 30  # far above everything (heat index > 120 F)
    tx = np.array([30, 30], dtype=dtype)
    r = impl(tmax, tmin, rh, tx, tx)
    assert r.counts[0, 0].tolist() == [0, 0, 0, 0]
    assert r.longest_run[0, 0].tolist() == [0, 0, 0, 0]
    assert r.counts[0, 1].tolist() == [days] * 4
    assert r.longest_run[0, 1].tolist() == [days] * 4


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_threshold_counts(impl: Evaluate, dtype: type) -> None:
    # Values exactly representable in float32 and float64; >= must count them.
    tx90 = np.array([31.5, 28.25], dtype=dtype)
    tx95 = np.array([33.0, 29.0], dtype=dtype)
    tmax = np.broadcast_to(tx90[None, :, None], (1, 2, 7)).astype(dtype)
    tmin = np.full((1, 2, 7), HOT_NIGHT_C, dtype=dtype)
    rh = np.full((1, 2, 7), 10, dtype=dtype)
    r = impl(tmax, tmin, rh, tx90, tx95)
    assert r.counts[0, :, TX90].tolist() == [7, 7]
    assert r.counts[0, :, TX95].tolist() == [0, 0]
    assert r.counts[0, :, HOT].tolist() == [7, 7]
    # One ulp below the threshold must not count.
    below = np.nextafter(tmax, np.array(-np.inf, dtype=dtype))
    r2 = impl(below, np.nextafter(tmin, np.array(-np.inf, dtype=dtype)), rh, tx90, tx95)
    assert r2.counts[0, :, TX90].tolist() == [0, 0]
    assert r2.counts[0, :, HOT].tolist() == [0, 0]


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_runs_match_scalar_oracle_with_nan_breaks(impl: Evaluate, dtype: type) -> None:
    rng = np.random.default_rng(3)
    s, n_loc, days = 2, 131, 90
    state = rng.choice(3, size=(s, n_loc, days), p=[0.35, 0.55, 0.10])  # cold, hot, NaN
    tmin = np.where(state == 1, 25.0, 15.0).astype(dtype)
    tmin[state == 2] = np.nan
    tmax = (tmin + 8).astype(dtype)
    rh = np.full_like(tmax, 40)
    tx = np.full(n_loc, 100, dtype=dtype)
    r = impl(tmax, tmin, rh, tx, tx)
    for i in range(s):
        for j in range(n_loc):
            flags = (state[i, j] == 1).tolist()
            assert r.counts[i, j, HOT] == sum(flags)
            assert r.longest_run[i, j, HOT] == longest_run_scalar(flags)


@pytest.mark.parametrize("impl", IMPLS)
def test_single_location_single_day(impl: Evaluate) -> None:
    one = np.array([[[23.0]]])
    r = impl(one + 10, one, np.full_like(one, 50), np.array([30.0]), np.array([40.0]))
    assert r.counts[0, 0].tolist() == [1, 0, 1, 1]
    assert r.longest_run[0, 0].tolist() == [1, 0, 1, 1]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("days", [1, 7, 8, 9, 15, 16, 17, 33])  # around 8 and 16 day tiles
@pytest.mark.parametrize("n_loc", [1, 127, 128, 129, 300])  # around the 128 row tile
def test_tiled_kernel_tile_edges(dtype: type, days: int, n_loc: int) -> None:
    args = _data(1, n_loc, days, dtype, seed=days * 1000 + n_loc)
    got = gpu_cuda.evaluate(*args, layout="tiled")
    _assert_same(got, cpu.evaluate(*args))
