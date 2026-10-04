import numpy as np
import pytest

from heat_risk.heat_index import heat_index_f
from heat_risk.spec import HEAT_INDEX_F, HOT_NIGHT_C
from heat_risk.synthetic import avoid_threshold_band, generate
from heat_risk.thresholds import percentile_thresholds


def test_seeded_and_shaped() -> None:
    a = generate(2, 5, 40, seed=7, baseline_days=100)
    b = generate(2, 5, 40, seed=7, baseline_days=100)
    assert a.tmax.shape == (2, 5, 40) and a.baseline_tmax.shape == (5, 100)
    np.testing.assert_array_equal(a.tmax, b.tmax)
    assert not np.array_equal(a.tmax, generate(2, 5, 40, seed=8, baseline_days=100).tmax)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_physical_ranges(dtype: type) -> None:
    d = generate(3, 20, 365, seed=1, dtype=dtype)
    assert d.tmax.dtype == dtype
    assert (d.tmin <= d.tmax).all()
    assert (d.rh >= 5).all() and (d.rh <= 100).all()
    assert -30 < d.tmax.min() and d.tmax.max() < 60


def test_seasonal_cycle_present() -> None:
    d = generate(1, 30, 365, seed=2)
    jan = d.tmax[0, :, :31].mean()
    jul = d.tmax[0, :, 181:212].mean()
    assert jul - jan > 4


def test_scenarios_warm() -> None:
    d = generate(4, 30, 365, seed=3)
    means = d.tmax.mean(axis=(1, 2))
    assert (np.diff(means) > 0).all()


def test_zero_days() -> None:
    d = generate(2, 3, 0, seed=0, baseline_days=10)
    assert d.tmax.shape == (2, 3, 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_avoid_threshold_band(dtype: type) -> None:
    d = generate(3, 25, 730, seed=4, dtype=dtype)
    tx90, tx95 = percentile_thresholds(d.baseline_tmax, dtype)
    a = avoid_threshold_band(d, tx90, tx95)
    t, n, r = (x.astype(np.float64) for x in (a.tmax, a.tmin, a.rh))
    ok = ~np.isnan(t)
    assert ok.mean() > 0.99
    for thr in (tx90, tx95):
        assert (np.abs(t - thr.astype(np.float64)[None, :, None])[ok] >= 0.5 - 1e-4).all()
    assert (np.abs(n - HOT_NIGHT_C)[ok] >= 0.5 - 1e-4).all()
    assert (np.abs(heat_index_f(t, r) - HEAT_INDEX_F)[ok] >= 0.5 - 1e-3).all()
    assert (n[ok] <= t[ok]).all()
