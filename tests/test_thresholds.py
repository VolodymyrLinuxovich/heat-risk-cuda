import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from heat_risk.thresholds import percentile_thresholds


def test_known_values_linear_method() -> None:
    # 0..10 has 11 points: linear 90th percentile is 9.0, 95th is 9.5.
    base = np.arange(11, dtype=np.float64)[None, :]
    tx90, tx95 = percentile_thresholds(base, np.float64)
    assert tx90.tolist() == [9.0]
    assert tx95.tolist() == [9.5]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cast_to_data_dtype(dtype: type) -> None:
    base = np.random.default_rng(0).normal(30, 5, size=(4, 100))
    tx90, tx95 = percentile_thresholds(base, dtype)
    assert tx90.dtype == dtype and tx95.dtype == dtype
    assert tx90.shape == (4,)


def test_nan_ignored_and_all_nan_row_gives_nan() -> None:
    base = np.array([[np.nan, 0.0, 10.0], [np.nan, np.nan, np.nan]])
    tx90, _ = percentile_thresholds(base, np.float64)
    assert tx90[0] == pytest.approx(9.0)
    assert np.isnan(tx90[1])


def test_empty_baseline_gives_nan() -> None:
    tx90, tx95 = percentile_thresholds(np.empty((3, 0)), np.float32)
    assert np.isnan(tx90).all() and np.isnan(tx95).all()


def test_rejects_bad_shape_and_dtype() -> None:
    with pytest.raises(ValueError):
        percentile_thresholds(np.zeros(5), np.float64)
    with pytest.raises(TypeError):
        percentile_thresholds(np.zeros((1, 5)), np.int32)


@given(arrays(np.float64, (3, 20), elements=st.floats(-40, 55)))
def test_tx95_at_least_tx90_within_baseline_range(base: np.ndarray) -> None:
    tx90, tx95 = percentile_thresholds(base, np.float64)
    assert (tx95 >= tx90).all()
    assert (tx90 >= base.min(axis=1)).all() and (tx95 <= base.max(axis=1)).all()
