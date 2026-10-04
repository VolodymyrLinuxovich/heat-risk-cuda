import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from test_cpu import EXPECTED_COUNTS, EXPECTED_RUNS, toy

from heat_risk import cpu, gpu_torch
from heat_risk.heat_index import heat_index_f
from heat_risk.synthetic import avoid_threshold_band, generate
from heat_risk.thresholds import percentile_thresholds


def test_not_a_custom_kernel() -> None:
    assert gpu_torch.is_custom_kernel() is False


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_toy_matches_hand_computed(dtype: type) -> None:
    r = gpu_torch.evaluate(*toy(dtype))
    assert r.counts.dtype == np.int32
    assert r.counts[0].tolist() == EXPECTED_COUNTS
    assert r.longest_run[0].tolist() == EXPECTED_RUNS


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_heat_index_close_on_cpu(dtype: type) -> None:
    # Bitwise equal on macOS arm64, but CI on Linux x86 found 2 of 20000 float32 values 1 ulp
    # apart, so only near-equality is guaranteed across platforms. Counts are protected by
    # keeping inputs away from the 89.6 F cutoff (see avoid_threshold_band).
    rng = np.random.default_rng(0)
    t = rng.uniform(-30, 50, 20000).astype(dtype)
    rh = rng.uniform(0, 100, 20000).astype(dtype)
    want = heat_index_f(t, rh)
    got = gpu_torch.heat_index_f(torch.from_numpy(t), torch.from_numpy(rh)).numpy()
    assert got.dtype == dtype
    np.testing.assert_array_max_ulp(got, want, maxulp=4)


def test_zero_days() -> None:
    z = np.zeros((2, 3, 0), dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    r = gpu_torch.evaluate(z, z, z, t, t)
    assert r.counts.shape == (2, 3, 4) and not r.counts.any()


@settings(max_examples=40, deadline=None)
@given(
    st.integers(1, 3),
    st.integers(1, 6),
    st.integers(0, 60),
    st.integers(0, 2**32 - 1),
    st.sampled_from([np.float32, np.float64]),
)
def test_random_parity_with_cpu_on_cpu_tensors(
    s: int, n_loc: int, days: int, seed: int, dtype: type
) -> None:
    d = generate(s, n_loc, days, seed=seed, baseline_days=60, dtype=dtype)
    tx90, tx95 = percentile_thresholds(d.baseline_tmax, dtype)
    # torch's CPU kernels can differ from NumPy by an ulp on some platforms, so stay off cutoffs.
    d = avoid_threshold_band(d, tx90, tx95)
    tmax = d.tmax.copy()
    tmax[np.random.default_rng(seed).random(tmax.shape) < 0.05] = np.nan  # exercise NaN policy
    want = cpu.evaluate(tmax, d.tmin, d.rh, tx90, tx95)
    got = gpu_torch.evaluate(tmax, d.tmin, d.rh, tx90, tx95)
    np.testing.assert_array_equal(got.counts, want.counts)
    np.testing.assert_array_equal(got.longest_run, want.longest_run)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_random_parity_with_cpu_on_cuda(dtype: type) -> None:
    # PyTorch's CUDA kernels may fuse multiply-adds, so inputs stay away from every cutoff.
    d = generate(4, 300, 400, seed=11, dtype=dtype)
    tx90, tx95 = percentile_thresholds(d.baseline_tmax, dtype)
    d = avoid_threshold_band(d, tx90, tx95)
    want = cpu.evaluate(d.tmax, d.tmin, d.rh, tx90, tx95)
    got = gpu_torch.evaluate(d.tmax, d.tmin, d.rh, tx90, tx95, device="cuda:0")
    np.testing.assert_array_equal(got.counts, want.counts)
    np.testing.assert_array_equal(got.longest_run, want.longest_run)
