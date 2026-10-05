"""``cpu_cpp``: the C++ CPU implementation, its build cache and its thread splitting.

Parity on the shared edge cases lives in test_parity.py; the native unit tests in
native/tests run through CTest.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_cpu import EXPECTED_COUNTS, EXPECTED_RUNS, toy

from heat_risk import cpu, cpu_cpp
from heat_risk.synthetic import generate
from heat_risk.thresholds import percentile_thresholds


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_toy_hand_computed(dtype: type) -> None:
    r = cpu_cpp.evaluate(*toy(dtype))
    assert r.counts[0].tolist() == EXPECTED_COUNTS
    assert r.longest_run[0].tolist() == EXPECTED_RUNS


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_threads", [0, 1, 2, 5, 64])
def test_bitwise_parity_without_threshold_band(dtype: type, n_threads: int) -> None:
    # No avoid_threshold_band: the heat index is evaluated in the same order and dtype as NumPy
    # with no fused multiply-add, so even values next to a cutoff must agree.
    d = generate(4, 300, 365, seed=11, baseline_days=200, dtype=dtype)
    tx90, tx95 = percentile_thresholds(d.baseline_tmax, dtype)
    args = (d.tmax, d.tmin, d.rh, tx90, tx95)
    got = cpu_cpp.evaluate(*args, n_threads=n_threads)
    want = cpu.evaluate(*args)
    np.testing.assert_array_equal(got.counts, want.counts)
    np.testing.assert_array_equal(got.longest_run, want.longest_run)


def test_non_contiguous_inputs() -> None:
    d = generate(2, 40, 120, seed=2, baseline_days=100, dtype=np.float64)
    tx90, tx95 = percentile_thresholds(d.baseline_tmax, np.float64)
    view = (d.tmax[:, ::2, ::3], d.tmin[:, ::2, ::3], d.rh[:, ::2, ::3], tx90[::2], tx95[::2])
    assert not view[0].flags.c_contiguous
    got = cpu_cpp.evaluate(*view)
    want = cpu.evaluate(*view)
    np.testing.assert_array_equal(got.counts, want.counts)
    np.testing.assert_array_equal(got.longest_run, want.longest_run)


def test_zero_rows() -> None:
    z = np.zeros((0, 3, 5))
    r = cpu_cpp.evaluate(z, z, z, np.zeros(3), np.zeros(3))
    assert r.counts.shape == (0, 3, 4)


def test_rejects_mixed_dtypes() -> None:
    a = np.zeros((1, 2, 3), dtype=np.float32)
    t = np.zeros(2, dtype=np.float64)
    with pytest.raises(TypeError):
        cpu_cpp.evaluate(a, a, a, t, t)


def test_library_is_cached_on_disk(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HEAT_RISK_CACHE_DIR", str(tmp_path))
    first = cpu_cpp.build_library()
    assert first.parent == tmp_path and first.exists()
    mtime = first.stat().st_mtime_ns
    assert cpu_cpp.build_library() == first
    assert first.stat().st_mtime_ns == mtime
    assert [p.name for p in tmp_path.iterdir()] == [first.name]  # no leftover temp dirs


def test_build_failure_reports_compiler_output(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HEAT_RISK_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cpu_cpp, "_source_dir", lambda: _broken_sources(tmp_path / "src"))
    with pytest.raises(RuntimeError, match="C\\+\\+ build failed"):
        cpu_cpp.build_library()


def _broken_sources(d):  # type: ignore[no-untyped-def]
    d.mkdir(exist_ok=True)
    (d / "heat_hazards.hpp").write_text("")
    (d / "heat_hazards.cpp").write_text("this is not C++;\n")
    return d
