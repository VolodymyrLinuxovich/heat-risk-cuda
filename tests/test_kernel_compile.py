import re
from dataclasses import replace

import numpy as np
import pytest
from test_cpu import EXPECTED_COUNTS, EXPECTED_RUNS, toy

from heat_risk import gpu_cuda


def test_is_custom_kernel() -> None:
    assert gpu_cuda.is_custom_kernel() is True


def test_source_is_packaged_with_both_entry_points() -> None:
    src = gpu_cuda.kernel_source()
    for name in gpu_cuda.KERNEL_NAMES.values():
        assert f"HEAT_HAZARDS_ENTRY({name}," in src


def test_cache_key_covers_source_options_and_nvrtc() -> None:
    o = gpu_cuda.CompileOptions()
    base = gpu_cuda.cache_key("src", o, "12.9")
    assert base == gpu_cuda.cache_key("src", o, "12.9")
    assert base != gpu_cuda.cache_key("src2", o, "12.9")
    assert base != gpu_cuda.cache_key("src", replace(o, fma=True), "12.9")
    assert base != gpu_cuda.cache_key("src", replace(o, arch="sm_80"), "12.9")
    assert base != gpu_cuda.cache_key("src", o, "12.8")


def test_default_options_disable_fma() -> None:
    assert gpu_cuda.DEFAULT_OPTIONS.fma is False and gpu_cuda.DEFAULT_OPTIONS.arch == "sm_75"


def _ptx(fma: bool) -> str:
    opts = replace(gpu_cuda.DEFAULT_OPTIONS, arch="compute_75", fma=fma)
    code = gpu_cuda.compile_kernel(opts, "ptx").code
    return code.decode() if isinstance(code, bytes) else str(code)


@pytest.mark.nvrtc
def test_compiles_to_ptx_for_sm75_without_fma() -> None:
    ptx = _ptx(fma=False)
    assert ".target sm_75" in ptx
    for name in gpu_cuda.KERNEL_NAMES.values():
        assert re.search(rf"\.entry\s+{name}\b", ptx), name
    assert not re.search(r"\bfma\.rn\.f(32|64)\b", ptx), "fma found despite fma=False"


@pytest.mark.nvrtc
def test_fma_flag_reaches_nvrtc() -> None:
    # Control for the test above: with contraction allowed, NVRTC does emit fma.
    assert re.search(r"\bfma\.rn\.f(32|64)\b", _ptx(fma=True))


@pytest.mark.nvrtc
def test_compiles_to_cubin_for_sm75() -> None:
    obj = gpu_cuda.compile_kernel(gpu_cuda.DEFAULT_OPTIONS, "cubin")
    assert len(bytes(obj.code)) > 0


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_toy_on_gpu(dtype: type) -> None:
    r = gpu_cuda.evaluate(*toy(dtype))
    assert r.counts[0].tolist() == EXPECTED_COUNTS
    assert r.longest_run[0].tolist() == EXPECTED_RUNS


@pytest.mark.gpu
def test_disk_cache_roundtrip(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HEAT_RISK_CACHE_DIR", str(tmp_path))
    gpu_cuda._load_module.cache_clear()
    first = gpu_cuda.evaluate(*toy(np.float32))
    assert len(list(tmp_path.glob("*.cubin"))) == 1
    gpu_cuda._load_module.cache_clear()
    second = gpu_cuda.evaluate(*toy(np.float32))  # loads the cached cubin
    np.testing.assert_array_equal(first.counts, second.counts)
