import re
from dataclasses import replace

import numpy as np
import pytest
from test_cpu import EXPECTED_COUNTS, EXPECTED_RUNS, toy

from heat_risk import gpu_cuda


def test_is_custom_kernel() -> None:
    assert gpu_cuda.is_custom_kernel() is True


def test_source_is_packaged_with_all_entry_points() -> None:
    src = gpu_cuda.kernel_source()
    for name in gpu_cuda.KERNEL_NAMES.values():
        assert f"HEAT_HAZARDS_ENTRY({name}," in src
    for name in gpu_cuda.TILED_KERNEL_NAMES.values():
        assert f"HEAT_HAZARDS_TILED_ENTRY({name}," in src


def test_tile_rows_match_kernel_source() -> None:
    assert f"#define TILE_ROWS {gpu_cuda.TILE_ROWS}" in gpu_cuda.kernel_source()
    assert gpu_cuda.BLOCK_SIZE == gpu_cuda.TILE_ROWS


def test_unknown_layout_rejected() -> None:
    import torch

    z = torch.zeros((1, 1, 1))
    with pytest.raises(ValueError):
        gpu_cuda.evaluate_tensors(z, z, z, z[0, 0], z[0, 0], layout="bogus")


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


def _nvrtc_flags(fma: bool) -> list[bytes]:
    opts = replace(gpu_cuda.DEFAULT_OPTIONS, arch="compute_75", fma=fma)
    flags: list[bytes] = opts.to_program_options().as_bytes("nvrtc", "ptx")
    return flags


def _ptx(fma: bool) -> str:
    """PTX from NVRTC with exactly the flags cuda.core would pass.

    cuda.core's own PTX path checks the driver version first, which needs libcuda; the CI runner
    has no driver (CI run 37214057672). Calling NVRTC directly needs no driver.
    """
    from cuda.bindings import nvrtc

    def check(result: tuple) -> tuple:  # type: ignore[type-arg]
        if result[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC error {result[0]}")
        return result

    src = gpu_cuda.kernel_source().encode()
    _, prog = check(nvrtc.nvrtcCreateProgram(src, b"heat_hazards.cu", 0, [], []))
    flags = _nvrtc_flags(fma)
    err = nvrtc.nvrtcCompileProgram(prog, len(flags), flags)[0]
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, n = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * n
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(log.decode(errors="replace"))
    _, size = check(nvrtc.nvrtcGetPTXSize(prog))
    buf = b" " * size
    check(nvrtc.nvrtcGetPTX(prog, buf))
    return buf.decode(errors="replace")


@pytest.mark.nvrtc
def test_cuda_core_translates_fma_false_to_fmad_flag() -> None:
    assert b"--fmad=false" in _nvrtc_flags(fma=False)
    assert not any(f.startswith(b"--use_fast_math") for f in _nvrtc_flags(fma=False))


@pytest.mark.nvrtc
def test_compiles_to_ptx_for_sm75_without_fma() -> None:
    ptx = _ptx(fma=False)
    assert ".target sm_75" in ptx
    names = [*gpu_cuda.KERNEL_NAMES.values(), *gpu_cuda.TILED_KERNEL_NAMES.values()]
    for name in names:
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
@pytest.mark.parametrize("layout", gpu_cuda.LAYOUTS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_toy_on_gpu(dtype: type, layout: str) -> None:
    r = gpu_cuda.evaluate(*toy(dtype), layout=layout)
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
