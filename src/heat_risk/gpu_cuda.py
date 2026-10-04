"""``gpu_cuda``: a custom CUDA C++ kernel compiled at runtime with NVIDIA cuda.core (NVRTC).

The kernel source is ``kernels/heat_hazards.cu``. It is compiled once per (source, options,
arch, NVRTC version) to a cubin for sm_75 (Tesla T4), cached on disk, and launched on torch
tensors via ``data_ptr()`` on torch's current CUDA stream. Two kernels exist: the default
shared-memory tiled kernel reads the original layout; the day-major kernel needs a transpose.

``cuda.core`` is imported lazily, so ``import heat_risk.gpu_cuda`` works without the ``gpu``
extra (for example on macOS, where no cuda-core wheel exists). Only compiling and launching need
it. Both kernels have passed the GPU test suite on a Tesla T4; see STATUS.md.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import torch

from heat_risk.spec import HEAT_INDEX_F, HOT_NIGHT_C, N_HAZARDS, HazardResult, validate_inputs

ARCH = "sm_75"
BLOCK_SIZE = 128
KERNEL_NAMES = {torch.float32: "heat_hazards_f32", torch.float64: "heat_hazards_f64"}
TILED_KERNEL_NAMES = {
    torch.float32: "heat_hazards_tiled_f32",
    torch.float64: "heat_hazards_tiled_f64",
}
TILE_ROWS = 128  # must match TILE_ROWS in heat_hazards.cu; the tiled kernel needs block == tile
LAYOUTS = ("day_major", "tiled")


def is_custom_kernel() -> bool:
    return True


def kernel_source() -> str:
    return resources.files("heat_risk").joinpath("kernels/heat_hazards.cu").read_text()


@dataclass(frozen=True)
class CompileOptions:
    """Options that change the generated code. All of them are part of the cache key."""

    arch: str = ARCH
    fma: bool = False
    std: str = "c++17"

    def to_program_options(self) -> Any:
        from cuda.core import ProgramOptions

        return ProgramOptions(
            name="heat_hazards",
            arch=self.arch,
            fma=self.fma,
            use_fast_math=False,
            prec_div=True,
            prec_sqrt=True,
            ftz=False,
            std=self.std,
        )


DEFAULT_OPTIONS = CompileOptions()


def nvrtc_version() -> str:
    from cuda.bindings import nvrtc

    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion failed: {err}")
    return f"{major}.{minor}"


def cache_key(source: str, options: CompileOptions, nvrtc: str) -> str:
    h = hashlib.sha256()
    for part in (source, repr(options), nvrtc):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


def cache_dir() -> Path:
    root = os.environ.get("HEAT_RISK_CACHE_DIR") or Path.home() / ".cache" / "heat_risk"
    return Path(root)


def compile_kernel(options: CompileOptions, target: str = "cubin") -> Any:
    """Compile the kernel source with NVRTC and return a cuda.core ``ObjectCode``.

    Needs only the NVRTC library, not a GPU, when ``options.arch`` is given.
    """
    from cuda.core import Program

    prog = Program(kernel_source(), code_type="c++", options=options.to_program_options())
    return prog.compile(target)


@cache
def _load_module(options: CompileOptions) -> Any:
    """Return an ``ObjectCode`` for the cubin, compiling only on a disk-cache miss."""
    from cuda.core import ObjectCode

    key = cache_key(kernel_source(), options, nvrtc_version())
    path = cache_dir() / f"heat_hazards-{options.arch}-{key}.cubin"
    if path.exists():
        return ObjectCode.from_cubin(path.read_bytes())
    obj = compile_kernel(options, "cubin")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(bytes(obj.code))
    tmp.replace(path)
    return obj


class _TorchStream:
    """Expose a torch CUDA stream through the ``__cuda_stream__`` protocol."""

    def __init__(self, stream: torch.cuda.Stream) -> None:
        self._handle = int(stream.cuda_stream)

    def __cuda_stream__(self) -> tuple[int, int]:
        return (0, self._handle)


def to_day_major(x: torch.Tensor) -> torch.Tensor:
    """[scenario, location, day] -> contiguous [day, scenario * location]."""
    s, n_loc, days = x.shape
    return x.reshape(s * n_loc, days).t().contiguous()


def _launch(
    kernel_name: str,
    tmax: torch.Tensor,
    tmin: torch.Tensor,
    rh: torch.Tensor,
    tx90: torch.Tensor,
    tx95: torch.Tensor,
    n_rows: int,
    n_locations: int,
    days: int,
    options: CompileOptions,
) -> tuple[torch.Tensor, torch.Tensor]:
    from cuda.core import Device, LaunchConfig, launch

    dev = tmax.device
    counts = torch.zeros((n_rows, N_HAZARDS), dtype=torch.int32, device=dev)
    runs = torch.zeros((n_rows, N_HAZARDS), dtype=torch.int32, device=dev)
    if n_rows == 0 or days == 0:
        return counts, runs
    for t in (tmax, tmin, rh, tx90, tx95):
        if not t.is_cuda or not t.is_contiguous() or t.device != dev:
            raise ValueError("all inputs must be contiguous CUDA tensors on one device")
    if n_rows * N_HAZARDS >= 2**31:
        raise ValueError("too many rows for int32 output indexing")

    device = Device(dev.index)
    device.set_current()
    stream = device.create_stream(_TorchStream(torch.cuda.current_stream(dev)))
    kernel = _load_module(options).get_kernel(kernel_name)
    scalar = np.float32 if tmax.dtype == torch.float32 else np.float64
    config = LaunchConfig(grid=(n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE, block=BLOCK_SIZE)
    launch(
        stream,
        config,
        kernel,
        tmax.data_ptr(),
        tmin.data_ptr(),
        rh.data_ptr(),
        tx90.data_ptr(),
        tx95.data_ptr(),
        counts.data_ptr(),
        runs.data_ptr(),
        np.int32(n_rows),
        np.int32(n_locations),
        np.int32(days),
        scalar(HOT_NIGHT_C),
        scalar(HEAT_INDEX_F),
    )
    return counts, runs


def launch_day_major(
    tmax_dm: torch.Tensor,
    tmin_dm: torch.Tensor,
    rh_dm: torch.Tensor,
    tx90: torch.Tensor,
    tx95: torch.Tensor,
    n_locations: int,
    options: CompileOptions = DEFAULT_OPTIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch on day-major CUDA tensors; returns int32 ``(counts, runs)`` shaped [row, hazard].

    Asynchronous on torch's current stream, like any torch op.
    """
    days, n_rows = tmax_dm.shape
    name = KERNEL_NAMES[tmax_dm.dtype]
    return _launch(name, tmax_dm, tmin_dm, rh_dm, tx90, tx95, n_rows, n_locations, days, options)


def launch_tiled(
    tmax: torch.Tensor,
    tmin: torch.Tensor,
    rh: torch.Tensor,
    tx90: torch.Tensor,
    tx95: torch.Tensor,
    options: CompileOptions = DEFAULT_OPTIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the shared-memory tiled kernel on the original [scenario, location, day] layout.

    No transpose is needed. Returns int32 ``(counts, runs)`` shaped [row, hazard].
    """
    if BLOCK_SIZE != TILE_ROWS:
        raise RuntimeError("the tiled kernel needs BLOCK_SIZE == TILE_ROWS")
    s, n_loc, days = tmax.shape
    name = TILED_KERNEL_NAMES[tmax.dtype]
    return _launch(name, tmax, tmin, rh, tx90, tx95, s * n_loc, n_loc, days, options)


def evaluate_tensors(
    tmax: torch.Tensor,
    tmin: torch.Tensor,
    rh: torch.Tensor,
    tx90: torch.Tensor,
    tx95: torch.Tensor,
    options: CompileOptions = DEFAULT_OPTIONS,
    layout: str = "tiled",
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA tensors [scenario, location, day] in; int32 [scenario, location, hazard] out.

    ``layout="tiled"`` (default) runs the shared-memory tiled kernel on the original layout.
    ``layout="day_major"`` transposes on the GPU first and runs the day-major kernel; on a
    Tesla T4 that transpose made end-to-end time about 2.2x slower at large sizes.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    s, n_loc, _ = tmax.shape
    if layout == "tiled":
        counts, runs = launch_tiled(
            tmax.contiguous(), tmin.contiguous(), rh.contiguous(),
            tx90.contiguous(), tx95.contiguous(), options,
        )
        return counts.reshape(s, n_loc, N_HAZARDS), runs.reshape(s, n_loc, N_HAZARDS)
    counts, runs = launch_day_major(
        to_day_major(tmax),
        to_day_major(tmin),
        to_day_major(rh),
        tx90.contiguous(),
        tx95.contiguous(),
        n_loc,
        options,
    )
    return counts.reshape(s, n_loc, N_HAZARDS), runs.reshape(s, n_loc, N_HAZARDS)


def evaluate(
    tmax: np.ndarray,
    tmin: np.ndarray,
    rh: np.ndarray,
    tx90: np.ndarray,
    tx95: np.ndarray,
    device: str | torch.device = "cuda:0",
    layout: str = "tiled",
) -> HazardResult:
    """NumPy in, NumPy out. Needs a CUDA device and the ``gpu`` extra."""
    validate_inputs(tmax, tmin, rh, tx90, tx95)
    arrays = (tmax, tmin, rh, tx90, tx95)
    t = [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays]
    counts, runs = evaluate_tensors(t[0], t[1], t[2], t[3], t[4], layout=layout)
    return HazardResult(counts=counts.cpu().numpy(), longest_run=runs.cpu().numpy())
