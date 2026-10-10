"""Benchmark cpu, cpu_cpp, gpu_torch and gpu_cuda on synthetic data.

Usage on a CUDA machine (e.g. Kaggle T4x2):
    python -m bench.run --sizes 4x1000x365 16x4000x365 --reps 20
CPU smoke test, never writes a file:
    python -m bench.run --dry-run

Stages are timed separately and reported as the median over ``--reps`` after ``--warmup``
untimed runs:
    h2d          host-to-device copy of the five inputs
    layout_prep  [scenario, location, day] -> day-major transpose (gpu_cuda only;
                 gpu_cuda_tiled reads the original layout and has no such stage)
    compute      the hazard evaluation itself
    d2h          device-to-host copy of counts and runs
    end_to_end   all of the above in one timed region
GPU stages use CUDA events on the current stream; cpu and cpu_cpp (C++, all host threads) use
time.perf_counter.

Results are written only from a real GPU run that reports a device name. Nothing here invents
or fills in numbers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from heat_risk import cpu, cpu_cpp, gpu_cuda, gpu_torch
from heat_risk.synthetic import avoid_threshold_band, generate
from heat_risk.thresholds import percentile_thresholds

SCHEMA_VERSION = 1
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_SIZES = ["4x1000x365", "16x4000x365", "32x16000x365"]
DRY_RUN_SIZES = ["2x50x30", "1x129x10"]


class ResultsRefused(RuntimeError):
    """Raised instead of writing results that do not come from a real GPU run."""


def parse_size(text: str) -> tuple[int, int, int]:
    parts = text.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"size must be SxLxD, got {text!r}")
    s, n_loc, days = (int(p) for p in parts)
    return s, n_loc, days


# --- timing -------------------------------------------------------------------------------


class Timer:
    def __init__(self, device: torch.device | None) -> None:
        self.cuda = device is not None and device.type == "cuda"

    def time(self, fn: Callable[[], Any]) -> tuple[float, Any]:
        """Run ``fn`` once; return (milliseconds, result)."""
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end)), out
        t0 = time.perf_counter()
        out = fn()
        return (time.perf_counter() - t0) * 1e3, out


def _median_stages(samples: list[dict[str, float]]) -> dict[str, float]:
    return {k: statistics.median(s[k] for s in samples) for k in samples[0]}


# --- one implementation, one size ---------------------------------------------------------


def run_cpu(
    arrays: tuple[np.ndarray, ...],
    reps: int,
    warmup: int,
    evaluate: Callable[..., Any] = cpu.evaluate,
) -> tuple[dict[str, float], Any]:
    timer = Timer(None)
    for _ in range(warmup):
        evaluate(*arrays)
    samples = []
    out = None
    for _ in range(reps):
        ms, out = timer.time(lambda: evaluate(*arrays))
        samples.append({"compute": ms, "end_to_end": ms})
    return _median_stages(samples), out


def run_device(
    impl: str,
    arrays: tuple[np.ndarray, ...],
    device: torch.device,
    reps: int,
    warmup: int,
    options: gpu_cuda.CompileOptions,
) -> tuple[dict[str, float], Any]:
    timer = Timer(device)
    host = [torch.from_numpy(np.ascontiguousarray(a)) for a in arrays]
    if device.type == "cuda":
        host = [h.pin_memory() for h in host]

    def h2d() -> list[torch.Tensor]:
        return [h.to(device, non_blocking=True) for h in host]

    def compute(d: list[torch.Tensor], prepped: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if impl == "gpu_torch":
            return gpu_torch.evaluate_tensors(*d)
        if impl == "gpu_cuda_tiled":
            return gpu_cuda.launch_tiled(d[0], d[1], d[2], d[3], d[4], options)
        tmax_dm, tmin_dm, rh_dm = prepped
        n_loc = d[0].shape[1]
        return gpu_cuda.launch_day_major(tmax_dm, tmin_dm, rh_dm, d[3], d[4], n_loc, options)

    def prep(d: list[torch.Tensor]) -> Any:
        if impl != "gpu_cuda":
            return None  # gpu_torch and gpu_cuda_tiled use the original layout
        return tuple(gpu_cuda.to_day_major(x) for x in d[:3])

    def d2h(out: tuple[torch.Tensor, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        return out[0].cpu().numpy(), out[1].cpu().numpy()

    def full() -> tuple[np.ndarray, np.ndarray]:
        d = h2d()
        return d2h(compute(d, prep(d)))

    for _ in range(warmup):
        full()
    samples = []
    result = None
    for _ in range(reps):
        stage: dict[str, float] = {}
        stage["h2d"], d = timer.time(h2d)
        prepped: Any = None
        if impl == "gpu_cuda":
            stage["layout_prep"], prepped = timer.time(partial(prep, d))
        stage["compute"], out = timer.time(partial(compute, d, prepped))
        stage["d2h"], _ = timer.time(partial(d2h, out))
        stage["end_to_end"], result = timer.time(full)
        samples.append(stage)
    return _median_stages(samples), result


# --- metadata -----------------------------------------------------------------------------


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _driver_version(index: int) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", f"-i={index}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def gpu_info(device: torch.device | None) -> dict[str, Any]:
    if device is None or device.type != "cuda":
        return {"name": None, "device_index": None, "driver": None, "compute_capability": None}
    index = device.index or 0
    major, minor = torch.cuda.get_device_capability(index)
    return {
        "name": torch.cuda.get_device_name(index),
        "device_index": index,
        "device_count": torch.cuda.device_count(),
        "driver": _driver_version(index),
        "compute_capability": f"{major}.{minor}",
    }


def software_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host_cpu_threads": os.cpu_count(),  # cpu_cpp uses all of them
        "cxx": " ".join(cpu_cpp.compiler()),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_core": _version("cuda-core"),
        "cuda_bindings": _version("cuda-bindings"),
        "heat_risk": _version("heat-risk-cuda"),
    }


# --- writing ------------------------------------------------------------------------------


def write_results(record: dict[str, Any], out_dir: Path = RESULTS_DIR) -> Path:
    """Write a results JSON, refusing anything that is not from a real GPU run."""
    if record.get("dry_run"):
        raise ResultsRefused("dry-run results are never written")
    gpu = record.get("gpu") or {}
    if not gpu.get("name"):
        raise ResultsRefused("no GPU name recorded; refusing to write benchmark results")
    if gpu.get("device_index") is None:
        raise ResultsRefused("no GPU device index recorded; refusing to write benchmark results")
    if not record.get("results"):
        raise ResultsRefused("no measurements in record")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = record["created_utc"].replace(":", "").replace("-", "")
    slug = "".join(c if c.isalnum() else "-" for c in gpu["name"]).strip("-").lower()
    base = f"{stamp}_{slug}_dev{gpu['device_index']}"
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    # The stamp has one second resolution, so two runs can share a name. Mode "x" never
    # replaces an existing file; a later run gets a numbered suffix instead.
    n = 1
    while True:
        path = out_dir / (f"{base}.json" if n == 1 else f"{base}_{n}.json")
        try:
            with path.open("x") as f:
                f.write(text)
        except FileExistsError:
            n += 1
            continue
        return path


# --- main ---------------------------------------------------------------------------------


def benchmark(
    sizes: list[tuple[int, int, int]],
    impls: list[str],
    device: torch.device | None,
    reps: int,
    warmup: int,
    seed: int,
    dtype: type,
    fma: bool,
    dry_run: bool,
) -> dict[str, Any]:
    options = gpu_cuda.CompileOptions(fma=fma)
    results = []
    for s, n_loc, days in sizes:
        data = generate(s, n_loc, days, seed=seed, dtype=dtype)
        tx90, tx95 = percentile_thresholds(data.baseline_tmax, dtype)
        data = avoid_threshold_band(data, tx90, tx95)  # so the parity check is meaningful
        arrays = (data.tmax, data.tmin, data.rh, tx90, tx95)
        reference = cpu.evaluate(*arrays)
        for impl in impls:
            if impl in ("cpu", "cpu_cpp"):
                fn = cpu.evaluate if impl == "cpu" else cpu_cpp.evaluate
                stages, out = run_cpu(arrays, reps, warmup, fn)
                got = (out.counts, out.longest_run)
            else:
                dev = device if device is not None else torch.device("cpu")
                if impl.startswith("gpu_cuda") and dev.type != "cuda":
                    continue  # the custom kernels need a CUDA device
                try:
                    stages, got = run_device(impl, arrays, dev, reps, warmup, options)
                except torch.cuda.OutOfMemoryError as exc:
                    # A real outcome on this GPU at this size: record it, do not hide it.
                    torch.cuda.empty_cache()
                    results.append(
                        {
                            "impl": impl,
                            "custom_kernel": impl.startswith("gpu_cuda"),
                            "scenarios": s,
                            "locations": n_loc,
                            "days": days,
                            "cells": s * n_loc * days,
                            "stages_ms": None,
                            "matches_cpu": None,
                            "error": f"CUDA out of memory: {str(exc).splitlines()[0]}",
                        }
                    )
                    print(f"{impl:>9} {s}x{n_loc}x{days}: CUDA out of memory")
                    continue
            got_counts = np.asarray(got[0]).reshape(reference.counts.shape)
            got_runs = np.asarray(got[1]).reshape(reference.longest_run.shape)
            matches = bool(
                np.array_equal(got_counts, reference.counts)
                and np.array_equal(got_runs, reference.longest_run)
            )
            results.append(
                {
                    "impl": impl,
                    "custom_kernel": impl.startswith("gpu_cuda"),
                    "scenarios": s,
                    "locations": n_loc,
                    "days": days,
                    "cells": s * n_loc * days,
                    "stages_ms": stages,
                    "matches_cpu": matches,
                }
            )
            print(f"{impl:>9} {s}x{n_loc}x{days}: {json.dumps(stages)} matches_cpu={matches}")
    return {
        "schema": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": dry_run,
        "git_commit": _git_commit(),
        "gpu": gpu_info(device),
        "software": software_info(),
        "config": {
            "reps": reps,
            "warmup": warmup,
            "seed": seed,
            "dtype": np.dtype(dtype).name,
            "fma": fma,
            "block_size": gpu_cuda.BLOCK_SIZE,
            "statistic": "median",
            "data": "synthetic, avoid_threshold_band applied",
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sizes", nargs="+", type=parse_size, default=None)
    p.add_argument(
        "--impls",
        nargs="+",
        default=["cpu", "cpu_cpp", "gpu_torch", "gpu_cuda", "gpu_cuda_tiled"],
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--fma", action="store_true", help="allow fused multiply-add (labelled)")
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    p.add_argument("--dry-run", action="store_true", help="CPU only, tiny sizes, writes nothing")
    a = p.parse_args(argv)

    if a.dry_run:
        device = None
        sizes = a.sizes or [parse_size(x) for x in DRY_RUN_SIZES]
        impls = [i for i in a.impls if not i.startswith("gpu_cuda")]
        reps, warmup = min(a.reps, 3), min(a.warmup, 1)
    else:
        if not torch.cuda.is_available():
            print("No CUDA device. Use --dry-run for a CPU smoke test.", file=sys.stderr)
            return 2
        device = torch.device(a.device)
        torch.cuda.set_device(device)
        sizes = a.sizes or [parse_size(x) for x in DEFAULT_SIZES]
        impls, reps, warmup = a.impls, a.reps, a.warmup

    record = benchmark(
        sizes, impls, device, reps, warmup, a.seed, getattr(np, a.dtype), a.fma, a.dry_run
    )
    if a.dry_run:
        print("dry run: results not written")
        return 0
    path = write_results(record, a.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
