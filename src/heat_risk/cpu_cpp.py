"""``cpu_cpp``: a multithreaded C++17 CPU implementation, loaded with ctypes.

The source is ``native/heat_hazards.hpp`` (the algorithm) and ``native/heat_hazards.cpp`` (a C
ABI). Like ``gpu_cuda``, it is compiled on first use, here with the host C++ compiler (``$CXX``,
else ``c++``), and the shared library is cached on disk keyed by source, flags and compiler
version. ``-ffp-contract=off`` keeps the compiler from fusing multiply-adds, so float32 results
match NumPy exactly. The same sources also build with CMake (``CMakeLists.txt``), which runs the
native unit tests in ``native/tests``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shlex
import subprocess
import sys
import tempfile
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from heat_risk.spec import HEAT_INDEX_F, HOT_NIGHT_C, N_HAZARDS, HazardResult, validate_inputs

Array = npt.NDArray[np.floating]

SOURCES = ("heat_hazards.hpp", "heat_hazards.cpp")
CXXFLAGS = ("-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared")
ABI_VERSION = 1


def compiler() -> list[str]:
    return shlex.split(os.environ.get("CXX") or "c++")


def _source_dir() -> Path:
    return Path(str(resources.files("heat_risk").joinpath("native")))


def _compiler_version(cxx: list[str]) -> str:
    out = subprocess.run([*cxx, "--version"], capture_output=True, text=True, check=True)
    return out.stdout


def _extra_flags() -> list[str]:
    return [] if sys.platform == "darwin" else ["-pthread"]


def cache_key(cxx: list[str]) -> str:
    h = hashlib.sha256()
    for name in SOURCES:
        h.update((_source_dir() / name).read_bytes())
        h.update(b"\0")
    for part in (*cxx, *CXXFLAGS, *_extra_flags(), _compiler_version(cxx)):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


def cache_dir() -> Path:
    root = os.environ.get("HEAT_RISK_CACHE_DIR") or Path.home() / ".cache" / "heat_risk"
    return Path(root)


def build_library() -> Path:
    """Compile the shared library unless a cached one with the same key exists."""
    cxx = compiler()
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    path = cache_dir() / f"heat_hazards_cpp-{cache_key(cxx)}{suffix}"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        out = Path(tmp) / path.name
        cmd = [
            *cxx,
            *CXXFLAGS,
            *_extra_flags(),
            str(_source_dir() / "heat_hazards.cpp"),
            "-o",
            str(out),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"C++ build failed: {shlex.join(cmd)}\n{res.stderr}")
        out.replace(path)
    return path


@cache
def _library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(build_library()))
    lib.heat_hazards_cpp_abi_version.restype = ctypes.c_int
    if lib.heat_hazards_cpp_abi_version() != ABI_VERSION:
        raise RuntimeError("heat_hazards_cpp ABI version mismatch")
    for name, scalar in (
        ("heat_hazards_cpp_f32", ctypes.c_float),
        ("heat_hazards_cpp_f64", ctypes.c_double),
    ):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int64] * 3 + [scalar] * 2 + [ctypes.c_int]
    return lib


def evaluate(
    tmax: Array,
    tmin: Array,
    rh: Array,
    tx90: Array,
    tx95: Array,
    n_threads: int = 0,
) -> HazardResult:
    """Same contract as ``cpu.evaluate``. ``n_threads <= 0`` uses every hardware thread."""
    dtype = validate_inputs(tmax, tmin, rh, tx90, tx95)
    s, n_loc, days = tmax.shape
    n_rows = s * n_loc
    counts = np.zeros((s, n_loc, N_HAZARDS), dtype=np.int32)
    runs = np.zeros((s, n_loc, N_HAZARDS), dtype=np.int32)
    if n_rows == 0:
        return HazardResult(counts=counts, longest_run=runs)
    ins = [np.ascontiguousarray(a) for a in (tmax, tmin, rh, tx90, tx95)]
    scalar: type[np.floating[Any]]
    if dtype == np.float32:
        fn, scalar = _library().heat_hazards_cpp_f32, np.float32
    else:
        fn, scalar = _library().heat_hazards_cpp_f64, np.float64
    fn(
        *(a.ctypes.data for a in ins),
        counts.ctypes.data,
        runs.ctypes.data,
        n_rows,
        n_loc,
        days,
        float(scalar(HOT_NIGHT_C)),
        float(scalar(HEAT_INDEX_F)),
        n_threads,
    )
    return HazardResult(counts=counts, longest_run=runs)
