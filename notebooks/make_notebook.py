"""Regenerate notebooks/kaggle_heat_risk_cuda.ipynb: python notebooks/make_notebook.py"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

INTRO = """\
# heat-risk-cuda on Kaggle T4

An independent student project using NVIDIA's open-source CUDA Python tooling. NVIDIA did not
create, endorse, or sponsor it. Repository: https://github.com/VolodymyrLinuxovich/heat-risk-cuda

**Settings before running:** Accelerator = **GPU T4 x2**, Internet = **on** (needed for the
clone and `pip install`). Then *Run all*. The run uses one of the two T4s (`cuda:0`).

Every GPU cell is guarded: without a CUDA device it prints why it is skipping instead of failing,
so the notebook can also be smoke-tested on a CPU machine."""

PROBE = """\
# 1. Probe the environment. Nothing here can fail; it only reports.
import os, shutil, subprocess, sys
from importlib import metadata

def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip() or f"(no output, exit {r.returncode})"
    except Exception as exc:
        return f"(failed: {exc})"

ON_KAGGLE = os.path.isdir("/kaggle")
SMOKE = os.environ.get("HEAT_RISK_NOTEBOOK_SMOKE") == "1"
print("python", sys.version.split()[0], "| on Kaggle:", ON_KAGGLE, "| smoke test:", SMOKE)
print("nvidia-smi:", sh("nvidia-smi") if shutil.which("nvidia-smi") else "not found")
NVCC = shutil.which("nvcc")
print("nvcc:", NVCC or "not found", "|", sh("nvcc --version | tail -1") if NVCC else "")
for dist in ("torch", "numpy", "cuda-python", "cuda-bindings", "cuda-core", "numba"):
    try:
        print(f"{dist:14s}", metadata.version(dist))
    except metadata.PackageNotFoundError:
        print(f"{dist:14s}", "not installed")
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    print("torch.cuda:", HAS_GPU, "|", torch.version.cuda,
          "|", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
except ImportError:
    HAS_GPU = False
    print("torch not installed")"""

INSTALL = """\
# 2. Get the code and install the pinned cuda-core (GPU machines only).
from pathlib import Path

def find_repo():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "pyproject.toml").exists() and (p / "src" / "heat_risk").exists():
            return p
    return None

REPO = find_repo()
if REPO is None and ON_KAGGLE:
    REPO = Path("/kaggle/working/heat-risk-cuda")
    URL = "https://github.com/VolodymyrLinuxovich/heat-risk-cuda"
    print(sh(f"git clone --depth 1 {URL} {REPO}"))
print("repo:", REPO)

if HAS_GPU and not SMOKE:
    print(sh(f'{sys.executable} -m pip install -q -e "{REPO}[dev,gpu]" 2>&1 | tail -3'))
    print("cuda-core", metadata.version("cuda-core"))
else:
    print("skipped: no CUDA device" if not HAS_GPU else "skipped: smoke test")
os.chdir(REPO)"""

CPU_TESTS = """\
# 3. CPU test suite.
if SMOKE:
    print("skipped: smoke test (the suite is what runs this notebook)")
else:
    print(sh(f'{sys.executable} -m pytest -m "not gpu" -q -rs 2>&1 | tail -15'))"""

GPU_TESTS = """\
# 4. GPU and NVRTC tests: the parity tests for gpu_torch[cuda] and gpu_cuda, kernel compile,
#    cubin cache. A skip here is reported as a skip, never as a pass.
if HAS_GPU and not SMOKE:
    cmd = f'{sys.executable} -m pytest -m "gpu or nvrtc" -rs -q 2>&1 | tail -30'
    print(sh(f"HEAT_RISK_REQUIRE_NVRTC=1 {cmd}"))
else:
    print("skipped: no CUDA device" if not HAS_GPU else "skipped: smoke test")"""

BENCH = """\
# 5. Benchmark on cuda:0. Writes bench/results/<time>_<gpu>_dev0.json and refuses to write
#    anything without a GPU name. On CPU this runs the --dry-run path, which writes nothing.
if HAS_GPU and not SMOKE:
    print(sh(f"{sys.executable} -m bench.run --device cuda:0 --reps 20 --warmup 3 "
             "--sizes 1x1000x365 4x1000x365 16x4000x365 32x16000x365 2>&1 | tail -20"))
else:
    print(sh(f"{sys.executable} -m bench.run --dry-run 2>&1 | tail -6"))"""

REPORT = """\
# 6. Regenerate the README table and plot from the JSON, then copy results out for download.
print(sh(f"{sys.executable} -m bench.make_table"))
if HAS_GPU and not SMOKE:
    print(sh(f"{sys.executable} -m bench.plot"))
    if ON_KAGGLE:
        OUT = "/kaggle/working/results"
        print(sh(f"mkdir -p {OUT} && cp bench/results/*.json bench/results/*.png README.md {OUT}/"))
        print(sh(f"ls -l {OUT}"))"""

OUTRO = """\
## After the run

Download `/kaggle/working/results/` (Output tab), copy the JSON into `bench/results/` in the
repository, run `python -m bench.make_table`, and commit the JSON and README together.
Report what was measured, including results where the GPU is slower."""


def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(INTRO),
        new_code_cell(PROBE),
        new_code_cell(INSTALL),
        new_code_cell(CPU_TESTS),
        new_code_cell(GPU_TESTS),
        new_code_cell(BENCH),
        new_code_cell(REPORT),
        new_markdown_cell(OUTRO),
    ]
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    return nb


if __name__ == "__main__":
    out = Path(__file__).with_name("kaggle_heat_risk_cuda.ipynb")
    nbformat.write(build(), out)
    print(f"wrote {out}")
