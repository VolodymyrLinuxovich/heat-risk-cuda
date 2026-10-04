# heat-risk-cuda

An independent student project using NVIDIA's open-source CUDA Python tooling. NVIDIA did not
create, endorse, or sponsor it.

It counts heat-hazard days (TX90, TX95, hot nights, heat index ≥ 32 °C) and their longest
consecutive runs over arrays shaped `[scenario, location, day]`, and compares three
implementations of the same specification:

| Name | What it is | Custom kernel? |
|---|---|---|
| `cpu` | NumPy reference implementation | no |
| `gpu_torch` | Ordinary PyTorch library ops | no: library ops baseline, not a custom kernel |
| `gpu_cuda` | CUDA C++ kernel compiled at runtime with NVIDIA `cuda.core` (NVRTC) | yes |

## Status and honesty

- All three implementations are written. `cpu` and `gpu_torch` are tested on CPU; the
  `gpu_cuda` kernel compiles with NVRTC in CI but has **not yet run on a GPU**. See STATUS.md.
- The hazard definitions are in [docs/hazards.md](docs/hazards.md). They are project
  simplifications, not ETCCDI indices.
- **Nothing has been measured on a GPU yet.** No speedup is claimed until it is measured, and
  any benchmark table in this README will be generated from JSON files written by real GPU runs.
- All data is synthetic (seeded generator).

## What has and has not been measured

| Claim | Status |
|---|---|
| `cpu` matches hand-computed cases, NWS chart values and property tests | verified on CPU (macOS and Linux CI) |
| `gpu_torch` matches `cpu` | verified on CPU tensors; also checked once by hand on an Apple M5 GPU via MPS (float32); not yet on CUDA |
| `gpu_cuda` kernel compiles for sm_75, float32 and float64, no `fma` in the PTX | verified in Linux CI with NVRTC, no GPU |
| `gpu_cuda` produces correct results | **not yet run**: needs a CUDA GPU |
| Any timing or speedup | **not yet measured** |

No speedup is claimed until it is measured. When it is, compute-only, layout prep, transfers and
end-to-end time are reported separately, and a GPU result slower than `cpu` is reported as data.

## How to reproduce on Kaggle

Kaggle's free GPU tier offers **T4x2** (two Tesla T4, 16 GB each, compute capability 7.5 /
`sm_75`). Per [Kaggle's GPU documentation](https://www.kaggle.com/docs/efficient-gpu-usage) (as summarised in October 2026) the quota is about **30 GPU hours per week** with sessions
of up to **12 hours**, and accelerators require a phone-verified account.

1. Create a new Kaggle notebook and import `notebooks/kaggle_heat_risk_cuda.ipynb` from this repo.
2. Settings: Accelerator **GPU T4 x2**, Internet **on**.
3. Run all. The notebook probes the environment (`nvidia-smi`, `nvcc`, package versions), clones
   the repo, installs it with the pinned `cuda-core[cu12]==1.2.1`, runs the CPU and GPU test
   suites, benchmarks on `cuda:0`, and copies results to `/kaggle/working/results/`.
4. Download the results, put the JSON in `bench/results/`, run `python -m bench.make_table`, and
   commit the JSON and README together.

The Triton language is not used (its Turing/sm_75 support was dropped in Triton 3.3). It is a
different thing from NVIDIA Triton Inference Server, which is also not used.

## Benchmark results

<!-- bench-table:start -->
**Not yet measured.** No benchmark has been run on a GPU yet, so there are no numbers here. No speedup is claimed until it is measured.
<!-- bench-table:end -->

## Development

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch
pip install -e ".[dev]"
ruff check && mypy src && pytest -m "not gpu"
```

The `gpu` extra (`cuda-core[cu12]`) installs only on Linux and Windows.

## Acknowledgments

- **Heat index:** the US National Weather Service heat index algorithm (the Rothfusz regression
  with Steadman's simple formula and the NWS adjustments), as published at
  https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml.
- **Index definitions:** the ETCCDI climate change indices, used as the reference point that this
  project's simplified TX90/TX95 and hot-night definitions are compared against.
- **Tooling:** NumPy, PyTorch, Hypothesis, pytest, ruff and mypy. The custom kernel (in progress)
  uses NVIDIA's open-source CUDA Python tooling (`cuda.core`). This is an independent project;
  NVIDIA did not create, endorse, or sponsor it.
- **Compute:** GitHub Actions runs the CPU test suite. GPU measurements are planned on Kaggle's free
  T4 notebooks and have not been run yet.
- Developed with AI coding assistance; the author specified the design, reviewed the code, and
  runs all GPU measurements.
