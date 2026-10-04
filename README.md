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

- Work in progress. Implemented so far: `cpu` and `gpu_torch`, tested on CPU only. `gpu_cuda`
  is not written yet.
- The hazard definitions are in [docs/hazards.md](docs/hazards.md). They are project
  simplifications, not ETCCDI indices.
- **Nothing has been measured on a GPU yet.** No speedup is claimed until it is measured, and
  any benchmark table in this README will be generated from JSON files written by real GPU runs.
- All data is synthetic (seeded generator).

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
