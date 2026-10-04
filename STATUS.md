# Status

As of 2026-10-04.

## Ran and passed

| Where | Command | Result |
|---|---|---|
| macOS arm64 (author's machine), Linux x86 CI (Python 3.11, 3.12) | `pytest -m "not gpu"` | 136 passed; the 4 `nvrtc` tests skip on macOS (no cuda-core wheel) |
| Linux x86 CI, no GPU, job `nvrtc` | `HEAT_RISK_REQUIRE_NVRTC=1 pytest -m nvrtc` | 4 passed: cuda.core passes `--fmad=false`; PTX for sm_75 has both entry points and no `fma.rn`; with fma allowed NVRTC does emit `fma.rn` (control); cubin for sm_75 compiles |
| macOS arm64 | `python -m bench.run --dry-run` | runs cpu and gpu_torch on CPU, both match cpu, writes nothing |
| Apple M5 GPU via PyTorch MPS, once by hand | not in the suite | gpu_torch float32 matched cpu on the toy case and 4×300×400 random data; heat index bitwise equal on 200,000 points |

The 136 CPU tests cover: thresholds, spec, heat index (NWS chart values, regime and adjustment
formulas, float32 vs float64), synthetic data, the cpu reference (hand-computed toy case and
properties), gpu_torch on CPU tensors, cross-implementation parity for `cpu` and
`gpu_torch[cpu]`, batching, the benchmark refusal path, dry run and table generator, and a CPU
smoke run of the Kaggle notebook.

## Ran on a GPU: Kaggle, 2 × Tesla T4 (2026-10-04)

Private Kaggle notebook run of `notebooks/kaggle_heat_risk_cuda.ipynb` on `cuda:0`, driver
580.178.04, torch 2.11.0+cu128, cuda-core 1.2.1, Python 3.13:

| Command | Result |
|---|---|
| `pytest -m "not gpu"` | 138 passed, 2 skipped (two tests that only apply without CUDA) |
| `HEAT_RISK_REQUIRE_NVRTC=1 pytest -m "gpu or nvrtc"` | **61 passed**: all 57 gpu tests (`gpu_cuda` and `gpu_torch[cuda]` parity on every edge case, batching, toy case, cubin disk cache) and the 4 NVRTC compile tests |

So the custom kernel gives exactly the cpu reference's counts and runs on a T4, in float32 and
float64.

## Not yet measured

The benchmark did not run in that session: the notebook's shell helper had a 120 s timeout,
which killed `bench.run`. Fixed (no timeout for the benchmark cell); see the next run.

## Next: on Kaggle (GPU T4 x2, Internet on)

Either run `notebooks/kaggle_heat_risk_cuda.ipynb` top to bottom, or in a Kaggle terminal:

```bash
git clone https://github.com/VolodymyrLinuxovich/heat-risk-cuda && cd heat-risk-cuda
nvidia-smi; which nvcc
pip install -e ".[dev,gpu]"
pytest -m "not gpu" -q
HEAT_RISK_REQUIRE_NVRTC=1 pytest -m "gpu or nvrtc" -rs
python -m bench.run --device cuda:0 --reps 20 --warmup 3 \
    --sizes 1x1000x365 4x1000x365 16x4000x365 32x16000x365
python -m bench.make_table && python -m bench.plot
```

Then copy `bench/results/*.json` into the repo, regenerate the README table with
`python -m bench.make_table`, commit, and update this file with what passed, what failed, and
the GPU name, driver and package versions recorded in the JSON.
