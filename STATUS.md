# Status

As of 2026-10-04. Counts come from `pytest --collect-only` by marker.

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

## Written but not yet run (needs a CUDA GPU)

57 tests marked `gpu`. They skip without a CUDA device; none has run.

| Test | Cases |
|---|---|
| `test_parity.py` (`gpu_torch[cuda]` and `gpu_cuda`) | 48 |
| `test_batch.py::test_batched_equals_unbatched[gpu_cuda]` | 4 |
| `test_gpu_torch.py::test_random_parity_with_cpu_on_cuda` | 2 |
| `test_kernel_compile.py::test_toy_on_gpu` | 2 |
| `test_kernel_compile.py::test_disk_cache_roundtrip` | 1 |

Also not yet run: the CUDA timing path of `bench/run.py`, `bench/plot.py` on real results, and
the notebook's GPU cells.

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
