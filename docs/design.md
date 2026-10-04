# Design

## Three implementations

| Name | Module | Custom kernel? |
|---|---|---|
| `cpu` | `heat_risk.cpu` | no: NumPy reference |
| `gpu_torch` | `heat_risk.gpu_torch` | no: library ops baseline, not a custom kernel |
| `gpu_cuda` | `heat_risk.gpu_cuda` + `kernels/heat_hazards.cu` | yes: CUDA C++ compiled with NVRTC via NVIDIA `cuda.core` |

All three take the same inputs (`[scenario, location, day]` arrays plus per-location thresholds
from `heat_risk.thresholds`) and return the same int32 `[scenario, location, hazard]` counts and
longest runs. The spec is docs/hazards.md.

## Kernel layout

One thread handles one `(scenario, location)` row and loops over all days, keeping a count, a
current run and a best run per hazard in registers. Rows are independent, so there is no
inter-thread communication and no atomics.

The kernel reads **day-major** data, `x[day * n_rows + row]`. On each loop iteration the threads
of a warp read 32 consecutive rows of the same day, so each warp load touches one contiguous
128-byte (float32) or 256-byte (float64) span. With the public `[scenario, location, day]` layout,
adjacent threads would read addresses `days * 4` bytes apart and every load would be scattered.

The price is a transpose (`gpu_cuda.to_day_major`), done on the GPU with one PyTorch copy per
input before the launch. It is not counted as kernel compute: the benchmark reports
`layout_prep` separately, next to `h2d`, `compute`, `d2h` and `end_to_end`. A caller that
already stores data day-major can call `launch_day_major` directly and skip it.

Each thread writes 4 counts and 4 runs at `row * 4 + hazard`. Those stores are not perfectly
coalesced, but they happen once per row, against `days` loads per row, so they do not matter.

Block size is 128 threads. For sizes that are not a multiple of 128, the extra threads in the
last block return immediately (`row >= n_rows`). The parity tests cover such sizes.

## Floating point

- **fma off.** NVRTC contracts `a * b + c` into a fused multiply-add by default; NumPy does not.
  Contraction changes rounding, which can flip a count when a value sits right at a cutoff. The
  kernel is compiled with `fma=False` (NVRTC `--fmad=false`), and a test checks that the PTX
  then contains no `fma.rn` instructions (and that it does when fma is allowed).
- **No fast math.** IEEE division and square root (`prec_div`, `prec_sqrt`), no flush-to-zero.
- **Same operation order** as `heat_index.py`, line by line, including how constants are rounded
  to float32 (`T(1.8)` matches `np.float32(1.8)`).
- Even so, exact bitwise agreement across platforms is not guaranteed: CI found PyTorch's CPU
  kernels 1 ulp off NumPy on Linux x86. Random-input parity tests therefore keep inputs away from
  every cutoff (`synthetic.avoid_threshold_band`). Exact-threshold tests use hand-picked values.

## Compilation and caching

The source is compiled once per process to a cubin for `sm_75` (Tesla T4, Kaggle's only free GPU
type). Compiling to cubin rather than PTX means the driver never has to JIT-compile PTX produced by
an NVRTC newer than itself. The cubin is cached on disk (`~/.cache/heat_risk`, or
`$HEAT_RISK_CACHE_DIR`) under a key built from the source, the compile options, and the NVRTC
version. The two kernel entry points are plain `extern "C"` names (`heat_hazards_f32`,
`heat_hazards_f64`) so a cached cubin can be loaded without NVRTC's template name mapping.

## Streams

`gpu_cuda` launches on torch's current CUDA stream, passed to `cuda.core.launch` through the
`__cuda_stream__` protocol. Kernel work is therefore ordered with the surrounding torch ops (the
transpose before it, the copy back after it) without extra synchronization.

## Not verified yet

Nothing in `gpu_cuda` has run on a GPU. See STATUS.md for what has run where.
