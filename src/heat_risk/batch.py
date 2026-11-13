"""Chunked evaluation so GPU memory use stays bounded.

Rows (scenario, location) are independent, so the input is split into blocks of scenarios and
locations and each block is evaluated separately. Days are never split, because runs cross days.
Works with any of the three ``evaluate`` functions. Pass ``bytes_per_cell`` for the one in use
(see ``device_bytes_per_cell``), because their device footprints differ a lot.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from heat_risk.spec import N_HAZARDS, HazardResult, validate_inputs

Array = npt.NDArray[np.floating]
Evaluate = Callable[[Array, Array, Array, Array, Array], HazardResult]

# Device bytes per input cell, in units of itemsize: three inputs plus the three day-major
# copies gpu_cuda makes. This is the default footprint.
BYTES_PER_CELL_FACTOR = 6

# gpu_torch work bytes per input cell on top of its three inputs, independent of dtype. Per
# hazard (4 of them): bool mask 1, int32 cumsum 4, zeros_like 4, where 4, cummax values 4 and
# cummax int64 indices 8. See the memory table in docs/design.md.
GPU_TORCH_WORK_BYTES_PER_CELL = 4 * (1 + 4 + 4 + 4 + 4 + 8)

IMPLS = ("cpu", "cpu_cpp", "gpu_torch", "gpu_cuda")


def device_bytes_per_cell(impl: str, itemsize: int) -> int:
    """Peak bytes per input cell that ``impl`` needs while evaluating a chunk."""
    if impl not in IMPLS:
        raise ValueError(f"impl must be one of {IMPLS}")
    if impl == "gpu_torch":
        return 3 * itemsize + GPU_TORCH_WORK_BYTES_PER_CELL
    return itemsize * BYTES_PER_CELL_FACTOR


def plan_chunks(
    n_scenarios: int,
    n_locations: int,
    n_days: int,
    itemsize: int,
    max_bytes: int,
    bytes_per_cell: int | None = None,
) -> list[tuple[slice, slice]]:
    """Split [scenario, location] into blocks whose device footprint fits in ``max_bytes``.

    ``bytes_per_cell`` defaults to ``itemsize * BYTES_PER_CELL_FACTOR``, the gpu_cuda footprint.
    """
    if bytes_per_cell is None:
        bytes_per_cell = itemsize * BYTES_PER_CELL_FACTOR
    per_row = max(n_days, 1) * bytes_per_cell
    if per_row > max_bytes:
        raise ValueError(f"one (scenario, location) row needs {per_row} bytes > max_bytes")
    rows_per_chunk = max_bytes // per_row
    if rows_per_chunk >= n_locations:
        loc_step = n_locations
        scen_step = max(1, rows_per_chunk // max(n_locations, 1))
    else:
        loc_step, scen_step = int(rows_per_chunk), 1
    return [
        (slice(s, min(s + scen_step, n_scenarios)), slice(loc, min(loc + loc_step, n_locations)))
        for s in range(0, n_scenarios, scen_step)
        for loc in range(0, n_locations, max(loc_step, 1))
    ]


def evaluate_batched(
    evaluate: Evaluate,
    tmax: Array,
    tmin: Array,
    rh: Array,
    tx90: Array,
    tx95: Array,
    max_bytes: int,
    bytes_per_cell: int | None = None,
) -> HazardResult:
    """Evaluate in chunks that each fit in ``max_bytes`` of device memory.

    Pass ``bytes_per_cell=device_bytes_per_cell(impl, itemsize)`` for the implementation in use.
    The default fits gpu_cuda; gpu_torch needs several times more per cell.
    """
    validate_inputs(tmax, tmin, rh, tx90, tx95)
    s, n_loc, days = tmax.shape
    counts = np.zeros((s, n_loc, N_HAZARDS), dtype=np.int32)
    runs = np.zeros((s, n_loc, N_HAZARDS), dtype=np.int32)
    for ss, ls in plan_chunks(s, n_loc, days, tmax.dtype.itemsize, max_bytes, bytes_per_cell):
        r = evaluate(
            np.ascontiguousarray(tmax[ss, ls]),
            np.ascontiguousarray(tmin[ss, ls]),
            np.ascontiguousarray(rh[ss, ls]),
            np.ascontiguousarray(tx90[ls]),
            np.ascontiguousarray(tx95[ls]),
        )
        counts[ss, ls] = r.counts
        runs[ss, ls] = r.longest_run
    return HazardResult(counts=counts, longest_run=runs)
