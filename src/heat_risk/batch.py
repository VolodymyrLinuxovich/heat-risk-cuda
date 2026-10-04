"""Chunked evaluation so GPU memory use stays bounded.

Rows (scenario, location) are independent, so the input is split into blocks of scenarios and
locations and each block is evaluated separately. Days are never split, because runs cross days.
Works with any of the three ``evaluate`` functions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from heat_risk.spec import N_HAZARDS, HazardResult, validate_inputs

Array = npt.NDArray[np.floating]
Evaluate = Callable[[Array, Array, Array, Array, Array], HazardResult]

# Device bytes per input cell: three inputs plus the three day-major copies gpu_cuda makes.
BYTES_PER_CELL_FACTOR = 6


def plan_chunks(
    n_scenarios: int, n_locations: int, n_days: int, itemsize: int, max_bytes: int
) -> list[tuple[slice, slice]]:
    """Split [scenario, location] into blocks whose device footprint fits in ``max_bytes``."""
    per_row = max(n_days, 1) * itemsize * BYTES_PER_CELL_FACTOR
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
) -> HazardResult:
    validate_inputs(tmax, tmin, rh, tx90, tx95)
    s, n_loc, days = tmax.shape
    counts = np.zeros((s, n_loc, N_HAZARDS), dtype=np.int32)
    runs = np.zeros((s, n_loc, N_HAZARDS), dtype=np.int32)
    for ss, ls in plan_chunks(s, n_loc, days, tmax.dtype.itemsize, max_bytes):
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
