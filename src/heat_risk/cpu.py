"""``cpu``: the NumPy reference implementation of docs/hazards.md."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from heat_risk.heat_index import heat_index_f
from heat_risk.spec import HEAT_INDEX_F, HOT_NIGHT_C, HazardResult, validate_inputs

Array = npt.NDArray[np.floating]


def hazard_masks(
    tmax: Array, tmin: Array, rh: Array, tx90: Array, tx95: Array
) -> npt.NDArray[np.bool_]:
    """Boolean [scenario, location, day, hazard] mask of days that count.

    Any comparison involving NaN is False, which implements the NaN policy.
    """
    dt = tmax.dtype.type
    return np.stack(
        [
            tmax >= tx90[None, :, None],
            tmax >= tx95[None, :, None],
            tmin >= dt(HOT_NIGHT_C),
            heat_index_f(tmax, rh) >= dt(HEAT_INDEX_F),
        ],
        axis=-1,
    )


def longest_runs(mask: npt.NDArray[np.bool_], axis: int) -> npt.NDArray[np.int32]:
    """Longest streak of consecutive True values along ``axis`` (0 when empty)."""
    m = np.moveaxis(mask, axis, -1)
    if m.shape[-1] == 0:
        return np.zeros(m.shape[:-1], dtype=np.int32)
    c = np.cumsum(m, axis=-1, dtype=np.int64)
    # Running length = cumulative count minus the count at the most recent False day.
    reset = np.maximum.accumulate(np.where(m, 0, c), axis=-1)
    runs: npt.NDArray[np.int32] = (c - reset).max(axis=-1).astype(np.int32)
    return runs


def evaluate(tmax: Array, tmin: Array, rh: Array, tx90: Array, tx95: Array) -> HazardResult:
    """Count hazard days and longest runs per (scenario, location, hazard).

    Inputs are [scenario, location, day] in one float dtype; thresholds are [location] in the
    same dtype, as returned by ``heat_risk.thresholds.percentile_thresholds``.
    """
    validate_inputs(tmax, tmin, rh, tx90, tx95)
    masks = hazard_masks(tmax, tmin, rh, tx90, tx95)
    counts = masks.sum(axis=2, dtype=np.int32)
    return HazardResult(counts=counts, longest_run=longest_runs(masks, axis=2))
