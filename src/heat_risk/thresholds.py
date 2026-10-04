"""Per-location TX90/TX95 thresholds, computed once and shared by every implementation."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

from heat_risk.spec import SUPPORTED_DTYPES, TX_PERCENTILES


def percentile_thresholds(
    baseline_tmax: npt.NDArray[np.floating],
    dtype: npt.DTypeLike,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return ``(tx90, tx95)``, each shaped [location] and cast to ``dtype``.

    ``baseline_tmax`` is a separate array shaped [location, baseline_day] in °C. Percentiles use
    ``np.nanpercentile(method="linear")``, computed in float64. A location whose baseline is all
    NaN (or empty) gets NaN thresholds.
    """
    target = np.dtype(dtype)
    if target not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported dtype {target}; use float32 or float64")
    base = np.asarray(baseline_tmax, dtype=np.float64)
    if base.ndim != 2:
        raise ValueError("baseline_tmax must be shaped [location, baseline_day]")
    n_loc, n_days = base.shape
    if n_days == 0:
        nan = np.full(n_loc, np.nan, dtype=target)
        return nan, nan.copy()
    with warnings.catch_warnings():
        # All-NaN rows are allowed and yield NaN thresholds by design.
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanpercentile(base, TX_PERCENTILES, axis=1, method="linear")
    return q[0].astype(target), q[1].astype(target)
