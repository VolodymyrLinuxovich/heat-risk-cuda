"""Hazard constants, input validation and the result container shared by every implementation.

The authoritative definitions live in docs/hazards.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

HAZARDS: tuple[str, ...] = ("tx90", "tx95", "hot_night", "heat_index")
N_HAZARDS = len(HAZARDS)

TX_PERCENTILES: tuple[float, float] = (90.0, 95.0)
HOT_NIGHT_C = 22.0
HEAT_INDEX_C = 32.0
HEAT_INDEX_F = 89.6  # 32 °C expressed in °F; the heat index is compared in °F

SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


@dataclass(frozen=True)
class HazardResult:
    """Per (scenario, location, hazard) outputs, hazards ordered as in ``HAZARDS``."""

    counts: npt.NDArray[np.int32]
    longest_run: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        if self.counts.shape != self.longest_run.shape:
            raise ValueError("counts and longest_run must have the same shape")
        if self.counts.ndim != 3 or self.counts.shape[2] != N_HAZARDS:
            raise ValueError(f"expected shape [scenario, location, {N_HAZARDS}]")

    def hazard(self, name: str) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Return ``(counts, longest_run)`` for one hazard, each shaped [scenario, location]."""
        i = HAZARDS.index(name)
        return self.counts[:, :, i], self.longest_run[:, :, i]


def validate_inputs(
    tmax: npt.NDArray[np.floating],
    tmin: npt.NDArray[np.floating],
    rh: npt.NDArray[np.floating],
    tx90: npt.NDArray[np.floating],
    tx95: npt.NDArray[np.floating],
) -> np.dtype[np.floating]:
    """Check shapes and dtypes; return the shared data dtype."""
    if tmax.ndim != 3:
        raise ValueError("tmax must be shaped [scenario, location, day]")
    if tmin.shape != tmax.shape or rh.shape != tmax.shape:
        raise ValueError("tmax, tmin and rh must have the same shape")
    dtype = tmax.dtype
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported dtype {dtype}; use float32 or float64")
    for name, arr in (("tmin", tmin), ("rh", rh), ("tx90", tx90), ("tx95", tx95)):
        if arr.dtype != dtype:
            raise TypeError(f"{name} has dtype {arr.dtype}, expected {dtype}")
    n_loc = tmax.shape[1]
    if tx90.shape != (n_loc,) or tx95.shape != (n_loc,):
        raise ValueError(f"thresholds must be shaped [location] = ({n_loc},)")
    return dtype
