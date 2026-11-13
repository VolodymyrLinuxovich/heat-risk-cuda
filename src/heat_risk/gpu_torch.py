"""``gpu_torch``: library ops baseline, not a custom kernel.

The same hazard spec written with ordinary PyTorch tensor operations. It runs on whatever device
its input tensors live on (CPU in tests, a CUDA device in the benchmark). On a GPU each operation
is a separate PyTorch kernel launch; nothing here is a hand-written kernel. The heat index mirrors
``heat_index.py`` line by line.
"""

from __future__ import annotations

import numpy as np
import torch

from heat_risk.spec import HEAT_INDEX_F, HOT_NIGHT_C, N_HAZARDS, HazardResult, validate_inputs


def is_custom_kernel() -> bool:
    return False


def heat_index_f(tmax_c: torch.Tensor, rh: torch.Tensor) -> torch.Tensor:
    """Heat index in °F; mirrors ``heat_risk.heat_index.heat_index_f``."""
    if tmax_c.dtype != rh.dtype:
        raise TypeError("tmax_c and rh must share a dtype")
    t = tmax_c * 1.8 + 32.0

    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)

    full = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * rh
        - 0.22475541 * t * rh
        - 0.00683783 * t * t
        - 0.05481717 * rh * rh
        + 0.00122874 * t * t * rh
        + 0.00085282 * t * rh * rh
        - 0.00000199 * t * t * rh * rh
    )

    in_low = (rh < 13.0) & (t >= 80.0) & (t <= 112.0)
    zero = torch.zeros((), dtype=t.dtype, device=t.device)
    low_arg = torch.where(in_low, (17.0 - torch.abs(t - 95.0)) / 17.0, zero)
    low_adj = ((13.0 - rh) / 4.0) * torch.sqrt(low_arg)
    full = torch.where(in_low, full - low_adj, full)

    in_high = (rh > 85.0) & (t >= 80.0) & (t <= 87.0)
    high_adj = ((rh - 85.0) / 10.0) * ((87.0 - t) / 5.0)
    full = torch.where(in_high, full + high_adj, full)

    use_full = (simple + t) / 2.0 >= 80.0
    return torch.where(use_full, full, simple)


SUPPORTED_DTYPES = (torch.float32, torch.float64)


def _check_tensors(
    tmax: torch.Tensor,
    tmin: torch.Tensor,
    rh: torch.Tensor,
    tx90: torch.Tensor,
    tx95: torch.Tensor,
) -> None:
    """Apply the dtype and shape rules of ``validate_inputs`` to tensors.

    Without this, thresholds shaped (1,) broadcast over every location, and float64 thresholds
    with float32 data compare in float64. Both give wrong counts with no error.
    """
    if tmax.ndim != 3:
        raise ValueError("tmax must be shaped [scenario, location, day]")
    if tmin.shape != tmax.shape or rh.shape != tmax.shape:
        raise ValueError("tmax, tmin and rh must have the same shape")
    dtype = tmax.dtype
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported dtype {dtype}; use float32 or float64")
    for name, t in (("tmin", tmin), ("rh", rh), ("tx90", tx90), ("tx95", tx95)):
        if t.dtype != dtype:
            raise TypeError(f"{name} has dtype {t.dtype}, expected {dtype}")
    n_loc = tmax.shape[1]
    if tx90.shape != (n_loc,) or tx95.shape != (n_loc,):
        raise ValueError(f"thresholds must be shaped [location] = ({n_loc},)")


def evaluate_tensors(
    tmax: torch.Tensor,
    tmin: torch.Tensor,
    rh: torch.Tensor,
    tx90: torch.Tensor,
    tx95: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(counts, longest_run)`` as int32 tensors [scenario, location, hazard] on the
    input device. Inputs follow the same layout and dtype rules as ``cpu.evaluate``."""
    _check_tensors(tmax, tmin, rh, tx90, tx95)
    s, n_loc, days = tmax.shape
    if days == 0:
        z = torch.zeros((s, n_loc, N_HAZARDS), dtype=torch.int32, device=tmax.device)
        return z, z.clone()
    masks = torch.stack(
        [
            tmax >= tx90[None, :, None],
            tmax >= tx95[None, :, None],
            tmin >= HOT_NIGHT_C,
            heat_index_f(tmax, rh) >= HEAT_INDEX_F,
        ],
        dim=-1,
    )  # [scenario, location, day, hazard]; comparisons with NaN are False
    counts = masks.sum(dim=2, dtype=torch.int32)
    c = torch.cumsum(masks, dim=2, dtype=torch.int32)
    reset = torch.cummax(torch.where(masks, torch.zeros_like(c), c), dim=2).values
    runs = (c - reset).amax(dim=2).to(torch.int32)
    return counts, runs


def evaluate(
    tmax: np.ndarray,
    tmin: np.ndarray,
    rh: np.ndarray,
    tx90: np.ndarray,
    tx95: np.ndarray,
    device: str | torch.device = "cpu",
) -> HazardResult:
    """NumPy in, NumPy out convenience wrapper around ``evaluate_tensors``."""
    validate_inputs(tmax, tmin, rh, tx90, tx95)
    arrays = (tmax, tmin, rh, tx90, tx95)
    args = [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays]
    counts, runs = evaluate_tensors(*args)
    return HazardResult(counts=counts.cpu().numpy(), longest_run=runs.cpu().numpy())
