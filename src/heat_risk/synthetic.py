"""Seeded synthetic daily weather shaped [scenario, location, day].

Each location has its own climate (annual mean, seasonal amplitude, humidity). Daily anomalies
follow an AR(1) process, and each scenario adds a warming offset. Daily data has no diurnal
cycle; ``tmin`` is ``tmax`` minus a noisy diurnal range, so ``tmin <= tmax`` always holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from heat_risk.heat_index import heat_index_f
from heat_risk.spec import HEAT_INDEX_F, HOT_NIGHT_C

Array = npt.NDArray[np.floating]


@dataclass(frozen=True)
class SyntheticData:
    tmax: Array  # [scenario, location, day], °C
    tmin: Array  # [scenario, location, day], °C
    rh: Array  # [scenario, location, day], %
    baseline_tmax: Array  # [location, baseline_day], °C


def _ar1(rng: np.random.Generator, shape: tuple[int, ...], phi: float, sd: float) -> Array:
    eps = rng.normal(0.0, sd * np.sqrt(1.0 - phi * phi), size=shape)
    out = np.empty(shape)
    if shape[-1] == 0:
        return out
    out[..., 0] = rng.normal(0.0, sd, size=shape[:-1])
    for d in range(1, shape[-1]):
        out[..., d] = phi * out[..., d - 1] + eps[..., d]
    return out


def generate(
    n_scenarios: int,
    n_locations: int,
    n_days: int,
    *,
    baseline_days: int = 3 * 365,
    seed: int = 0,
    dtype: npt.DTypeLike = np.float64,
    warming_per_scenario_c: float = 0.75,
) -> SyntheticData:
    """Generate seeded synthetic data. Days start on 1 January of a 365-day year."""
    rng = np.random.default_rng(seed)
    mean_c = rng.uniform(16.0, 30.0, size=n_locations)
    amp_c = rng.uniform(4.0, 12.0, size=n_locations)
    rh_base = rng.uniform(30.0, 80.0, size=n_locations)

    def seasonal(n: int) -> Array:
        doy = np.arange(n) % 365
        cyc: Array = -np.cos(2.0 * np.pi * (doy - 20) / 365.0)  # coolest near 20 January
        return mean_c[:, None] + amp_c[:, None] * cyc[None, :]

    base_anom = _ar1(rng, (n_locations, baseline_days), phi=0.7, sd=2.5)
    baseline_tmax = seasonal(baseline_days) + base_anom

    anom = _ar1(rng, (n_scenarios, n_locations, n_days), phi=0.7, sd=2.5)
    warming = warming_per_scenario_c * np.arange(n_scenarios)[:, None, None]
    tmax = seasonal(n_days)[None, :, :] + warming + anom
    dtr = np.clip(rng.normal(11.0, 2.0, size=tmax.shape), 2.0, None)
    tmin = tmax - dtr
    rh = np.clip(
        rh_base[None, :, None] - 1.5 * anom + rng.normal(0.0, 8.0, size=tmax.shape), 5.0, 100.0
    )

    dt = np.dtype(dtype)
    return SyntheticData(
        tmax=tmax.astype(dt),
        tmin=tmin.astype(dt),
        rh=rh.astype(dt),
        baseline_tmax=baseline_tmax.astype(dt),
    )


def _near(
    tmax: Array, tmin: Array, rh: Array, thresholds: Sequence[Array], m_c: float, m_f: float
) -> npt.NDArray[np.bool_]:
    near: npt.NDArray[np.bool_] = np.abs(tmin - HOT_NIGHT_C) < m_c
    for thr in thresholds:
        near |= np.abs(tmax - thr[None, :, None]) < m_c
    near |= np.abs(heat_index_f(tmax, rh) - HEAT_INDEX_F) < m_f
    return near


def avoid_threshold_band(
    data: SyntheticData,
    tx90: Array,
    tx95: Array,
    *,
    margin_c: float = 0.5,
    margin_f: float = 0.5,
) -> SyntheticData:
    """Shift values so no day is near a threshold, checked in float64.

    Afterwards every Tmax is at least ``margin_c`` from TX90 and TX95, every Tmin is at least
    ``margin_c`` from 22 °C, and the float64 heat index is at least ``margin_f`` °F from 89.6 °F.
    Days that cannot be fixed by the candidate shifts become NaN in all three variables. NaN days
    are left as they are. This keeps fp32 vs fp64 and fused vs unfused arithmetic from changing
    any count.
    """
    tmax = data.tmax.astype(np.float64)
    tmin = data.tmin.astype(np.float64)
    rh = data.rh.astype(np.float64)
    thr = [tx90.astype(np.float64), tx95.astype(np.float64)]
    shifts = [s * k for k in (1.2, 2.5, 4.0, 6.0) for s in (1.0, -1.0)]
    bad = _near(tmax, tmin, rh, thr, margin_c, margin_f)
    for s in shifts:
        if not bad.any():
            break
        cand_tmax = np.where(bad, tmax + s, tmax)
        cand_tmin = np.where(bad, tmin + s, tmin)
        ok = bad & ~_near(cand_tmax, cand_tmin, rh, thr, margin_c, margin_f)
        tmax = np.where(ok, cand_tmax, tmax)
        tmin = np.where(ok, cand_tmin, tmin)
        bad &= ~ok
    tmax[bad] = np.nan
    tmin[bad] = np.nan
    rh[bad] = np.nan
    dt = data.tmax.dtype
    return SyntheticData(tmax.astype(dt), tmin.astype(dt), rh.astype(dt), data.baseline_tmax)
