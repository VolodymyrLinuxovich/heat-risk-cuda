"""NWS heat index in °F, evaluated in the input dtype.

The operation order here is the reference that ``gpu_torch.py`` and ``kernels/heat_hazards.cu``
mirror line by line, so float32 results agree across implementations. See docs/hazards.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def c_to_f(t_c: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """°C to °F as ``t * 1.8 + 32``, in the input dtype."""
    dt = t_c.dtype.type
    out: npt.NDArray[np.floating] = t_c * dt(1.8) + dt(32.0)
    return out


def heat_index_f(
    tmax_c: npt.NDArray[np.floating], rh: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Heat index in °F from Tmax in °C and relative humidity in percent.

    NaN in either input gives NaN. Every intermediate stays in the input dtype.
    """
    if tmax_c.dtype != rh.dtype:
        raise TypeError("tmax_c and rh must share a dtype")
    dt = tmax_c.dtype.type
    t = c_to_f(tmax_c)

    simple = dt(0.5) * (t + dt(61.0) + (t - dt(68.0)) * dt(1.2) + rh * dt(0.094))

    full = (
        dt(-42.379)
        + dt(2.04901523) * t
        + dt(10.14333127) * rh
        - dt(0.22475541) * t * rh
        - dt(0.00683783) * t * t
        - dt(0.05481717) * rh * rh
        + dt(0.00122874) * t * t * rh
        + dt(0.00085282) * t * rh * rh
        - dt(0.00000199) * t * t * rh * rh
    )

    in_low = (rh < dt(13.0)) & (t >= dt(80.0)) & (t <= dt(112.0))
    # Only evaluated where in_low holds, where the sqrt argument is >= 0.
    low_arg = np.where(in_low, (dt(17.0) - np.abs(t - dt(95.0))) / dt(17.0), dt(0.0))
    low_adj = ((dt(13.0) - rh) / dt(4.0)) * np.sqrt(low_arg)
    full = np.where(in_low, full - low_adj, full)

    in_high = (rh > dt(85.0)) & (t >= dt(80.0)) & (t <= dt(87.0))
    high_adj = ((rh - dt(85.0)) / dt(10.0)) * ((dt(87.0) - t) / dt(5.0))
    full = np.where(in_high, full + high_adj, full)

    use_full = (simple + t) / dt(2.0) >= dt(80.0)
    out: npt.NDArray[np.floating] = np.where(use_full, full, simple).astype(tmax_c.dtype)
    return out
