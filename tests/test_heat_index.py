import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from heat_risk.heat_index import c_to_f, heat_index_f


def f_to_c(f: float) -> float:
    return (f - 32.0) / 1.8


# Values read from the NWS heat index chart (https://www.weather.gov/safety/heat-index),
# which rounds to whole °F, as (temperature °F, RH %, heat index °F).
NWS_CHART = [
    (80, 40, 80),
    (90, 40, 91),
    (100, 40, 109),
    (90, 50, 95),
    (90, 70, 106),
    (86, 90, 105),
    (80, 90, 86),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("t_f", "rh", "expected"), NWS_CHART)
def test_nws_chart_spot_values(t_f: float, rh: float, expected: float, dtype: type) -> None:
    got = heat_index_f(np.array([f_to_c(t_f)], dtype=dtype), np.array([rh], dtype=dtype))
    assert got.dtype == dtype
    assert got[0] == pytest.approx(expected, abs=1.0)


def test_simple_formula_below_80() -> None:
    # 70 °F at 50 %: simple estimate is 0.5*(70+61+2.4+4.7) = 69.05, so it is used directly.
    got = heat_index_f(np.array([f_to_c(70.0)]), np.array([50.0]))
    assert got[0] == pytest.approx(69.05, abs=1e-9)


def test_low_humidity_adjustment() -> None:
    t_f, rh = 100.0, 10.0
    got = heat_index_f(np.array([f_to_c(t_f)]), np.array([rh]))[0]
    t, r = t_f, rh
    full = (
        -42.379 + 2.04901523 * t + 10.14333127 * r - 0.22475541 * t * r - 0.00683783 * t * t
        - 0.05481717 * r * r + 0.00122874 * t * t * r + 0.00085282 * t * r * r
        - 0.00000199 * t * t * r * r
    )
    adj = ((13 - r) / 4) * np.sqrt((17 - abs(t - 95)) / 17)
    assert got == pytest.approx(full - adj, abs=1e-9)


def test_high_humidity_adjustment() -> None:
    t_f, rh = 82.0, 95.0
    got = heat_index_f(np.array([f_to_c(t_f)]), np.array([rh]))[0]
    t, r = t_f, rh
    full = (
        -42.379 + 2.04901523 * t + 10.14333127 * r - 0.22475541 * t * r - 0.00683783 * t * t
        - 0.05481717 * r * r + 0.00122874 * t * t * r + 0.00085282 * t * r * r
        - 0.00000199 * t * t * r * r
    )
    assert got == pytest.approx(full + ((r - 85) / 10) * ((87 - t) / 5), abs=1e-9)


def test_nan_propagates() -> None:
    got = heat_index_f(np.array([np.nan, 30.0]), np.array([50.0, np.nan]))
    assert np.isnan(got).all()


def test_dtype_mismatch_rejected() -> None:
    with pytest.raises(TypeError):
        heat_index_f(np.zeros(1, np.float32), np.zeros(1, np.float64))


def test_c_to_f() -> None:
    assert c_to_f(np.array([0.0, 100.0, 32.0])).tolist() == pytest.approx([32.0, 212.0, 89.6])


temps = st.floats(-30.0, 50.0)
rhs = st.floats(0.0, 100.0)


@given(temps, rhs)
def test_float32_close_to_float64(t: float, rh: float) -> None:
    hi64 = heat_index_f(np.array([t]), np.array([rh]))[0]
    hi32 = heat_index_f(np.array([t], np.float32), np.array([rh], np.float32))[0]
    assert np.isfinite(hi64)
    assert hi32 == pytest.approx(hi64, abs=0.05)


@given(temps, rhs)
def test_regime_selection(t: float, rh: float) -> None:
    t_f = t * 1.8 + 32.0
    simple = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)
    hi = heat_index_f(np.array([t]), np.array([rh]))[0]
    if (simple + t_f) / 2.0 < 80.0:
        assert hi == pytest.approx(simple, abs=1e-9)
