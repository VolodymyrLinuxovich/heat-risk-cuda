import numpy as np
import pytest

from heat_risk.spec import HAZARDS, HEAT_INDEX_C, HEAT_INDEX_F, HazardResult, validate_inputs


def test_heat_index_cutoff_is_32c_in_f() -> None:
    assert HEAT_INDEX_C * 9 / 5 + 32 == pytest.approx(HEAT_INDEX_F)


def test_hazard_result_shape_checks() -> None:
    z = np.zeros((2, 3, len(HAZARDS)), dtype=np.int32)
    r = HazardResult(z, z.copy())
    assert r.hazard("hot_night")[0].shape == (2, 3)
    with pytest.raises(ValueError):
        HazardResult(z, z[:, :, :2])


def test_hazard_result_equality() -> None:
    # == used to raise "truth value of an array is ambiguous" (issue #23).
    z = np.zeros((2, 3, 4), dtype=np.int32)
    r = HazardResult(z, z.copy())
    assert r == HazardResult(z.copy(), z.copy())
    assert r in [HazardResult(z.copy(), z.copy())]
    other = z.copy()
    other[1, 2, 3] = 1
    assert r != HazardResult(other, z.copy())
    assert r != HazardResult(z.copy(), other)
    assert r != HazardResult(z[:1], z[:1].copy())
    assert r != (z, z)
    with pytest.raises(TypeError):
        hash(r)


def test_validate_inputs() -> None:
    a = np.zeros((1, 2, 5), dtype=np.float32)
    t = np.zeros(2, dtype=np.float32)
    assert validate_inputs(a, a, a, t, t) == np.float32
    with pytest.raises(TypeError):
        validate_inputs(a, a, a, t.astype(np.float64), t)
    with pytest.raises(ValueError):
        validate_inputs(a, a, a, t[:1], t[:1])
