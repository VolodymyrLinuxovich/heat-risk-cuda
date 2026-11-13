"""``cpu_cpp`` reports a missing or broken C++ compiler with a clear error."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from heat_risk import cpu_cpp


@pytest.mark.parametrize(
    "cxx",
    [
        "heat-risk-no-such-compiler",  # not on PATH
        f"{sys.executable} -c 'raise SystemExit(1)'",  # exists, but --version fails
    ],
)
def test_unusable_compiler_raises_runtime_error(tmp_path, monkeypatch, cxx) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HEAT_RISK_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CXX", cxx)
    with pytest.raises(RuntimeError, match="no usable C\\+\\+ compiler.*set CXX"):
        cpu_cpp.build_library()


def test_evaluate_reports_missing_compiler(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HEAT_RISK_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CXX", "heat-risk-no-such-compiler")
    cpu_cpp._library.cache_clear()
    a = np.zeros((1, 1, 3))
    t = np.zeros(1)
    try:
        with pytest.raises(RuntimeError, match="heat-risk-no-such-compiler"):
            cpu_cpp.evaluate(a, a, a, t, t)
    finally:
        cpu_cpp._library.cache_clear()
