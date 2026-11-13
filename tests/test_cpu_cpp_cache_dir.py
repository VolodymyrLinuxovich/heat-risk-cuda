"""``cpu_cpp`` loads its library from a relative cache directory."""

from __future__ import annotations

import numpy as np
import pytest

from heat_risk import cpu_cpp


@pytest.mark.parametrize("cache", [".", "cache"])
def test_relative_cache_dir(tmp_path, monkeypatch, cache) -> None:  # type: ignore[no-untyped-def]
    # Path(".") / name drops the "./", and dlopen on Linux never searches the working
    # directory for a bare file name.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEAT_RISK_CACHE_DIR", cache)
    cpu_cpp._library.cache_clear()
    try:
        assert cpu_cpp.build_library().is_absolute()
        a = np.full((1, 1, 3), 30.0)
        t = np.zeros(1)
        r = cpu_cpp.evaluate(a, a, a, t, t)
        assert r.counts[0, 0].tolist() == [3, 3, 3, 0]
    finally:
        cpu_cpp._library.cache_clear()
