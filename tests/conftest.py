"""Shared test configuration.

Tests that need a CUDA device are marked ``@pytest.mark.gpu``. They are skipped, not passed,
when no device is available. CI and local runs use ``pytest -m "not gpu"``.

Tests marked ``@pytest.mark.nvrtc`` need the ``gpu`` extra (cuda-core + NVRTC) but no device.
They skip when cuda.core cannot be imported, unless ``HEAT_RISK_REQUIRE_NVRTC=1`` is set, in
which case they fail instead. The CI ``nvrtc`` job sets it so a skip can never look like a pass.
"""

from __future__ import annotations

import os

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _cuda_core_error() -> str | None:
    try:
        import cuda.core  # noqa: F401
    except Exception as exc:  # ImportError, or a driver-loading error on import
        return f"{type(exc).__name__}: {exc}"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not _cuda_available():
        skip_gpu = pytest.mark.skip(reason="no CUDA device available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    if any("nvrtc" in item.keywords for item in items):
        err = _cuda_core_error()
        if err is not None:
            if os.environ.get("HEAT_RISK_REQUIRE_NVRTC") == "1":
                raise pytest.UsageError(f"HEAT_RISK_REQUIRE_NVRTC=1 but cuda.core failed: {err}")
            skip = pytest.mark.skip(reason=f"cuda.core unavailable ({err})")
            for item in items:
                if "nvrtc" in item.keywords:
                    item.add_marker(skip)
