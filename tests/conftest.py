"""Shared test configuration.

Tests that need a CUDA device are marked ``@pytest.mark.gpu``. They are skipped, not passed,
when no device is available. CI and local runs use ``pytest -m "not gpu"``.
"""

from __future__ import annotations

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
