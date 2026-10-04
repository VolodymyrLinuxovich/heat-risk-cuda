"""Smoke-execute the Kaggle notebook on CPU with nbconvert. GPU cells must skip, not fail."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "kaggle_heat_risk_cuda.ipynb"
README = Path(__file__).parent.parent / "README.md"


def test_notebook_is_up_to_date_with_generator() -> None:
    import importlib.util

    generator = NOTEBOOK.with_name("make_notebook.py")
    spec = importlib.util.spec_from_file_location("make_notebook", generator)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    want = [c.source for c in mod.build().cells]
    got = [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells]
    assert got == want, "run python notebooks/make_notebook.py"


def test_notebook_runs_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("smoke test is for CPU machines; on a GPU run the notebook itself")
    readme_before = README.read_text()
    monkeypatch.setenv("HEAT_RISK_NOTEBOOK_SMOKE", "1")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    try:
        ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(NOTEBOOK.parent)}}
        )
    finally:
        os.chdir(Path(__file__).parent.parent)
    text = "\n".join(
        o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
    )
    assert "torch.cuda: False" in text
    assert "skipped: no CUDA device" in text
    assert "dry run: results not written" in text
    assert not list((NOTEBOOK.parent.parent / "bench" / "results").glob("*.json"))
    # make_table may rewrite README; with no results it must be unchanged.
    assert README.read_text() == readme_before
