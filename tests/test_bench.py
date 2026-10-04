import argparse
import json
import shutil
from pathlib import Path

import pytest

from bench import make_table, run

FIXTURE = Path(__file__).parent / "fixtures" / "FAKE_bench_result.json"


def _record(**gpu: object) -> dict:
    base = {"name": "Tesla T4", "device_index": 0}
    base.update(gpu)
    return {
        "dry_run": False,
        "created_utc": "2026-10-04T00:00:00Z",
        "gpu": base,
        "results": [{"impl": "cpu"}],
    }


@pytest.mark.parametrize(
    "record",
    [
        _record(name=None),
        _record(name=""),
        _record(device_index=None),
        {**_record(), "dry_run": True},
        {**_record(), "results": []},
        {"dry_run": False, "created_utc": "x", "results": [{}]},  # no gpu block at all
    ],
)
def test_refuses_to_write_without_real_gpu_run(record: dict, tmp_path: Path) -> None:
    with pytest.raises(run.ResultsRefused):
        run.write_results(record, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_writes_when_gpu_named(tmp_path: Path) -> None:
    path = run.write_results(_record(), tmp_path)
    assert path.parent == tmp_path and "tesla-t4_dev0" in path.name
    assert json.loads(path.read_text())["gpu"]["name"] == "Tesla T4"


def test_dry_run_writes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    before = sorted(run.RESULTS_DIR.glob("*.json"))
    assert run.main(["--dry-run", "--reps", "2"]) == 0
    assert sorted(run.RESULTS_DIR.glob("*.json")) == before
    out = capsys.readouterr().out
    assert "dry run: results not written" in out
    assert "matches_cpu=True" in out and "matches_cpu=False" not in out
    assert "gpu_cuda" not in out  # the custom kernel is never faked on CPU


def test_benchmark_record_schema_on_cpu() -> None:
    import numpy as np

    rec = run.benchmark(
        [(1, 129, 12)], ["cpu", "gpu_torch"], None, 2, 0, 0, np.float32, False, True
    )
    assert rec["schema"] == run.SCHEMA_VERSION and rec["dry_run"] is True
    assert rec["gpu"]["name"] is None
    assert {r["impl"] for r in rec["results"]} == {"cpu", "gpu_torch"}
    for r in rec["results"]:
        assert set(r) >= {"impl", "custom_kernel", "cells", "stages_ms", "matches_cpu"}
        assert r["matches_cpu"] is True
    gt = next(r for r in rec["results"] if r["impl"] == "gpu_torch")
    assert set(gt["stages_ms"]) == {"h2d", "compute", "d2h", "end_to_end"}


def test_no_cuda_without_dry_run_exits_nonzero() -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("has CUDA")
    assert run.main([]) == 2


def test_parse_size() -> None:
    assert run.parse_size("4x1000x365") == (4, 1000, 365)
    with pytest.raises(argparse.ArgumentTypeError):
        run.parse_size("4x1000")


def test_table_reads_only_results_dir() -> None:
    assert make_table.RESULTS_DIR == Path(run.__file__).parent / "results"


def test_table_not_measured_when_empty(tmp_path: Path) -> None:
    assert make_table.render(make_table.load_records(tmp_path)) == make_table.NOT_MEASURED


def test_table_from_fake_fixture(tmp_path: Path) -> None:
    shutil.copy(FIXTURE, tmp_path / FIXTURE.name)
    # Records that are dry runs or name no GPU are ignored.
    (tmp_path / "dry.json").write_text(json.dumps({**_record(), "dry_run": True}))
    (tmp_path / "noname.json").write_text(json.dumps(_record(name=None)))
    records = make_table.load_records(tmp_path)
    assert [r["_file"] for r in records] == [FIXTURE.name]
    table = make_table.render(records)
    assert "FAKE-GPU-FOR-TESTS" in table
    assert "| gpu_cuda | 1×2×3 | 2.000 | 3.000 | 4.000 | 5.000 | 6.000 | **no** |" in table
    assert "| cpu | 1×2×3 |  |  | 1.000 |  | 1.000 | yes |" in table


def test_update_readme_between_markers() -> None:
    text = f"a\n{make_table.START}\nold\n{make_table.END}\nb\n"
    want = f"a\n{make_table.START}\nnew\n{make_table.END}\nb\n"
    assert make_table.update_readme(text, "new") == want
    with pytest.raises(ValueError):
        make_table.update_readme("no markers", "x")


def test_fixture_is_labelled_fake() -> None:
    rec = json.loads(FIXTURE.read_text())
    assert "FAKE" in rec["_comment"] and "FAKE" in rec["gpu"]["name"]


def test_out_of_memory_is_recorded_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    import torch

    def oom(*args: object, **kwargs: object) -> None:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 6.00 GiB")

    monkeypatch.setattr(run, "run_device", oom)
    rec = run.benchmark([(1, 3, 5)], ["cpu", "gpu_torch"], None, 1, 0, 0, np.float32, False, True)
    gt = next(r for r in rec["results"] if r["impl"] == "gpu_torch")
    assert gt["error"].startswith("CUDA out of memory")
    assert gt["stages_ms"] is None and gt["matches_cpu"] is None
    table = make_table.render([{**rec, "gpu": {"name": "x", "device_index": 0}, "_file": "f"}])
    assert "| gpu_torch | 1×3×5 | CUDA out of memory |" in table


def test_plot_skips_error_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("matplotlib")
    from bench import plot

    rec = json.loads(FIXTURE.read_text())
    rec["results"].append({**rec["results"][1], "stages_ms": None, "error": "CUDA out of memory"})
    rec["_file"] = "f"
    monkeypatch.setattr(plot, "load_records", lambda: [rec])
    out = tmp_path / "p.png"
    assert plot.main(["--out", str(out)]) == 0 and out.exists()
