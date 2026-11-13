import pytest

from bench import run


def test_cpp_library_built_before_first_timed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # With --warmup 0 the first timed cpu_cpp run used to include the compiler run.
    import numpy as np

    events: list[str] = []
    monkeypatch.setattr(run.cpu_cpp, "build_library", lambda: events.append("build"))

    def fake_run_cpu(arrays, reps, warmup, evaluate):  # type: ignore[no-untyped-def]
        events.append("timed")
        return {"compute": 1.0, "end_to_end": 1.0}, run.cpu.evaluate(*arrays)

    monkeypatch.setattr(run, "run_cpu", fake_run_cpu)
    run.benchmark([(1, 4, 4)], ["cpu_cpp"], None, 1, 0, 0, np.float32, False, True)
    assert events == ["build", "timed"]


def test_cuda_kernel_compiled_before_first_timed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    loaded: list[object] = []
    monkeypatch.setattr(run.gpu_cuda, "_load_module", loaded.append)
    monkeypatch.setattr(run.gpu_cuda, "device_arch", lambda dev: "sm_75")
    run.compile_ahead("gpu_cuda_tiled", torch.device("cuda:0"), run.gpu_cuda.CompileOptions())
    assert loaded == [run.gpu_cuda.CompileOptions(arch="sm_75")]
    run.compile_ahead("gpu_cuda", None, run.gpu_cuda.CompileOptions())
    run.compile_ahead("gpu_torch", torch.device("cuda:0"), run.gpu_cuda.CompileOptions())
    assert len(loaded) == 1
