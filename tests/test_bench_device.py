import pytest

from bench import run


@pytest.mark.parametrize("text", ["cpu", "cuda:1", "cuda:7", "gpu0"])
def test_device_must_be_a_visible_cuda_device(
    text: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # These used to end in a traceback from torch.cuda.set_device.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def no_set_device(device: object) -> None:
        raise AssertionError("set_device must not be reached")

    monkeypatch.setattr(torch.cuda, "set_device", no_set_device)
    assert run.main(["--device", text]) == 2
    assert "1 visible CUDA devices" in capsys.readouterr().err


def test_cuda_device_accepts_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert run.cuda_device("cuda") == torch.device("cuda")
    assert run.cuda_device("cuda:1") == torch.device("cuda:1")
    assert run.cuda_device("cuda:2") is None
