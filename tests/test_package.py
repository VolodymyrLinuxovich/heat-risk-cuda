import heat_risk


def test_package_imports_without_gpu_extra() -> None:
    assert heat_risk.__version__ == "0.1.0"
