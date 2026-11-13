from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from heat_risk import cpu, gpu_cuda, gpu_torch
from heat_risk.batch import (
    BYTES_PER_CELL_FACTOR,
    device_bytes_per_cell,
    evaluate_batched,
    plan_chunks,
)
from heat_risk.synthetic import avoid_threshold_band, generate
from heat_risk.thresholds import percentile_thresholds

IMPLS = [
    pytest.param(cpu.evaluate, id="cpu"),
    pytest.param(lambda *a: gpu_torch.evaluate(*a, device="cpu"), id="gpu_torch[cpu]"),
    pytest.param(gpu_cuda.evaluate, id="gpu_cuda[tiled]", marks=pytest.mark.gpu),
    pytest.param(
        lambda *a: gpu_cuda.evaluate(*a, layout="day_major"),
        id="gpu_cuda[day_major]",
        marks=pytest.mark.gpu,
    ),
]


def _args(s: int, n_loc: int, days: int, seed: int) -> tuple[np.ndarray, ...]:
    d = generate(s, n_loc, days, seed=seed, baseline_days=100, dtype=np.float32)
    tx90, tx95 = percentile_thresholds(d.baseline_tmax, np.float32)
    d = avoid_threshold_band(d, tx90, tx95)
    return d.tmax, d.tmin, d.rh, tx90, tx95


@settings(max_examples=40, deadline=None)
@given(
    st.integers(1, 4),
    st.integers(1, 9),
    st.integers(1, 30),
    st.integers(1, 40),
    st.integers(0, 2**32 - 1),
)
def test_chunks_cover_every_row_once(s: int, n_loc: int, days: int, rows: int, seed: int) -> None:
    max_bytes = rows * days * 4 * BYTES_PER_CELL_FACTOR
    seen = np.zeros((s, n_loc), dtype=int)
    for ss, ls in plan_chunks(s, n_loc, days, 4, max_bytes):
        seen[ss, ls] += 1
        n_rows = (ss.stop - ss.start) * (ls.stop - ls.start)
        assert n_rows * days * 4 * BYTES_PER_CELL_FACTOR <= max_bytes
    assert (seen == 1).all()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("rows_per_chunk", [1, 7, 128, 10_000])
def test_batched_equals_unbatched(impl: Callable[..., object], rows_per_chunk: int) -> None:
    args = _args(3, 50, 120, seed=rows_per_chunk)
    max_bytes = rows_per_chunk * 120 * 4 * BYTES_PER_CELL_FACTOR
    want = cpu.evaluate(*args)
    got = evaluate_batched(impl, *args, max_bytes=max_bytes)  # type: ignore[arg-type]
    np.testing.assert_array_equal(got.counts, want.counts)
    np.testing.assert_array_equal(got.longest_run, want.longest_run)


def test_row_larger_than_budget_rejected() -> None:
    with pytest.raises(ValueError):
        plan_chunks(1, 1, 1000, 8, max_bytes=100)


def _peak_new_storage_bytes(fn: Callable[[], object], known: list[int]) -> int:
    """Peak bytes held by tensors that ``fn`` allocates, tracked op by op."""
    import weakref

    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    live: dict[int, int] = {}
    state = {"cur": 0, "peak": 0}

    def free(ptr: int) -> None:
        state["cur"] -= live.pop(ptr, 0)

    class Track(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
            out = func(*args, **(kwargs or {}))
            for t in tree_flatten(out)[0]:
                if not isinstance(t, torch.Tensor):
                    continue
                storage = t.untyped_storage()
                ptr, n = storage.data_ptr(), storage.nbytes()
                if n == 0 or ptr in known or ptr in live:
                    continue  # views of inputs or of tensors already counted
                live[ptr] = n
                state["cur"] += n
                state["peak"] = max(state["peak"], state["cur"])
                weakref.finalize(t, free, ptr)
            return out

    with Track():
        fn()
    return state["peak"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gpu_torch_chunk_fits_its_budget(dtype: type) -> None:
    # The default footprint is sized for gpu_cuda; gpu_torch needs about 4x more (issue #9).
    import torch

    s, n_loc, days = 2, 40, 90
    d = generate(s, n_loc, days, seed=1, baseline_days=100, dtype=dtype)
    tx90, tx95 = percentile_thresholds(d.baseline_tmax, dtype)
    t = [torch.from_numpy(np.ascontiguousarray(a)) for a in (d.tmax, d.tmin, d.rh, tx90, tx95)]
    work = _peak_new_storage_bytes(
        lambda: gpu_torch.evaluate_tensors(*t), [x.untyped_storage().data_ptr() for x in t]
    )
    itemsize = np.dtype(dtype).itemsize
    needed = 3 * s * n_loc * days * itemsize + work

    # Every chunk the planner allows must really fit: the budget per cell covers the measured
    # peak per cell. The old gpu_cuda sized default does not.
    cells = s * n_loc * days
    assert device_bytes_per_cell("gpu_torch", itemsize) * cells >= needed
    assert itemsize * BYTES_PER_CELL_FACTOR * cells < needed


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_device_bytes_per_cell(dtype: type) -> None:
    itemsize = np.dtype(dtype).itemsize
    assert device_bytes_per_cell("gpu_cuda", itemsize) == itemsize * BYTES_PER_CELL_FACTOR
    assert device_bytes_per_cell("gpu_torch", itemsize) == 3 * itemsize + 100
    with pytest.raises(ValueError):
        device_bytes_per_cell("bogus", itemsize)


def test_batched_gpu_torch_with_its_budget() -> None:
    args = _args(3, 50, 120, seed=5)
    budget = device_bytes_per_cell("gpu_torch", 4)
    want = cpu.evaluate(*args)
    got = evaluate_batched(
        lambda *a: gpu_torch.evaluate(*a, device="cpu"),
        *args,
        max_bytes=7 * 120 * budget,
        bytes_per_cell=budget,
    )
    np.testing.assert_array_equal(got.counts, want.counts)
    np.testing.assert_array_equal(got.longest_run, want.longest_run)
