from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from heat_risk import cpu, gpu_cuda, gpu_torch
from heat_risk.batch import BYTES_PER_CELL_FACTOR, evaluate_batched, plan_chunks
from heat_risk.synthetic import avoid_threshold_band, generate
from heat_risk.thresholds import percentile_thresholds

IMPLS = [
    pytest.param(cpu.evaluate, id="cpu"),
    pytest.param(lambda *a: gpu_torch.evaluate(*a, device="cpu"), id="gpu_torch[cpu]"),
    pytest.param(gpu_cuda.evaluate, id="gpu_cuda", marks=pytest.mark.gpu),
    pytest.param(
        lambda *a: gpu_cuda.evaluate(*a, layout="tiled"),
        id="gpu_cuda[tiled]",
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
    st.integers(1, 4), st.integers(1, 9), st.integers(1, 30), st.integers(1, 40),
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
