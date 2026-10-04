"""Scalar pure-Python oracle for run lengths, shared by tests."""

from __future__ import annotations

from collections.abc import Iterable


def longest_run_scalar(flags: Iterable[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best
