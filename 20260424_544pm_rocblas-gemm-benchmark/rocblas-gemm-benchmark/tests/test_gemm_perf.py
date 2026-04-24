"""Performance assertions for rocBLAS GEMM.

For each (precision, matrix_size) cell, the MEDIAN GFLOPS across recorded
runs must meet the per-precision threshold in ``config.yaml``. Median is
used (rather than max or mean) because it is robust to occasional
cold-cache or thermal-throttling outliers — a single slow run should not
fail the test, but a regression affecting the typical run should.

Assertions are per-cell so a regression in one cell cannot mask passes
elsewhere; failures read like::

    FAILED tests/test_gemm_perf.py::test_gflops_meets_threshold[f32-m16384]

Thresholds in ``config.yaml`` are PRACTICAL floors (roughly 70% of peak),
not theoretical peaks. Raw theoretical peak is rarely achieved; see
config.yaml for the reasoning.
"""

from __future__ import annotations

import sqlite3
import statistics
from typing import Any

import pytest


def _fetch_gflops(
    conn: sqlite3.Connection, precision: str, size: int
) -> list[float]:
    """Return all non-NULL GFLOPS samples for the given cell."""
    rows = conn.execute(
        """
        SELECT gflops FROM gemm_runs
        WHERE precision = ? AND M = ? AND N = ? AND K = ?
          AND gflops IS NOT NULL
        """,
        (precision, size, size, size),
    ).fetchall()
    return [float(r["gflops"]) for r in rows]


def test_gflops_meets_threshold(
    db_conn: sqlite3.Connection,
    config: dict[str, Any],
    cell: tuple[str, int],
):
    precision, size = cell
    threshold = float(config["thresholds"]["gflops"][precision])  # GFLOPS

    samples = _fetch_gflops(db_conn, precision, size)
    if not samples:
        pytest.skip(f"no GFLOPS samples recorded for {precision}/{size}")

    median = statistics.median(samples)
    assert median >= threshold, (
        f"{precision} m=n=k={size}: median GFLOPS {median:,.1f} "
        f"below threshold {threshold:,.1f} "
        f"(n={len(samples)} samples, "
        f"min={min(samples):,.1f}, max={max(samples):,.1f})"
    )
