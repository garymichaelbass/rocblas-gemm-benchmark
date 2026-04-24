"""Correctness assertions for rocBLAS GEMM kernels.

Each (precision, matrix_size) cell's worst-case ``rocblas-error`` across
all recorded runs must not exceed 1e-3.

The 1e-3 threshold is fixed by the benchmark specification and is
intentionally NOT loaded from config.yaml — making it configurable risks
accidental loosening over time ("just bump it to make CI green"). If the
threshold ever legitimately needs to change, the change should be an
explicit code edit reviewed in its own commit.

``rocblas-error`` is the relative error norm reported when rocblas-bench
is invoked with ``-v 1`` (verification enabled). It compares the GPU GEMM
result against a reference implementation.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import pytest

# Spec-fixed. Do NOT make this configurable.
ERROR_NORM_THRESHOLD: float = 1e-3


def _fetch_max_error(
    conn: sqlite3.Connection, precision: str, size: int
) -> tuple[int, Optional[float]]:
    """Return ``(n_rows, max_error_norm)`` for a given cell.

    ``max_error_norm`` is NULL-propagating: if every row has NULL error_norm
    (verification disabled), MAX returns NULL and we report that as an
    explicit failure rather than a pass.
    """
    row = conn.execute(
        """
        SELECT COUNT(*) AS n, MAX(error_norm) AS worst
        FROM gemm_runs
        WHERE precision = ? AND M = ? AND N = ? AND K = ?
        """,
        (precision, size, size, size),
    ).fetchone()
    return int(row["n"]), row["worst"]


def test_cell_has_data(db_conn: sqlite3.Connection, cell: tuple[str, int]):
    """Every sweep-grid cell must have at least one recorded run."""
    precision, size = cell
    n, _ = _fetch_max_error(db_conn, precision, size)
    assert n > 0, (
        f"no runs in DB for precision={precision} size={size}; "
        "sweep may have been incomplete"
    )


def test_error_norm_within_threshold(
    db_conn: sqlite3.Connection, cell: tuple[str, int]
):
    """Worst-case rocblas-error across runs of this cell must be ≤ 1e-3."""
    precision, size = cell
    n, worst = _fetch_max_error(db_conn, precision, size)
    if n == 0:
        pytest.skip(f"no data for precision={precision} size={size}")

    # A NULL worst means every row's error_norm was NULL: verification was
    # not enabled (bench was run without -v 1). That means correctness is
    # unvalidated, which is a failure — we cannot silently pass.
    assert worst is not None, (
        f"{precision} m=n=k={size}: error_norm is NULL across all {n} runs. "
        "Was rocblas-bench run with -v 1 (verification enabled)?"
    )
    assert worst <= ERROR_NORM_THRESHOLD, (
        f"{precision} m=n=k={size}: worst error_norm {worst:.3e} "
        f"exceeds threshold {ERROR_NORM_THRESHOLD:.3e} (across {n} runs)"
    )
