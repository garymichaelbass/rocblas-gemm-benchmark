"""Pytest fixtures for rocBLAS GEMM benchmark tests.

Pre-conditions (the DB must be populated before pytest runs):

    bench/sweep.sh                  # generate logs/
    python -m parse.parser          # logs/ → results/results.csv
    python -m parse.db              # CSV → results/results.db
    pytest tests/                   # <-- here

Tests read EXCLUSIVELY from the DB (not the CSV) — a single source of
truth keeps tests decoupled from the intermediate CSV format.

Fixtures:
    config    (session)  : parsed config.yaml; fails loudly on missing/bad.
    db_path   (session)  : resolved results DB path (ROCBLAS_BENCH_DB overrides).
    db_conn   (session)  : read-only sqlite3.Connection, opened once per run.
    cell      (function) : parametrized (precision, matrix_size) tuple — one
                           per (prec × size) combination in the sweep.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_DB_PATH = REPO_ROOT / "results" / "results.db"

# Must match the sweep grid in bench/sweep.sh.
PRECISIONS: tuple[str, ...] = ("f16", "bf16", "f32")
SIZES: tuple[int, ...] = (4096, 8192, 16384)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Load and validate ``config.yaml``. Fails loudly on missing/malformed."""
    if not CONFIG_PATH.is_file():
        pytest.fail(f"config.yaml not found at {CONFIG_PATH}")
    try:
        with CONFIG_PATH.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        pytest.fail(f"config.yaml is malformed: {exc}")

    if not isinstance(data, dict):
        pytest.fail("config.yaml must define a top-level mapping")

    thresholds = data.get("thresholds")
    if not isinstance(thresholds, dict):
        pytest.fail("config.yaml missing required key 'thresholds'")

    gflops = thresholds.get("gflops")
    if not isinstance(gflops, dict):
        pytest.fail("config.yaml missing required key 'thresholds.gflops'")

    for prec in PRECISIONS:
        if prec not in gflops:
            pytest.fail(f"config.yaml missing thresholds.gflops.{prec}")
        try:
            float(gflops[prec])
        except (TypeError, ValueError):
            pytest.fail(
                f"config.yaml: thresholds.gflops.{prec} must be a number, "
                f"got {gflops[prec]!r}"
            )
    return data


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def db_path() -> Path:
    override = os.environ.get("ROCBLAS_BENCH_DB")
    path = Path(override) if override else DEFAULT_DB_PATH
    if not path.is_file():
        pytest.fail(
            f"results DB not found at {path}. "
            "Run sweep.sh → parser.py → db.py before invoking pytest."
        )
    return path


@pytest.fixture(scope="session")
def db_conn(db_path: Path):
    """Session-scoped read connection. Opened once, closed at session teardown."""
    # uri=True + mode=ro lets us open the file read-only and fail fast if it
    # doesn't exist (we've already checked, but it's a cheap safety net).
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Cell parametrization
# ---------------------------------------------------------------------------
def _cell_params() -> list[tuple[str, int]]:
    return [(p, s) for p in PRECISIONS for s in SIZES]


@pytest.fixture(
    params=_cell_params(),
    ids=lambda p: f"{p[0]}-m{p[1]}",
)
def cell(request) -> tuple[str, int]:
    """One parametrized ``(precision, matrix_size)`` cell from the sweep grid."""
    return request.param
