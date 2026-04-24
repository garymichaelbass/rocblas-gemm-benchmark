#!/usr/bin/env python3
"""SQLite storage for rocblas-bench sweep results.

Schema stores one row per rocblas-bench invocation with M/N/K split into
separate columns (so non-square GEMMs can be added later without migration).
Rows are always APPENDED (never upserted) so repeated runs build a trend
history — delete the DB file if you want a clean slate.

Indexes:
    idx_gemm_precision_size    : fast filtering by (precision, M, N, K)
    idx_gemm_timestamp         : fast "latest runs" queries

Usage:
    python -m parse.db --csv results/results.csv --db results/results.db

Configurable via env:
    ROCBLAS_BENCH_DB   DB path (overridden by --db on the CLI)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("db")

DEFAULT_DB = os.environ.get("ROCBLAS_BENCH_DB", "results/results.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS gemm_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    precision     TEXT    NOT NULL,
    M             INTEGER NOT NULL,
    N             INTEGER NOT NULL,
    K             INTEGER NOT NULL,
    gflops        REAL,
    gb_s          REAL,
    exec_time_us  REAL,
    error_norm    REAL,
    timestamp     TEXT    NOT NULL,
    log_file      TEXT    NOT NULL,
    inserted_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_gemm_precision_size
    ON gemm_runs (precision, M, N, K);

CREATE INDEX IF NOT EXISTS idx_gemm_timestamp
    ON gemm_runs (timestamp);
"""

INSERT_SQL = """
INSERT INTO gemm_runs
    (precision, M, N, K, gflops, gb_s, exec_time_us, error_norm,
     timestamp, log_file)
VALUES
    (:precision, :M, :N, :K, :gflops, :gb_s, :exec_time_us, :error_norm,
     :timestamp, :log_file)
"""


def connect(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open (and migrate) the SQLite DB at ``db_path``. Creates parent dirs."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _maybe_float(x: str) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _coerce(row: dict[str, str]) -> dict[str, object]:
    """Convert a CSV row (all str) to properly typed values for SQLite."""
    return {
        "precision":    row["precision"],
        "M":            int(row["M"]),
        "N":            int(row["N"]),
        "K":            int(row["K"]),
        "gflops":       _maybe_float(row["gflops"]),
        "gb_s":         _maybe_float(row["gb_s"]),
        "exec_time_us": _maybe_float(row["exec_time_us"]),
        "error_norm":   _maybe_float(row["error_norm"]),
        "timestamp":    row["timestamp"],
        "log_file":     row["log_file"],
    }


def load_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    """Append every row in ``csv_path`` to the gemm_runs table. Returns count."""
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = [_coerce(r) for r in reader]
    with conn:
        conn.executemany(INSERT_SQL, rows)
    return len(rows)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("results/results.csv"),
                    help="parsed CSV to load (default: results/results.csv)")
    ap.add_argument("--db",  type=Path, default=Path(DEFAULT_DB),
                    help=f"SQLite DB path (default: {DEFAULT_DB})")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.csv.is_file():
        logger.error("CSV not found: %s (run parse/parser.py first)", args.csv)
        return 2

    conn = connect(args.db)
    n = load_csv(conn, args.csv)
    logger.info("inserted %d rows into %s", n, args.db)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
