#!/usr/bin/env python3
"""Parse rocblas-bench stdout logs into a structured CSV.

rocblas-bench emits a CSV-like block to stdout: a header row containing
column names like 'rocblas-Gflops', 'rocblas-GB/s', 'us', 'rocblas-error',
followed by one or more data rows. We locate the header, then read the
first matching data row and project out the four metric columns we care
about.

Precision and matrix size are recovered from the log FILENAME (not by
re-parsing them from stdout, which does not always echo them back cleanly).
The filename convention is produced by ``bench/sweep.sh``:

    bench_<precision>_<size>_run<n>_<timestamp>.log

CSV schema (one row per rocblas-bench invocation, not aggregated):
    precision, M, N, K, gflops, gb_s, exec_time_us, error_norm,
    timestamp, log_file

Usage:
    python -m parse.parser --log-dir logs/ --output results/results.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("parser")

# ---------------------------------------------------------------------------
# Filename convention (must match bench/sweep.sh)
# ---------------------------------------------------------------------------
FILENAME_RE = re.compile(
    r"^bench_"
    r"(?P<precision>f16|bf16|f32)_"
    r"(?P<size>\d+)_"
    r"run(?P<run>\d+)_"
    r"(?P<ts>\d{8}T\d{6}\d*Z)"
    r"\.log$"
)

# rocblas-bench column header → our CSV/DB field name
METRIC_HEADERS: dict[str, str] = {
    "rocblas-Gflops": "gflops",
    "rocblas-GB/s":   "gb_s",
    "us":             "exec_time_us",
    "rocblas-error":  "error_norm",
}


@dataclass
class Row:
    precision: str
    M: int
    N: int
    K: int
    gflops: Optional[float]
    gb_s: Optional[float]
    exec_time_us: Optional[float]
    error_norm: Optional[float]
    timestamp: str          # ISO-8601 UTC, e.g. "2026-04-24T12:34:56+00:00"
    log_file: str

    @classmethod
    def csv_header(cls) -> list[str]:
        return [f.name for f in fields(cls)]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_filename(path: Path) -> tuple[str, int, int, int, str]:
    """Return ``(precision, M, N, K, iso_timestamp)`` from a sweep log filename."""
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"filename does not match sweep convention: {path.name}")
    precision = m.group("precision")
    size = int(m.group("size"))
    raw_ts = m.group("ts")  # e.g. 20260424T123456123456789Z

    # Parse just the seconds-resolution prefix; sub-second digits (nanoseconds
    # from GNU date %N) are kept as metadata in the filename but not in the
    # stored timestamp (second resolution is plenty for benchmark trending).
    seconds_part = raw_ts[:15]  # YYYYMMDDTHHMMSS
    try:
        parsed = dt.datetime.strptime(seconds_part, "%Y%m%dT%H%M%S").replace(
            tzinfo=dt.timezone.utc
        )
        iso = parsed.isoformat()
    except ValueError:
        # Shouldn't happen given the regex, but fall back gracefully.
        logger.warning("could not parse timestamp %r; storing raw", raw_ts)
        iso = raw_ts
    return precision, size, size, size, iso


def _parse_log_body(text: str) -> dict[str, Optional[float]]:
    """Extract metric values from rocblas-bench stdout.

    Strategy: scan lines for the header row (identifiable by the presence of
    'rocblas-Gflops' AND a comma), then take the first subsequent line whose
    comma-split arity matches the header. This is robust to preambles like
    'rocBLAS command line: ...' and any trailing whitespace.
    """
    out: dict[str, Optional[float]] = {v: None for v in METRIC_HEADERS.values()}

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if "rocblas-Gflops" in line and "," in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("no rocblas-Gflops header row found in log")

    header = [h.strip() for h in lines[header_idx].split(",")]
    data_row: Optional[list[str]] = None
    for line in lines[header_idx + 1:]:
        candidate = [c.strip() for c in line.split(",")]
        if len(candidate) == len(header):
            data_row = candidate
            break
    if data_row is None:
        raise ValueError("header found but no matching data row")

    col_index = {name: idx for idx, name in enumerate(header)}
    for src_header, dst_field in METRIC_HEADERS.items():
        if src_header not in col_index:
            # rocblas-error is only present when -v 1 was passed. That's
            # expected for f32, less common for f16. Leave as None.
            logger.debug("header missing column %r; leaving %s as NULL",
                         src_header, dst_field)
            continue
        raw = data_row[col_index[src_header]]
        try:
            out[dst_field] = float(raw)
        except ValueError:
            logger.warning("non-numeric value for %s: %r — leaving as NULL",
                           src_header, raw)
    return out


def parse_log(path: Path) -> Row:
    """Parse one rocblas-bench log file into a ``Row``."""
    precision, M, N, K, ts = _parse_filename(path)
    text = path.read_text(errors="replace")
    metrics = _parse_log_body(text)
    return Row(
        precision=precision,
        M=M, N=N, K=K,
        gflops=metrics["gflops"],
        gb_s=metrics["gb_s"],
        exec_time_us=metrics["exec_time_us"],
        error_norm=metrics["error_norm"],
        timestamp=ts,
        log_file=str(path),
    )


def iter_logs(log_dir: Path) -> Iterator[Path]:
    """Yield sweep log files in deterministic (sorted) order."""
    for p in sorted(log_dir.glob("bench_*.log")):
        if FILENAME_RE.match(p.name):
            yield p
        else:
            logger.debug("skipping non-matching file: %s", p.name)


def write_csv(rows: list[Row], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=Row.csv_header())
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", type=Path, default=Path("logs"),
                    help="directory of bench_*.log files (default: logs)")
    ap.add_argument("--output", type=Path, default=Path("results/results.csv"),
                    help="CSV output path (default: results/results.csv)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.log_dir.is_dir():
        logger.error("log directory not found: %s", args.log_dir)
        return 2

    rows: list[Row] = []
    failures = 0
    for path in iter_logs(args.log_dir):
        try:
            rows.append(parse_log(path))
        except Exception as exc:  # noqa: BLE001 — we want to continue past bad logs
            logger.error("failed to parse %s: %s", path, exc)
            failures += 1

    if not rows:
        logger.error("no rows parsed from %s", args.log_dir)
        return 3

    write_csv(rows, args.output)
    logger.info("wrote %d rows to %s (parse failures: %d)",
                len(rows), args.output, failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
