#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end driver: build rocblas-bench → sweep → parse → load → pytest.
#
# Required env:
#   ROCM_VERSION     e.g. "6.1.0"
#
# Optional env passes through to sweep.sh:
#   REPEATS, ITERATIONS, COLD_ITERATIONS, LOG_DIR, RESULTS_DIR,
#   TRANSPOSE_A, TRANSPOSE_B
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

: "${ROCM_VERSION:?ROCM_VERSION must be set, e.g. export ROCM_VERSION=6.1.0}"

log() { printf '[run_all] %s\n' "$*" >&2; }

log "Step 1/5: build rocblas-bench"
# Last line of build_rocblas.sh stdout is the binary path.
BENCH_PATH="$(install/build_rocblas.sh | tail -n 1)"
export ROCBLAS_BENCH="${BENCH_PATH}"
log "  rocblas-bench: ${ROCBLAS_BENCH}"

log "Step 2/5: sweep"
bench/sweep.sh

log "Step 3/5: parse logs → CSV"
python -m parse.parser --log-dir logs --output results/results.csv

log "Step 4/5: load CSV → SQLite"
python -m parse.db --csv results/results.csv --db results/results.db

log "Step 5/5: pytest"
pytest -v tests/

log "Done."
