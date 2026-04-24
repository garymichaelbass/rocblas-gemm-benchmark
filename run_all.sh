#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end driver for AMD Developer Cloud MI300X droplets.
#
# Designed to be the one-and-only command you need on a fresh droplet:
#
#     ssh root@<DROPLET_IP>
#     git clone <this-repo-url> rocblas-gemm-benchmark
#     cd rocblas-gemm-benchmark
#     bash run_all.sh
#
# Handles automatically:
#   * apt-installing python3-venv / python3-pip / sqlite3 / git
#   * creating + activating a local .venv
#   * pip-installing pinned deps (pytest, PyYAML)
#   * detecting pre-installed rocblas-bench at /opt/rocm/bin/rocblas-bench
#   * auto-detecting ROCM_VERSION from /opt/rocm/.info/version
#   * running sweep → parse → load → pytest
#
# Override behavior with env vars:
#   ROCBLAS_BENCH    Path to rocblas-bench (auto-detected if unset).
#   ROCM_VERSION     ROCm version string (auto-detected if unset).
#   FORCE_BUILD=1    Build rocblas-bench from source even if pre-installed.
#                    Requires ROCM_VERSION.
#   SKIP_APT=1       Skip the apt-get install step (e.g. inside containers
#                    where you've already provisioned packages).
#   SKIP_VENV=1      Skip venv creation/activation (use current interpreter).
#   Any sweep var:   REPEATS, ITERATIONS, COLD_ITERATIONS, LOG_DIR,
#                    RESULTS_DIR, TRANSPOSE_A, TRANSPOSE_B
#                    (passed through to bench/sweep.sh).
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

log() { printf '[run_all] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# --- sudo helper ---------------------------------------------------------
# AMD Developer Cloud droplets log you in as root, so SUDO is empty.
# On a non-root user with sudo available, we prefix apt with sudo.
if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
  log "WARNING: not running as root and sudo not found; apt-get may fail"
fi

# --- Step 1: system packages --------------------------------------------
if [[ "${SKIP_APT:-0}" != "1" ]]; then
  log "Step 1/7: install system packages (python3-venv, python3-pip, sqlite3, git)"
  ${SUDO} apt-get update -y >/dev/null
  ${SUDO} apt-get install -y --no-install-recommends \
    python3-venv python3-pip sqlite3 git ca-certificates >/dev/null
else
  log "Step 1/7: skipping apt-get (SKIP_APT=1)"
fi

# --- Step 2: Python venv + deps ------------------------------------------
if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  log "Step 2/7: set up Python venv at .venv/"
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  log "Step 2/7: skipping venv (SKIP_VENV=1); using current Python: $(command -v python3)"
fi
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
log "  python: $(python --version 2>&1)"
log "  pytest: $(pytest --version 2>&1 | head -n1)"

# --- Step 3: ensure scripts are executable -------------------------------
chmod +x install/build_rocblas.sh bench/sweep.sh

# --- Step 4: locate rocblas-bench ---------------------------------------
PREINSTALLED="/opt/rocm/bin/rocblas-bench"

if [[ "${FORCE_BUILD:-0}" == "1" ]]; then
  : "${ROCM_VERSION:?ROCM_VERSION must be set when FORCE_BUILD=1}"
  log "Step 3/7: FORCE_BUILD=1 — building rocblas-bench from source (ROCM_VERSION=${ROCM_VERSION})"
  ROCBLAS_BENCH="$(install/build_rocblas.sh | tail -n 1)"
elif [[ -n "${ROCBLAS_BENCH:-}" && -x "${ROCBLAS_BENCH}" ]]; then
  log "Step 3/7: using ROCBLAS_BENCH from env: ${ROCBLAS_BENCH}"
elif [[ -x "${PREINSTALLED}" ]]; then
  ROCBLAS_BENCH="${PREINSTALLED}"
  log "Step 3/7: using pre-installed rocblas-bench at ${ROCBLAS_BENCH}"
else
  log "Step 3/7: no pre-installed rocblas-bench found at ${PREINSTALLED}, building from source"
  : "${ROCM_VERSION:?ROCM_VERSION must be set to build rocblas-bench from source}"
  ROCBLAS_BENCH="$(install/build_rocblas.sh | tail -n 1)"
fi
export ROCBLAS_BENCH

# Detect ROCM_VERSION for forensic logging if user didn't set it.
if [[ -z "${ROCM_VERSION:-}" ]]; then
  if [[ -f /opt/rocm/.info/version ]]; then
    ROCM_VERSION="$(cut -d'-' -f1 < /opt/rocm/.info/version)"
    log "  detected ROCM_VERSION=${ROCM_VERSION} from /opt/rocm/.info/version"
  else
    ROCM_VERSION="unknown"
    log "  WARNING: could not auto-detect ROCM_VERSION (no /opt/rocm/.info/version)"
  fi
fi
export ROCM_VERSION

# --- Step 5: sweep -------------------------------------------------------
log "Step 4/7: run rocblas-bench sweep"
bench/sweep.sh

# --- Step 6: parse logs --------------------------------------------------
log "Step 5/7: parse logs → CSV"
python -m parse.parser --log-dir logs --output results/results.csv

# --- Step 7: load DB -----------------------------------------------------
log "Step 6/7: load CSV → SQLite"
python -m parse.db --csv results/results.csv --db results/results.db

# --- Step 8: pytest ------------------------------------------------------
log "Step 7/7: pytest"
pytest -v tests/

log "Done. Results in results/results.db; logs in logs/."
