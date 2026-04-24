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
#   * apt-installing librocblas0-tests if rocblas-bench isn't on the system
#     (this is the package that provides /opt/rocm-X.Y.Z/bin/rocblas-bench
#     on the AMD Developer Cloud's pre-installed ROCm image)
#   * creating + activating a local .venv
#   * pip-installing pinned deps (pytest, PyYAML)
#   * locating rocblas-bench across /opt/rocm*, /usr/bin, /usr/local/bin
#   * auto-detecting ROCM_VERSION from /opt/rocm/.info/version OR /opt/rocm-X.Y.Z
#   * running sweep → parse → load → pytest
#
# Override behavior with env vars:
#   ROCBLAS_BENCH    Path to rocblas-bench (auto-detected if unset).
#   ROCM_VERSION     ROCm version string (auto-detected if unset).
#   FORCE_BUILD=1    Build rocblas-bench from source even if pre-installed.
#                    Requires ROCM_VERSION (auto-detected if not given).
#   SKIP_APT=1       Skip the apt-get install step.
#   SKIP_VENV=1      Skip venv creation; use the active Python.
#   Any sweep var:   REPEATS, ITERATIONS, COLD_ITERATIONS, LOG_DIR,
#                    RESULTS_DIR, TRANSPOSE_A, TRANSPOSE_B  → bench/sweep.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

log() { printf '[run_all] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# --- sudo helper ---------------------------------------------------------
# AMD Developer Cloud droplets log you in as root, so SUDO is empty.
if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
  log "WARNING: not running as root and sudo not found; apt-get may fail"
fi

# --- Helpers -------------------------------------------------------------
# Echo a path to a usable rocblas-bench, or empty string if none found.
find_rocblas_bench() {
  # 1. Honor explicit env if it points to an executable file.
  if [[ -n "${ROCBLAS_BENCH:-}" && -x "${ROCBLAS_BENCH}" ]]; then
    echo "${ROCBLAS_BENCH}"; return 0
  fi
  # 2. Standard symlinked location.
  if [[ -x /opt/rocm/bin/rocblas-bench ]]; then
    echo "/opt/rocm/bin/rocblas-bench"; return 0
  fi
  # 3. Versioned ROCm directories (e.g. /opt/rocm-7.2.0/bin/).
  local d found
  for d in /opt/rocm-*/bin/rocblas-bench; do
    [[ -x "$d" ]] && { echo "$d"; return 0; }
  done
  # 4. Broader filesystem search as a last resort (Ubuntu may put it in /usr/bin).
  found="$(find /opt/rocm /opt/rocm-* /usr/bin /usr/local/bin \
              -maxdepth 4 -name rocblas-bench -type f -executable \
              2>/dev/null | head -n1 || true)"
  [[ -n "$found" ]] && { echo "$found"; return 0; }
  return 1
}

# Echo a detected ROCm version (e.g. "7.2.0"), or empty if none.
detect_rocm_version() {
  if [[ -f /opt/rocm/.info/version ]]; then
    cut -d'-' -f1 < /opt/rocm/.info/version
    return 0
  fi
  local d
  for d in /opt/rocm-*; do
    if [[ -d "$d" && "$d" =~ /opt/rocm-([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
      echo "${BASH_REMATCH[1]}"
      return 0
    fi
  done
  return 1
}

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
ROCBLAS_BENCH_PATH="$(find_rocblas_bench || true)"

# If the user set FORCE_BUILD=1, ignore whatever's installed and build.
if [[ "${FORCE_BUILD:-0}" == "1" ]]; then
  if [[ -z "${ROCM_VERSION:-}" ]]; then
    ROCM_VERSION="$(detect_rocm_version || true)"
    [[ -n "${ROCM_VERSION}" ]] || die "FORCE_BUILD=1 set but ROCM_VERSION unset and could not auto-detect"
  fi
  export ROCM_VERSION
  log "Step 3/7: FORCE_BUILD=1 — building rocblas-bench from source (ROCM_VERSION=${ROCM_VERSION})"
  ROCBLAS_BENCH_PATH="$(install/build_rocblas.sh | tail -n 1)"
elif [[ -n "${ROCBLAS_BENCH_PATH}" ]]; then
  log "Step 3/7: found rocblas-bench at ${ROCBLAS_BENCH_PATH}"
else
  # Not found anywhere. On the AMD Developer Cloud, the bench client lives
  # in the librocblas0-tests package, which is in the apt repos but not
  # installed by default. Try that first — it's ~30 sec vs ~30 min source build.
  if [[ "${SKIP_APT:-0}" != "1" ]]; then
    log "Step 3/7: rocblas-bench not found; trying apt-get install librocblas0-tests"
    if ${SUDO} apt-get install -y --no-install-recommends librocblas0-tests >/dev/null 2>&1; then
      log "  installed librocblas0-tests successfully"
      ROCBLAS_BENCH_PATH="$(find_rocblas_bench || true)"
      [[ -n "${ROCBLAS_BENCH_PATH}" ]] && log "  located: ${ROCBLAS_BENCH_PATH}"
    else
      log "  apt-get install librocblas0-tests failed (likely a package conflict)"
    fi
  fi

  # Still not found → source build, with auto-detected ROCM_VERSION.
  if [[ -z "${ROCBLAS_BENCH_PATH}" ]]; then
    if [[ -z "${ROCM_VERSION:-}" ]]; then
      ROCM_VERSION="$(detect_rocm_version || true)"
      [[ -n "${ROCM_VERSION}" ]] || die "rocblas-bench not found, apt install failed, and ROCM_VERSION could not be auto-detected. Set it manually: export ROCM_VERSION=7.2.0"
      log "  auto-detected ROCM_VERSION=${ROCM_VERSION} for source build"
    fi
    export ROCM_VERSION
    log "  building rocblas-bench from source (this can take 10–30 minutes)"
    ROCBLAS_BENCH_PATH="$(install/build_rocblas.sh | tail -n 1)"
  fi
fi

export ROCBLAS_BENCH="${ROCBLAS_BENCH_PATH}"

# Auto-detect ROCM_VERSION for forensic logging if still unset.
if [[ -z "${ROCM_VERSION:-}" ]]; then
  ROCM_VERSION="$(detect_rocm_version || echo unknown)"
  log "  detected ROCM_VERSION=${ROCM_VERSION}"
fi
export ROCM_VERSION

# --- Step 5: sweep -------------------------------------------------------
log "Step 4/7: run rocblas-bench sweep (using ${ROCBLAS_BENCH})"
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
