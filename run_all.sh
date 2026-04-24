#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end driver for AMD Developer Cloud MI300X droplets.
#
#     ssh root@<DROPLET_IP>
#     git clone <this-repo-url> rocblas-gemm-benchmark
#     cd rocblas-gemm-benchmark
#     bash run_all.sh
#
# Handles automatically:
#   * apt-installing core deps: python3-venv, python3-pip, sqlite3, git
#   * creating + activating a local .venv + pip-installing pinned deps
#   * locating rocblas-bench via:
#       1. $ROCBLAS_BENCH env (if set and executable)
#       2. /opt/rocm/bin/ and /opt/rocm-*/bin/
#       3. dpkg -L on rocBLAS-related packages
#       4. broad /opt + /usr filesystem search
#       5. apt-get install librocblas0-tests (NB: Ubuntu's package does NOT
#          ship the bench binary on most images — typically falls through)
#       6. source build via install/build_rocblas.sh — auto-installs the
#          full build toolchain (cmake, gfortran, libgtest-dev, python3-dev,
#          libnuma-dev, etc.) before invoking rocBLAS's install.sh.
#          After build, sets LD_LIBRARY_PATH to include the build tree so
#          the source-built bench can find Tensile kernel libraries.
#   * auto-detecting ROCM_VERSION from /opt/rocm/.info/version OR /opt/rocm-X.Y.Z
#   * running sweep → parse → load → pytest
#
# Override behavior with env vars:
#   ROCBLAS_BENCH    Path to rocblas-bench (auto-detected if unset).
#   ROCM_VERSION     ROCm version string (auto-detected if unset).
#                    Try 7.2.1 if 7.2.0 source has build issues.
#   FORCE_BUILD=1    Build rocblas-bench from source even if one is installed.
#   USE_KITWARE_CMAKE=1   Pull cmake from Kitware's apt repo before building
#                    rocBLAS (recommended for older Ubuntu; usually
#                    unnecessary on noble 24.04 which ships cmake 3.28).
#   SKIP_APT=1       Skip apt-get install steps.
#   SKIP_VENV=1      Skip venv creation; use the active Python.
#   Any sweep var:   REPEATS, ITERATIONS, COLD_ITERATIONS, LOG_DIR,
#                    RESULTS_DIR, TRANSPOSE_A, TRANSPOSE_B → bench/sweep.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

log() { printf '[run_all] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# --- sudo helper ---------------------------------------------------------
if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
  log "WARNING: not running as root and sudo not found; apt-get may fail"
fi

# Track whether we built rocblas-bench from source (changes LD_LIBRARY_PATH later).
BUILT_FROM_SOURCE=0

# --- Helpers -------------------------------------------------------------
# Echo a path to a usable rocblas-bench, or empty if none found.
find_rocblas_bench() {
  if [[ -n "${ROCBLAS_BENCH:-}" && -x "${ROCBLAS_BENCH}" ]]; then
    echo "${ROCBLAS_BENCH}"; return 0
  fi
  if [[ -x /opt/rocm/bin/rocblas-bench ]]; then
    echo "/opt/rocm/bin/rocblas-bench"; return 0
  fi
  local d
  for d in /opt/rocm-*/bin/rocblas-bench; do
    [[ -x "$d" ]] && { echo "$d"; return 0; }
  done
  if command -v dpkg >/dev/null 2>&1; then
    local pkg bench
    for pkg in librocblas0-tests rocblas-dev rocblas; do
      if dpkg -s "${pkg}" >/dev/null 2>&1; then
        bench="$(dpkg -L "${pkg}" 2>/dev/null | grep -E '/rocblas-bench$' | head -n1 || true)"
        if [[ -n "${bench}" && -x "${bench}" ]]; then
          echo "${bench}"; return 0
        fi
      fi
    done
  fi
  local found
  found="$(find /opt /usr -maxdepth 6 -name rocblas-bench -type f -executable \
              2>/dev/null | head -n1 || true)"
  [[ -n "${found}" ]] && { echo "${found}"; return 0; }
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

# Optional: install cmake from Kitware's apt repo (ensures cmake >= 3.30).
# Stock Ubuntu 24.04 (noble) ships cmake 3.28 which is fine for ROCm 7.2,
# so this is opt-in via USE_KITWARE_CMAKE=1. Mirrors the user's known-good
# bare-metal recipe.
install_kitware_cmake() {
  log "  setting up Kitware apt repo for cmake"
  ${SUDO} apt-get install -y --no-install-recommends \
    wget gpg ca-certificates lsb-release >/dev/null
  local codename
  codename="$(lsb_release -cs)"
  local keyring=/usr/share/keyrings/kitware-archive-keyring.gpg
  wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc | \
    gpg --dearmor 2>/dev/null | ${SUDO} tee "${keyring}" >/dev/null
  echo "deb [signed-by=${keyring}] https://apt.kitware.com/ubuntu/ ${codename} main" | \
    ${SUDO} tee /etc/apt/sources.list.d/kitware.list >/dev/null
  ${SUDO} apt-get update -y >/dev/null
  ${SUDO} apt-get install -y cmake >/dev/null
}

# Install the build toolchain rocBLAS source build needs. Idempotent (apt
# is no-op if everything's already installed). Mirrors the user's proven
# bare-metal recipe.
install_build_toolchain() {
  [[ "${SKIP_APT:-0}" == "1" ]] && return 0
  log "  installing build toolchain (build-essential, gfortran, libgtest-dev, python3-dev, ...)"
  ${SUDO} apt-get install -y --no-install-recommends \
    build-essential gfortran libgtest-dev libnuma-dev \
    python3-dev python3-yaml python3-setuptools \
    >/dev/null

  if [[ "${USE_KITWARE_CMAKE:-0}" == "1" ]]; then
    install_kitware_cmake
  else
    ${SUDO} apt-get install -y --no-install-recommends cmake >/dev/null
  fi
}

# --- Step 1: core system packages ----------------------------------------
if [[ "${SKIP_APT:-0}" != "1" ]]; then
  log "Step 1/7: install core packages (python3-venv, python3-pip, sqlite3, git)"
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
ROCBLAS_BENCH_PATH=""

if [[ "${FORCE_BUILD:-0}" != "1" ]]; then
  ROCBLAS_BENCH_PATH="$(find_rocblas_bench || true)"
fi

if [[ -n "${ROCBLAS_BENCH_PATH}" ]]; then
  log "Step 3/7: found rocblas-bench at ${ROCBLAS_BENCH_PATH}"
else
  # Try the apt package first (cheap if it works).
  if [[ "${FORCE_BUILD:-0}" != "1" && "${SKIP_APT:-0}" != "1" ]]; then
    log "Step 3/7: rocblas-bench not found; trying apt-get install librocblas0-tests"
    if ${SUDO} apt-get install -y --no-install-recommends librocblas0-tests >/dev/null 2>&1; then
      log "  installed librocblas0-tests successfully"
      ROCBLAS_BENCH_PATH="$(find_rocblas_bench || true)"
      if [[ -n "${ROCBLAS_BENCH_PATH}" ]]; then
        log "  located: ${ROCBLAS_BENCH_PATH}"
      else
        log "  package installed but does not ship rocblas-bench (expected on most Ubuntu builds)"
      fi
    else
      log "  apt-get install librocblas0-tests failed (likely a package conflict)"
    fi
  fi

  # Fall through to source build.
  if [[ -z "${ROCBLAS_BENCH_PATH}" ]]; then
    if [[ -z "${ROCM_VERSION:-}" ]]; then
      ROCM_VERSION="$(detect_rocm_version || true)"
      [[ -n "${ROCM_VERSION}" ]] || die "Could not find rocblas-bench and could not auto-detect ROCM_VERSION. Set manually: export ROCM_VERSION=7.2.1"
      log "  auto-detected ROCM_VERSION=${ROCM_VERSION} for source build"
    fi
    export ROCM_VERSION
    install_build_toolchain
    log "  building rocblas-bench from source (~5–10 minutes on a 20-core droplet)"
    ROCBLAS_BENCH_PATH="$(install/build_rocblas.sh | tail -n 1)"
    BUILT_FROM_SOURCE=1
  fi
fi

export ROCBLAS_BENCH="${ROCBLAS_BENCH_PATH}"

# Auto-detect ROCM_VERSION for forensic logging if still unset.
if [[ -z "${ROCM_VERSION:-}" ]]; then
  ROCM_VERSION="$(detect_rocm_version || echo unknown)"
fi
export ROCM_VERSION
log "  ROCM_VERSION=${ROCM_VERSION}"

# --- Step 4.5: configure runtime library path for source-built bench -----
# The clients-only build doesn't install Tensile kernel libraries to a
# system location, so the bench binary needs LD_LIBRARY_PATH pointing at
# the build tree (mirrors the user's known-good bare-metal recipe).
if [[ "${BUILT_FROM_SOURCE}" -eq 1 ]]; then
  BUILD_RELEASE_DIR="${REPO_ROOT}/local/src/rocBLAS/build/release"
  if [[ -d "${BUILD_RELEASE_DIR}" ]]; then
    export LD_LIBRARY_PATH="${BUILD_RELEASE_DIR}/rocblas/library:${BUILD_RELEASE_DIR}:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    log "  LD_LIBRARY_PATH set for source-built bench"
  fi
fi

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
