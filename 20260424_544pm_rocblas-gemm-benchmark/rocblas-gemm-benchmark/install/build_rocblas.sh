#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build rocBLAS + rocblas-bench clients for MI300X (gfx942).
#
# Strategy:
#   * --clients-only builds just the benchmark/test client binaries against
#     the system-installed rocBLAS runtime (assumed at /opt/rocm). This is
#     ~10x faster than a full library rebuild and is what the spec asks for.
#   * Source and build artifacts live under a LOCAL prefix inside the repo
#     (default ./local), so this script never needs root and is safe in CI.
#   * Idempotent: if a stamp file matching ROCM_VERSION is present and the
#     rocblas-bench binary exists, the build is skipped.
#
# Required env:
#   ROCM_VERSION             e.g. "6.1.0" (maps to git tag rocm-6.1.0)
#
# Optional env:
#   ROCBLAS_INSTALL_PREFIX   default: <repo>/local
#   ROCBLAS_SRC_DIR          default: <prefix>/src/rocBLAS
#   GPU_ARCH                 default: gfx942 (MI300X)
#   JOBS                     default: $(nproc)
#
# On success, prints the absolute path to rocblas-bench on stdout.
# ---------------------------------------------------------------------------

set -euo pipefail

log() { printf '[build_rocblas] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# --- Preconditions --------------------------------------------------------
[[ -n "${ROCM_VERSION:-}" ]] || die \
  "ROCM_VERSION is not set. Example: export ROCM_VERSION=6.1.0"

command -v git >/dev/null 2>&1 || die "git is required but not on PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_PREFIX="${ROCBLAS_INSTALL_PREFIX:-${REPO_ROOT}/local}"
SRC_DIR="${ROCBLAS_SRC_DIR:-${INSTALL_PREFIX}/src/rocBLAS}"
GPU_ARCH="${GPU_ARCH:-gfx942}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
STAMP_FILE="${INSTALL_PREFIX}/.rocblas-${ROCM_VERSION}-${GPU_ARCH}.stamp"
BENCH_BIN="${SRC_DIR}/build/release/clients/staging/rocblas-bench"

mkdir -p "${INSTALL_PREFIX}"

# --- Log header (forensic reproducibility) -------------------------------
log "repo_root=${REPO_ROOT}"
log "hostname=$(hostname)"
log "rocm_version=${ROCM_VERSION}"
log "gpu_arch=${GPU_ARCH}"
log "install_prefix=${INSTALL_PREFIX}"
log "jobs=${JOBS}"

# --- Idempotency check ----------------------------------------------------
if [[ -f "${STAMP_FILE}" && -x "${BENCH_BIN}" ]]; then
  log "rocblas-bench already built for ROCm ${ROCM_VERSION} / ${GPU_ARCH}"
  log "  stamp: ${STAMP_FILE}"
  log "  bench: ${BENCH_BIN}"
  log "  (remove stamp to force rebuild)"
  echo "${BENCH_BIN}"
  exit 0
fi

# --- Fetch source ---------------------------------------------------------
ROCBLAS_TAG="rocm-${ROCM_VERSION}"
if [[ ! -d "${SRC_DIR}/.git" ]]; then
  log "Cloning rocBLAS @ ${ROCBLAS_TAG} into ${SRC_DIR}"
  mkdir -p "$(dirname "${SRC_DIR}")"
  git clone --depth 1 --branch "${ROCBLAS_TAG}" \
    https://github.com/ROCm/rocBLAS.git "${SRC_DIR}" \
    || die "git clone failed — check that tag '${ROCBLAS_TAG}' exists"
else
  log "Updating existing checkout at ${SRC_DIR} to ${ROCBLAS_TAG}"
  git -C "${SRC_DIR}" fetch --depth 1 origin "${ROCBLAS_TAG}"
  git -C "${SRC_DIR}" checkout "${ROCBLAS_TAG}"
fi

# --- Build ----------------------------------------------------------------
# install.sh flags:
#   --clients-only  : build client binaries (rocblas-bench, rocblas-test) only
#   -a <arch>       : target GPU architecture
#   -j <n>          : parallel build jobs
log "Building rocblas-bench (clients-only) for ${GPU_ARCH} with ${JOBS} jobs"
pushd "${SRC_DIR}" >/dev/null
./install.sh --clients-only -a "${GPU_ARCH}" -j "${JOBS}"
popd >/dev/null

# --- Verify artifact ------------------------------------------------------
[[ -x "${BENCH_BIN}" ]] || die \
  "build completed but rocblas-bench binary not found at ${BENCH_BIN}"

# --- Stamp ---------------------------------------------------------------
{
  echo "rocm_version=${ROCM_VERSION}"
  echo "gpu_arch=${GPU_ARCH}"
  echo "built_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "bench_bin=${BENCH_BIN}"
  echo "src_dir=${SRC_DIR}"
  echo "hostname=$(hostname)"
} > "${STAMP_FILE}"

log "Build complete. rocblas-bench → ${BENCH_BIN}"
echo "${BENCH_BIN}"
