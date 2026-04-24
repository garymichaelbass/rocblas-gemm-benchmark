#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sweep rocblas-bench across {f16, bf16, f32} × {4096, 8192, 16384} square
# GEMM on MI300X, one log file per invocation.
#
# Each run captures GFLOPS, GB/s, wall time, and the relative error norm
# (verification enabled via -v 1). rocm-smi snapshots are taken once before
# and once after the full sweep and saved to sidecar files (GPU state is a
# sweep-level observation, not per-run).
#
# Required (for reproducibility in logs, not for function):
#   ROCM_VERSION    e.g. "6.1.0"
#
# Optional env:
#   ROCBLAS_BENCH   path to rocblas-bench binary
#                   (default: build tree under <repo>/local/...)
#   REPEATS         external repetitions per (prec, size) cell   (default 3)
#   ITERATIONS      internal rocblas-bench iterations per run    (default 10)
#   COLD_ITERATIONS internal warmup iterations                    (default 2)
#   LOG_DIR         per-run stdout logs                           (default <repo>/logs)
#   RESULTS_DIR     sidecar outputs (rocm-smi snapshots, etc.)    (default <repo>/results)
#   TRANSPOSE_A     'N' or 'T' (default N — non-transposed)
#   TRANSPOSE_B     'N' or 'T' (default N)
#
# Exits non-zero on any rocblas-bench failure or empty log.
# ---------------------------------------------------------------------------

set -euo pipefail

log() { printf '[sweep] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# rocblas-bench location: prefer env, else the clients-only build tree.
DEFAULT_BENCH="${REPO_ROOT}/local/src/rocBLAS/build/release/clients/staging/rocblas-bench"
ROCBLAS_BENCH="${ROCBLAS_BENCH:-${DEFAULT_BENCH}}"

REPEATS="${REPEATS:-3}"
ITERATIONS="${ITERATIONS:-10}"
COLD_ITERATIONS="${COLD_ITERATIONS:-2}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
TRANSPOSE_A="${TRANSPOSE_A:-N}"
TRANSPOSE_B="${TRANSPOSE_B:-N}"

[[ -x "${ROCBLAS_BENCH}" ]] \
  || die "rocblas-bench not found or not executable at ${ROCBLAS_BENCH}
Hint: run install/build_rocblas.sh first, or set ROCBLAS_BENCH."

# Validate integer params
for v in REPEATS ITERATIONS COLD_ITERATIONS; do
  [[ "${!v}" =~ ^[0-9]+$ ]] && (( ${!v} > 0 )) \
    || die "${v} must be a positive integer (got '${!v}')"
done

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

SWEEP_TS="$(date -u +'%Y%m%dT%H%M%SZ')"
SWEEP_HEADER="${LOG_DIR}/sweep_header_${SWEEP_TS}.txt"

# --- Environment / forensic snapshot -------------------------------------
{
  echo "sweep_timestamp_utc=${SWEEP_TS}"
  echo "hostname=$(hostname)"
  echo "rocm_version=${ROCM_VERSION:-unset}"
  echo "rocblas_bench=${ROCBLAS_BENCH}"
  echo "repeats=${REPEATS}"
  echo "iterations=${ITERATIONS}"
  echo "cold_iterations=${COLD_ITERATIONS}"
  echo "transpose_a=${TRANSPOSE_A}"
  echo "transpose_b=${TRANSPOSE_B}"
  echo "kernel=$(uname -a)"
} > "${SWEEP_HEADER}"
log "Sweep header written: ${SWEEP_HEADER}"

# rocm-smi pre-sweep snapshot. Tolerate absence on dev hosts.
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showclocks --showtemp --showpower --showproductname \
    > "${RESULTS_DIR}/rocm-smi_pre_${SWEEP_TS}.txt" 2>&1 || true
  log "rocm-smi pre snapshot → ${RESULTS_DIR}/rocm-smi_pre_${SWEEP_TS}.txt"
else
  log "WARNING: rocm-smi not found on PATH; skipping GPU state capture"
fi

# --- Sweep grid -----------------------------------------------------------
PRECISIONS=(f16 bf16 f32)
SIZES=(4096 8192 16384)

# Map our short precision names → rocblas-bench -r flag values.
# -r sets compute_type and all of {a,b,c,d}_type to the same value, which is
# exactly what the spec calls for (uniform precision sweep).
precision_flag() {
  case "$1" in
    f16)  echo "f16_r"  ;;
    bf16) echo "bf16_r" ;;
    f32)  echo "f32_r"  ;;
    *) die "unknown precision: $1" ;;
  esac
}

total=$(( ${#PRECISIONS[@]} * ${#SIZES[@]} * REPEATS ))
run=0

for prec in "${PRECISIONS[@]}"; do
  r_flag="$(precision_flag "${prec}")"
  for size in "${SIZES[@]}"; do
    for ((i = 1; i <= REPEATS; i++)); do
      run=$((run + 1))
      # nanosecond-resolution timestamp keeps log filenames unique across
      # tight back-to-back runs (a per-second stamp is not unique enough).
      ts="$(date -u +'%Y%m%dT%H%M%S%NZ')"
      log_file="${LOG_DIR}/bench_${prec}_${size}_run${i}_${ts}.log"

      log "[${run}/${total}] prec=${prec} M=N=K=${size} run=${i}/${REPEATS} → ${log_file##*/}"

      # -v 1 enables verification; without it rocblas-error is NOT emitted.
      # NOTE: The transpose flag is --transposeA / --transposeB in current
      # rocBLAS (6.x). Older (pre-5.0) builds used --transA / --transB; if
      # using an older rocBLAS, update these flags accordingly.
      if ! "${ROCBLAS_BENCH}" \
             -f gemm \
             -r "${r_flag}" \
             --transposeA "${TRANSPOSE_A}" \
             --transposeB "${TRANSPOSE_B}" \
             -m "${size}" -n "${size}" -k "${size}" \
             --alpha 1 --beta 0 \
             -i "${ITERATIONS}" -j "${COLD_ITERATIONS}" \
             -v 1 \
             > "${log_file}" 2>&1; then
        die "rocblas-bench exited non-zero for prec=${prec} size=${size} run=${i}
See ${log_file} for details."
      fi

      # Sanity check: guard the parser downstream from empty files.
      [[ -s "${log_file}" ]] || die "empty log produced: ${log_file}"
    done
  done
done

# rocm-smi post-sweep snapshot
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showclocks --showtemp --showpower --showproductname \
    > "${RESULTS_DIR}/rocm-smi_post_${SWEEP_TS}.txt" 2>&1 || true
  log "rocm-smi post snapshot → ${RESULTS_DIR}/rocm-smi_post_${SWEEP_TS}.txt"
fi

log "Sweep complete. ${total} runs logged under ${LOG_DIR}"
