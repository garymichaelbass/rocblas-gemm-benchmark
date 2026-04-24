# rocblas-gemm-benchmark

Sweep `rocblas-bench` across **{f16, bf16, f32} × {4096, 8192, 16384}**
square GEMM on AMD Instinct MI300X (gfx942), capture GFLOPS / GB·s⁻¹ /
wall time / error norm, persist into SQLite, and assert correctness +
performance thresholds via pytest.

This README is written specifically for the **AMD Developer Cloud**
(DigitalOcean-backed GPU droplets) running **Ubuntu 22.04 LTS (jammy) or
24.04 LTS (noble)** with **ROCm 7.2.x pre-installed**. Every command
below is copy-pasteable into a fresh droplet SSH session as the default
`root` user.

---

## TL;DR — run everything on a fresh droplet

```bash
# 0. SSH into the droplet from your workstation
ssh root@<DROPLET_IP>

# 1. Verify the GPU and ROCm are live
rocm-smi                        # should list 1× or 8× MI300X at idle ~40–50°C
cat /opt/rocm/.info/version     # e.g. 7.2.2-XX

# 2. Grab the code (either clone or scp from your workstation)
#    Option A: from a git remote you control
git clone <your-remote-url> rocblas-gemm-benchmark
#    Option B: from your local machine, on the workstation side:
#    scp -r rocblas-gemm-benchmark root@<DROPLET_IP>:~/

cd ~/rocblas-gemm-benchmark

# 3. Python deps (droplets ship with python3.12 on 24.04, python3.10 on 22.04)
apt-get update -y
apt-get install -y python3-venv python3-pip sqlite3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Point the sweep at the pre-installed rocblas-bench
export ROCBLAS_BENCH=/opt/rocm/bin/rocblas-bench
export ROCM_VERSION="$(cat /opt/rocm/.info/version | cut -d'-' -f1)"   # e.g. 7.2.2

# 5. Run the pipeline
chmod +x install/build_rocblas.sh bench/sweep.sh run_all.sh
bench/sweep.sh                                                  # ~2–5 min with defaults
python -m parse.parser --log-dir logs --output results/results.csv
python -m parse.db     --csv results/results.csv --db results/results.db
pytest -v tests/                                                # 27 tests, per-cell
```

If all 27 assertions pass, the droplet's rocBLAS GEMM kernels meet the
configured MI300X thresholds. If any fail, pytest reports the exact cell
(e.g. `test_gflops_meets_threshold[f32-m16384]`).

---

## Prerequisites on the droplet

The AMD Developer Cloud's ROCm-pre-installed image ships with almost
everything needed. Verify each item before starting:

| Component             | Check command                                | Expected                                |
| --------------------- | -------------------------------------------- | --------------------------------------- |
| GPU visible           | `rocm-smi`                                   | MI300X device(s) listed                 |
| ROCm version          | `cat /opt/rocm/.info/version`                | `7.2.x-*`                               |
| rocblas-bench present | `ls /opt/rocm/bin/rocblas-bench`             | path exists and is executable           |
| Python ≥ 3.10         | `python3 --version`                          | `Python 3.10+`                          |
| Ubuntu release        | `lsb_release -ds`                            | `Ubuntu 22.04 LTS` or `Ubuntu 24.04 LTS`|

If **`rocblas-bench` is missing** (rare on the pre-installed image, but
possible on custom images), install the dev package:

```bash
apt-get install -y rocblas-dev    # provides /opt/rocm/bin/rocblas-bench
```

If you need to build `rocblas-bench` from source anyway (e.g. to test an
unreleased commit), use the provided builder — see "Building from source"
below.

---

## Environment variables reference

All variables are optional unless marked **required**.

### For using the pre-installed `rocblas-bench`

| Variable             | Default                       | Purpose                                       |
| -------------------- | ----------------------------- | --------------------------------------------- |
| `ROCBLAS_BENCH`      | *(none)*                      | **Set this** to `/opt/rocm/bin/rocblas-bench` to use the pre-installed binary and skip the build. |
| `ROCM_VERSION`       | *(none)*                      | Only used for logging/forensic metadata when using the pre-installed binary. Recommended: `$(cat /opt/rocm/.info/version \| cut -d'-' -f1)`. |

### For building `rocblas-bench` from source (`install/build_rocblas.sh`)

| Variable                  | Default                      | Purpose                                                 |
| ------------------------- | ---------------------------- | ------------------------------------------------------- |
| `ROCM_VERSION`            | *(none, **required**)*       | git tag suffix, e.g. `7.2.2` → checks out `rocm-7.2.2`. |
| `ROCBLAS_INSTALL_PREFIX`  | `<repo>/local`               | Where source and build artifacts live.                  |
| `ROCBLAS_SRC_DIR`         | `<prefix>/src/rocBLAS`       | rocBLAS git checkout location.                          |
| `GPU_ARCH`                | `gfx942`                     | MI300X architecture. Do not change on MI300X.           |
| `JOBS`                    | `$(nproc)`                   | Parallel build jobs.                                    |

### For the sweep (`bench/sweep.sh`)

| Variable          | Default                  | Purpose                                            |
| ----------------- | ------------------------ | -------------------------------------------------- |
| `ROCBLAS_BENCH`   | `<prefix>/.../staging/rocblas-bench` | Binary to invoke.                          |
| `REPEATS`         | `3`                      | External repetitions per (prec, size) cell.        |
| `ITERATIONS`      | `10`                     | Internal `rocblas-bench` iterations per invocation.  |
| `COLD_ITERATIONS` | `2`                      | Warmup iterations per invocation.                  |
| `LOG_DIR`         | `<repo>/logs`            | Per-run stdout logs.                               |
| `RESULTS_DIR`     | `<repo>/results`         | `rocm-smi` snapshots, CSV, SQLite DB.              |
| `TRANSPOSE_A`     | `N`                      | `N` or `T`. Default is non-transposed.             |
| `TRANSPOSE_B`     | `N`                      | `N` or `T`.                                        |

### For the parser / DB / tests

| Variable                | Default                         | Purpose                                    |
| ----------------------- | ------------------------------- | ------------------------------------------ |
| `ROCBLAS_BENCH_DB`      | `results/results.db`            | SQLite path. Overrides both the `db.py --db` default and the pytest fixture default. |

---

## Stage-by-stage execution

Each stage is independent and idempotent (except the DB loader, which
appends). You can stop and resume at any boundary.

### Stage 1 — Choose your `rocblas-bench`

**Recommended on AMD Developer Cloud:** use the pre-installed binary.

```bash
export ROCBLAS_BENCH=/opt/rocm/bin/rocblas-bench
export ROCM_VERSION="$(cat /opt/rocm/.info/version | cut -d'-' -f1)"
"${ROCBLAS_BENCH}" --version       # smoke test
```

### Stage 2 — Run the sweep

```bash
# Default: 27 invocations (3 prec × 3 sizes × 3 repeats), ~2–5 min wall time.
bench/sweep.sh

# Quick smoke (1 repeat per cell, single-digit seconds for f16/bf16 but
# slower for f32 at 16384):
REPEATS=1 bench/sweep.sh

# Heavier run for more stable medians:
REPEATS=5 ITERATIONS=20 bench/sweep.sh
```

Logs land in `logs/bench_<prec>_<size>_run<n>_<ts>.log`. A
`sweep_header_*.txt` captures the full environment. `rocm-smi` snapshots
are written to `results/rocm-smi_{pre,post}_<ts>.txt`.

### Stage 3 — Parse logs into CSV

```bash
python -m parse.parser --log-dir logs --output results/results.csv
# With debug output:
python -m parse.parser --log-dir logs --output results/results.csv -v
```

Exit codes: `0` clean, `1` parsed with failures, `2` log dir missing,
`3` no parseable files.

### Stage 4 — Load CSV into SQLite

```bash
python -m parse.db --csv results/results.csv --db results/results.db
```

Inspect the DB directly (useful for ad-hoc investigation):

```bash
sqlite3 results/results.db <<'SQL'
.mode column
.headers on
SELECT precision, M,
       COUNT(*)               AS runs,
       ROUND(AVG(gflops),1)   AS mean_gflops,
       ROUND(MIN(gflops),1)   AS min_gflops,
       ROUND(MAX(gflops),1)   AS max_gflops,
       MAX(error_norm)        AS worst_err
FROM gemm_runs
GROUP BY precision, M
ORDER BY precision, M;
SQL
```

Rows are always **appended** (never upserted), so re-running the
pipeline builds a trend history. For a clean slate: `rm results/results.db`.

### Stage 5 — Run the assertions

```bash
pytest -v tests/
```

Filter to a subset when debugging:

```bash
# Only f32 cells
pytest -v tests/ -k "f32"

# Only the perf tests, all cells
pytest -v tests/test_gemm_perf.py

# Only the 16384 × 16384 f16 cell
pytest -v "tests/test_gemm_perf.py::test_gflops_meets_threshold[f16-m16384]"
```

---

## Building `rocblas-bench` from source (optional)

Only needed if the droplet image lacks the binary **or** you want to
benchmark an unreleased rocBLAS commit. Note that `--clients-only` builds
only the benchmark/test clients against the **system-installed** rocBLAS
runtime — it does not rebuild the library.

```bash
export ROCM_VERSION=7.2.2        # required — maps to git tag rocm-7.2.2
install/build_rocblas.sh         # prints the built binary path on stdout
export ROCBLAS_BENCH="$(install/build_rocblas.sh | tail -n 1)"
```

The build is idempotent: repeated calls with the same `ROCM_VERSION`
skip rebuilding. Force a rebuild with `rm local/.rocblas-*.stamp`.

---

## Full one-shot driver

`run_all.sh` chains everything, intended for CI or a clean benchmark
pass. It **requires `ROCM_VERSION`** because it will call
`install/build_rocblas.sh` by default.

```bash
export ROCM_VERSION=7.2.2
./run_all.sh
```

To skip the build step (use the pre-installed binary) and still run the
rest of the pipeline end-to-end, just run the stages manually as in the
TL;DR — the driver script is only a convenience.

---

## Directory layout

```
rocblas-gemm-benchmark/
├── install/build_rocblas.sh          # idempotent clients-only build
├── bench/sweep.sh                    # parameterized sweep driver
├── parse/
│   ├── parser.py                     # stdout → CSV (regex + dataclass)
│   └── db.py                         # CSV → SQLite (append-only)
├── tests/
│   ├── conftest.py                   # session fixtures + cell parametrize
│   ├── test_gemm_correctness.py      # error_norm ≤ 1e-3
│   └── test_gemm_perf.py             # median GFLOPS ≥ config threshold
├── config.yaml                       # MI300X thresholds (GFLOPS, not TFLOPS)
├── requirements.txt                  # pinned: pytest, PyYAML
├── run_all.sh                        # chains all steps
└── .gitignore
```

`logs/`, `results/`, and `local/` are created at runtime and gitignored.

---

## Configuration

Thresholds live in `config.yaml`:

```yaml
thresholds:
  gflops:
    f16:  900000    # ~69% of 1,307,400 peak (MI300X matrix FP16/BF16)
    bf16: 900000
    f32:  110000    # ~67% of 163,400 peak (MI300X matrix FP32)
  error_norm: 1.0e-3
```

Values are **practical floors (~70% of peak)**, not theoretical peaks.
See the inline comments in `config.yaml` for the MI300X peak-GFLOPS
table and rationale. Tune upward only after observing sustained better
numbers on a warmed, unshared MI300X.

> **Note re: AMD Developer Cloud:** AMD engineers have stated that
> Developer Cloud droplets are not intended for public performance
> claims due to potential noisy-neighbor variance on shared tenancy.
> The thresholds above are set conservatively to pass on a typical
> droplet; they are useful for regression detection and correctness
> validation rather than for quoting peak numbers externally.

---

## Database schema

```sql
CREATE TABLE gemm_runs (
    id            INTEGER PRIMARY KEY,
    precision     TEXT    NOT NULL,          -- 'f16' | 'bf16' | 'f32'
    M, N, K       INTEGER NOT NULL,
    gflops        REAL,
    gb_s          REAL,
    exec_time_us  REAL,
    error_norm    REAL,
    timestamp     TEXT    NOT NULL,          -- ISO-8601 UTC, per-run
    log_file      TEXT    NOT NULL,          -- path to source log
    inserted_at   TEXT    NOT NULL           -- DB insert time
);
-- Indexes: (precision, M, N, K) and (timestamp)
```

---

## Test semantics

Tests parametrize over all nine `(precision, matrix_size)` cells. A
failure in one cell does not mask others — the pytest report identifies
the exact failing cell.

| Test                                  | What it asserts                                                        |
| ------------------------------------- | ---------------------------------------------------------------------- |
| `test_cell_has_data`                  | Every sweep-grid cell has ≥ 1 recorded run in the DB.                  |
| `test_error_norm_within_threshold`    | Worst-case `rocblas-error` across runs of the cell ≤ `1e-3`. Threshold is **hard-coded** (spec-fixed) to prevent accidental loosening. |
| `test_gflops_meets_threshold`         | **Median** GFLOPS across runs of the cell ≥ `thresholds.gflops.<precision>`. Median is robust to single cold-cache / thermal outliers. |

---

## Troubleshooting

### `rocm-smi: command not found`
The droplet didn't boot with the GPU image. Re-provision from the AMD
Developer Cloud dashboard choosing the **ROCm Software** image tier.

### `rocblas-bench: No such file or directory`
The pre-installed image omitted rocBLAS dev tooling. Install it:
```bash
apt-get update -y && apt-get install -y rocblas-dev
```
Or build from source: `export ROCM_VERSION=7.2.2 && install/build_rocblas.sh`.

### `rocblas-bench` exits non-zero with `unknown option '--transposeA'`
You're on a very old rocBLAS (< 5.0) where the flags were `--transA` /
`--transB`. Edit `bench/sweep.sh` accordingly or upgrade ROCm. (Not
expected on AMD Developer Cloud with ROCm 7.2.)

### `error_norm is NULL for <cell>` in the correctness test
`rocblas-bench` was invoked without `-v 1`. The sweep script enables
verification by default — if you see this, confirm `bench/sweep.sh`
wasn't modified locally, or that you're not pointing at logs from a
different benchmarking run.

### Perf test failing slightly below threshold
Check `results/rocm-smi_post_*.txt` for thermal throttling (sustained
`Temp` > 85°C or `SCLK` below nominal). Also verify no other process is
on the GPU: `rocm-smi --showpids`. Then rerun with more cold iterations
and more repeats:
```bash
REPEATS=5 COLD_ITERATIONS=5 bench/sweep.sh
python -m parse.parser --log-dir logs --output results/results.csv
python -m parse.db     --csv results/results.csv --db results/results.db
pytest -v tests/
```

### `tag 'rocm-X.Y.Z' not found` from `build_rocblas.sh`
`ROCM_VERSION` doesn't match a rocBLAS upstream tag. List available tags:
```bash
git ls-remote --tags https://github.com/ROCm/rocBLAS.git | grep rocm- | tail -20
```

### Tests complain the DB is missing
The DB must be populated before pytest runs. Confirm in order:
```bash
ls -la logs/bench_*.log | head -5                              # sweep produced logs
ls -la results/results.csv                                     # parser produced CSV
sqlite3 results/results.db "SELECT COUNT(*) FROM gemm_runs;"   # DB has rows
```

### `pip install` fails with `externally-managed-environment`
Ubuntu 24.04's system Python is PEP 668-locked. Always use a venv (as in
the TL;DR) rather than `pip install --break-system-packages`.

### Session drops during long sweeps
Run the sweep inside `tmux` or `screen` so an SSH disconnect doesn't
kill the job:
```bash
apt-get install -y tmux
tmux new -s bench
# inside tmux: run the sweep normally
# detach with Ctrl-B then D; reattach later with: tmux attach -t bench
```
