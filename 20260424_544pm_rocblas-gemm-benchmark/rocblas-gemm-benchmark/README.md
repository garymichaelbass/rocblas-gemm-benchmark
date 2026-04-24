# rocblas-gemm-benchmark

Sweep `rocblas-bench` across **{f16, bf16, f32} × {4096, 8192, 16384}**
square GEMM on AMD Instinct MI300X (gfx942), capture GFLOPS / GB·s⁻¹ /
wall time / error norm, persist into SQLite, and assert correctness +
performance thresholds via pytest.

This README is written specifically for the **AMD Developer Cloud**
(DigitalOcean-backed GPU droplets) running **Ubuntu 22.04 LTS (jammy) or
24.04 LTS (noble)** with **ROCm 7.2.x pre-installed**.

---

## TL;DR — three commands on a fresh droplet

```bash
ssh root@<DROPLET_IP>
git clone https://github.com/<your-org>/rocblas-gemm-benchmark.git
cd rocblas-gemm-benchmark && bash run_all.sh
```

That's it. `run_all.sh` will:
1. `apt-get install` the system packages it needs (`python3-venv`, `python3-pip`, `sqlite3`, `git`)
2. Create a local `.venv/` and `pip install` pinned dependencies
3. **Auto-detect** the pre-installed `rocblas-bench` at `/opt/rocm/bin/rocblas-bench`
4. **Auto-detect** `ROCM_VERSION` from `/opt/rocm/.info/version`
5. Run the full pipeline: **sweep → parse → load → pytest**

Total wall time: ~3–6 minutes on a single MI300X droplet (mostly the f32 16384³ runs).

If all 27 assertions pass, the droplet's rocBLAS GEMM kernels meet the
configured MI300X thresholds. If any fail, pytest reports the exact cell
(e.g. `test_gflops_meets_threshold[f32-m16384]`).

> **Tip:** wrap the whole thing in `tmux` so an SSH disconnect doesn't
> kill the run — see [Troubleshooting](#troubleshooting).

---

## Prerequisites on the droplet

The AMD Developer Cloud's ROCm-pre-installed image ships with everything
needed. Verify before starting:

| Component             | Check command                                | Expected                                |
| --------------------- | -------------------------------------------- | --------------------------------------- |
| GPU visible           | `rocm-smi`                                   | MI300X device(s) listed                 |
| ROCm version          | `cat /opt/rocm/.info/version`                | `7.2.x-*`                               |
| rocblas-bench present | `ls /opt/rocm/bin/rocblas-bench`             | path exists and is executable           |
| Ubuntu release        | `lsb_release -ds`                            | `Ubuntu 22.04 LTS` or `Ubuntu 24.04 LTS`|

If `rocblas-bench` is missing (rare on the pre-installed image), install it:
```bash
apt-get install -y rocblas-dev    # provides /opt/rocm/bin/rocblas-bench
```

`run_all.sh` will fall back to building from source if no pre-installed
binary is found and `ROCM_VERSION` is set.

---

## How `run_all.sh` decides what to do

The script chooses behavior based on environment variables. Defaults are
correct for a fresh AMD Developer Cloud droplet — you only need to set
these to override.

| Variable          | Default behavior                                                       |
| ----------------- | ---------------------------------------------------------------------- |
| `ROCBLAS_BENCH`   | If set & executable, used directly. Otherwise auto-detect `/opt/rocm/bin/rocblas-bench`. |
| `ROCM_VERSION`    | Auto-detected from `/opt/rocm/.info/version`. Required only when building from source. |
| `FORCE_BUILD=1`   | Build `rocblas-bench` from source even if pre-installed. Requires `ROCM_VERSION`. |
| `SKIP_APT=1`      | Skip the `apt-get install` step (e.g. inside a container with packages already present). |
| `SKIP_VENV=1`     | Skip venv creation; use the active Python. Useful in CI with a managed environment. |
| Sweep tunables    | `REPEATS`, `ITERATIONS`, `COLD_ITERATIONS`, `LOG_DIR`, `RESULTS_DIR`, `TRANSPOSE_A`, `TRANSPOSE_B` pass through to `bench/sweep.sh`. |

### Examples

```bash
# Default — fresh AMD Developer Cloud droplet
bash run_all.sh

# Heavier run for tighter medians
REPEATS=5 ITERATIONS=20 bash run_all.sh

# Test against an unreleased rocBLAS commit (build from source)
FORCE_BUILD=1 ROCM_VERSION=7.2.2 bash run_all.sh

# Quick smoke test (one repeat per cell, ~1–2 min)
REPEATS=1 bash run_all.sh
```

---

## Stage-by-stage execution (manual / debugging)

If you want to run individual stages — e.g. to re-parse logs without
re-sweeping, or to inspect intermediate output — every stage is
independent and idempotent (except the DB loader, which appends).

```bash
# One-time setup
apt-get update -y && apt-get install -y python3-venv python3-pip sqlite3
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ROCBLAS_BENCH=/opt/rocm/bin/rocblas-bench

# Stage 2: sweep
bench/sweep.sh

# Stage 3: parse logs into CSV
python -m parse.parser --log-dir logs --output results/results.csv

# Stage 4: load CSV into SQLite (always appends)
python -m parse.db --csv results/results.csv --db results/results.db

# Stage 5: assertions
pytest -v tests/
```

Inspect the DB directly:

```bash
sqlite3 results/results.db <<'SQL'
.mode column
.headers on
SELECT precision, M,
       COUNT(*)             AS runs,
       ROUND(AVG(gflops),1) AS mean_gflops,
       ROUND(MIN(gflops),1) AS min_gflops,
       ROUND(MAX(gflops),1) AS max_gflops,
       MAX(error_norm)      AS worst_err
FROM gemm_runs
GROUP BY precision, M
ORDER BY precision, M;
SQL
```

Filter pytest to a subset when debugging:

```bash
pytest -v tests/ -k "f32"                                            # only f32 cells
pytest -v tests/test_gemm_perf.py                                    # only perf tests
pytest -v "tests/test_gemm_perf.py::test_gflops_meets_threshold[f16-m16384]"  # one cell
```

For a clean slate: `rm -rf logs/ results/ .venv/`.

---

## Building `rocblas-bench` from source (optional)

Only needed to test an unreleased rocBLAS commit. `--clients-only`
builds just the benchmark/test clients against the system-installed
rocBLAS runtime — it does not rebuild the library.

```bash
export ROCM_VERSION=7.2.2
install/build_rocblas.sh             # prints built binary path on stdout

# Or have run_all.sh do it:
FORCE_BUILD=1 ROCM_VERSION=7.2.2 bash run_all.sh
```

Idempotent — repeated calls with the same `ROCM_VERSION` skip
rebuilding. Force a rebuild with `rm local/.rocblas-*.stamp`.

| Variable                  | Default                | Purpose                                |
| ------------------------- | ---------------------- | -------------------------------------- |
| `ROCM_VERSION`            | *(required)*           | Maps to git tag `rocm-${ROCM_VERSION}`.|
| `ROCBLAS_INSTALL_PREFIX`  | `<repo>/local`         | Source and build artifacts location.   |
| `GPU_ARCH`                | `gfx942`               | MI300X. Don't change on MI300X.        |
| `JOBS`                    | `$(nproc)`             | Parallel build jobs.                   |

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
├── run_all.sh                        # one-shot driver (this is what you run)
└── .gitignore
```

`logs/`, `results/`, `local/`, and `.venv/` are created at runtime and gitignored.

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
> droplet; they're useful for regression detection and correctness
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
failure in one cell does not mask others — pytest reports the exact
failing cell.

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
apt-get install -y rocblas-dev
```
Or build from source: `FORCE_BUILD=1 ROCM_VERSION=7.2.2 bash run_all.sh`.

### `rocblas-bench` exits with `unknown option '--transposeA'`
You're on a very old rocBLAS (< 5.0) where flags were `--transA` /
`--transB`. Edit `bench/sweep.sh` accordingly or upgrade ROCm. Not
expected on AMD Developer Cloud with ROCm 7.2.

### `error_norm is NULL for <cell>` in the correctness test
`rocblas-bench` was invoked without `-v 1`. The sweep script enables
verification by default — if you see this, confirm `bench/sweep.sh`
wasn't modified locally, or that you're not pointing at logs from a
different benchmarking run.

### Perf test failing slightly below threshold
Check `results/rocm-smi_post_*.txt` for thermal throttling (sustained
`Temp` > 85°C or `SCLK` below nominal). Verify no other process is on
the GPU: `rocm-smi --showpids`. Then rerun with more cold iterations
and more repeats:
```bash
REPEATS=5 COLD_ITERATIONS=5 bash run_all.sh
```

### Tests complain the DB is missing
Confirm in order:
```bash
ls -la logs/bench_*.log | head -5                              # sweep produced logs
ls -la results/results.csv                                     # parser produced CSV
sqlite3 results/results.db "SELECT COUNT(*) FROM gemm_runs;"   # DB has rows
```

### `pip install` fails with `externally-managed-environment`
Ubuntu 24.04's system Python is PEP 668-locked. `run_all.sh` handles
this automatically by creating a venv. If you're running stages
manually, always activate `.venv` first.

### Session drops during long sweeps
Use `tmux` so an SSH disconnect doesn't kill the job:
```bash
apt-get install -y tmux
tmux new -s bench
# inside tmux:
bash run_all.sh
# detach: Ctrl-B then D
# reattach later: tmux attach -t bench
```

### `tag 'rocm-X.Y.Z' not found` from `build_rocblas.sh`
`ROCM_VERSION` doesn't match a rocBLAS upstream tag. List available tags:
```bash
git ls-remote --tags https://github.com/ROCm/rocBLAS.git | grep rocm- | tail -20
```
