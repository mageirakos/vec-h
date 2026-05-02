#!/bin/bash
#SBATCH --job-name=pgvec_smoke
#SBATCH --account=sm94
#SBATCH --environment=pgvector
#SBATCH --container-writable
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=slurm-pgsmoke-%j.out
#SBATCH --error=slurm-pgsmoke-%j.err

# Quick smoke test of the pgvector setup on CSCS daint.
# Verifies, in order:
#   1. Postgres can start against $SCRATCH/pg_data and tables are non-empty
#   2. vech_benchmark.py imports cleanly and can connect (list + check-indexes)
#   3. A single-query, single-rep ENN run completes end-to-end and writes output
#
# Use this BEFORE the full sbatch run_cscs_pg_vech.sh sweep — it catches missing
# python deps, broken db credentials, missing tables, or EXPLAIN-parsing
# regressions in seconds instead of failing 30 minutes into a real run.
#
# Prereqs (same as run_cscs_pg_vech.sh):
#   1. $SCRATCH/ce-images/pgvector.sqsh exists (built via podman + enroot import)
#   2. ~/.edf/pgvector.toml points at it
#   3. $SCRATCH/pg_data/ already loaded via `sbatch setup_pgvector_cscs.sh 1`
#
# Usage:
#   sbatch smoke_test_cscs.sh [SF]
#
# Examples:
#   sbatch smoke_test_cscs.sh           # SF1
#   sbatch smoke_test_cscs.sh 1         # same

set -euo pipefail

SF="${1:-1}"
DB_NAME="vech"
SYSTEM="cscs-gh200"
DATASETS="$SCRATCH/datasets/sf${SF}"
REVIEWS_QUERIES_FILE="$DATASETS/industrial_and_scientific_sf${SF}_reviews_queries.parquet"
IMAGES_QUERIES_FILE="$DATASETS/industrial_and_scientific_sf${SF}_images_queries.parquet"

# --- Repo-relative paths (same SLURM_SUBMIT_DIR pattern as the other scripts) ---
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
BENCH_SCRIPT="${SCRIPT_DIR}/../postgres-scripts/vech_benchmark.py"
PG_TUNING_CONF_PATH="${SCRIPT_DIR}/../postgres-scripts/pgtuning/cscs-gh200.conf"

# --- Postgres / container env ---
export PGDATA="$SCRATCH/pg_data"
export PG_TUNING_CONF="$PG_TUNING_CONF_PATH"
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=1234
export POSTGRES_DB="$DB_NAME"

# --- Output dir for the smoke test results (separate from real runs) ---
SMOKE_DIR="$SCRATCH/results/pgvector/smoke/job${SLURM_JOB_ID:-local}"
mkdir -p "$SMOKE_DIR"

echo "=========================================="
echo "PGVECTOR SMOKE TEST: $(date)"
echo "  SYSTEM:    $SYSTEM"
echo "  SF:        $SF"
echo "  DB:        $DB_NAME"
echo "  PGDATA:    $PGDATA"
echo "  BENCH:     $BENCH_SCRIPT"
echo "  OUTPUT:    $SMOKE_DIR"
echo "=========================================="

# Verify pg_data exists — if not, the user hasn't run setup_pgvector_cscs.sh yet.
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "ERROR: $PGDATA not initialized."
    echo "       Run 'sbatch setup_pgvector_cscs.sh $SF' first to load data."
    exit 1
fi

# --- Run all checks inside one entrypoint invocation ---
/usr/local/bin/entrypoint.sh bash -c "
set -euo pipefail

DB_NAME='$DB_NAME'
SF='$SF'
BENCH_SCRIPT='$BENCH_SCRIPT'
SMOKE_DIR='$SMOKE_DIR'
REVIEWS_QUERIES_FILE='$REVIEWS_QUERIES_FILE'
IMAGES_QUERIES_FILE='$IMAGES_QUERIES_FILE'

step() { echo ''; echo '----------------------------------------'; echo \"\$*\"; echo '----------------------------------------'; }

step '[1/3] Table sanity check (psql counts)'
psql -h localhost -U postgres -d \"\$DB_NAME\" <<'SQL'
SELECT 'reviews'         AS t, count(*) FROM reviews
UNION ALL SELECT 'images',          count(*) FROM images
UNION ALL SELECT 'reviews_queries', count(*) FROM reviews_queries
UNION ALL SELECT 'images_queries',  count(*) FROM images_queries
UNION ALL SELECT 'lineitem',        count(*) FROM lineitem;
SQL

step '[2/3] vech_benchmark.py imports + DB connection'
python3 \"\$BENCH_SCRIPT\" list
echo ''
python3 \"\$BENCH_SCRIPT\" check-indexes --db_name \"\$DB_NAME\"

step '[3/3] Single-query end-to-end smoke (q10_mid, 1 rep, ENN, no warmup/flush/prewarm)'
python3 \"\$BENCH_SCRIPT\" run \\
    --sf \"\$SF\" \\
    --query q10_mid \\
    --nreps 1 --no_warmup --no_flush --no_prewarm \\
    --index_label enn \\
    --db_name \"\$DB_NAME\" \\
    --reviews_queries_file \"\$REVIEWS_QUERIES_FILE\" \\
    --images_queries_file \"\$IMAGES_QUERIES_FILE\" \\
    --output_dir \"\$SMOKE_DIR\"

echo ''
echo '----------------------------------------'
echo 'Smoke test outputs:'
ls -la \"\$SMOKE_DIR\" 2>/dev/null || echo '(no files written)'
find \"\$SMOKE_DIR\" -maxdepth 3 -type f 2>/dev/null | head -20

echo ''
echo '=========================================='
echo 'SMOKE TEST PASSED'
echo '=========================================='
"
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "SMOKE TEST COMPLETE: $(date)"
    echo "  Results: $SMOKE_DIR"
    echo ""
    echo "Next: sbatch run_cscs_pg_vech.sh $SF"
else
    echo "SMOKE TEST FAILED (exit $EXIT_CODE): $(date)"
    echo "  Inspect: slurm-pgsmoke-${SLURM_JOB_ID:-local}.{out,err}"
fi
echo "=========================================="

exit $EXIT_CODE
