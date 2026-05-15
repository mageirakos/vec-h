#!/bin/bash
#SBATCH --job-name=pgvec_setup
#SBATCH --account=sm94
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=slurm-pgsetup-%j.out
#SBATCH --error=slurm-pgsetup-%j.err
#SBATCH --partition=debug

# --- Container ---
#SBATCH --environment=pgvector
#SBATCH --container-writable

# One-time job: initialize postgres, load TPC-H + VECH data, run ANALYZE.
# After this, pg_data on $SCRATCH persists across jobs.

set -euo pipefail

# --- Configuration ---
SF="${1:-1}"
# DATASET = directory name under $SCRATCH/datasets/
# $SCRATCH stays the base on daint). Override via env:
#   DATASET=vech-sf1 DB_NAME=vech_sf1 sbatch setup_pgvector_cscs.sh
DATASET="${DATASET:-vech-industrial_and_scientific-sf_1}"
PGDATA="$SCRATCH/pg_data"
DATASETS="$SCRATCH/datasets/$DATASET"

# --- Repo-relative paths (work inside the container because /capstor is mounted) ---
# Under sbatch, $0 points at the spooled copy under /var/spool/slurmd/, so
# dirname $0 is useless for finding sibling files in the repo. SLURM_SUBMIT_DIR
# is the directory the user ran sbatch from — when invoked from postgres-cscs/
# it equals what dirname $0 would have been outside SLURM. Fall back to $0 for
# direct (non-sbatch) execution.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
LOAD_SCRIPT="${SCRIPT_DIR}/../load_to_postgres_database.py"
PG_TUNING_CONF="${SCRIPT_DIR}/../postgres-scripts/pgtuning/cscs-gh200.conf"

DB_NAME="${DB_NAME:-vech}"
export PGDATA PG_TUNING_CONF
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=1234
export POSTGRES_DB="$DB_NAME"

echo "=========================================="
echo "PGVECTOR SETUP: $(date)"
echo "  SF:       $SF"
echo "  DATASET:  $DATASET"
echo "  DB_NAME:  $DB_NAME"
echo "  PGDATA:   $PGDATA"
echo "  DATASETS: $DATASETS"
echo "=========================================="

# --- Start postgres via entrypoint ---
# entrypoint.sh will init PGDATA if empty, or start if existing.
# We run it with "bash" as the command so it starts postgres and gives us a shell.
/usr/local/bin/entrypoint.sh bash -c "
    echo '--- Loading data ---'
    python3 $LOAD_SCRIPT \
        --sf $SF \
        --db_name $POSTGRES_DB \
        --db_host localhost \
        --db_user $POSTGRES_USER \
        --db_password $POSTGRES_PASSWORD \
        --tpch_dir $DATASETS \
        --reviews_file $DATASETS/reviews.parquet \
        --images_file $DATASETS/images.parquet \
        --reviews_queries_file $DATASETS/reviews_queries.parquet \
        --images_queries_file $DATASETS/images_queries.parquet

    echo '--- Running ANALYZE ---'
    psql -U $POSTGRES_USER -d $POSTGRES_DB -c 'ANALYZE;'

    echo '--- Table sizes ---'
    psql -U $POSTGRES_USER -d $POSTGRES_DB -c \"
        SELECT tablename, pg_size_pretty(pg_total_relation_size(quote_ident(tablename)))
        FROM pg_tables WHERE schemaname = 'public' ORDER BY pg_total_relation_size(quote_ident(tablename)) DESC;
    \"

    echo '=== Setup complete: $(date) ==='
"
