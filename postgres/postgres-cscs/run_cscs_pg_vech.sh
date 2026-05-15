#!/bin/bash
#SBATCH --job-name=pgvec_full
#SBATCH --account=sm94
#SBATCH --environment=pgvector
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH --output=slurm-run_cscs_pg_vech-%j.out
#SBATCH --error=slurm-run_cscs_pg_vech-%j.err
#SBATCH --signal=TERM@120

# NOTE:
# - check db logs: $ tail -f $SCRATCH/pg_data/logfile

# Full pgvector benchmark on CSCS daint GH200.
# Runs HNSW32 + IVF1024 + ENN (no index) in sequence inside the pgvector enroot
# container, properly building/dropping indexes between phases.
#
# Prereqs (one-time, see NEW_CONTAINER_TESTING.md phases 1-3):
#   1. pgvector image built and imported as $SCRATCH/ce-images/pgvector.sqsh
#   2. ~/.edf/pgvector.toml points at it
#   3. $SCRATCH/datasets/ has the parquets and tpch CSVs
#   4. $SCRATCH/pg_data/ initialized + loaded via setup_pgvector_cscs.sh
#
# Usage:
#   sbatch run_cscs_pg_vech.sh [SF] [INDEXES]
#
# Examples:
#   sbatch run_cscs_pg_vech.sh                  # SF1, all 3 phases (hnsw, ivf, enn)
#   sbatch run_cscs_pg_vech.sh 1 hnsw,ivf       # SF1, only hnsw + ivf
#   sbatch run_cscs_pg_vech.sh 1 hnsw           # SF1, only hnsw
#   sbatch run_cscs_pg_vech.sh 1 hnsw,ivf,ivf4096,enn  # opt-in: add IVF4096; off by default
#
# --- cscs-gh200 (Alps / Daint) NUMA pinning ---
# We run on cores 0-71 (NUMA node 0). Both postgres backends and the python
# client inherit the SLURM cpuset, so the cache flush hits the same L3.
# See run_cscs.sh in Maximus for the same pattern.

# exec > >(tee -a /capstor/scratch/cscs/vmageira/slurm-logs/pgvec-boot-$SLURM_JOB_ID.log) 2>&1
# echo "BOOT $(date) host=$(hostname) pwd=$(pwd) user=$(id)"
# echo "SCRATCH=${SCRATCH:-UNSET}"
# echo "SLURM_JOB_ID=${SLURM_JOB_ID:-UNSET}"
# echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-UNSET}"
# ls -la /usr/local/bin/entrypoint.sh || echo "entrypoint missing — container not active"


set -euo pipefail
# set -x

# --- Args ---
SF="${1:-1}"
# Phase order is arbitrary — each phase unconditionally lands the DB in the
# state it needs before running, so any starting state is fine:
#   - hnsw → `build-indexes --index hnsw --clean` (clean drops any existing
#            HNSW/IVF index on reviews/images first, then builds HNSW).
#   - ivf  → `build-indexes --index ivfflat --clean` (same, but builds IVFFlat).
#   - enn  → `build-indexes --drop_vec_indexes` (drops HNSW/IVFFlat, builds
#            nothing; btree primary keys / non-vector indexes are untouched).
INDEXES="${2:-hnsw,ivf,enn}"

SYSTEM="cscs-gh200"
DB_NAME="${DB_NAME:-vech}"
DATASET="${DATASET:-vech-industrial_and_scientific-sf_1}"

# --- Dataset dir on $SCRATCH (where we rsynced everything) ---
# Query parquet is passed into vech_benchmark.py via --reviews_queries_file /
# --images_queries_file so it doesn't fall back to the hardcoded
DATASETS="$SCRATCH/datasets/$DATASET"
REVIEWS_QUERIES_FILE="${REVIEWS_QUERIES_FILE:-$DATASETS/reviews_queries.parquet}"
IMAGES_QUERIES_FILE="${IMAGES_QUERIES_FILE:-$DATASETS/images_queries.parquet}"

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
BENCH_SCRIPT="${SCRIPT_DIR}/../postgres-scripts/vech_benchmark.py"
PG_TUNING_CONF_PATH="${SCRIPT_DIR}/../postgres-scripts/pgtuning/cscs-gh200.conf"

# --- Postgres / container env ---
export PGDATA="$SCRATCH/pg_data"
export PG_TUNING_CONF="$PG_TUNING_CONF_PATH"
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=1234
export POSTGRES_DB="$DB_NAME"

# --- Auto-increment run ID (race-safe via SLURM_JOB_ID) ---
DATE_TAG=$(LC_TIME=C date +%d%b%y)
RUN_DIR_BASE="$SCRATCH/results/pgvector/runs/${DATE_TAG}-${SYSTEM}"
mkdir -p "$RUN_DIR_BASE"
MAX=$(ls -d "$RUN_DIR_BASE"/run* 2>/dev/null | sed 's|.*/run\([0-9]*\).*|\1|' | sort -n | tail -1 || true)
RUN_ID=$(( ${MAX:-0} + 1 ))
RESULTS_BASE="$RUN_DIR_BASE/run${RUN_ID}_job${SLURM_JOB_ID:-local}"
mkdir -p "$RESULTS_BASE"

ORCH_LOG="$RESULTS_BASE/orchestrator.log"
ulimit -c 0

echo "==========================================" | tee -a "$ORCH_LOG"
echo "FULL PGVECTOR RUN ON CSCS GH200: $(date)" | tee -a "$ORCH_LOG"
echo "  SYSTEM:    $SYSTEM" | tee -a "$ORCH_LOG"
echo "  SF:        $SF" | tee -a "$ORCH_LOG"
echo "  INDEXES:   $INDEXES" | tee -a "$ORCH_LOG"
echo "  DB:        $DB_NAME" | tee -a "$ORCH_LOG"
echo "  PGDATA:    $PGDATA" | tee -a "$ORCH_LOG"
echo "  RESULTS:   $RESULTS_BASE" | tee -a "$ORCH_LOG"
echo "  JOB_ID:    ${SLURM_JOB_ID:-local}" | tee -a "$ORCH_LOG"
echo "==========================================" | tee -a "$ORCH_LOG"

# Verify pg_data exists (must have been initialized by setup_pgvector_cscs.sh)
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "ERROR: $PGDATA not initialized." | tee -a "$ORCH_LOG"
    echo "       Run 'sbatch setup_pgvector_cscs.sh $SF' first to load data." | tee -a "$ORCH_LOG"
    exit 1
fi

# --- Run all phases inside one entrypoint invocation ---
# entrypoint.sh starts postgres (instant — pg_data already populated), then
# execs the orchestrator command. Postgres stops cleanly when the script exits
# (entrypoint trap).
#
# `taskset -c 0-71` is redundant under SLURM (the cpuset already constrains us)
# but explicit is harmless and matches Maximus run_cscs.sh.

# run from image :
# taskset -c 0-71 /usr/local/bin/entrypoint.sh bash -c "
# run from repo :
taskset -c 0-71 "${SCRIPT_DIR}/../postgres-scripts/entrypoint.sh" bash -c "
set -euo pipefail

SF='$SF'
INDEXES='$INDEXES'
DB_NAME='$DB_NAME'
RESULTS_BASE='$RESULTS_BASE'
ORCH_LOG='$ORCH_LOG'
BENCH_SCRIPT='$BENCH_SCRIPT'
REVIEWS_QUERIES_FILE='$REVIEWS_QUERIES_FILE'
IMAGES_QUERIES_FILE='$IMAGES_QUERIES_FILE'

# q11_end is by far the longest (~12 min at SF1) — keep it last.
QUERIES=(q2_start q10_mid q13_mid q15_end q16_start q18_mid q19_start q11_end)

log() { echo \"\$(date +%FT%T) \$*\" | tee -a \"\$ORCH_LOG\"; }
fmt_dur() { local s=\$1; printf '%dh%02dm%02ds' \$((s/3600)) \$(((s%3600)/60)) \$((s%60)); }

OVERALL_START=\$SECONDS

run_phase() {
    local phase=\"\$1\"        # 'enn' | 'hnsw' | 'ivf'
    local label=\"\$2\"        # 'enn' | 'HNSW32' | 'IVF1024'
    local phase_start=\$SECONDS

    log ''
    log '=========================================='
    log \"PHASE: \$phase  (--index_label \$label)\"
    log '=========================================='

    case \"\$phase\" in
        enn)
            log \"[\$phase] ENN baseline — dropping any existing HNSW/IVFFlat index on reviews/images (primary keys and other btree indexes are left alone)\"
            python3 \"\$BENCH_SCRIPT\" build-indexes --sf \"\$SF\" --drop_vec_indexes \\
                --db_name \"\$DB_NAME\" 2>&1 | tee -a \"\$ORCH_LOG\"
            ;;
        hnsw)
            log \"[\$phase] Building HNSW32 (clean: drops any existing vector indexes first)\"
            python3 \"\$BENCH_SCRIPT\" build-indexes --sf \"\$SF\" --index hnsw --clean \\
                --db_name \"\$DB_NAME\" 2>&1 | tee -a \"\$ORCH_LOG\"
            ;;
        ivf)
            log \"[\$phase] Building IVF1024 (clean: drops any existing vector indexes first)\"
            python3 \"\$BENCH_SCRIPT\" build-indexes --sf \"\$SF\" --index ivfflat --clean \\
                --db_name \"\$DB_NAME\" 2>&1 | tee -a \"\$ORCH_LOG\"
            ;;
        ivf4096)
            log \"[\$phase] Building IVF4096 (nlist=4096, clean: drops any existing vector indexes first)\"
            python3 \"\$BENCH_SCRIPT\" build-indexes --sf \"\$SF\" --index ivfflat --clean \\
                --n_lists 4096 --db_name \"\$DB_NAME\" 2>&1 | tee -a \"\$ORCH_LOG\"
            ;;
        *)
            log \"ERROR: unknown phase '\$phase'\"
            exit 1
            ;;
    esac

    log \"[\$phase] Verifying indexes\"
    python3 \"\$BENCH_SCRIPT\" check-indexes --db_name \"\$DB_NAME\" 2>&1 | tee -a \"\$ORCH_LOG\"

    log \"[\$phase] Running benchmark (queries: \${QUERIES[*]})\"
    # pipefail off for the benchmark only: if one phase's run crashes
    # (e.g. a stray unraisable at shutdown, a single bad query), we still
    # want the remaining phases to execute. Build/check failures above
    # still halt — a broken index state would produce garbage results.
    set +o pipefail
    python3 \"\$BENCH_SCRIPT\" run --sf \"\$SF\" \\
        --index_label \"\$label\" \\
        --query \"\${QUERIES[@]}\" \\
        --timeout_min 20 \\
        --save_csv --save_plans \\
        --db_name \"\$DB_NAME\" \\
        --reviews_queries_file \"\$REVIEWS_QUERIES_FILE\" \\
        --images_queries_file \"\$IMAGES_QUERIES_FILE\" \\
        --output_dir \"\$RESULTS_BASE\" \\
        2>&1 | tee -a \"\$ORCH_LOG\"
    set -o pipefail

    log \"[\$phase] DONE in \$(fmt_dur \$((SECONDS - phase_start))) (build + bench)\"
}

IFS=',' read -ra PHASES <<< \"\$INDEXES\"
for phase in \"\${PHASES[@]}\"; do
    case \"\$phase\" in
        enn)  run_phase enn  enn ;;
        hnsw) run_phase hnsw HNSW32 ;;
        ivf)  run_phase ivf  IVF1024 ;;
        ivf4096) run_phase ivf4096 IVF4096 ;;
        *)    log \"ERROR: unknown phase '\$phase' (valid: enn, hnsw, ivf, ivf4096)\" && exit 1 ;;
    esac
done

log ''
log '=========================================='
log 'ALL PHASES COMPLETE'
log \"  Total:   \$(fmt_dur \$((SECONDS - OVERALL_START)))\"
log \"  Results: \$RESULTS_BASE\"
log \"  Log:     \$ORCH_LOG\"
log '=========================================='

if [ -d \"\$RESULTS_BASE/csv/enn\" ]; then
    log 'Computing intra-pgvector recall (ENN vs ANN)...'
    python3 \"${SCRIPT_DIR}/../postgres-scripts/recall_pg_intra.py\" \"\$RESULTS_BASE\" --sf \"\$SF\" \\
        2>&1 | tee -a \"\$ORCH_LOG\" || log '[WARN] recall_pg_intra failed'
fi
"
EXIT_CODE=$?

echo "" | tee -a "$ORCH_LOG"
echo "==========================================" | tee -a "$ORCH_LOG"
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "RUN COMPLETE: $(date)" | tee -a "$ORCH_LOG"
else
    echo "RUN FAILED (exit $EXIT_CODE): $(date)" | tee -a "$ORCH_LOG"
fi
echo "  Results: $RESULTS_BASE" | tee -a "$ORCH_LOG"
echo "==========================================" | tee -a "$ORCH_LOG"

exit $EXIT_CODE
