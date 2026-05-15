#!/bin/bash
# Full pgvector benchmark on default.
# Runs HNSW32 + IVF1024 + ENN (no index) in sequence, properly building/dropping
# indexes between each phase.
#
# Usage:
#   ./run_pg_vech.sh [SF] [INDEXES]
#
# Examples:
#   ./run_pg_vech.sh 1                    # SF1, all 3 phases (hnsw, ivf, enn)
#   ./run_pg_vech.sh 1 hnsw,ivf           # SF1, only hnsw + ivf
#   ./run_pg_vech.sh 1 hnsw               # SF1, only hnsw
#   ./run_pg_vech.sh 1 hnsw,ivf,ivf4096,enn   # opt-in: add IVF4096 (nlist=4096); off by default
#
# Prereq:
#   - postgres-default container running (`make setup` in postgres-default/)
#   - data loaded (`make load-data SF=1`)

set -euo pipefail

SF="${1:-1}"
# Phase order is arbitrary — each phase unconditionally lands the DB in the
# state it needs before running, so any starting state is fine:
#   - hnsw → `build-indexes --index hnsw --clean` (clean drops any existing
#            HNSW/IVF index on reviews/images first, then builds HNSW).
#   - ivf  → `build-indexes --index ivfflat --clean` (same, but builds IVFFlat).
#   - enn  → `build-indexes --drop_vec_indexes` (drops HNSW/IVFFlat, builds
#            nothing; btree primary keys / non-vector indexes are untouched).
INDEXES="${2:-hnsw,ivf,enn}"

# DB_NAME / DATASET are env-overridable and default to the Makefile defaults,
# so `make load-data` then `./run_pg_vech.sh` Just Works:
#   DB_NAME=vech_sf1 DATASET=vech-sf1 ./run_pg_vech.sh 1
DB_NAME="${DB_NAME:-vech}"
DATASET="${DATASET:-vech-industrial_and_scientific-sf_1}"

# q11_end is by far the longest (~12 min at SF1) — keep it last.
QUERIES=(q2_start q10_mid q13_mid q15_end q16_start q18_mid q19_start q11_end)

# Where vech_benchmark.py lives (host-side, not in container — we run client from host)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Query vectors are read from parquet on the HOST (benchmark runs client-side).
# (SCRIPT_DIR = postgres-default/ → ../../data = repo-root/data)
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/../../data/$DATASET}"
REVIEWS_QUERIES_FILE="${REVIEWS_QUERIES_FILE:-$DATA_DIR/reviews_queries.parquet}"
IMAGES_QUERIES_FILE="${IMAGES_QUERIES_FILE:-$DATA_DIR/images_queries.parquet}"
BENCH_SCRIPT="${SCRIPT_DIR}/../postgres-scripts/vech_benchmark.py"

# Output root: under postgres-default/results/<DATE>-default/
DATE_TAG=$(LC_TIME=C date +%d%b%y)
RUN_DIR_BASE="${SCRIPT_DIR}/results/runs/${DATE_TAG}-default"
mkdir -p "$RUN_DIR_BASE"
MAX=$(ls -d "$RUN_DIR_BASE"/run* 2>/dev/null | sed 's|.*/run\([0-9]*\)|\1|' | sort -n | tail -1 || echo 0)
RUN_ID=$(( ${MAX:-0} + 1 ))
RESULTS_BASE="$RUN_DIR_BASE/run${RUN_ID}"
mkdir -p "$RESULTS_BASE"

ORCH_LOG="${RESULTS_BASE}/orchestrator.log"

log() {
    echo "$(date +%FT%T) $*" | tee -a "$ORCH_LOG"
}

OVERALL_START=$SECONDS

# hh:mm:ss for a duration in seconds
fmt_dur() { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

log "=========================================="
log "FULL PGVECTOR RUN ON DEFAULT"
log "  SF:        $SF"
log "  INDEXES:   $INDEXES"
log "  DB:        $DB_NAME"
log "  DATASET:   $DATASET"
log "  Queries:   $REVIEWS_QUERIES_FILE"
log "             $IMAGES_QUERIES_FILE"
log "  Output:    $RESULTS_BASE"
log "=========================================="

run_phase() {
    local phase="$1"        # "enn" | "hnsw" | "ivf"
    local label="$2"        # "none" | "HNSW32" | "IVF1024"
    local phase_start=$SECONDS

    log ""
    log "=========================================="
    log "PHASE: $phase  (--index_label $label)"
    log "=========================================="

    case "$phase" in
        enn)
            log "[$phase] ENN baseline — dropping any existing HNSW/IVFFlat index on reviews/images (primary keys and other btree indexes are left alone)"
            python3 "$BENCH_SCRIPT" build-indexes --sf "$SF" --drop_vec_indexes \
                --db_name "$DB_NAME" 2>&1 | tee -a "$ORCH_LOG"
            ;;
        hnsw)
            log "[$phase] Building HNSW32 (clean: drops any existing vector indexes first)"
            python3 "$BENCH_SCRIPT" build-indexes --sf "$SF" --index hnsw --clean \
                --db_name "$DB_NAME" 2>&1 | tee -a "$ORCH_LOG"
            ;;
        ivf)
            log "[$phase] Building IVF1024 (clean: drops any existing vector indexes first)"
            python3 "$BENCH_SCRIPT" build-indexes --sf "$SF" --index ivfflat --clean \
                --db_name "$DB_NAME" 2>&1 | tee -a "$ORCH_LOG"
            ;;
        ivf4096)
            log "[$phase] Building IVF4096 (nlist=4096, clean: drops any existing vector indexes first)"
            python3 "$BENCH_SCRIPT" build-indexes --sf "$SF" --index ivfflat --clean \
                --n_lists 4096 --db_name "$DB_NAME" 2>&1 | tee -a "$ORCH_LOG"
            ;;
        *)
            log "ERROR: unknown phase '$phase'"
            exit 1
            ;;
    esac

    log "[$phase] Verifying indexes"
    python3 "$BENCH_SCRIPT" check-indexes --db_name "$DB_NAME" 2>&1 | tee -a "$ORCH_LOG"

    log "[$phase] Running benchmark (queries: ${QUERIES[*]})"
    # pipefail off for the benchmark only: if one phase's run crashes
    # (e.g. a stray unraisable at shutdown, a single bad query), we still
    # want the remaining phases to execute. Build/check failures above
    # still halt — a broken index state would produce garbage results.
    set +o pipefail
    python3 "$BENCH_SCRIPT" run --sf "$SF" \
        --index_label "$label" \
        --query "${QUERIES[@]}" \
        --timeout_min 20 \
        --save_csv --save_plans \
        --db_name "$DB_NAME" \
        --reviews_queries_file "$REVIEWS_QUERIES_FILE" \
        --images_queries_file "$IMAGES_QUERIES_FILE" \
        --output_dir "$RESULTS_BASE" \
        2>&1 | tee -a "$ORCH_LOG"
    set -o pipefail

    log "[$phase] DONE in $(fmt_dur $((SECONDS - phase_start))) (build + bench)"
}

# Parse and run requested phases
IFS=',' read -ra PHASES <<< "$INDEXES"
for phase in "${PHASES[@]}"; do
    case "$phase" in
        enn)  run_phase enn  enn ;;
        hnsw) run_phase hnsw HNSW32 ;;
        ivf)  run_phase ivf  IVF1024 ;;
        ivf4096) run_phase ivf4096 IVF4096 ;;
        *)    log "ERROR: unknown phase '$phase' (valid: enn, hnsw, ivf, ivf4096)" && exit 1 ;;
    esac
done

log ""
log "=========================================="
log "ALL PHASES COMPLETE"
log "  Total:   $(fmt_dur $((SECONDS - OVERALL_START)))"
log "  Results: $RESULTS_BASE"
log "  Log:     $ORCH_LOG"
log "=========================================="

# Intra-pgvector recall (ENN vs ANN). No-op if enn phase was skipped.
if [ -d "$RESULTS_BASE/csv/enn" ]; then
    log "Computing intra-pgvector recall (ENN vs ANN)..."
    python3 "${SCRIPT_DIR}/../postgres-scripts/recall_pg_intra.py" "$RESULTS_BASE" --sf "$SF" \
        2>&1 | tee -a "$ORCH_LOG" || log "[WARN] recall_pg_intra failed"
fi
