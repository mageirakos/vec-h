#!/bin/bash
set -euo pipefail

# Orchestrator script for full experiment runs.
# Called by `make full-run`, `run_cscs.sh`, and `run_cscs_parallel.sh`.
#
# Usage:
#   ./results/full_run.sh <RESULTS_BASE> <SYSTEM> [options]
#
# Options:
#   --sf=<N>              Scale factor (default: 1)
#   --stages=<LIST>       Comma-separated subset of: vsds,varbatch,fullsweep
#                         Default: vsds,varbatch,fullsweep (i.e. all)
#   --analyze-only        Skip build + benchmarks; parse+plot+recall only
#   --no-csv              Don't persist CSV result files (skips recall step)
#
# Examples:
#   ./results/full_run.sh results/runs/18Apr26-dgx/run1 dgx-spark-02
#   ./results/full_run.sh results/runs/18Apr26-dgx/run1 dgx-spark-02 --sf=0.01 --stages=vsds
#   ./results/full_run.sh results/runs/18Apr26-dgx/run1 dgx-spark-02 --analyze-only --stages=vsds,varbatch

USAGE="Usage: full_run.sh <RESULTS_BASE> <SYSTEM> [--sf=<N>] [--stages=vsds,varbatch,fullsweep] [--analyze-only] [--no-csv]"

if [ $# -lt 2 ]; then
    echo "$USAGE" >&2
    exit 2
fi

RESULTS_BASE="$1"
SYSTEM="$2"
shift 2

RUN_SF="1"
STAGES="vsds,varbatch,fullsweep"
ANALYZE_ONLY="false"
PERSIST_CSV="true"

for arg in "$@"; do
    case "$arg" in
        --sf=*)          RUN_SF="${arg#--sf=}" ;;
        --stages=*)      STAGES="${arg#--stages=}" ;;
        --analyze-only)  ANALYZE_ONLY="true" ;;
        --no-csv)        PERSIST_CSV="false" ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "$USAGE" >&2
            exit 2
            ;;
    esac
done

# Parse stages into associative flags
RUN_VSDS="false"
RUN_VARBATCH="false"
RUN_FULLSWEEP="false"
IFS=',' read -ra STAGE_ARR <<< "$STAGES"
for s in "${STAGE_ARR[@]}"; do
    case "$s" in
        vsds)      RUN_VSDS="true" ;;
        varbatch)  RUN_VARBATCH="true" ;;
        fullsweep) RUN_FULLSWEEP="true" ;;
        "")        ;;
        *)
            echo "Unknown stage: $s (valid: vsds,varbatch,fullsweep)" >&2
            exit 2
            ;;
    esac
done

mkdir -p "$RESULTS_BASE"
LOG_FILE="${RESULTS_BASE}/full_run.log"
exec >> "$LOG_FILE" 2>&1

trap 'echo ""; echo "=========================================="; echo "KILLED at $(date)"; echo "==========================================" ' TERM INT

FULL_RUN_START_EPOCH=$(date +%s)
step_start() { STEP_START_EPOCH=$(date +%s); }
step_done() {
    local now elapsed_s elapsed_m total_s total_m
    now=$(date +%s)
    elapsed_s=$((now - STEP_START_EPOCH))
    elapsed_m=$(awk "BEGIN{printf \"%.1f\", ${elapsed_s}/60}")
    total_s=$((now - FULL_RUN_START_EPOCH))
    total_m=$(awk "BEGIN{printf \"%.1f\", ${total_s}/60}")
    echo "  => Done in ${elapsed_s}s (${elapsed_m}m) | Total elapsed: ${total_m}m"
}

echo "=========================================="
if [ "$ANALYZE_ONLY" = "true" ]; then
    echo "ANALYZE-ONLY START: $(date)"
else
    echo "FULL RUN START: $(date)"
fi
echo "  SYSTEM:        $SYSTEM"
echo "  SF:            $RUN_SF"
echo "  RESULTS_BASE:  $RESULTS_BASE"
echo "  STAGES:        $STAGES (vsds=$RUN_VSDS varbatch=$RUN_VARBATCH fullsweep=$RUN_FULLSWEEP)"
echo "  ANALYZE_ONLY:  $ANALYZE_ONLY"
echo "  PERSIST_CSV:   $PERSIST_CSV"
echo "=========================================="

# ---- Step 1: Build (skipped with --analyze-only) ----
if [ "$ANALYZE_ONLY" != "true" ]; then
    echo "[1/5] Building..."
    step_start
    ninja -C ./build/Release
    step_done
else
    echo "[1/5] Skipping build (--analyze-only)"
fi

# ---- Step 2: Run benchmarks (skipped with --analyze-only) ----
if [ "$ANALYZE_ONLY" != "true" ]; then
    if [ "$RUN_VSDS" = "true" ]; then
        echo "[2a/5] Running VSDS benchmarks..."
        step_start
        ./results/run_vsds.sh all "$RUN_SF" all "$SYSTEM" 100 "$RESULTS_BASE" "$PERSIST_CSV"
        step_done
    else
        echo "[2a/5] Skipping VSDS (not in --stages)"
    fi

    if [ "$RUN_VARBATCH" = "true" ]; then
        echo "[2b/5] Running varbatch (vary-batch-in-maxbench)..."
        step_start
        ./results/run_other_experiments.sh "$RUN_SF" vary-batch-in-maxbench "$SYSTEM" all "$RESULTS_BASE" "$PERSIST_CSV"
        step_done
    else
        echo "[2b/5] Skipping varbatch (not in --stages)"
    fi

    if [ "$RUN_FULLSWEEP" = "true" ]; then
        echo "[2c/5] Running fullsweep (bs1_fullsweep / CASE 6)..."
        step_start
        ./results/run_other_experiments.sh "$RUN_SF" bs1_fullsweep "$SYSTEM" all "$RESULTS_BASE" "$PERSIST_CSV"
        step_done
    else
        echo "[2c/5] Skipping fullsweep (not in --stages)"
    fi
else
    echo "[2/5] Skipping benchmarks (--analyze-only)"
fi

# ---- Step 3: Parse results ----
echo "[3/5] Parsing results..."
step_start
PARSE_BASE="../${RESULTS_BASE}"
cd results

if [ "$RUN_VSDS" = "true" ]; then
    echo "  parse_caliper (vsds)..."
    python parse_caliper.py --sf "$RUN_SF" --benchmark vsds --system "$SYSTEM" \
        --base_dir "$PARSE_BASE" || echo "  [WARN] parse_caliper vsds failed"
fi

if [ "$RUN_VARBATCH" = "true" ]; then
    echo "  parse_varbatch (vary-batch)..."
    python parse_varbatch.py --sf "$RUN_SF" --system "$SYSTEM" \
        --result_dir other-vary-batch-in-maxbench \
        --base_dir "$PARSE_BASE" --parse_caliper || echo "  [WARN] parse_varbatch vary-batch failed"
fi

if [ "$RUN_FULLSWEEP" = "true" ]; then
    echo "  parse_varbatch (bs1_fullsweep)..."
    python parse_varbatch.py --sf "$RUN_SF" --system "$SYSTEM" \
        --result_dir other-bs1_fullsweep \
        --base_dir "$PARSE_BASE" || echo "  [WARN] parse_varbatch bs1_fullsweep failed"
fi
cd ..
step_done

# ---- Step 4: Plot results ----
echo "[4/5] Plotting results..."
step_start
cd results

if [ "$RUN_VSDS" = "true" ]; then
    echo "  plot_caliper (vsds)..."
    python plot_caliper.py --sf "$RUN_SF" --benchmark vsds --system "$SYSTEM" \
        --in_dir "$PARSE_BASE/parse_caliper" \
        --out_dir "$PARSE_BASE/plots" \
        --metric median mean \
        || echo "  [WARN] plot_caliper failed"

    echo "  plot_paper (vsds)..."
    python plot_paper.py --sf "$RUN_SF" --benchmark vsds --system "$SYSTEM" \
        --in_dir "$PARSE_BASE/parse_caliper" \
        --out_dir "$PARSE_BASE/plots_paper" \
        --format pdf \
        || echo "  [WARN] plot_paper failed"
fi

if [ "$RUN_VARBATCH" = "true" ]; then
    echo "  plot_varbatch (operator breakdown)..."
    python plot_varbatch.py --sf "$RUN_SF" --system "$SYSTEM" \
        --in_dir "$PARSE_BASE/parse_caliper" \
        --out_dir "$PARSE_BASE/plots" \
        --result_dir other-vary-batch-in-maxbench \
        --plot_operator_breakdown \
        || echo "  [WARN] plot_varbatch vary-batch failed"
fi

if [ "$RUN_FULLSWEEP" = "true" ]; then
    echo "  plot_varbatch (bs1_fullsweep cumulative)..."
    python plot_varbatch.py --sf "$RUN_SF" --system "$SYSTEM" \
        --in_dir "$PARSE_BASE/parse_caliper" \
        --out_dir "$PARSE_BASE/plots" \
        --result_dir other-bs1_fullsweep \
        --plot_bs1_fullsweep \
        || echo "  [WARN] plot_varbatch bs1_fullsweep failed"
fi
cd ..
step_done

# ---- Step 5: Recall (only if CSV persisted) ----
if [ "$PERSIST_CSV" = "true" ]; then
    echo "[5/5] Calculating recall..."
    step_start
    RECALL_BASE="../${RESULTS_BASE}/recall"
    cd results

    if [ "$RUN_VSDS" = "true" ]; then
        echo "  recall (vsds)..."
        python recall_vsds.py --results_dir "../${RESULTS_BASE}/vsds" \
            --sf "$RUN_SF" --system "$SYSTEM" \
            --recall_dir "${RECALL_BASE}/vsds" \
            --out "${SYSTEM}_vsds_recall_sf_${RUN_SF}.csv" \
            || echo "  [WARN] recall vsds failed"
    fi

    if [ "$RUN_VARBATCH" = "true" ]; then
        echo "  recall (vary-batch)..."
        python recall_vsds.py --results_dir "../${RESULTS_BASE}/other-vary-batch-in-maxbench" \
            --sf "$RUN_SF" --system "$SYSTEM" \
            --recall_dir "${RECALL_BASE}/vary-batch" \
            --out "${SYSTEM}_vary-batch_recall_sf_${RUN_SF}.csv" \
            --batch_filter 10000 \
            || echo "  [WARN] recall vary-batch failed"
    fi
    cd ..
    step_done
else
    echo "[5/5] Skipping recall (--no-csv)"
fi

echo "=========================================="
echo "FULL RUN COMPLETE: $(date)"
echo "Results in: $RESULTS_BASE"
echo "  Parsed CSVs:  $RESULTS_BASE/parse_caliper/"
echo "  Plots:        $RESULTS_BASE/plots/"
echo "  Paper plots:  $RESULTS_BASE/plots_paper/"
echo "  Recall:       $RESULTS_BASE/recall/"
echo "Re-analyze:"
echo "  ./results/full_run.sh $RESULTS_BASE $SYSTEM --sf=$RUN_SF --analyze-only --stages=$STAGES"
echo "=========================================="
