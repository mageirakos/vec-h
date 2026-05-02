#!/bin/bash
#
# get_index_movement_table.sh
#
# Collects data for the index-movement / H2D-transfer table (Flat, IVF-1K,
# IVF-4K, IVF-4K^H, CAGRA, CAGRA^C, CAGRA^C+H). Each row runs twice: once
# under nsys (CUPTI hardware timestamps for H2D) and once clean (true Total).
#
# Alias note: CagraC, CagraCH, IVFH* are aliases resolved downstream by
# results/_vsds_transforms.sh (sourced by both quick_run.sh and run_vsds.sh).
# It rewrites the index name and force-sets cagra_cache_graph /
# index_data_on_gpu per variant. This script just passes strings through.
#
# Usage:
#   ./get_index_movement_table.sh --system=sgs
#   ./get_index_movement_table.sh --system=cscs [--partition=normal|debug]

set -o pipefail

SYSTEM=""
QUERIES="ann_reviews"
SF="1"
N_REPS="4"
N_RUNS="3"
CACHE_ROOT="/tmp/maximus_faiss_cache_idxmove"
# CSCS default: normal partition. run_quick_cscs_serial.sh hardcodes
# --partition=debug which only allows 1 concurrent job; we override so all
# 16 sbatch jobs can run in parallel. sbatch CLI --partition wins over #SBATCH.
PARTITION="normal"
SBATCH_TIME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --system=*)     SYSTEM="${1#--system=}" ;;
        --queries=*)    QUERIES="${1#--queries=}" ;;
        --sf=*)         SF="${1#--sf=}" ;;
        --n_reps=*)     N_REPS="${1#--n_reps=}" ;;
        --n_runs=*)     N_RUNS="${1#--n_runs=}" ;;
        --cache_root=*) CACHE_ROOT="${1#--cache_root=}" ;;
        --partition=*)  PARTITION="${1#--partition=}" ;;
        --time=*)       SBATCH_TIME="${1#--time=}" ;;
        -h|--help)      sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ -z "$SYSTEM" ]]; then
    echo "Error: --system=sgs|cscs is required" >&2
    exit 2
fi

# CagraC / CagraCH / IVFH* are aliases: _vsds_transforms.sh (sourced by quick_run.sh
# and run_vsds.sh) rewrites the index name and force-sets cagra_cache_graph /
# index_data_on_gpu per variant. H variants skipped on sgs (no unified memory).
ROWS=(
    "flat|GPU,Flat|-"
    "ivf1024|GPU,IVF1024,Flat|-"
    "ivf4096|GPU,IVF4096,Flat|-"
    "ivfh1024|GPU,IVFH1024,Flat|h"
    "ivfh4096|GPU,IVFH4096,Flat|h"
    "cagra|GPU,Cagra,64,32,NN_DESCENT|-"
    "cagra_c|GPU,CagraC,64,32,NN_DESCENT|-"
    "cagra_ch|GPU,CagraCH,64,32,NN_DESCENT|h"
)

PARAMS='k=100,query_count=1,query_start=0,ivf_nprobe=30,cagra_itopksize=128,cagra_searchwidth=1,hnsw_efsearch=128,use_cuvs=1,metric=IP,index_data_on_gpu=1'

cache_dir_for() { echo "${CACHE_ROOT}/$1/$2"; }

case "$SYSTEM" in
    sgs)
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        MAXIMUS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
        if [[ ! -x "${MAXIMUS_DIR}/results/quick_run.sh" ]]; then
            echo "Error: ${MAXIMUS_DIR}/results/quick_run.sh not found or not executable" >&2
            exit 1
        fi
        cd "$MAXIMUS_DIR" || exit 1

        FILTERED=()
        for ROW in "${ROWS[@]}"; do
            IFS='|' read -r LABEL IDX HTAG <<< "$ROW"
            if [[ "$HTAG" == "h" ]]; then
                echo "[sgs] skipping (H) variant: $LABEL ($IDX)"
                continue
            fi
            FILTERED+=("$ROW")
        done

        echo "=== sgs local loop: $(date) ==="
        echo "Maximus dir: $(pwd)"
        echo "Rows: ${#FILTERED[@]}, modes: nsys+clean, pin_modes: unpinned+pinned, runs: $N_RUNS"
        echo "Total invocations: $((${#FILTERED[@]} * 2 * 2 * N_RUNS))"
        echo ""

        # RUN is outermost so each iter produces a complete table (both
        # pin_modes × both nsys/clean modes × all rows) before the next run
        # starts. N_RUNS=1 yields exactly one full table.
        for RUN in $(seq 1 "$N_RUNS"); do
            echo "======== RUN $RUN / $N_RUNS ========"
            for PIN_MODE in unpinned pinned; do
                if [[ "$PIN_MODE" == "pinned" ]]; then
                    STORAGE="cpu-pinned"
                    PIN_TAG="_pin"
                    PIN_SEGMENT="pinned/"
                else
                    STORAGE="cpu"
                    PIN_TAG=""
                    PIN_SEGMENT=""
                fi
                echo "==== pin_mode=$PIN_MODE (storage=$STORAGE) ===="
                for MODE in nsys clean; do
                    echo "### mode=$MODE"
                    for ROW in "${FILTERED[@]}"; do
                        IFS='|' read -r LABEL IDX _ <<< "$ROW"
                        CDIR="${CACHE_ROOT}/${PIN_SEGMENT}${MODE}/${LABEL}"
                        mkdir -p "$CDIR"
                        RUN_LABEL="im_table_${LABEL}_${MODE}${PIN_TAG}_iter${RUN}"
                        NSYS_FLAG=()
                        [[ "$MODE" == "nsys" ]] && NSYS_FLAG=(--nsys)

                        echo "--- $RUN_LABEL  (cache=$CDIR) ---"
                        ./results/quick_run.sh \
                            -l "$RUN_LABEL" \
                            --queries "$QUERIES" --index "$IDX" \
                            --device gpu --storage_device "$STORAGE" --index_storage_device "$STORAGE" \
                            --sf "$SF" --n_reps "$N_REPS" "${NSYS_FLAG[@]}" \
                            --params "$PARAMS" \
                            --use_index_cache_dir "$CDIR"
                        echo ""
                    done
                done
            done
        done

        echo "=== sgs DONE: $(date) ==="
        echo "Outputs: results/0_quick_run/im_table_*_{nsys,clean}{,_pin}_iter{1..$N_RUNS}/"
        ;;

    cscs)
        if [[ -z "${SCRATCH:-}" ]]; then
            echo "Error: \$SCRATCH is not set (expected on cscs-gh200 login node)" >&2
            exit 1
        fi
        MAXIMUS_DIR="${SCRATCH}/maxvec/Maximus"
        if [[ ! -f "${MAXIMUS_DIR}/results/run_quick_cscs_serial.sh" ]]; then
            echo "Error: ${MAXIMUS_DIR}/results/run_quick_cscs_serial.sh does not exist" >&2
            exit 1
        fi
        cd "$MAXIMUS_DIR" || exit 1

        SBATCH_OVERRIDES=(--partition="$PARTITION")
        [[ -n "$SBATCH_TIME" ]] && SBATCH_OVERRIDES+=(--time="$SBATCH_TIME")
        if [[ "$PARTITION" == "debug" ]]; then
            echo "WARNING: partition=debug only allows 1 concurrent job; 16 jobs will run serially." >&2
            echo "         pass --partition=normal to parallelize." >&2
        fi

        echo "=== cscs sbatch submission: $(date) ==="
        echo "Maximus dir: $(pwd)"
        echo "Partition:   $PARTITION"
        echo "Rows: ${#ROWS[@]}, modes: nsys+clean, total sbatch jobs: $((${#ROWS[@]} * 2))"
        echo ""

        for MODE in nsys clean; do
            echo "### mode=$MODE"
            for ROW in "${ROWS[@]}"; do
                IFS='|' read -r LABEL IDX _ <<< "$ROW"
                CDIR=$(cache_dir_for "$MODE" "$LABEL")
                RUN_LABEL="im_table_${LABEL}_${MODE}"
                NSYS_FLAG=()
                [[ "$MODE" == "nsys" ]] && NSYS_FLAG=(--nsys)

                echo "--- sbatch $RUN_LABEL  (cache=$CDIR) ---"
                sbatch "${SBATCH_OVERRIDES[@]}" ./results/run_quick_cscs_serial.sh \
                    --label="$RUN_LABEL" --n_runs="$N_RUNS" "${NSYS_FLAG[@]}" \
                    --queries "$QUERIES" --index "$IDX" \
                    --device gpu --storage_device cpu --index_storage_device cpu \
                    --sf "$SF" --n_reps "$N_REPS" \
                    --params "$PARAMS" \
                    --use_index_cache_dir "$CDIR"
            done
        done

        echo ""
        echo "=== cscs sbatch queued: $(date) ==="
        echo "Check with: squeue -u \$USER"
        echo "Outputs: results/0_quick_run/im_table_*_{nsys,clean}_iter{1..$N_RUNS}/"
        ;;

    *)
        echo "Error: unknown system '$SYSTEM' (expected sgs|cscs)" >&2
        exit 2
        ;;
esac
