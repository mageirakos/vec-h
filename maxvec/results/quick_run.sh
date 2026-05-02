#!/bin/bash
# quick_run.sh — Single-command run + parse + plot for quick experiments.
#
# Mode A (run + parse + plot):
#   ./quick_run.sh --label test1 --queries q10_mid --index "GPU,Cagra,64,32,NN_DESCENT" \
#     --device gpu --storage_device cpu --index_storage_device cpu --sf 1 --n_reps 10 \
#     --params "k=100,query_count=1,..."
#
# Mode B (parse + plot existing log):
#   ./quick_run.sh --label test1 --log /path/to/run.log
#
# Output goes to results/0_quick_run/<label>/ with key plots symlinked at top level.

##### EXAMPLE USAGE #####
# CUSTOM MODE (direct maxbench invocation):
# $  ./quick_run.sh -l test_cagra_default --queries q10_mid --index 'GPU,Cagra,64,32,NN_DESCENT' --device gpu --storage_device cpu --index_storage_device cpu --sf 1 --n_reps 10 --params 'k=100,query_count=1,query_start=0,postfilter_ksearch=100,cagra_itopksize=128,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=1,metric=IP,index_data_on_gpu=1,cagra_cache_graph=0'
#
# PRESET MODE (delegates to run_vsds.sh — per-query params / case mapping baked in there):
# Recognized --index_filter tokens: cagra, cagrach, ivf1024, ivfh1024, ivf4096, ivfh4096, flat.
#
# Plain variants:
# $  ./quick_run.sh -l q10_cagra_preset --filter=q10_c1 --index_filter=cagra --system=sgs-gpu05 --sf 1 --n_reps 20
# $  ./quick_run.sh -l q15_ivf1024 --filter=q15_c0 --index_filter=ivf1024 --scenario=ann --sf 1 --n_reps 20
# $  ./quick_run.sh -l allcase1_flat --filter=c1 --index_filter=flat --scenario=enn --system=sgs-gpu05 --sf 0.01 --n_reps 4
#
# ATS / host-view variants (cached graph / host-resident inverted lists):
# $  ./quick_run.sh -l q10_cagrach --filter=q10_c1 --index_filter=cagrach --system=cscs-gh200 --sf 1 --n_reps 20
# $  ./quick_run.sh -l q15_ivfh1024 --filter=q15_c1 --index_filter=ivfh1024 --system=cscs-gh200 --sf 1 --n_reps 20
# $  ./quick_run.sh -l q10_ivfh4096 --filter=q10_c3 --index_filter=ivfh4096 --system=cscs-gh200 --sf 1 --n_reps 20
#
# "Custom" way to do the same variants (when you need to override run_vsds.sh's baked params).
# Use the index aliases handled by _vsds_transforms.sh (CagraCH / CagraC / IVFH). --index_filter not used here.
# $  ./quick_run.sh -l q10_cagrach_custom --queries q10_mid --index 'GPU,CagraCH,64,32,NN_DESCENT' --device gpu --storage_device cpu --index_storage_device cpu --sf 1 --n_reps 10 --params 'k=100,query_count=1,query_start=0,metric=IP,cagra_itopksize=128'
# $  ./quick_run.sh -l q10_cagrac_custom  --queries q10_mid --index 'GPU,CagraC,64,32,NN_DESCENT'  --device gpu --storage_device cpu --index_storage_device cpu --sf 1 --n_reps 10 --params 'k=100,query_count=1,query_start=0,metric=IP,cagra_itopksize=128'
# $  ./quick_run.sh -l q10_ivfh1024_custom --queries q10_mid --index 'GPU,IVFH1024,Flat'             --device gpu --storage_device cpu --index_storage_device cpu --sf 1 --n_reps 10 --params 'k=100,query_count=1,query_start=0,metric=IP,ivf_nprobe=30,postfilter_ksearch=100'
#
# OPTIONS:
# --use_index_cache_dir /tmp/maximus_faiss_cache : for loading from cache
# --nsys : for profiling output

###### MINIMAL QUICK RUN -- WITHOUT THIS SCRIPT 
# Quick nsys profile via run_vsds.sh "as-is" (EG. on sgs-gpu05)

# # CagraCH on q10_c1 (GPU-only alias; not valid for case 0)
# export NSYS_OUT_DIR=/tmp/nsys_q10_c1_cagrach REPS=4 RUN_INDEX_FILTER=cagrach
# taskset -c 0-3,16-19 ./results/run_vsds.sh q10_c1 1 ann sgs-gpu05 100 ./results/quick_nsys true
# unset NSYS_OUT_DIR REPS RUN_INDEX_FILTER

# # IVF1024 on q11_c0 (pure CPU, q11 special params inherited from run_vsds.sh)
# export NSYS_OUT_DIR=/tmp/nsys_q11_c0_ivf1024 REPS=4 RUN_INDEX_FILTER=ivf1024
# taskset -c 0-3,16-19 ./results/run_vsds.sh q11_c0 1 ann sgs-gpu05 100 ./results/quick_nsys true
# unset NSYS_OUT_DIR REPS RUN_INDEX_FILTER

set -euo pipefail

# ---- Defaults ----
LABEL=""
LOG_FILE=""
QUERIES="" ; INDEX="" ; DEVICE="gpu" ; STORAGE="cpu" ; INDEX_STORAGE="cpu"
VS_DEVICE=""  # empty = inherit from DEVICE; set to "cpu" for Case 4 (hybrid GPU relational + CPU VS)
SF="1" ; N_REPS="4" ; PARAMS="" ; K="100" ; EXTRA_TAG=""
BUILD_DIR="" ; NSYS="" ; INDEX_CACHE_DIR="" ; DO_BUILD="1"
DATASET="industrial_and_scientific"
# Preset-mode defaults (empty FILTER → custom mode). SYSTEM="quick" keeps custom-mode paths byte-identical.
FILTER="" ; INDEX_FILTER="" ; SCENARIO="ann" ; SYSTEM="quick"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label|-l)         LABEL="$2"; shift 2;;
        --log)              LOG_FILE="$2"; shift 2;;
        --queries)          QUERIES="$2"; shift 2;;
        --index)            INDEX="$2"; shift 2;;
        --device)           DEVICE="$2"; shift 2;;
        --storage_device)   STORAGE="$2"; shift 2;;
        --index_storage_device) INDEX_STORAGE="$2"; shift 2;;
        --vs_device)        VS_DEVICE="$2"; shift 2;;
        --sf)               SF="$2"; shift 2;;
        --n_reps)           N_REPS="$2"; shift 2;;
        --params)           PARAMS="$2"; shift 2;;
        --k)                K="$2"; shift 2;;
        --extra_tag)        EXTRA_TAG="$2"; shift 2;;
        --build_dir)        BUILD_DIR="$2"; shift 2;;
        --nsys)             NSYS="1"; shift 1;;
        --use_index_cache_dir) INDEX_CACHE_DIR="$2"; shift 2;;
        --build)            DO_BUILD="1"; shift 1;;
        --no-build)         DO_BUILD=""; shift 1;;
        --dataset)          DATASET="$2"; shift 2;;
        --filter=*)         FILTER="${1#--filter=}"; shift 1;;
        --filter)           FILTER="$2"; shift 2;;
        --index_filter=*)   INDEX_FILTER="${1#--index_filter=}"; shift 1;;
        --index_filter)     INDEX_FILTER="$2"; shift 2;;
        --scenario=*)       SCENARIO="${1#--scenario=}"; shift 1;;
        --scenario)         SCENARIO="$2"; shift 2;;
        --system=*)         SYSTEM="${1#--system=}"; shift 1;;
        --system)           SYSTEM="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# ---- Mode B: auto-detect from log header ----
if [ -n "$LOG_FILE" ]; then
    [ -f "$LOG_FILE" ] || { echo "Error: $LOG_FILE not found"; exit 1; }
    LOG_FILE="$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
    [ -z "$QUERIES" ]       && QUERIES=$(grep "^---> queries:" "$LOG_FILE" | head -1 | awk -F: '{print $NF}' | xargs)
    [ -z "$INDEX" ]         && INDEX=$(grep "^---> VSDS Index:" "$LOG_FILE" | head -1 | awk -F: '{print $NF}' | xargs)
    [ -z "$DEVICE" ]        && DEVICE=$(grep "^---> Device:" "$LOG_FILE" | head -1 | awk -F: '{print $NF}' | xargs | tr '[:upper:]' '[:lower:]')
    STORAGE=$(grep "^---> Storage Device:" "$LOG_FILE" | head -1 | awk -F: '{print $NF}' | xargs | tr '[:upper:]' '[:lower:]')
    INDEX_STORAGE=$(grep "^---> Index Storage Device:" "$LOG_FILE" | head -1 | awk -F: '{print $NF}' | xargs | tr '[:upper:]' '[:lower:]')
    [ "$SF" = "1" ] && SF=$(grep -oP "sf_[\d.]+" "$LOG_FILE" | head -1 | sed 's/sf_//' || echo "$SF")
fi

# ---- Label (default: timestamp) ----
[ -z "$LABEL" ] && LABEL="$(date +%Y-%m-%d_%H-%M-%S)_${FILTER:-$QUERIES}"

# ---- Paths (all absolute, safe for srun) ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
[ -z "$BUILD_DIR" ] && BUILD_DIR="${PROJECT_ROOT}/build/Release"
BASE="${SCRIPT_DIR}/0_quick_run/${LABEL}"
DATA_DIR="${BASE}/_data"

# ---- Custom-mode-only: validate, apply transforms, build filename, cleanup ----
if [ -z "$FILTER" ]; then
    [ -z "$QUERIES" ] && { echo "Error: --queries required (or use --log)"; exit 1; }
    [ -z "$INDEX" ]   && { echo "Error: --index required (or use --log)"; exit 1; }

    # Apply CagraCH/IVFH/use_cuvs alias transforms (shared with run_vsds.sh).
    # Placed after Mode B so auto-detected INDEX is also transformed (no-op if INDEX is post-transform).
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/_vsds_transforms.sh"
    apply_index_transforms "$INDEX" "$PARAMS" "$EXTRA_TAG"
    INDEX_FOR_MAXBENCH="$ACTUAL_INDEX"
    PARAMS="$TRANSFORMED_PARAMS"
    EXTRA_TAG="$(join_extra_tag "$EXTRA_TAG" "$TRANSFORM_TAG")"

    SYS_DIR="${DATA_DIR}/vsds/${DEVICE}-${SYSTEM}/sf_${SF}"

    TAG_PART="k_${K}"
    [ -n "$EXTRA_TAG" ] && TAG_PART="${TAG_PART}-${EXTRA_TAG}"
    # Case 4 marker: when --vs_device differs from --device, annotate the filename
    # so parse_caliper identifies the case (via the `vsd` extra-tag key).
    if [ -n "$VS_DEVICE" ] && [ "$VS_DEVICE" != "$DEVICE" ]; then
        TAG_PART="${TAG_PART}-vsd_${VS_DEVICE}"
    fi
    FILENAME="q_${QUERIES}-i_${INDEX_FOR_MAXBENCH}-d_${DEVICE}-s_${STORAGE}-is_${INDEX_STORAGE}-sf_${SF}-${TAG_PART}.log"

    if [ -d "${DATA_DIR}" ]; then
        echo "Cleaning up previous logs and parsed CSVs for label '${LABEL}'..."
        rm -rf "${DATA_DIR}/vsds"
        rm -rf "${DATA_DIR}/parse_caliper"
        # leaving ${DATA_DIR}/plots intact

        # clean up old symlinks in the base dir to avoid broken links
        find "${BASE}" -maxdepth 1 -type l -name "*.log" -delete
    fi
fi

mkdir -p "$BASE" "$DATA_DIR"
[ -z "$FILTER" ] && mkdir -p "$SYS_DIR"
LOG_PATH="${SYS_DIR:-}/${FILENAME:-}"

echo "============================================"
echo "  Quick Run: ${LABEL}"
echo "============================================"
if [ -n "$FILTER" ]; then
    echo "Mode:    PRESET (delegates to run_vsds.sh)"
    echo "Filter:  ${FILTER}   Index filter: ${INDEX_FILTER:-<all>}   Scenario: ${SCENARIO}"
    echo "System:  ${SYSTEM}   SF: ${SF}   K: ${K}   Reps: ${N_REPS}"
else
    echo "Query:   ${QUERIES}"
    echo "Index:   ${INDEX}"
    echo "Device:  ${DEVICE} / Storage: ${STORAGE} / IndexStorage: ${INDEX_STORAGE}"
    echo "SF:      ${SF}"
    echo "Log:     ${LOG_PATH}"
fi
echo "PID:     $$ | Cores: $(taskset -cp $$ 2>/dev/null || echo 'N/A') | GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo ""

# ---- Step 1: Run or copy log ----
if [ -n "$FILTER" ]; then
    # Preset mode: run_vsds.sh runs ninja at its top; no explicit build here.
    echo "[1/3] Preset mode: delegating to run_vsds.sh (filter=${FILTER}, index_filter=${INDEX_FILTER:-<all>})..."
    export RUN_INDEX_FILTER="$INDEX_FILTER"
    export REPS="$N_REPS"
    if [ -n "$NSYS" ]; then
        export NSYS_OUT_DIR="${DATA_DIR}/nsys"
        mkdir -p "$NSYS_OUT_DIR"
    else
        unset NSYS_OUT_DIR
    fi
    (
        cd "$PROJECT_ROOT"
        ./results/run_vsds.sh "$FILTER" "$SF" "$SCENARIO" "$SYSTEM" "$K" "$DATA_DIR" true
    )
elif [ -n "$LOG_FILE" ]; then
    echo "[1/3] Copying log to standard path..."
    cp "$LOG_FILE" "$LOG_PATH"
else
    echo "[1/3] Running maxbench (${N_REPS} reps)..."
    if [ -n "$DO_BUILD" ]; then
        echo "Building first (--build)..."
        ninja -C "${BUILD_DIR}"
    fi
    DATA_PATH="${PROJECT_ROOT}/tests/vsds/data-${DATASET}-sf_${SF}/"
    MAXBENCH_CMD="${BUILD_DIR}/benchmarks/maxbench \
        --benchmark vsds --n_reps ${N_REPS} \
        --storage_device ${STORAGE} --index_storage_device ${INDEX_STORAGE} --device ${DEVICE} \
        --queries ${QUERIES} --index '${INDEX_FOR_MAXBENCH}' \
        --params '${PARAMS}' \
        --profile 'runtime-report(calc.inclusive=true,output=stdout)' \
        --path ${DATA_PATH} \
        --using_large_list --flush_cache"
    [ -n "$VS_DEVICE" ] && MAXBENCH_CMD="${MAXBENCH_CMD} --vs_device ${VS_DEVICE}"
    [ -n "$INDEX_CACHE_DIR" ] && MAXBENCH_CMD="${MAXBENCH_CMD} --use_index_cache_dir ${INDEX_CACHE_DIR}"
    # Case 4 only (vs_device=cpu AND device=gpu): keep glibc heap pages resident
    # across reps so cuDF's pageable D2H destination doesn't pay first-touch
    # page faults every rep. See run_vsds.sh for the full rationale.
    MALLOC_PREFIX=""
    if [ "$VS_DEVICE" = "cpu" ] && [ "$DEVICE" = "gpu" ]; then
        MALLOC_PREFIX="MALLOC_ARENA_MAX=1 MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=-1 "
    fi
    if [ -n "$NSYS" ]; then
        NSYS_OUTPUT="${BASE}/${LABEL}.nsys-rep"
        CMD="${MALLOC_PREFIX}CALI_SERVICES_ENABLE=nvtx nsys profile --trace=cuda,nvtx --force-overwrite=true -o ${NSYS_OUTPUT} ${MAXBENCH_CMD}"
        echo "CMD (nsys): ${CMD}"
    else
        CMD="${MALLOC_PREFIX}${MAXBENCH_CMD}"
        echo "CMD: ${CMD}"
    fi
    echo ""
    eval "$CMD" 2>&1 | tee "$LOG_PATH" || true
fi

# ---- Step 2: Parse ----
echo ""
echo "[2/3] Parsing caliper output..."
cd "$SCRIPT_DIR"
python parse_caliper.py --base_dir "0_quick_run/${LABEL}/_data" --sf "$SF" --system "$SYSTEM" --benchmark vsds

# ---- Step 3: Plot ----
echo ""
echo "[3/3] Plotting..."
python plot_caliper.py \
    --in_dir "0_quick_run/${LABEL}/_data/parse_caliper" \
    --out_dir "0_quick_run/${LABEL}/_data/plots" \
    --sf "$SF" --system "$SYSTEM" --benchmark vsds --k "$K" \
    --metric median min \
    --quick --title --skip-cases ""

# ---- Step 4: Symlink key outputs to label root ----
echo ""
if [ -n "$FILTER" ]; then
    # Preset-mode: symlink all caliper logs, nsys traces, and per-query plots produced by run_vsds.sh.
    # Mirrors run_quick_cscs_serial.sh:250–265.
    for f in "${DATA_DIR}"/vsds/*/sf_*/*.log; do
        [ -f "$f" ] || continue
        ln -sf "$(realpath --relative-to="$BASE" "$f")" "${BASE}/$(basename "$f")" 2>/dev/null || true
    done
    for f in "${DATA_DIR}"/nsys/*.nsys-rep; do
        [ -f "$f" ] || continue
        ln -sf "$(realpath --relative-to="$BASE" "$f")" "${BASE}/$(basename "$f")" 2>/dev/null || true
    done
    for img in "${DATA_DIR}"/plots/vsds/sf_${SF}/${SYSTEM}/*/plot_1/median/per_query/*.jpeg \
               "${DATA_DIR}"/plots/vsds/sf_${SF}/${SYSTEM}/*/plot_2/median/per_query/*.jpeg \
               "${DATA_DIR}"/plots/vsds/sf_${SF}/${SYSTEM}/*/plot_1/min/per_query/*.jpeg \
               "${DATA_DIR}"/plots/vsds/sf_${SF}/${SYSTEM}/*/plot_2/min/per_query/*.jpeg; do
        [ -f "$img" ] || continue
        [[ "$img" == *"coarse"* ]] && continue
        ln -sf "$(realpath --relative-to="$BASE" "$img")" "${BASE}/$(basename "$img")" 2>/dev/null || true
    done
else
    ln -sf "_data/vsds/${DEVICE}-${SYSTEM}/sf_${SF}/${FILENAME}" "${BASE}/run.log" 2>/dev/null || true

    # Symlink per-query plots to label root for quick access (median + min)
    for metric in median min; do
        suffix=""
        [ "$metric" = "min" ] && suffix="_min"
        for img in "${DATA_DIR}"/plots/vsds/sf_${SF}/${SYSTEM}/*/plot_1/${metric}/per_query/*"${QUERIES}"*.jpeg; do
            [ -f "$img" ] || continue
            rel=$(python3 -c "import os; print(os.path.relpath('$img', '$BASE'))")
            ln -sf "$rel" "${BASE}/${QUERIES}_breakdown${suffix}.jpeg" 2>/dev/null || true
            break
        done
        for img in "${DATA_DIR}"/plots/vsds/sf_${SF}/${SYSTEM}/*/plot_2/${metric}/per_query/*"${QUERIES}"*.jpeg; do
            [ -f "$img" ] || continue
            [[ "$img" == *"coarse"* ]] && continue
            rel=$(python3 -c "import os; print(os.path.relpath('$img', '$BASE'))")
            ln -sf "$rel" "${BASE}/${QUERIES}_operators${suffix}.jpeg" 2>/dev/null || true
            break
        done
    done
fi

echo "============================================"
echo "  Done: ${LABEL}"
echo "============================================"
echo "Log:        ${BASE}/run.log"
for f in "${BASE}"/*.jpeg; do
    [ -f "$f" ] && echo "Plot:       $f"
done
echo "All plots:  ${DATA_DIR}/plots/"
echo "CSV:        $(ls "${DATA_DIR}"/parse_caliper/vsds/*.csv 2>/dev/null | head -1)"
echo "============================================"
