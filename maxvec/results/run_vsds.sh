#!/bin/bash
###################################
# GLOBAL Configuration
###################################
# NOTE:
# cases: # 1: cpu store gpu exec, 2: index gpu, data cpu, gpu exec, 3: all gpu
RUN_Q="${1:-all}" # "<query>_c<caseid>"
RUN_SF="${2:-all}"
RUN_VS="${3:-all}" # enn/ann
SYSTEM="${4:-dgx-spark-02}"
K_ARG="${5:-100}"  # comma-separated: "1,10,100,1000,10000" or single "100"
RESULTS_BASE="${6:-./results}"  # output root directory (default: ./results)
PERSIST_CSV="${7:-true}"        # "true" to save CSV result files (needed for recall checks)
# Optional 8th positional arg OR RUN_INDEX_FILTER env var. Empty = no filter (default).
# Recognized tokens: cagra, cagrach, ivf1024, ivfh1024, ivf4096, ivfh4096, flat.
# Any other value falls back to case-insensitive substring match on the INDEX string.
RUN_INDEX_FILTER="${8:-${RUN_INDEX_FILTER:-}}"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_vsds_transforms.sh"

# Optional nsys wrap. Set to a directory and each maxbench call gets a
# matching <TAGGED_BASENAME>.nsys-rep alongside the caliper log.
NSYS_OUT_DIR="${NSYS_OUT_DIR:-}"
[ -n "$NSYS_OUT_DIR" ] && mkdir -p "$NSYS_OUT_DIR"

# Parse comma-separated K values into an array
IFS=',' read -ra K_VALUES <<< "$K_ARG"

echo "Running on system: ${SYSTEM} <------------"
echo "K values: ${K_VALUES[*]}"
if [ -n "$RUN_INDEX_FILTER" ]; then
    echo "Index filter: ${RUN_INDEX_FILTER}"
fi

REPS="${REPS:-20}"
BENCH_TIMEOUT=1200  # per-benchmark timeout in seconds (20 minutes) (keep in mind index build times..)
BUILD_DIR="./build/Release"
USE_CUVS=1
# SCALE_FACTOR=("0.01" "1") # `"all`" ( 0.01 just for quick testing... )
# SCALE_FACTOR=("0.01") # "small"
SCALE_FACTOR=("1")

# Which indexes to test
# ENN :
CPU_ENN_INDEX="Flat"
GPU_ENN_INDEX="GPU,Flat"
# [] which nprobe to use?  Run ann_reviews and ann_images.
# [] what recall do i get
# [] verify outputs?
# ANN :
CPU_ANN_INDEXES=("IVF1024,Flat" "IVF4096,Flat" "GPU,Cagra,64,32,NN_DESCENT")        # CPU ANN indexes. No CagraCH/IVFH: those optimizations only apply to GPU to_gpu() path. Parse copies CPU data from base variant.
GPU_ANN_INDEXES=("GPU,IVF1024,Flat" "GPU,IVF4096,Flat" "GPU,IVFH1024,Flat" "GPU,IVFH4096,Flat" "GPU,Cagra,64,32,NN_DESCENT" "GPU,CagraCH,64,32,NN_DESCENT")       # GPU ANN indexes; use_cuvs auto-set per index in run_benchmark

ninja -C "$BUILD_DIR"


##### HELPER
run_benchmark() {
    local QUERY="$1"
    local DEVICE="$2"
    local STORAGE_DEVICE="$3"
    local INDEX_STORAGE_DEVICE="$4"
    local INDEX="$5"
    local PARAMS="$6"
    local REPS_LOCAL="${7:-$REPS}"
    local SCALE_LOCAL="${8:-$SCALE_FACTOR}"
    local EXTRA_TAG="${9}" # Optional parameter to append (e.g. k_100)
    local EXTRA_ARGS="${10}" # Optional extra maxbench flags (e.g. "--csv_batch_size max")
    local VS_DEVICE="${11}" # Optional --vs_device (defaults to DEVICE when empty)

    # Skip (H) variants on systems without unified memory (sgs-gpu05/06).
    if [[ "$SYSTEM" == "sgs-gpu05" || "$SYSTEM" == "sgs-gpu06" ]] \
        && [[ "$INDEX" == *"IVFH"* || "$INDEX" == *"CagraCH"* ]]; then
        echo "Skipping (H) variant ${INDEX} on ${SYSTEM} (no unified memory)"
        return 0
    fi

    # Index-alias transforms (CagraCH/IVFH/use_cuvs). See _vsds_transforms.sh.
    apply_index_transforms "$INDEX" "$PARAMS" "$EXTRA_TAG"
    local ACTUAL_INDEX="$ACTUAL_INDEX"
    PARAMS="$TRANSFORMED_PARAMS"
    EXTRA_TAG="$(join_extra_tag "$EXTRA_TAG" "$TRANSFORM_TAG")"

    if [ "$DEVICE" = "cpu" ]; then
        OUT_DIR="${RESULTS_BASE}/vsds/cpu-${SYSTEM}/sf_${SCALE_LOCAL}"
    elif [ "$DEVICE" = "gpu" ]; then
        OUT_DIR="${RESULTS_BASE}/vsds/gpu-${SYSTEM}/sf_${SCALE_LOCAL}"
    else
        OUT_DIR="${RESULTS_BASE}/vsds/other-${SYSTEM}/sf_${SCALE_LOCAL}"
    fi
    mkdir -p "$OUT_DIR"

    # Standardized tagged filename format to avoid ambiguity
    # q_<query>-i_<index>-d_<device>-s_<storage>-is_<index_storage>-sf_<sf>.log
    TAGGED_BASENAME="q_${QUERY}-i_${ACTUAL_INDEX}-d_${DEVICE}-s_${STORAGE_DEVICE}-is_${INDEX_STORAGE_DEVICE}-sf_${SCALE_LOCAL}"
    # Append EXTRA_TAG if provided (e.g. for varying k)
    if [ -n "$EXTRA_TAG" ]; then
        TAGGED_BASENAME="${TAGGED_BASENAME}-${EXTRA_TAG}"
    fi
    OUTPUT_FILE="${OUT_DIR}/${TAGGED_BASENAME}.log"

    # CSV output (optional, off by default)
    if [ "$PERSIST_CSV" = "true" ]; then
        CSV_DIR="${OUT_DIR}/csv"
        mkdir -p "$CSV_DIR"
        CSV_FILE="${CSV_DIR}/${TAGGED_BASENAME}.csv"
    fi

    echo "------------------------------------------------"
    echo "Running ${QUERY} with index:${INDEX^^} on device:${DEVICE^^} storage:${STORAGE_DEVICE^^} index_storage:${INDEX_STORAGE_DEVICE^^} "
    echo "Output: ${OUTPUT_FILE}"
    echo ""

    CMD=(
        ./build/Release/benchmarks/maxbench
        --benchmark vsds
        --queries "$QUERY"
        --n_reps "$REPS_LOCAL"
        --storage_device "$STORAGE_DEVICE"
        --index_storage_device "$INDEX_STORAGE_DEVICE"
        --device "$DEVICE"
        --index "$ACTUAL_INDEX"
        --params "$PARAMS"
        --profile "runtime-report(calc.inclusive=true,output=stdout)"
        --path ./tests/vsds/data-industrial_and_scientific-sf_${SCALE_LOCAL}/
        --using_large_list
        --flush_cache
    )
    if [ -n "$VS_DEVICE" ]; then
        CMD+=("--vs_device" "$VS_DEVICE")
    fi
    # Conditionally append CSV persistence
    if [ "$PERSIST_CSV" = "true" ]; then
        CMD+=("--persist_results" "yes" "--out_file" "$CSV_FILE")
    fi
    # Conditionally append the index cache
    # NOTE: was previously disabled for Cagra (FAISS issue #4742); now enabled for all indexes
    # if [[ "$INDEX" != *"Cagra"* ]] && ([[ "$SYSTEM" == "dgx-spark-02" ]] || [[ "$SYSTEM" == "cscs-gh200" ]] || [[ "$SYSTEM" == "sgs-gpu06" ]]); then
    if [ -n "$SLURM_LOCALID" ]; then
        CMD+=("--use_index_cache_dir" "/tmp/maximus_faiss_cache_${SLURM_LOCALID}")
    else
        CMD+=("--use_index_cache_dir" "/tmp/maximus_faiss_cache")
    fi
    # fi
    # Append extra maxbench arguments (e.g. --csv_batch_size max for single-chunk table loading)
    if [ -n "$EXTRA_ARGS" ]; then
        # Word splitting intentional — EXTRA_ARGS contains flag + value pairs
        # shellcheck disable=SC2206
        CMD+=($EXTRA_ARGS)
    fi
    # Case 4 only (vs_device=cpu AND device=gpu): keep glibc's heap pages
    # resident across reps so cuDF's pageable D2H destination doesn't pay
    # first-touch page-fault costs on every rep. Without this, rep timings
    # for the GPU->CPU transfer of filtered embeddings (~1.7 GiB at sf=1) are
    # bimodal (~100 ms vs ~2.5 s, 25x swing). cuDF's to_arrow_host has no
    # host memory-resource parameter, so we cannot inject the pinned pool.
    local malloc_env=()
    if [ "$VS_DEVICE" = "cpu" ] && [ "$DEVICE" = "gpu" ]; then
        malloc_env=(
            MALLOC_ARENA_MAX=1
            MALLOC_MMAP_MAX_=0
            MALLOC_TRIM_THRESHOLD_=-1
        )
    fi

    # Convert the array to a string
    CMD_STR="${CMD[*]}"
    # Wrap the specific profile argument in literal double quotes just for the output string
    CMD_STR="${CMD_STR/runtime-report(calc.inclusive=true,output=stdout)/\"runtime-report(calc.inclusive=true,output=stdout)\"}"
    # Prefix env-var assignments to the logged string when present
    if [ ${#malloc_env[@]} -gt 0 ]; then
        CMD_STR="${malloc_env[*]} ${CMD_STR}"
    fi

    echo "Executing: ${CMD_STR} > $OUTPUT_FILE"

    # log start to progress file
    PROGRESS_FILE="${OUT_DIR}/progress.log"
    echo "$(date +%FT%T) START (${TAGGED_BASENAME}) CMD: ${CMD_STR}" >> "$PROGRESS_FILE"
        # prepare error file under OUT_DIR/errors
        ERROR_DIR="${OUT_DIR}/errors"
        mkdir -p "$ERROR_DIR"
        ERROR_FILE="${ERROR_DIR}/${TAGGED_BASENAME}.err"
    # nsys wrap goes inside timeout so wall-clock covers profiler+benchmark together.
    local nsys_prefix=()
    if [ -n "$NSYS_OUT_DIR" ]; then
        local nsys_file="${NSYS_OUT_DIR}/${TAGGED_BASENAME}.nsys-rep"
        nsys_prefix=(
            env CALI_SERVICES_ENABLE=nvtx
            nsys profile
                --trace=cuda,nvtx
                --force-overwrite=true
                -o "$nsys_file"
                --
        )
        echo "nsys wrap -> ${nsys_file}"
    fi

    # run the command and capture exit code (stdout -> OUTPUT_FILE, stderr -> ERROR_FILE)
    local start_epoch
    start_epoch=$(date +%s)
    timeout --kill-after=30s "$BENCH_TIMEOUT" env "${malloc_env[@]}" "${nsys_prefix[@]}" "${CMD[@]}" > "$OUTPUT_FILE" 2> "$ERROR_FILE"
    rc=$?
    local end_epoch elapsed_s elapsed_m
    end_epoch=$(date +%s)
    elapsed_s=$((end_epoch - start_epoch))
    elapsed_m=$(awk "BEGIN{printf \"%.1f\", ${elapsed_s}/60}")
    if [ $rc -eq 124 ]; then
        echo "$(date +%FT%T) TIMEOUT after ${BENCH_TIMEOUT}s (${elapsed_s}s / ${elapsed_m}m) ${TAGGED_BASENAME}" >> "$PROGRESS_FILE"
        echo "  Killed: ${CMD_STR}" >> "$PROGRESS_FILE"
        echo "*** TIMEOUT (${BENCH_TIMEOUT}s) — skipping ${QUERY} ***"
    else
        echo "$(date +%FT%T) DONE RC=${rc} (${elapsed_s}s / ${elapsed_m}m) ${TAGGED_BASENAME}" >> "$PROGRESS_FILE"
    fi
    if [ $rc -ne 0 ] && [ $rc -ne 124 ]; then
        echo "Command failed (RC=${rc}): ${CMD_STR}" >> "$PROGRESS_FILE"
        echo "Error output saved: ${ERROR_FILE}" >> "$PROGRESS_FILE"
        echo "--- Error excerpt (last 50 lines) for ${QUERY} ---" >> "$PROGRESS_FILE"
        tail -n 50 "$ERROR_FILE" >> "$PROGRESS_FILE" 2>/dev/null
        echo "--- End error excerpt ---" >> "$PROGRESS_FILE"
    elif [ $rc -eq 0 ]; then
        rm -f "$ERROR_FILE"
    fi
}


# Returns 0 (match) if INDEX passes the RUN_INDEX_FILTER token, 1 otherwise.
# Empty filter => match all (preserves default behavior).
# Smart tokens distinguish Cagra vs CagraCH and IVFN vs IVFHN (substring alone would be ambiguous).
index_matches_filter() {
    local idx_lc flt_lc
    idx_lc="${1,,}"
    flt_lc="${2,,}"
    [ -z "$flt_lc" ] && return 0
    case "$flt_lc" in
        cagra)    [[ "$idx_lc" == *"cagra"* && "$idx_lc" != *"cagrach"* ]] ;;
        cagrach)  [[ "$idx_lc" == *"cagrach"* ]] ;;
        ivf1024)  [[ "$idx_lc" == "ivf1024,"* || "$idx_lc" == *",ivf1024,"* ]] ;;
        ivfh1024) [[ "$idx_lc" == "ivfh1024,"* || "$idx_lc" == *",ivfh1024,"* ]] ;;
        ivf4096)  [[ "$idx_lc" == "ivf4096,"* || "$idx_lc" == *",ivf4096,"* ]] ;;
        ivfh4096) [[ "$idx_lc" == "ivfh4096,"* || "$idx_lc" == *",ivfh4096,"* ]] ;;
        flat)     [[ "$idx_lc" == "flat" || "$idx_lc" == "gpu,flat" ]] ;;
        *)        [[ "$idx_lc" == *"$flt_lc"* ]] ;;
    esac
}


get_safe_itopk() {
    local target_k="$1"
    local device="${2:-gpu}"  # "gpu" or "cpu"; GPU has cuVS itopk ceiling
    # If target is small, echo the original input (do not force 100)
    if [ "${target_k}" -le 100 ]; then
        echo "${target_k}"
        return 0
    fi
    # Rounds up to the nearest multiple of 32 for optimal CAGRA hardware alignment
    local itopk=$(( (target_k + 31) / 32 * 32 ))
    # Clamp for CAGRA/cuVS hard limit (GPU only — CPU HNSW-Cagra has no ceiling)
    if [ "$device" = "gpu" ]; then
        local max_itopk=1024
        if [ "${itopk}" -gt "${max_itopk}" ]; then
            itopk=${max_itopk}
        fi
    fi
    echo "${itopk}"
}
get_safe_k_vals() {
    local base_k="$1"
    local use_post="$2"
    local device="${3:-gpu}"  # pass through to get_safe_itopk
    local post_mult="${4:-10}"
    local index_hint="${5:-}"   # pass $INDEX so we mirror run_benchmark's use_cuvs auto-correction
    local post_k
    local cagra_k
    local max_postfilter

    # GPU faiss k-ceiling per index (CPU has no such ceiling):
    #   IVF   → use_cuvs=0 → cap 2048
    #   Cagra → cap 1024
    #   Flat  → use_cuvs=1 → cap 16384
    if [ "$device" = "gpu" ]; then
        if [[ "$index_hint" == *"IVF"* ]]; then
            max_postfilter=2048
        elif [[ "$index_hint" == *"Cagra"* ]]; then
            max_postfilter=1024
        else
            max_postfilter=16384
        fi
    else
        max_postfilter=-1   # CPU: no cap
    fi

    if [ "$use_post" -eq 1 ]; then
        post_k=$(( base_k * post_mult ))
        if [ "$max_postfilter" -gt 0 ] && [ "$post_k" -gt "$max_postfilter" ]; then
            post_k=$max_postfilter
        fi
    else
        # If no post-filter, default to the base k
        post_k="$base_k"
    fi

    # CAGRA must track at least the post-filter amount (clamped for cuVS/CAGRA limits on GPU)
    cagra_k=$(get_safe_itopk "$post_k" "$device")

    # Return the three values separated by spaces
    echo "$base_k $post_k $cagra_k"
}


# #################################################################
# #              START LOOP OVER SCALE FACTORS
# #################################################################


for current_scale_factor in "${SCALE_FACTOR[@]}"; do

    if [ "$RUN_SF" != "all" ] && [ "$RUN_SF" != "$current_scale_factor" ]; then
        # echo "Skipping scale factor ${current_scale_factor} as per RUN_SF=${RUN_SF}"
        continue
    fi

    echo "Running benchmarks for scale factor: ${current_scale_factor}"

    for run_scenario in "enn" "ann"; do
        if [ "$RUN_VS" != "$run_scenario" ] && [ "$RUN_VS" != "all" ]; then
            continue
        fi

        echo "Running benchmarks for scenario: ${run_scenario}"

        # =====================================================================
        # K LOOP: iterate over all requested K values (except q11_end)
        # =====================================================================
        for current_k in "${K_VALUES[@]}"; do
        EXTRA_TAG="k_${current_k}"
        echo "  K=${current_k} (EXTRA_TAG=${EXTRA_TAG})"


        # #################################################################
        # #         CASE 0: CPU, CPU, CPU - single query at a time
        # #################################################################

        DEVICE="cpu"
        STORAGE_DEVICE="cpu"
        INDEX_STORAGE_DEVICE="$STORAGE_DEVICE"
        if [ "$run_scenario" = "enn" ]; then
            ANN_INDEXES_FOR_CASE=("$CPU_ENN_INDEX")
        else
            ANN_INDEXES_FOR_CASE=("${CPU_ANN_INDEXES[@]}")
        fi

        for INDEX in "${ANN_INDEXES_FOR_CASE[@]}"; do
        index_matches_filter "$INDEX" "$RUN_INDEX_FILTER" || continue
        CASE_EXTRA_TAG="${EXTRA_TAG}"

        # SKIP Q1 for now...
        # # q1_start
        # if [ "$RUN_Q" = "q1_c0" ] || [ "$RUN_Q" = "c0" ] || [ "$RUN_Q" = "all" ]; then
        #     QUERY="q1_start"
        #     k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE"))
        #     PARAMS="k=${k_vals[0]},query_count=10,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # fi

        # q2_start  [use_post=1 for ann] [get_safe_k_vals: 1 = uses postfilter]
        if [ "$RUN_Q" = "q2_c0" ] || [ "$RUN_Q" = "c0" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q2_start"
            if [ "$run_scenario" = "ann" ]; then
                USE_POST=1
            else
                USE_POST=0
            fi
            # out: current_k="${k_vals[0]}" current_postk="${k_vals[1]}" current_itopk="${k_vals[2]}"
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE")) # 1: uses postfilter_ksearch
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q10_mid  [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q10_c0" ] || [ "$RUN_Q" = "c0"  ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q10_mid"
            k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q13_mid  [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q13_c0" ] || [ "$RUN_Q" = "c0" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q13_mid"
            k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q15_end  [use_post=1 for ann] [get_safe_k_vals: 1 = uses postfilter]
        if [ "$RUN_Q" = "q15_c0" ] || [ "$RUN_Q" = "c0" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q15_end"
            # NOTE: you need a huge postfilter_ksearch to get good recall for q15
            # - you're looking for reviews from top supplier. Since ~2.5M reviews, what's the likelihood the 50k most similar all include the ones from top supplier?
            #       - i.e. you're missing "queries". So this is a query where "prefilter" is WAY better. (but engine limitation due to acero join w/ list<float>)
            if [ "$run_scenario" = "ann" ]; then
                USE_POST=1
            else
                USE_POST=0
            fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE" 500 "$INDEX")) # q15: force postfilter cap
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG" "--csv_batch_size max"
        fi

        # q16_start  [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q16_c0" ] || [ "$RUN_Q" = "c0" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q16_start"
            k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q18_mid  [use_post=1 for ann] [get_safe_k_vals: 1 = uses postfilter]
        if [ "$RUN_Q" = "q18_c0" ] || [ "$RUN_Q" = "c0" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q18_mid"
            if [ "$run_scenario" = "ann" ]; then
                USE_POST=1
            else
                USE_POST=0
            fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE")) # 1: uses postfilter_ksearch
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q19_start  [use_post=1 for ann] [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q19_c0" ] || [ "$RUN_Q" = "c0" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q19_start"
            if [ "$run_scenario" = "ann" ]; then
                USE_POST=1
            else
                USE_POST=0
            fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE" 10 "$INDEX")) # 1: uses postfilter_ksearch
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi


        done # end index loop for case 0


        # #################################################################
        # #         CASE 1:  GPU, CPU, CPU - single query at a time
        # #################################################################

        DEVICE="gpu"
        STORAGE_DEVICE="cpu"
        INDEX_STORAGE_DEVICE="$STORAGE_DEVICE"
        if [ "$run_scenario" = "enn" ]; then
            ANN_INDEXES_FOR_CASE=("$GPU_ENN_INDEX")
        else
            ANN_INDEXES_FOR_CASE=("${GPU_ANN_INDEXES[@]}")
        fi

        for INDEX in "${ANN_INDEXES_FOR_CASE[@]}"; do
        index_matches_filter "$INDEX" "$RUN_INDEX_FILTER" || continue

        # # q1_start
        # if [ "$RUN_Q" = "q1_c1" ] || [ "$RUN_Q" = "c1" ] || [ "$RUN_Q" = "all" ]; then
        #     QUERY="q1_start"
        #     k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE"))
        #     PARAMS="k=${k_vals[0]},query_count=10,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # fi

        # q2_start  [use_post=1 for ann] [get_safe_k_vals: 1 = uses postfilter]
        if [ "$RUN_Q" = "q2_c1" ] || [ "$RUN_Q" = "c1" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q2_start"
            if [ "$run_scenario" = "ann" ]; then
                USE_POST=1
            else
                USE_POST=0
            fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE")) # 1: uses postfilter_ksearch
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q10_mid  [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q10_c1" ] || [ "$RUN_Q" = "c1" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q10_mid"
            k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q13_mid  [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q13_c1" ] || [ "$RUN_Q" = "c1" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q13_mid"
            k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q15_end  [use_post=1 always on gpu] [get_safe_k_vals: 1 = uses postfilter]
        if [ "$RUN_Q" = "q15_c1" ] || [ "$RUN_Q" = "c1" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q15_end"
            # - use_post=1 must for q15 on gpu
            # NOTE: you need a huge postfilter_ksearch to get good recall for q15
            # postfilter_ksearch=16384 #  16384 is the max "k" allowed on GPU (for use_cuvs=1)
            use_cuvs=1 # must for q15 on gpu
            # postfilter_ksearch=2048  #  2048 is the max "k" allowed on GPU (for use_cuvs=0)
            # use_cuvs=0 # default
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE" 500 "$INDEX")) # q15: force postfilter cap
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=$use_cuvs,metric=IP,use_post=1"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG" "--csv_batch_size max"
        fi

        # q16_start  [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q16_c1" ] || [ "$RUN_Q" = "c1" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q16_start"
            k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q18_mid  [use_post=1 for ann] [get_safe_k_vals: 1 = uses postfilter]
        if [ "$RUN_Q" = "q18_c1" ] || [ "$RUN_Q" = "c1" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q18_mid"
            if [ "$run_scenario" = "ann" ]; then
                USE_POST=1
            else
                USE_POST=0
            fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE")) # 1: uses postfilter_ksearch
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q19_start  [use_post=1 for ann] [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q19_c1" ] || [ "$RUN_Q" = "c1" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q19_start"
            if [ "$run_scenario" = "ann" ]; then
                USE_POST=1
            else
                USE_POST=0
            fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE" 10 "$INDEX")) # 1: uses postfilter_ksearch
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi


        done # end index loop for case 1


        #################################################################
        # #  [SKIPPING]         CASE 2: GPU, CPU, GPU - So only index starts on GPU
        #################################################################

        # DEVICE="gpu"
        # STORAGE_DEVICE="cpu"
        # INDEX_STORAGE_DEVICE="gpu"
        # if [ "$run_scenario" = "enn" ]; then
        #     ANN_INDEXES_FOR_CASE=("$GPU_ENN_INDEX")
        # else
        #     ANN_INDEXES_FOR_CASE=("${GPU_ANN_INDEXES[@]}")
        # fi

        # for INDEX in "${ANN_INDEXES_FOR_CASE[@]}"; do

        # # # q1_start
        # # if [ "$RUN_Q" = "q1_c2" ] || [ "$RUN_Q" = "c2" ] || [ "$RUN_Q" = "all" ]; then
        # #     QUERY="q1_start"
        # #     k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE"))
        # #     PARAMS="k=${k_vals[0]},query_count=10,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
        # #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # # fi

        # # q2_start  [use_post=1 for ann] [get_safe_k_vals: 1 = uses postfilter]
        # if [ "$RUN_Q" = "q2_c2" ] || [ "$RUN_Q" = "c2" ] || [ "$RUN_Q" = "all" ]; then
        #     QUERY="q2_start"
        #     if [ "$run_scenario" = "ann" ]; then
        #         USE_POST=1
        #     else
        #         USE_POST=0
        #     fi
        #     k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE")) # 1: uses postfilter_ksearch
        #     PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # fi

        # # q10_mid  [get_safe_k_vals: 0 = no postfilter scaling]
        # if [ "$RUN_Q" = "q10_c2" ] || [ "$RUN_Q" = "c2" ] || [ "$RUN_Q" = "all" ]; then
        #     QUERY="q10_mid"
        #     k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
        #     PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # fi

        # # q13_mid  [get_safe_k_vals: 0 = no postfilter scaling]
        # if [ "$RUN_Q" = "q13_c2" ] || [ "$RUN_Q" = "c2" ] || [ "$RUN_Q" = "all" ]; then
        #     QUERY="q13_mid"
        #     k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
        #     PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # fi

        # # q15_end  [use_post=1 always on gpu] [get_safe_k_vals: 1 = uses postfilter]
        # if [ "$RUN_Q" = "q15_c2" ] || [ "$RUN_Q" = "c2" ] || [ "$RUN_Q" = "all" ]; then
        #     QUERY="q15_end"
        #     # NOTE: you need a huge postfilter_ksearch to get good recall for q15
        #     # postfilter_ksearch=16384 #  16384 is the max "k" allowed on GPU (for use_cuvs=1)
        #     # postfilter_ksearch=2048  #  2048 is the max "k" allowed on GPU (for use_cuvs=0)
        #     use_cuvs=1 # must for q15 on gpu
        #     # use_cuvs=0 # default
        #     k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE")) # 1: uses postfilter_ksearch
        #     PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=$use_cuvs,metric=IP,use_post=1"
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # fi

        # # q16_start  [get_safe_k_vals: 0 = no postfilter scaling]
        # if [ "$RUN_Q" = "q16_c2" ] || [ "$RUN_Q" = "c2" ] || [ "$RUN_Q" = "all" ]; then
        #     QUERY="q16_start"
        #     k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
        #     PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # fi

        # # q18_mid  [use_post=1 for ann] [get_safe_k_vals: 1 = uses postfilter]
        # if [ "$RUN_Q" = "q18_c2" ] || [ "$RUN_Q" = "c2" ] || [ "$RUN_Q" = "all" ]; then
        #     QUERY="q18_mid"
        #     if [ "$run_scenario" = "ann" ]; then
        #         USE_POST=1
        #     else
        #         USE_POST=0
        #     fi
        #     k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE")) # 1: uses postfilter_ksearch
        #     PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # fi

        # # q19_start  [use_post=1 for ann] [get_safe_k_vals: 0 = no postfilter scaling]
        # if [ "$RUN_Q" = "q19_c2" ] || [ "$RUN_Q" = "c2" ] || [ "$RUN_Q" = "all" ]; then
        #     QUERY="q19_start"
        #     if [ "$run_scenario" = "ann" ]; then
        #         USE_POST=1
        #     else
        #         USE_POST=0
        #     fi
        #     k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE")) # 1: uses postfilter_ksearch
        #     PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # fi


        # done # end index loop for case 2


        #################################################################
        # #         CASE 3: GPU, GPU, GPU - single query at a time
        #################################################################

        DEVICE="gpu"
        STORAGE_DEVICE="gpu"
        INDEX_STORAGE_DEVICE="$STORAGE_DEVICE"
        if [ "$run_scenario" = "enn" ]; then
            ANN_INDEXES_FOR_CASE=("$GPU_ENN_INDEX")
        else
            ANN_INDEXES_FOR_CASE=("${GPU_ANN_INDEXES[@]}")
        fi

        for INDEX in "${ANN_INDEXES_FOR_CASE[@]}"; do
        index_matches_filter "$INDEX" "$RUN_INDEX_FILTER" || continue

        # # q1_start
        # if [ "$RUN_Q" = "q1_c3" ] || [ "$RUN_Q" = "c3" ] || [ "$RUN_Q" = "all" ]; then
        #     QUERY="q1_start"
        #     k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE"))
        #     PARAMS="k=${k_vals[0]},query_count=10,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        # fi

        # q2_start  [use_post=1 for ann] [get_safe_k_vals: 1 = uses postfilter]
        if [ "$RUN_Q" = "q2_c3" ] || [ "$RUN_Q" = "c3" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q2_start"
            if [ "$run_scenario" = "ann" ]; then
                USE_POST=1
            else
                USE_POST=0
            fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE")) # 1: uses postfilter_ksearch
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q10_mid  [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q10_c3" ] || [ "$RUN_Q" = "c3" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q10_mid"
            k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q13_mid  [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q13_c3" ] || [ "$RUN_Q" = "c3" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q13_mid"
            k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q15_end  [use_post=1 always on gpu] [get_safe_k_vals: 1 = uses postfilter]
        if [ "$RUN_Q" = "q15_c3" ] || [ "$RUN_Q" = "c3" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q15_end"
            # NOTE: you need a huge postfilter_ksearch to get good recall for q15
            # postfilter_ksearch=16384 #  16384 is the max "k" allowed on GPU (for use_cuvs=1)
            # postfilter_ksearch=2048  #  2048 is the max "k" allowed on GPU (for use_cuvs=0)
            use_cuvs=1 # must for q15 on gpu
            # use_cuvs=0 # default
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE" 500 "$INDEX")) # q15: force postfilter cap
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=$use_cuvs,metric=IP,use_post=1"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q16_start  [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q16_c3" ] || [ "$RUN_Q" = "c3" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q16_start"
            k_vals=($(get_safe_k_vals "$current_k" "0" "$DEVICE")) # 0: no postfilter scaling
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q18_mid  [use_post=1 for ann] [get_safe_k_vals: 1 = uses postfilter]
        if [ "$RUN_Q" = "q18_c3" ] || [ "$RUN_Q" = "c3" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q18_mid"
            if [ "$run_scenario" = "ann" ]; then
                USE_POST=1
            else
                USE_POST=0
            fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE")) # 1: uses postfilter_ksearch
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        # q19_start  [use_post=1 for ann] [get_safe_k_vals: 0 = no postfilter scaling]
        if [ "$RUN_Q" = "q19_c3" ] || [ "$RUN_Q" = "c3" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q19_start"
            if [ "$run_scenario" = "ann" ]; then
                USE_POST=1
            else
                USE_POST=0
            fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "$DEVICE" 10 "$INDEX"))
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$EXTRA_TAG"
        fi

        done # end index loop for case 3


        # #################################################################
        # #         CASE 4: GPU relational + CPU vector-search (hybrid)
        # #         device=gpu, storage=cpu, index_storage=cpu, vs_device=cpu
        # #         No index movement. ENN=Flat, ANN=IVF/Cagra (no IVFH/CagraCH).
        # #################################################################

        DEVICE="gpu"
        STORAGE_DEVICE="cpu"
        INDEX_STORAGE_DEVICE="$STORAGE_DEVICE"
        VS_DEVICE_C4="cpu"
        if [ "$run_scenario" = "enn" ]; then
            ANN_INDEXES_FOR_CASE=("$CPU_ENN_INDEX")
        else
            ANN_INDEXES_FOR_CASE=("${CPU_ANN_INDEXES[@]}")
        fi

        for INDEX in "${ANN_INDEXES_FOR_CASE[@]}"; do
        index_matches_filter "$INDEX" "$RUN_INDEX_FILTER" || continue
        # Case 4 is identified by the "vsd_cpu" extra_tag marker. parse_caliper
        # strips it from the bucket key so Case 4 aggregates alongside Case 1
        # in the same CSV (as a separate case column).
        CASE_EXTRA_TAG="${EXTRA_TAG}-vsd_cpu"

        # q2_start
        if [ "$RUN_Q" = "q2_c4" ] || [ "$RUN_Q" = "c4" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q2_start"
            if [ "$run_scenario" = "ann" ]; then USE_POST=1; else USE_POST=0; fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "cpu"))
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$CASE_EXTRA_TAG" "" "$VS_DEVICE_C4"
        fi

        # q10_mid
        if [ "$RUN_Q" = "q10_c4" ] || [ "$RUN_Q" = "c4" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q10_mid"
            k_vals=($(get_safe_k_vals "$current_k" "0" "cpu"))
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$CASE_EXTRA_TAG" "" "$VS_DEVICE_C4"
        fi

        # q13_mid
        if [ "$RUN_Q" = "q13_c4" ] || [ "$RUN_Q" = "c4" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q13_mid"
            k_vals=($(get_safe_k_vals "$current_k" "0" "cpu"))
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$CASE_EXTRA_TAG" "" "$VS_DEVICE_C4"
        fi

        # q15_end
        if [ "$RUN_Q" = "q15_c4" ] || [ "$RUN_Q" = "c4" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q15_end"
            if [ "$run_scenario" = "ann" ]; then USE_POST=1; else USE_POST=0; fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "cpu" 500 "$INDEX"))
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$CASE_EXTRA_TAG" "--csv_batch_size max" "$VS_DEVICE_C4"
        fi

        # q16_start
        if [ "$RUN_Q" = "q16_c4" ] || [ "$RUN_Q" = "c4" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q16_start"
            k_vals=($(get_safe_k_vals "$current_k" "0" "cpu"))
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$CASE_EXTRA_TAG" "" "$VS_DEVICE_C4"
        fi

        # q18_mid
        if [ "$RUN_Q" = "q18_c4" ] || [ "$RUN_Q" = "c4" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q18_mid"
            if [ "$run_scenario" = "ann" ]; then USE_POST=1; else USE_POST=0; fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "cpu"))
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$CASE_EXTRA_TAG" "" "$VS_DEVICE_C4"
        fi

        # q19_start
        if [ "$RUN_Q" = "q19_c4" ] || [ "$RUN_Q" = "c4" ] || [ "$RUN_Q" = "all" ]; then
            QUERY="q19_start"
            if [ "$run_scenario" = "ann" ]; then USE_POST=1; else USE_POST=0; fi
            k_vals=($(get_safe_k_vals "$current_k" "1" "cpu" 10 "$INDEX"))
            PARAMS="k=${k_vals[0]},query_count=1,query_start=0,postfilter_ksearch=${k_vals[1]},cagra_itopksize=${k_vals[2]},ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_post=$USE_POST"
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$CASE_EXTRA_TAG" "" "$VS_DEVICE_C4"
        fi

        done # end index loop for case 4

        done # end K loop

        # =================================================================
        # q11_end: EXCLUDED from K loop. Q11 is fixed k value. (We only tag it as k_100 to be included with the others when plotting)
        # =================================================================
        QUERY="q11_end"

        # CASE 0: CPU, CPU, CPU
        if [ "$RUN_Q" = "q11_c0" ] || [ "$RUN_Q" = "c0" ] || [ "$RUN_Q" = "all" ]; then
            DEVICE="cpu"; STORAGE_DEVICE="cpu"; INDEX_STORAGE_DEVICE="cpu"
            if [ "$run_scenario" = "enn" ]; then Q11_INDEXES=("$CPU_ENN_INDEX"); else Q11_INDEXES=("${CPU_ANN_INDEXES[@]}"); fi
            for INDEX in "${Q11_INDEXES[@]}"; do
            index_matches_filter "$INDEX" "$RUN_INDEX_FILTER" || continue
            # NOTE: we are "overprovisioning" number of partitions "k" even though we may have fewer results
            #  - 'k' controls the number of "queries".
            #  - for sf=0.01 k > ~350 ( industrial_and_scientific_sf0.01 , seed 98 )
            #  - for sf=1 k > ~1000 ?
            # NOTE: Remember to do the same from postgres side to be fair!
            # NOTE: postfilter ksearch doesn't matter here, it's "k" that controls.
            if [ "$current_scale_factor" = "0.01" ]; then
                # overshoot k a bit just to be safe
                # limit per group on CPU is worse than partition by / union all
                PARAMS="k=400,query_count=1,query_start=0,postfilter_ksearch=500,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_limit_per_group=0"
            else
                PARAMS="k=1050,query_count=1,query_start=0,postfilter_ksearch=500,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_limit_per_group=0"
            fi
            # k=100 is the "default" for others, so I want q11 to be grouped with them (based on the extra tag)
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "k_100"
            done # end index loop q11 case 0
        fi

        # CASE 1: GPU, CPU, CPU
        if [ "$RUN_Q" = "q11_c1" ] || [ "$RUN_Q" = "c1" ] || [ "$RUN_Q" = "all" ]; then
            DEVICE="gpu"; STORAGE_DEVICE="cpu"; INDEX_STORAGE_DEVICE="cpu"
            if [ "$run_scenario" = "enn" ]; then Q11_INDEXES=("$GPU_ENN_INDEX"); else Q11_INDEXES=("${GPU_ANN_INDEXES[@]}"); fi
            for INDEX in "${Q11_INDEXES[@]}"; do
            index_matches_filter "$INDEX" "$RUN_INDEX_FILTER" || continue
            if [ "$current_scale_factor" = "0.01" ]; then
                # limit_per_group in GPU provides a speed up over partitionby/union
                PARAMS="k=400,query_count=1,query_start=0,postfilter_ksearch=500,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_limit_per_group=1"
            else
                PARAMS="k=1050,query_count=1,query_start=0,postfilter_ksearch=500,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_limit_per_group=1"
            fi
            # k=100 is the "default" for others, so I want q11 to be grouped with them (based on the extra tag)
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "k_100"
            done # end index loop q11 case 1
        fi

        # [SKIPPING] CASE 2: GPU, CPU, GPU
        # if [ "$RUN_Q" = "q11_c2" ] || [ "$RUN_Q" = "c2" ] || [ "$RUN_Q" = "all" ]; then
        #     DEVICE="gpu"; STORAGE_DEVICE="cpu"; INDEX_STORAGE_DEVICE="gpu"
        #     if [ "$run_scenario" = "enn" ]; then Q11_INDEXES=("$GPU_ENN_INDEX"); else Q11_INDEXES=("${GPU_ANN_INDEXES[@]}"); fi
        #     for INDEX in "${Q11_INDEXES[@]}"; do
        #     if [ "$current_scale_factor" = "0.01" ]; then
        #         # limit_per_group in GPU provides a speed up over partitionby/union
        #         PARAMS="k=400,query_count=1,query_start=0,postfilter_ksearch=500,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_limit_per_group=1"
        #     else
        #         PARAMS="k=1050,query_count=1,query_start=0,postfilter_ksearch=500,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_limit_per_group=1"
        #     fi
        #     # k=100 is the "default" for others, so I want q11 to be grouped with them (based on the extra tag)
        #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "k_100"
        #     done # end index loop q11 case 2
        # fi

        # CASE 3: GPU, GPU, GPU
        if [ "$RUN_Q" = "q11_c3" ] || [ "$RUN_Q" = "c3" ] || [ "$RUN_Q" = "all" ]; then
            DEVICE="gpu"; STORAGE_DEVICE="gpu"; INDEX_STORAGE_DEVICE="gpu"
            if [ "$run_scenario" = "enn" ]; then Q11_INDEXES=("$GPU_ENN_INDEX"); else Q11_INDEXES=("${GPU_ANN_INDEXES[@]}"); fi
            for INDEX in "${Q11_INDEXES[@]}"; do
            index_matches_filter "$INDEX" "$RUN_INDEX_FILTER" || continue
            if [ "$current_scale_factor" = "0.01" ]; then
                # limit_per_group in GPU provides a speed up over partitionby/union
                PARAMS="k=400,query_count=1,query_start=0,postfilter_ksearch=500,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_limit_per_group=1"
            else
                PARAMS="k=1050,query_count=1,query_start=0,postfilter_ksearch=500,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_limit_per_group=1"
            fi
            # k=100 is the "default" for others, so I want q11 to be grouped with them (based on the extra tag)
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "k_100"
            done # end index loop q11 case 3
        fi

        # CASE 4: GPU relational + CPU VS
        if [ "$RUN_Q" = "q11_c4" ] || [ "$RUN_Q" = "c4" ] || [ "$RUN_Q" = "all" ]; then
            DEVICE="gpu"; STORAGE_DEVICE="cpu"; INDEX_STORAGE_DEVICE="cpu"; VS_DEVICE_C4="cpu"
            if [ "$run_scenario" = "enn" ]; then Q11_INDEXES=("$CPU_ENN_INDEX"); else Q11_INDEXES=("${CPU_ANN_INDEXES[@]}"); fi
            for INDEX in "${Q11_INDEXES[@]}"; do
            index_matches_filter "$INDEX" "$RUN_INDEX_FILTER" || continue
            if [ "$current_scale_factor" = "0.01" ]; then
                PARAMS="k=400,query_count=1,query_start=0,postfilter_ksearch=500,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_limit_per_group=1"
            else
                PARAMS="k=1050,query_count=1,query_start=0,postfilter_ksearch=500,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=${USE_CUVS},metric=IP,use_limit_per_group=1"
            fi
            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "k_100-vsd_cpu" "" "$VS_DEVICE_C4"
            done # end index loop q11 case 4
        fi

    done

    # TODO: add more as needed
echo "Completed benchmarks for scale factor: ${current_scale_factor}"

done
echo "All benchmarks completed."
