#!/bin/bash

###################################
#           HOW TO RUN:
###################################

# 1) Change the SYSTEM variable
# 2) Decide on the enviroment variables
# 3) run  from top-level : $./results/run_vsds.sh


### -- dgx-spark-02 -- ###

# figure out if you want to set these based on your system
# export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=18
# export MAXIMUS_NUM_INNER_THREADS=18
# export MAXIMUS_NUM_OUTER_THREADS=1
# export MAXIMUS_OPERATORS_FUSION=0
# export MAXIMUS_DATASET_API=0
# export MAXIMUS_NUM_IO_THREADS=18

### -- cscs-gh200 -- ###

# figure out if you want to set these based on your system
# export CUDA_VISIBLE_DEVICES=0
# # NOTE: conda OpenBLAS is compiled with NUM_THREADS=128. OMP_NUM_THREADS > 128
# # causes heap corruption ("corrupted size vs. prev_size") and core dumps because
# # OpenBLAS can't safely track >128 concurrent callers. GH200 Grace has 72 cores
# # (no SMT), so 72-128 is optimal. Rebuild OpenBLAS from source if you need more.
# export OMP_NUM_THREADS=128
# export OPENBLAS_NUM_THREADS=1     # BLAS calls single-threaded; OMP handles parallelism
# export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
# export MAXIMUS_NUM_INNER_THREADS=128
# export MAXIMUS_NUM_OUTER_THREADS=1
# export MAXIMUS_OPERATORS_FUSION=0
# export MAXIMUS_DATASET_API=0
# export MAXIMUS_NUM_IO_THREADS=128


### -- sgs-gpu-05 -- ###

# figure out if you want to set these based on your system
# export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=32
# export MAXIMUS_NUM_INNER_THREADS=32
# export MAXIMUS_NUM_OUTER_THREADS=1
# export MAXIMUS_OPERATORS_FUSION=0
# export MAXIMUS_DATASET_API=0
# export MAXIMUS_NUM_IO_THREADS=32

###################################
# GLOBAL Configuration 
###################################
# NOTE: 
# cases: # 1: cpu store gpu exec, 2: index gpu, data cpu, gpu exec, 3: all gpu
RUN_SF="${1:-all}"
RUN_VS="${2:-all}" # enn/ann/enn-batch/ann-batch/enn-ground-truth/ann-vary-ivf/ann-vary-hnsw/ann-vary-cagra
SYSTEM="${3:-dgx-spark-02}"
RUN_CASE="${4:-all}" # case to run: 0, 1, 2, 3, or "all"
RESULTS_BASE="${5:-./results}"  # output root directory (default: ./results)
PERSIST_CSV="${6:-true}"        # "true" to save CSV result files (needed for recall checks)
echo "Running on system: ${SYSTEM} <------------"
echo "RUN_CASE: ${RUN_CASE}"

REPS=20
BUILD_DIR="./build/Release"
METRIC="IP" # "L2" or "IP"
K=100
USE_CUVS=1

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_vsds_transforms.sh"

SCALE_FACTOR=("0.01" "1") # "all" ( 0.01 just for quick testing... )
# SCALE_FACTOR=("0.01") # "small"
# SCALE_FACTOR=("1") # "mid"

# Which indexes to test
# ENN : 
# [] which nprobe to use?  Run ann_reviews and ann_images. 
# [] what recall do i get
# [] verify outputs? 
# ANN : 
CPU_ANN_INDEX="IVF1024,Flat"
GPU_ANN_INDEX="GPU,IVF1024,Flat"
# IVF4096 variants (used alongside IVF1024 in index arrays below)
CPU_ANN_INDEX_4096="IVF4096,Flat"
GPU_ANN_INDEX_4096="GPU,IVF4096,Flat"

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
    local RUN_SCENARIO="${9:-$RUN_VS}"
    local EXTRA_TAG="${10}" # Optional parameter to append (e.g. nprobe_20)
    local USE_CACHE="${11:-true}" # Optional parameter to toggle cache, defaults to true
    local VARY_BATCH="${12}" # Optional: comma-separated batch sizes for vary-batch mode
    local TOTAL_Q="${13}"    # Optional: total queries for vary-batch mode
    local CASE6_MODE="${14}" # Optional: "true" to append --case6_persist_gpu_index (CASE 6 only)
    local FORCE_NO_CSV="${15:-false}" # Optional: "true" to skip CSV persistence regardless of PERSIST_CSV

    # Skip (H) variants on systems without unified memory (sgs-gpu05/06).
    if [[ "$SYSTEM" == "sgs-gpu05" || "$SYSTEM" == "sgs-gpu06" ]] \
        && [[ "$INDEX" == *"IVFH"* || "$INDEX" == *"CagraCH"* ]]; then
        echo "Skipping (H) variant ${INDEX} on ${SYSTEM} (no unified memory)"
        return 0
    fi

    # -------------------------------------------------------------
    # Skip CASE 2 for everything EXCEPT hybrid post query
    # -------------------------------------------------------------
    if [ "$CASE_ID" = "2" ] && [[ "$QUERY" != *"hybrid"* ]]; then
        # Silently skip, or uncomment the next line if you want logging
        # echo "Skipping non-hybrid query ${QUERY} for CASE 2."
        return 0
    fi

    # Index-alias transforms (CagraCH/IVFH/use_cuvs). See _vsds_transforms.sh.
    apply_index_transforms "$INDEX" "$PARAMS" "$EXTRA_TAG"
    local ACTUAL_INDEX="$ACTUAL_INDEX"
    PARAMS="$TRANSFORMED_PARAMS"
    EXTRA_TAG="$(join_extra_tag "$EXTRA_TAG" "$TRANSFORM_TAG")"

    if [ "$DEVICE" = "cpu" ]; then
        OUT_DIR="${RESULTS_BASE}/other-${RUN_SCENARIO}/cpu-${SYSTEM}/sf_${SCALE_LOCAL}"
    elif [ "$DEVICE" = "gpu" ]; then
        OUT_DIR="${RESULTS_BASE}/other-${RUN_SCENARIO}/gpu-${SYSTEM}/sf_${SCALE_LOCAL}"
    else
        OUT_DIR="${RESULTS_BASE}/other-${RUN_SCENARIO}/other-${SYSTEM}/sf_${SCALE_LOCAL}"
    fi
    mkdir -p "$OUT_DIR"

    # Standardized tagged filename format to avoid ambiguity
    # q_<query>-i_<index>-d_<device>-s_<storage>-is_<index_storage>-sf_<sf>.log
    TAGGED_BASENAME="q_${QUERY}-i_${ACTUAL_INDEX}-d_${DEVICE}-s_${STORAGE_DEVICE}-is_${INDEX_STORAGE_DEVICE}-sf_${SCALE_LOCAL}"

    # Append EXTRA_TAG if provided (e.g. for varying nprobe/efsearch)
    if [ -n "$EXTRA_TAG" ]; then
        TAGGED_BASENAME="${TAGGED_BASENAME}-${EXTRA_TAG}"
    fi

    OUTPUT_FILE="${OUT_DIR}/${TAGGED_BASENAME}.log"

    # CSV output (optional, off by default).
    # Per-call FORCE_NO_CSV ($15) overrides PERSIST_CSV — used by bs1_fullsweep where
    # we never want to keep per-query CSVs (10k tiny files per index).
    local effective_persist_csv="$PERSIST_CSV"
    if [ "$FORCE_NO_CSV" = "true" ]; then
        effective_persist_csv="false"
    fi
    if [ "$effective_persist_csv" = "true" ]; then
        CSV_DIR="${OUT_DIR}/csv"
        mkdir -p "$CSV_DIR"
        CSV_FILE="${CSV_DIR}/${TAGGED_BASENAME}.csv"
    fi

    echo "------------------------------------------------"
    echo "Running ${QUERY} with index:${INDEX^^} on device:${DEVICE^^} storage:${STORAGE_DEVICE^^} index_storage:${INDEX_STORAGE_DEVICE^^} "
    echo "Output: ${OUTPUT_FILE}"
    echo ""

    # Disable index cache for Cagra (FAISS issue #4742). use_cuvs is force-set by helper above.
    if [[ "$INDEX" == *"Cagra"* ]]; then
        USE_CACHE="false"
    fi

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
    # Conditionally append CSV persistence (respects per-call FORCE_NO_CSV override)
    if [ "$effective_persist_csv" = "true" ]; then
        CMD+=("--persist_results" "yes" "--out_file" "$CSV_FILE")
    fi

    # Conditionally append the index cache (defaults to true; can be disabled via $11)
    # NOTE: was previously disabled for Cagra (FAISS issue #4742); now enabled for all indexes
    if [ "$USE_CACHE" = "true" ]; then
        if [ -n "$SLURM_LOCALID" ]; then
            CMD+=("--use_index_cache_dir" "/tmp/maximus_faiss_cache_${SLURM_LOCALID}")
        else
            CMD+=("--use_index_cache_dir" "/tmp/maximus_faiss_cache")
        fi
    fi

    # Conditionally append vary-batch mode flags
    if [ -n "$VARY_BATCH" ]; then
        CMD+=("--vary_batch_sizes" "$VARY_BATCH" "--total_queries" "$TOTAL_Q")
    fi

    # CASE 6 only: persist index on GPU across all vary_batch iterations
    if [ "$CASE6_MODE" = "true" ]; then
        CMD+=("--case6_persist_gpu_index")
    fi

    # Convert the array to a string
    CMD_STR="${CMD[*]}"
    # Wrap the specific profile argument in literal double quotes just for the output string
    CMD_STR="${CMD_STR/runtime-report(calc.inclusive=true,output=stdout)/\"runtime-report(calc.inclusive=true,output=stdout)\"}"

    echo "Executing: ${CMD_STR} > $OUTPUT_FILE"
    # log start to progress file
    PROGRESS_FILE="${OUT_DIR}/progress.log"
    echo "$(date +%FT%T) START (${QUERY}, i_${INDEX}, d_${DEVICE}, s_${STORAGE_DEVICE}, is_${INDEX_STORAGE_DEVICE}, sf_${SCALE_LOCAL}) CMD: ${CMD_STR}" >> "$PROGRESS_FILE"
        # prepare error file under OUT_DIR/errors
    ERROR_DIR="${OUT_DIR}/errors"
    mkdir -p "$ERROR_DIR"
    ERROR_FILE="${ERROR_DIR}/${TAGGED_BASENAME}.err"
    # run the command and capture exit code (stdout -> OUTPUT_FILE, stderr -> ERROR_FILE)
    local start_epoch
    start_epoch=$(date +%s)
    "${CMD[@]}" > "$OUTPUT_FILE" 2> "$ERROR_FILE"
    rc=$?
    local end_epoch elapsed_s elapsed_m
    end_epoch=$(date +%s)
    elapsed_s=$((end_epoch - start_epoch))
    elapsed_m=$(awk "BEGIN{printf \"%.1f\", ${elapsed_s}/60}")
    echo "$(date +%FT%T) DONE RC=${rc} (${elapsed_s}s / ${elapsed_m}m) ${QUERY}, i_${INDEX}, d_${DEVICE}, s_${STORAGE_DEVICE}, is_${INDEX_STORAGE_DEVICE}, sf_${SCALE_LOCAL}" >> "$PROGRESS_FILE"
    if [ $rc -ne 0 ]; then
        echo "Command failed (RC=${rc}): ${CMD_STR}" >> "$PROGRESS_FILE"
        echo "Error output saved: ${ERROR_FILE}" >> "$PROGRESS_FILE"
        echo "--- Error excerpt (last 50 lines) for ${QUERY} ---" >> "$PROGRESS_FILE"
        tail -n 50 "$ERROR_FILE" >> "$PROGRESS_FILE" 2>/dev/null
        echo "--- End error excerpt ---" >> "$PROGRESS_FILE"
    else
        # Remove error file if command succeeded (no actual errors)
        rm -f "$ERROR_FILE"
    fi
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
    local post_k
    local cagra_k

    if [ "$use_post" -eq 1 ]; then
        # Post-filter: search 10x more candidates to compensate for filtering loss
        post_k=$(( base_k * 10 ))
        # Clamp to GPU faiss backend limits:
        #   use_cuvs=1 (cuVS):  max k = 16384
        #   use_cuvs=0 (legacy): max k = 2048
        # Use cuVS limit here (index-agnostic); IVF's own limit is enforced by Faiss.
        local max_postfilter=16384
        if [ "$post_k" -gt "$max_postfilter" ]; then
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

#################################################################
#              START LOOP OVER SCALE FACTORS
#################################################################

for current_scale_factor in "${SCALE_FACTOR[@]}"; do

    if [ "$RUN_SF" != "all" ] && [ "$RUN_SF" != "$current_scale_factor" ]; then
        # echo "Skipping scale factor ${current_scale_factor} as per RUN_SF=${RUN_SF}"
        continue
    fi

    echo "Running benchmarks for scale factor: ${current_scale_factor}"

    for run_scenario in "enn" "ann" "enn-ground-truth" "ann-vary-ivf" "ann-vary-hnsw" "ann-vary-cagra" "enn-batch" "ann-batch" "enn-vary-batch-in-maxbench" "ann-vary-batch-in-maxbench" "vary-batch-in-maxbench" "TEST-ann-vary-batch-in-maxbench" "bs1_fullsweep"; do
        
        if [ "$RUN_VS" != "$run_scenario" ] && [ "$RUN_VS" != "all" ]; then
            continue
        fi

        echo "Running benchmarks for scenario: ${run_scenario}"
        
        # We define cases as: "CaseID Device Storage IndexStorage"
        # 0:  CPU-CPU-CPU
        # 1:  GPU-CPU-CPU
        # 1P: GPU-CPU(pinned)-CPU(pinned) — varbatch-only
        # 2:  GPU-CPU-GPU
        # 3:  GPU-GPU-GPU
        CASES=(
            "0 cpu cpu cpu"
            "1 gpu cpu cpu"
            "1P gpu cpu-pinned cpu-pinned"
            "2 gpu cpu gpu"
            "3 gpu gpu gpu"
        )

        for case_config in "${CASES[@]}"; do
            read -r CASE_ID DEVICE STORAGE_DEVICE INDEX_STORAGE_DEVICE <<< "$case_config"

            # 1P (cpu-pinned) is scoped to CagraC under ANN-varbatch scenarios only.
            if [ "$CASE_ID" = "1P" ] && \
               [ "$run_scenario" != "vary-batch-in-maxbench" ] && \
               [ "$run_scenario" != "ann-vary-batch-in-maxbench" ] && \
               [ "$run_scenario" != "TEST-ann-vary-batch-in-maxbench" ]; then
                continue
            fi

            if [ "$RUN_CASE" != "all" ] && [ "$RUN_CASE" != "$CASE_ID" ]; then
                continue
            fi

            echo "Running Case $CASE_ID: Device=$DEVICE, Storage=$STORAGE_DEVICE, IndexStorage=$INDEX_STORAGE_DEVICE"

            #################################################################
            #                          ENN                                  #
            #################################################################
            
            if [ "$run_scenario" = "enn" ] || [ "$run_scenario" = "all" ] || [ "$run_scenario" = "main" ]; then
                PARAMS="k=100,query_count=1,query_start=0,use_cuvs=${USE_CUVS},metric=${METRIC}"
                if [ "$DEVICE" = "cpu" ]; then
                    INDEX="Flat"
                else
                    INDEX="GPU,Flat"
                fi

                # Reviews ENN
                # NOTE: for "all gpu case 3" the Reviews table in sf1 needs large_list. We can't do storage GPU for large_list. (cuDF int32 limitation)
                if [ "$CASE_ID" = "3" ]; then
                    QUERY="enn_reviews"
                    run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario"

                    QUERY="enn_reviews_project_distance"
                    run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario"
                fi

                QUERY="ann_reviews"
                run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario"

                # Images ENN
                QUERY="enn_images"
                run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario"

                QUERY="enn_images_project_distance"
                run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario"

                QUERY="ann_images"
                run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario"
            fi


            #################################################################
            #                          ANN                                  #
            #################################################################

            # TODO: Set the final parameters for these.
            
            if [ "$run_scenario" = "ann" ] || [ "$run_scenario" = "all" ] || [ "$run_scenario" = "main" ]; then
                CPU_ANN_INDEX=("IVF1024,Flat" "IVF4096,Flat" "IVF1024,PQ32" "HNSW32,Flat" "GPU,Cagra,64,32,NN_DESCENT")
                GPU_ANN_INDEX=("GPU,IVF1024,Flat" "GPU,IVF4096,Flat" "GPU,IVF1024,PQ32" "GPU,Cagra,64,32,NN_DESCENT" "GPU,CagraCH,64,32,NN_DESCENT")

                if [ "$DEVICE" = "cpu" ]; then
                    INDEXES=("${CPU_ANN_INDEX[@]}")
                else
                    INDEXES=("${GPU_ANN_INDEX[@]}")
                fi

                for INDEX in "${INDEXES[@]}"; do
                    if [[ "$INDEX" == *"Cagra"* ]]; then
                        USE_CUVS=1
                    else
                        USE_CUVS=0
                    fi

                    PARAMS="k=${K},query_count=1,query_start=0,ivf_nprobe=30,hnsw_efsearch=128,cagra_searchwidth=1,cagra_itopksize=128,use_cuvs=${USE_CUVS},metric=${METRIC}"

                    QUERY="ann_reviews"
                    run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario"

                    QUERY="ann_images"
                    run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario"
                done
            fi

            #################################################################
            #                     GET GROUND TRUTH (cpu and gpu)            #
            #################################################################
            
            if [ "$run_scenario" = "enn-ground-truth" ] || [ "$run_scenario" = "all" ]; then
                # skip if case 1 or 2 (only keep case 0 or 3) (all cpu or all gpu for now)
                if [ "$CASE_ID" = "1" ] || [ "$CASE_ID" = "2" ]; then
                    continue
                fi
                LOCAL_REPS=1 # only need to run once for GT

                PARAMS="query_count=0,query_start=0,use_cuvs=${USE_CUVS},metric=${METRIC}"
                EXTRA_TAG="k_${K}-metric_${METRIC}_ground_truth"
                
                QUERY="enn_reviews"
                run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "Flat" "$PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"
                
                QUERY="enn_images"
                run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "Flat" "$PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"
                
                # TODO: filtered
                PARAMS="query_count=0,query_start=0,use_cuvs=${USE_CUVS},metric=${METRIC}"
                EXTRA_TAG="k_${K}-metric_${METRIC}_ground_truth"
                
                QUERY="pre_reviews"
                run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "Flat" "$PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"
                
                QUERY="pre_images"
                run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "Flat" "$PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"
            fi


            #################################################################
            #                     ANN - VARY IVF                            #
            #################################################################
            
            if [ "$run_scenario" = "ann-vary-ivf" ] || [ "$run_scenario" = "all" ]; then
                # Ground truth
                for PARTS in 1024 2048; do
                    if [ "$DEVICE" = "cpu" ]; then
                        INDEX="IVF${PARTS},Flat"
                    else
                        INDEX="GPU,IVF${PARTS},Flat"
                    fi
                    
                    # skip if case 1 or 2 (only keep case 0 or 3) (all cpu or all gpu for now)
                    if [ "$CASE_ID" = "1" ] || [ "$CASE_ID" = "2" ]; then
                        continue
                    fi
                    
                    for NPROBE in 1 10 15 20 30 40 50 60; do
                        # query_count=0 : all queries
                        PARAMS="k=${K},query_count=0,query_start=0,ivf_nprobe=${NPROBE},use_cuvs=${USE_CUVS},metric=${METRIC}"
                        EXTRA_TAG="k_${K}-metric_${METRIC}_nprobe_${NPROBE}"
                        
                        QUERY="ann_reviews"
                        # "false" -> distable cache to not store all the above indexes....
                        run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"

                        QUERY="ann_images"
                        run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"
                    done
                done
            fi
            
            #################################################################
            #                     ANN - VARY HNSW/GRAPH                     #
            #################################################################
            
            if [ "$run_scenario" = "ann-vary-hnsw" ] || [ "$run_scenario" = "all" ]; then
                if [ "$DEVICE" = "cpu" ] && [ "$CASE_ID" = "0" ]; then
                    # CPU uses HNSW
                    # Vary M (connections per node): 32 and 64
                    for M in 32 64; do
                        INDEX="HNSW${M},Flat"
                        
                        # efSearch must be >= K. Since K=100, we start at 100.
                        for EFSEARCH in 100 128 256 512; do
                            # query_count=0 : all queries
                            PARAMS="k=${K},query_count=0,query_start=0,hnsw_efsearch=${EFSEARCH},use_cuvs=${USE_CUVS},metric=${METRIC}"
                            EXTRA_TAG="k_${K}-metric_${METRIC}_efsearch_${EFSEARCH}"
                            
                            QUERY="ann_reviews"
                            # "false" -> disable cache
                            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"

                            QUERY="ann_images"
                            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"
                        done
                    done
                fi
            fi

            #################################################################
            #                     ANN - VARY CAGRA/GRAPH                    #
            #################################################################
            
            if [ "$run_scenario" = "ann-vary-cagra" ] || [ "$run_scenario" = "all" ]; then
                # check only "all gpu"
                if [ "$CASE_ID" = "3" ]; then
                        
                    # GPU uses Cagra
                    # Vary Graph Degree: 32 and 64. 
                    # (Intermediate degree is usually 2x the final degree during Cagra build)
                    for DEGREE in 32 64; do
                        INTERMEDIATE_DEGREE=$((DEGREE * 2))
                        INDEX="GPU,Cagra,${INTERMEDIATE_DEGREE},${DEGREE},NN_DESCENT"
                        # TODO: figure out what the correct thing to vary is for Cagra.
                        # itopk must be >= K. Since K=100, we start at 128 to be safe/standard.
                        for ITOPK in 128 256 512; do
                            # query_count=0 : all queries
                            PARAMS="k=${K},query_count=0,query_start=0,cagra_searchwidth=1,cagra_itopksize=${ITOPK},use_cuvs=${USE_CUVS},metric=${METRIC}"
                            EXTRA_TAG="itopk_${ITOPK}-metric_${METRIC}"
                            USE_CACHE="true"
                            
                            if [ "$SYSTEM" = "dgx-spark-02" ]; then
                                USE_CACHE="false" # disable cache for Cagra, see FAISS ISSUE #4742 : https://github.com/facebookresearch/faiss/issues/4742
                                echo "Skipping ann_images with GPU Cagra on dgx-spark-02. See FAISS issue #4742."
                            fi

                            QUERY="ann_reviews"
                            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG" "$USE_CACHE"

                            QUERY="ann_images"
                            run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG" "$USE_CACHE"
                        done
                    done
                fi

            fi

            #################################################################
            #                     ENN - Full Batch
            #################################################################
            if [ "$run_scenario" = "enn-batch" ] || [ "$run_scenario" = "all" ] || [ "$run_scenario" = "main" ]; then
                    # query_count=batch size = 10k for sf1
                    if [ "$current_scale_factor" = "1" ]; then
                        QCNT=10000
                    else
                        QCNT=1000
                    fi

                    PARAMS="k=100,query_count=${QCNT},query_start=0,use_cuvs=${USE_CUVS},metric=${METRIC}"
                    if [ "$DEVICE" = "cpu" ]; then
                        INDEX="Flat"
                    else
                        INDEX="GPU,Flat"
                    fi
                    
                    EXTRA_TAG="k_${K}-batch_${QCNT}"

                    # Reviews ENN
                    # NOTE: for "all gpu case 3" the Reviews table in sf1 needs large_list. We can't do storage GPU for large_list. (cuDF int32 limitation)
                    if [ "$CASE_ID" != "3" ]; then
                        QUERY="enn_reviews"
                        run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"
                        
                        # TODO: Batching project distance not finished iirc
                        # QUERY="enn_reviews_project_distance"
                        # run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"
                    fi

                    QUERY="ann_reviews"
                    run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"

                    # Images ENN
                    QUERY="enn_images"
                    run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"

                    # QUERY="enn_images_project_distance"
                    # run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"

                    QUERY="ann_images"
                    run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"
                fi

            #################################################################
            #          ENN - Vary Batch Size (in-maxbench)
            #################################################################
            if [ "$run_scenario" = "enn-vary-batch-in-maxbench" ] || [ "$run_scenario" = "vary-batch-in-maxbench" ]; then
                if [ "$current_scale_factor" = "1" ]; then
                    # TOTAL_QUERIES=10000 # full sweep
                    TOTAL_QUERIES=0 # only one batch but nreps times
                else
                    # TOTAL_QUERIES=1000 # full sweep
                    TOTAL_QUERIES=0 # only one batch but nreps times
                fi
                VARY_BATCH_SIZES="1,10,100,1000,10000"
                # LOCAL_REPS=1 # <------ if full sweep
                LOCAL_REPS=3 # no need to do more

                if [ "$DEVICE" = "cpu" ]; then INDEX="Flat"; else INDEX="GPU,Flat"; fi
                BASE_PARAMS="k=${K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                EXTRA_TAG="vary_batch_in_maxbench"

                # avoiding: "enn_reviews"  due to large list issue on GPU
                # for QUERY in "ann_reviews" "enn_images" "ann_images" "pre_reviews" "pre_images" "pre_images_partitioned" "pre_reviews_partitioned"; do

                #---------------------------------------------------------------
                # Pure ENN (no filter)
                #---------------------------------------------------------------
                CURR_PARAMS="$BASE_PARAMS"
                run_benchmark "ann_reviews" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                    "$INDEX" "$CURR_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG" \
                    "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"

                #---------------------------------------------------------------
                # Pure PRE-FILTER QUERIES (main variant/rv_rating filter )
                #---------------------------------------------------------------

                # # keep reviews for paper because we can play with selecticity more easily:
                # for QUERY in "enn_reviews" "ann_reviews" "pre_reviews" "pre_reviews_partitioned"; do
                #     # Skip enn_reviews on all-GPU case 3 (large_list limitation)
                #     if [ "$CASE_ID" = "3" ] && [ "$QUERY" = "enn_reviews" ]; then
                #         continue
                #     fi
                #     CURR_PARAMS="$BASE_PARAMS"
                #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                #         "$INDEX" "$CURR_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG" \
                #         "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"
                # done
                # # HYBRID QUERY:
                # # TODO: add pre hybrid implementation
                # if [ "$CASE_ID" = "1" ]; then
                #     # for query in "pre_images_hybrid" "pre_reviews_hybrid"; do
                #     # keep reviews for paper because we can play with selecticity more easily:
                #     for query in "pre_reviews_hybrid"; do
                #         CURR_PARAMS="${BASE_PARAMS},use_limit_per_group=1"
                #         run_benchmark "$query" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                #             "$INDEX" "$CURR_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "${EXTRA_TAG}_pre_hybrid" \
                #             "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"
                #     done
                # fi

                # #---------------------------------------------------------------
                # # Image selectivity experiments (sf_1 only — thresholds calibrated for sf_1)
                # #---------------------------------------------------------------
                # if [ "$current_scale_factor" = "1" ]; then
                #     for SEL_KEY in "low" "high"; do
                #         if [ "$SEL_KEY" = "low" ]; then
                #             FILTER_PARTKEY=4000    # ~2% selectivity
                #         else
                #             FILTER_PARTKEY=180000  # ~90% selectivity
                #         fi
                #         CURR_PARAMS="${BASE_PARAMS},filter_partkey=${FILTER_PARTKEY}"
                #         run_benchmark "pre_images" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                #             "$INDEX" "$CURR_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" \
                #             "${EXTRA_TAG}-sel_${SEL_KEY}" \
                #             "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"
                #         # Hybrid (case 1 only)
                #         if [ "$CASE_ID" = "1" ]; then
                #             HYBRID_PARAMS="${BASE_PARAMS},filter_partkey=${FILTER_PARTKEY},use_limit_per_group=1"
                #             run_benchmark "pre_images_hybrid" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                #                 "$INDEX" "$HYBRID_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" \
                #                 "${EXTRA_TAG}_pre_hybrid-sel_${SEL_KEY}" \
                #                 "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"
                #         fi
                #     done
                # fi
            fi

            #################################################################
            #          ANN - Vary Batch Size (in-maxbench)
            #################################################################
            if [ "$run_scenario" = "ann-vary-batch-in-maxbench" ] || [ "$run_scenario" = "vary-batch-in-maxbench" ]; then
                if [ "$current_scale_factor" = "1" ]; then
                    # TOTAL_QUERIES=10000 # full sweep
                    TOTAL_QUERIES=0 # only one batch but nreps times
                else
                    # TOTAL_QUERIES=1000 # full sweep
                    TOTAL_QUERIES=0 # only one batch but nreps times
                fi
                VARY_BATCH_SIZES="1,10,100,1000,10000"
                # LOCAL_REPS=1 # since we go over all 10k queries... dont repeat each one # <--- IF FULL SWEEP
                LOCAL_REPS=50

                # # GPU,Cagra has a CPU version -> we can build the index move it to CPU and FAISS uses HNSWCagra index internally
                # CPU_ANN_INDEX_VBM=("IVF1024,Flat" "IVF2048,Flat" "IVF4096,Flat" "IVF8192,Flat" "IVF16384,Flat" "IVF1024,PQ64" "HNSW32,Flat" "GPU,Cagra,64,32,NN_DESCENT")
                # GPU_ANN_INDEX_VBM=("GPU,IVF1024,Flat" "GPU,IVF2048,Flat" "GPU,IVF4096,Flat" "GPU,IVF8192,Flat" "GPU,IVF16384,Flat" "GPU,IVF1024,PQ64" "GPU,Cagra,64,32,NN_DESCENT")

                CPU_ANN_INDEX_VBM=("IVF1024,Flat" "IVF4096,Flat" "GPU,CagraC,64,32,NN_DESCENT")
                GPU_ANN_INDEX_VBM=("GPU,IVF1024,Flat" "GPU,IVF4096,Flat" "GPU,IVFH1024,Flat" "GPU,IVFH4096,Flat" "GPU,CagraC,64,32,NN_DESCENT" "GPU,CagraCH,64,32,NN_DESCENT")

                if [ "$DEVICE" = "cpu" ]; then
                    INDEXES_VBM=("${CPU_ANN_INDEX_VBM[@]}")
                else
                    INDEXES_VBM=("${GPU_ANN_INDEX_VBM[@]}")
                fi

                for INDEX in "${INDEXES_VBM[@]}"; do
                    # Case 1P (cpu-pinned) is restricted to the CagraC index.
                    if [ "$CASE_ID" = "1P" ] && [[ "$INDEX" != *"CagraC,"* ]]; then
                        continue
                    fi
                    if [[ "$INDEX" == *"Cagra"* ]]; then
                        USE_CUVS=1
                    else
                        USE_CUVS=0
                    fi
                    if [[ "$INDEX" == *"PQ"* ]]; then
                        nprobe=80 # PQ128 needs more clusters to compensate for quantization error
                    elif [[ "$INDEX" == *"IVF4096"* || "$INDEX" == *"IVFH4096"* ]]; then
                        nprobe=15 # nlist=4096 needs slightly more probes than 1024 to reach ~90%
                    else
                        nprobe=11 # ~90% recall for nlist=1024
                    fi
                    k_vals=($(get_safe_k_vals "$K" "1" "$DEVICE"))
                    ANN_CAGRA_K=$(get_safe_itopk "$K" "$DEVICE")  # pure ANN: itopk just above k (no postfilter)
                    # post_* search params: higher nprobe/efsearch + larger k' to compensate for
                    # filter selectivity (postfilter_ksearch = 20x K so enough candidates survive the filter)
                    POST_NPROBE=$(( nprobe * 2 ))
                    POST_KSEARCH=$(( K * 20 ))
                    POST_HNSW_EFSEARCH=384 # images ~88%, reviews ~93%
                    POST_CAGRA_K=$(get_safe_itopk "$POST_KSEARCH" "$DEVICE")  # must be >= POST_KSEARCH
                    EXTRA_TAG="vary_batch_in_maxbench"

                    #---------------------------------------------------------------
                    # Pure ANN (no filter)
                    #---------------------------------------------------------------
                    CURR_PARAMS="k=${k_vals[0]},ivf_nprobe=${nprobe},hnsw_efsearch=64,postfilter_ksearch=${k_vals[1]},cagra_searchwidth=1,cagra_itopksize=${ANN_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                    run_benchmark "ann_reviews" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                        "$INDEX" "$CURR_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG" \
                        "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"

                    #---------------------------------------------------------------
                    # Pure POST-FILTER QUERIES (main variant/rv_rating filter )
                    #---------------------------------------------------------------

                    # # for QUERY in "ann_reviews" "ann_images" "post_reviews" "post_images" "post_images_partitioned" "post_reviews_partitioned"; do
                    # # keep reviews for paper because we can play with selecticity more easily:
                    # for QUERY in "ann_reviews" "post_reviews" "post_reviews_partitioned" "post_reviews_filter_partitioned"; do
                    #     if [[ "$QUERY" == post_* ]]; then
                    #         if [[ "$INDEX" == *"Cagra"* ]]; then
                    #             # Cagra constraint: postfilter_ksearch must equal cagra_itopksize (both capped at 1024)
                    #             CURR_PARAMS="k=${k_vals[0]},ivf_nprobe=${POST_NPROBE},hnsw_efsearch=${POST_HNSW_EFSEARCH},postfilter_ksearch=${POST_CAGRA_K},cagra_searchwidth=1,cagra_itopksize=${POST_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                    #         else
                    #             CURR_PARAMS="k=${k_vals[0]},ivf_nprobe=${POST_NPROBE},hnsw_efsearch=${POST_HNSW_EFSEARCH},postfilter_ksearch=${POST_KSEARCH},cagra_searchwidth=1,cagra_itopksize=${POST_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                    #         fi
                    #     else
                    #         # ann_reviews needs efsearch=200 to clear 95% (was 0.930 at 128); ann_images fine at 128
                    #         if [ "$QUERY" = "ann_reviews" ]; then
                    #             ann_efsearch=96
                    #         else
                    #             ann_efsearch=64
                    #         fi
                    #         CURR_PARAMS="k=${k_vals[0]},ivf_nprobe=${nprobe},hnsw_efsearch=${ann_efsearch},postfilter_ksearch=${k_vals[1]},cagra_searchwidth=1,cagra_itopksize=${ANN_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                    #     fi
                    #     if [[ "$DEVICE" = "gpu" && ( "$QUERY" = "post_reviews" || "$QUERY" = "post_images" ) ]]; then
                    #         CURR_PARAMS="${CURR_PARAMS},use_limit_per_group=1"
                    #     fi
                    #     run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                    #         "$INDEX" "$CURR_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG" \
                    #         "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"
                    # done

                    # # HYBRID QUERIES have special setup anyways.
                    # # - The post is the only one that requires index on GPU. The pre requires everything CPU.
                    # if [ "$CASE_ID" = "2" ]; then
                    #     if [[ "$INDEX" == *"Cagra"* ]]; then
                    #         # Cagra constraint: postfilter_ksearch must equal cagra_itopksize (both capped at 1024)
                    #         POST_PARAMS="k=${k_vals[0]},ivf_nprobe=${POST_NPROBE},hnsw_efsearch=${POST_HNSW_EFSEARCH},postfilter_ksearch=${POST_CAGRA_K},cagra_searchwidth=1,cagra_itopksize=${POST_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                    #     else
                    #         POST_PARAMS="k=${k_vals[0]},ivf_nprobe=${POST_NPROBE},hnsw_efsearch=${POST_HNSW_EFSEARCH},postfilter_ksearch=${POST_KSEARCH},cagra_searchwidth=1,cagra_itopksize=${POST_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                    #     fi
                    #     for query in "post_images_hybrid" "post_reviews_hybrid"; do
                    #         CURR_PARAMS="${POST_PARAMS},use_limit_per_group=0"
                    #         run_benchmark "$query" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                    #             "$INDEX" "$CURR_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "${EXTRA_TAG}_post_hybrid" \
                    #             "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"
                    #     done
                    # fi

                    #---------------------------------------------------------------
                    # Image selectivity experiments (sf_1 only — thresholds calibrated for sf_1)
                    #---------------------------------------------------------------
                    # if [ "$current_scale_factor" = "1" ]; then
                    #     for SEL_KEY in "low" "high"; do
                    #         if [ "$SEL_KEY" = "low" ]; then
                    #             FILTER_PARTKEY=4000    # ~2% selectivity
                    #         else
                    #             FILTER_PARTKEY=180000  # ~90% selectivity
                    #         fi
                    #         # post_images with selectivity
                    #         if [[ "$INDEX" == *"Cagra"* ]]; then
                    #             SEL_PARAMS="k=${k_vals[0]},ivf_nprobe=${POST_NPROBE},hnsw_efsearch=${POST_HNSW_EFSEARCH},postfilter_ksearch=${POST_CAGRA_K},cagra_searchwidth=1,cagra_itopksize=${POST_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC},filter_partkey=${FILTER_PARTKEY}"
                    #         else
                    #             SEL_PARAMS="k=${k_vals[0]},ivf_nprobe=${POST_NPROBE},hnsw_efsearch=${POST_HNSW_EFSEARCH},postfilter_ksearch=${POST_KSEARCH},cagra_searchwidth=1,cagra_itopksize=${POST_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC},filter_partkey=${FILTER_PARTKEY}"
                    #         fi
                    #         if [ "$DEVICE" = "gpu" ]; then
                    #             SEL_PARAMS="${SEL_PARAMS},use_limit_per_group=1"
                    #         fi
                    #         run_benchmark "post_images" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                    #             "$INDEX" "$SEL_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" \
                    #             "${EXTRA_TAG}-sel_${SEL_KEY}" \
                    #             "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"
                    #         # Hybrid (case 2 only)
                    #         if [ "$CASE_ID" = "2" ]; then
                    #             HYBRID_SEL_PARAMS="${SEL_PARAMS//use_limit_per_group=1/use_limit_per_group=0}"
                    #             run_benchmark "post_images_hybrid" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                    #                 "$INDEX" "$HYBRID_SEL_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" \
                    #                 "${EXTRA_TAG}_post_hybrid-sel_${SEL_KEY}" \
                    #                 "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"
                    #         fi
                    #     done
                    # fi
                done
            fi


            #################################################################
            #          TESTING ANN - Vary Batch Size (in-maxbench)
            #################################################################
            if [ "$run_scenario" = "TEST-ann-vary-batch-in-maxbench" ]; then
                if [ "$current_scale_factor" = "1" ]; then
                    # TOTAL_QUERIES=10000 # full sweep
                    TOTAL_QUERIES=0 # only one batch but nreps times
                else
                    # TOTAL_QUERIES=1000 # full sweep
                    TOTAL_QUERIES=0 # only one batch but nreps times
                fi
                VARY_BATCH_SIZES="10000"
                LOCAL_REPS=1
                # LOCAL_REPS=10

                # GPU,Cagra has a CPU version -> we can build the index move it to CPU and FAISS uses HNSWCagra index internally
                CPU_ANN_INDEX_VBM=("IVF1024,Flat" "IVF4096,Flat")
                GPU_ANN_INDEX_VBM=("GPU,IVF1024,Flat" "GPU,IVF4096,Flat" "GPU,IVFH1024,Flat" "GPU,IVFH4096,Flat")

                if [ "$DEVICE" = "cpu" ]; then
                    INDEXES_VBM=("${CPU_ANN_INDEX_VBM[@]}")
                else
                    INDEXES_VBM=("${GPU_ANN_INDEX_VBM[@]}")
                fi

                for INDEX in "${INDEXES_VBM[@]}"; do
                    # Case 1P (cpu-pinned) is restricted to the CagraC index.
                    if [ "$CASE_ID" = "1P" ] && [[ "$INDEX" != *"CagraC,"* ]]; then
                        continue
                    fi
                    if [[ "$INDEX" == *"Cagra"* ]]; then
                        USE_CUVS=1
                    else
                        USE_CUVS=0
                    fi
                    if [[ "$INDEX" == *"PQ"* ]]; then
                        nprobe=80 # PQ128 needs more clusters to compensate for quantization error
                    else
                        nprobe=11 # ~90% recall for nlist=1024
                    fi
                    k_vals=($(get_safe_k_vals "$K" "1" "$DEVICE"))
                    ANN_CAGRA_K=$(get_safe_itopk "$K" "$DEVICE")  # pure ANN: itopk just above k (no postfilter)
                    # post_* search params: higher nprobe/efsearch + larger k' to compensate for
                    # filter selectivity (postfilter_ksearch = 20x K so enough candidates survive the filter)
                    POST_NPROBE=$(( nprobe * 2 ))
                    POST_KSEARCH=$(( K * 20 ))
                    POST_HNSW_EFSEARCH=384 # images ~88%, reviews ~93%
                    POST_CAGRA_K=$(get_safe_itopk "$POST_KSEARCH" "$DEVICE")  # must be >= POST_KSEARCH
                    EXTRA_TAG="vary_batch_in_maxbench"

                    # for QUERY in "ann_reviews" "ann_images" "post_reviews" "post_images" "post_images_partitioned" "post_reviews_partitioned"; do # original
                    # for QUERY in "post_reviews" "post_images"; do
                    for QUERY in "post_images"; do
                        if [[ "$QUERY" == post_* ]]; then
                            if [[ "$INDEX" == *"Cagra"* ]]; then
                                # Cagra constraint: postfilter_ksearch must equal cagra_itopksize (both capped at 1024)
                                CURR_PARAMS="k=${k_vals[0]},ivf_nprobe=${POST_NPROBE},hnsw_efsearch=${POST_HNSW_EFSEARCH},postfilter_ksearch=${POST_CAGRA_K},cagra_searchwidth=1,cagra_itopksize=${POST_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                            else
                                CURR_PARAMS="k=${k_vals[0]},ivf_nprobe=${POST_NPROBE},hnsw_efsearch=${POST_HNSW_EFSEARCH},postfilter_ksearch=${POST_KSEARCH},cagra_searchwidth=1,cagra_itopksize=${POST_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                            fi
                        else
                            # ann_reviews needs efsearch=200 to clear 95% (was 0.930 at 128); ann_images fine at 128
                            if [ "$QUERY" = "ann_reviews" ]; then
                                ann_efsearch=96
                            else
                                ann_efsearch=64
                            fi
                            CURR_PARAMS="k=${k_vals[0]},ivf_nprobe=${nprobe},hnsw_efsearch=${ann_efsearch},postfilter_ksearch=${k_vals[1]},cagra_searchwidth=1,cagra_itopksize=${ANN_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                        fi
                        if [[ "$DEVICE" = "gpu" && ( "$QUERY" = "post_reviews" || "$QUERY" = "post_images" ) ]]; then
                            CURR_PARAMS="${CURR_PARAMS},use_limit_per_group=1"
                        fi
                        run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                            "$INDEX" "$CURR_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG" \
                            "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"
                    done

                    # HYBRID QUERIES have special setup anyways.
                    # - The post is the only one that requires index on GPU. The pre requires everything CPU.
                    if [ "$CASE_ID" = "2" ]; then
                        if [[ "$INDEX" == *"Cagra"* ]]; then
                            # Cagra constraint: postfilter_ksearch must equal cagra_itopksize (both capped at 1024)
                            POST_PARAMS="k=${k_vals[0]},ivf_nprobe=${POST_NPROBE},hnsw_efsearch=${POST_HNSW_EFSEARCH},postfilter_ksearch=${POST_CAGRA_K},cagra_searchwidth=1,cagra_itopksize=${POST_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                        else
                            POST_PARAMS="k=${k_vals[0]},ivf_nprobe=${POST_NPROBE},hnsw_efsearch=${POST_HNSW_EFSEARCH},postfilter_ksearch=${POST_KSEARCH},cagra_searchwidth=1,cagra_itopksize=${POST_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"
                        fi
                        for query in "post_images_hybrid" "post_reviews_hybrid"; do
                            for LPG in 0 1; do
                                CURR_PARAMS="${POST_PARAMS},use_limit_per_group=0"
                                run_benchmark "$query" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                                    "$INDEX" "$CURR_PARAMS" "$LOCAL_REPS" "$current_scale_factor" "$run_scenario" "${EXTRA_TAG}_post_hybrid--limit_per_group_${LPG}" \
                                    "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES"
                            done
                        done
                    fi
                done
            fi


            #################################################################
            #          BS1 FULL SWEEP (CASE 6 — index persists on GPU)
            #################################################################
            # Batch size 1 full sweep of 10k distinct queries (ann_reviews, ann_images).
            # Uses CASE 6: index is swapped to GPU once via --case6_persist_gpu_index
            # (wrapped in the 'setup_index_movement' Caliper region). All per-query iterations
            # then skip the move — only the 1-row post-limit query embedding moves to GPU.
            # Rep 0 is warmup (discarded by parse convention); rep 1 is the measured query.
            if [ "$run_scenario" = "bs1_fullsweep" ]; then
                # The outer case loop iterates global CASES (0,1,2,3). bs1_fullsweep defines its
                # own scenario-local cases below. Gate to outer CASE_ID==0 so the block runs
                # only once per scenario iteration (then overrides CASE_ID locally).
                if [ "$CASE_ID" != "0" ]; then continue; fi
                # Scenario-local cases — does NOT modify the global CASES array.
                # Case 0 (CPU baseline, no index move) for comparison vs Case 6 (GPU index persist).
                RUN_SCENARIO_CASES=("0 cpu cpu cpu" "6 gpu cpu cpu")
                for case_config in "${RUN_SCENARIO_CASES[@]}"; do
                    read -r CASE_ID DEVICE STORAGE_DEVICE INDEX_STORAGE_DEVICE <<< "$case_config"
                    if [ "$RUN_CASE" != "all" ] && [ "$RUN_CASE" != "$CASE_ID" ]; then continue; fi

                    VARY_BATCH_SIZES="1"
                    LOCAL_REPS=2                        # rep 0 warmup (discarded), rep 1 measured
                    EXTRA_TAG="bs1_fullsweep"
                    if [ "$current_scale_factor" = "1" ]; then
                        ANN_TOTAL_QUERIES=10000
                    else
                        ANN_TOTAL_QUERIES=1000
                    fi

                    if [ "$DEVICE" = "cpu" ]; then
                        # CPU equivalents: ENN=Flat; ANN=IVF/HNSW-Cagra (auto from GPU,Cagra).
                        INDEX_LIST=("Flat" "IVF1024,Flat" "IVF4096,Flat" "GPU,Cagra,64,32,NN_DESCENT")
                        CASE6_MODE_FLAG=""   # CASE 0: no index move, no flag
                    else
                        # GPU full matrix (CASE 6: persist index on GPU).
                        INDEX_LIST=("GPU,Flat" "GPU,IVF1024,Flat" "GPU,IVF4096,Flat" "GPU,IVFH1024,Flat" "GPU,IVFH4096,Flat" "GPU,Cagra,64,32,NN_DESCENT" "GPU,CagraCH,64,32,NN_DESCENT")
                        CASE6_MODE_FLAG="true"
                    fi

                    for INDEX in "${INDEX_LIST[@]}"; do
                        # ENN (Flat / GPU,Flat) is constant per-query — 3 reps + plot extrapolates.
                        # ANN variants have real steady-state shape — full sweep.
                        if [ "$INDEX" = "Flat" ] || [ "$INDEX" = "GPU,Flat" ]; then
                            TOTAL_QUERIES=5 # enn doesnt really vary...
                        else
                            TOTAL_QUERIES=$ANN_TOTAL_QUERIES
                        fi

                        if [[ "$INDEX" == *"Cagra"* ]]; then
                            USE_CUVS=1
                        elif [[ "$INDEX" == *"IVF"* ]]; then
                            USE_CUVS=0
                        else
                            USE_CUVS=1   # Flat uses cuVS brute-force (non-interleaved, safe)
                        fi
                        if [[ "$INDEX" == *"IVF4096"* || "$INDEX" == *"IVFH4096"* ]]; then
                            nprobe=15 # nlist=4096 needs slightly more probes than 1024 to reach ~90%
                        else
                            nprobe=11 # ~90% recall for nlist=1024
                        fi
                        k_vals=($(get_safe_k_vals "$K" "1" "$DEVICE"))
                        ANN_CAGRA_K=$(get_safe_itopk "$K" "$DEVICE")

                        CURR_PARAMS="k=${k_vals[0]},ivf_nprobe=${nprobe},hnsw_efsearch=64,postfilter_ksearch=${k_vals[1]},cagra_searchwidth=1,cagra_itopksize=${ANN_CAGRA_K},use_cuvs=${USE_CUVS},metric=${METRIC}"

                        for Q in "ann_reviews"; do  # SMOKE: ann_images skipped
                            run_benchmark "$Q" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" \
                                "$INDEX" "$CURR_PARAMS" "$LOCAL_REPS" "$current_scale_factor" \
                                "$run_scenario" "$EXTRA_TAG" "true" "$VARY_BATCH_SIZES" "$TOTAL_QUERIES" \
                                "$CASE6_MODE_FLAG" \
                                "true"   # $15 = FORCE_NO_CSV (bs1_fullsweep never persists CSVs)
                        done
                    done
                done
            fi


            #################################################################
            #                     ANN - Full Batch
            #################################################################
            if [ "$run_scenario" = "ann-batch" ] || [ "$run_scenario" = "all" ] || [ "$run_scenario" = "main" ]; then
                if [ "$current_scale_factor" = "1" ]; then
                    QCNT=10000
                else
                    QCNT=1000
                fi
                
                CPU_ANN_INDEX=("IVF1024,Flat" "IVF4096,Flat" "IVF1024,PQ32" "HNSW32,Flat" "GPU,Cagra,64,32,NN_DESCENT")
                GPU_ANN_INDEX=("GPU,IVF1024,Flat" "GPU,IVF4096,Flat" "GPU,IVF1024,PQ32" "GPU,Cagra,64,32,NN_DESCENT" "GPU,CagraCH,64,32,NN_DESCENT")

                EXTRA_TAG="k_${K}-batch_${QCNT}"

                if [ "$DEVICE" = "cpu" ]; then
                    INDEXES=("${CPU_ANN_INDEX[@]}")
                else
                    INDEXES=("${GPU_ANN_INDEX[@]}")
                fi

                for INDEX in "${INDEXES[@]}"; do
                    PARAMS="k=${K},query_count=${QCNT},query_start=0,ivf_nprobe=30,hnsw_efsearch=128,cagra_searchwidth=1,cagra_itopksize=128,use_cuvs=${USE_CUVS},metric=${METRIC}"

                    QUERY="ann_reviews"
                    run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"

                    QUERY="ann_images"
                    run_benchmark "$QUERY" "$DEVICE" "$STORAGE_DEVICE" "$INDEX_STORAGE_DEVICE" "$INDEX" "$PARAMS" "$REPS" "$current_scale_factor" "$run_scenario" "$EXTRA_TAG"
                done
            fi

        done # close "case" config loop

    done # close run scenario loop

    # TODO: add more as needed
    echo "Completed benchmarks for scale factor: ${current_scale_factor}"

done # close scale factor loop

echo "All benchmarks completed."
