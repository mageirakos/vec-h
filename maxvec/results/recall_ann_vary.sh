#!/bin/bash

#################################################################
# Script to calculate recall after running run_vsds.sh
#################################################################

SYSTEM='dgx-spark-02'
SCALE_FACTORS=("0.01" "1")
QUERIES=("ann_reviews" "ann_images")
K=100
METRIC="IP"

RESULTS_BASE="${1:-./results}"

PYTHON_SCRIPT="./results/recall_from_two_csvs.py"

echo "============================================================"
echo " Starting Recall Calculation Pipeline"
echo " System: $SYSTEM"
echo "============================================================"

shopt -s nullglob

for SF in "${SCALE_FACTORS[@]}"; do
    echo ""
    echo "Processing Scale Factor: $SF"
    echo "------------------------------------------------------------"

    for QUERY in "${QUERIES[@]}"; do
        
        GT_QUERY="${QUERY/ann/enn}"
        
        # 1. Define the paths to the CPU Ground Truth (GT) file
        CPU_GT_DIR="${RESULTS_BASE}/other-enn-ground-truth/cpu-${SYSTEM}/sf_${SF}/csv"
        CPU_GT_MATCH=( "${CPU_GT_DIR}/"*q_${GT_QUERY}-i_Flat-d_cpu-s_cpu-is_cpu-sf_${SF}-k_${K}_metric_${METRIC}_ground_truth.csv )
        CPU_GT_CSV="${CPU_GT_MATCH[0]}"
        
        # Check if CPU GT file exists before proceeding
        if [ ! -f "$CPU_GT_CSV" ]; then
            echo "⚠️  Missing CPU Ground Truth file for ${QUERY} at SF=${SF}. Skipping..."
            continue
        fi

        # Make CPU Ground Truth the absolute master reference for everything
        REFERENCE_GT="$CPU_GT_CSV"
        echo " [i] Using CPU Ground Truth as the master reference for ${QUERY}"

        # Look for the GPU GT file just for the comparison check
        GPU_GT_DIR="${RESULTS_BASE}/other-enn-ground-truth/gpu-${SYSTEM}/sf_${SF}/csv"
        GPU_GT_MATCH=( "${GPU_GT_DIR}/"*q_${GT_QUERY}-i_Flat-d_gpu-s_gpu-is_gpu-sf_${SF}-k_${K}_metric_${METRIC}_ground_truth.csv )
        GPU_GT_CSV="${GPU_GT_MATCH[0]}"

        # ==========================================================
        # (a) Calculate recall between CPU GT and GPU GT (if GPU exists)
        # ==========================================================
        if [ -f "$GPU_GT_CSV" ]; then
            GT_RECALL_DIR="${GPU_GT_DIR}/recall"
            mkdir -p "$GT_RECALL_DIR"
            GT_COMPARE_OUT="${GT_RECALL_DIR}/recall_cpu_gt_vs_gpu_gt_${QUERY}.txt"
            
            echo " > Comparing CPU GT vs GPU GT for ${QUERY}..."
            python "$PYTHON_SCRIPT" "$CPU_GT_CSV" "$GPU_GT_CSV" > "$GT_COMPARE_OUT"
        else
            echo " > [!] GPU GT missing (likely cuDF int32 limit). Skipping CPU vs GPU GT comparison."
        fi

        # ==========================================================
        # (b) Calculate recall for all Indexes against the REFERENCE_GT
        # ==========================================================
        SCENARIOS=("ann-vary-ivf" "ann-vary-hnsw")
        
        for SCENARIO in "${SCENARIOS[@]}"; do
            for DEVICE in "cpu" "gpu"; do
                
                INDEX_CSV_DIR="${RESULTS_BASE}/other-${SCENARIO}/${DEVICE}-${SYSTEM}/sf_${SF}/csv"
                
                if [ -d "$INDEX_CSV_DIR" ]; then
                    RECALL_OUT_DIR="${INDEX_CSV_DIR}/recall"
                    mkdir -p "$RECALL_OUT_DIR"

                    for INDEX_CSV in "$INDEX_CSV_DIR"/*q_${QUERY}*nprobe*.csv \
                                     "$INDEX_CSV_DIR"/*q_${QUERY}*efsearch*.csv \
                                     "$INDEX_CSV_DIR"/*q_${QUERY}*itopk*.csv; do
                        
                        BASENAME=$(basename "$INDEX_CSV" .csv)
                        RECALL_FILE="${RECALL_OUT_DIR}/recall_${BASENAME}.txt"
                        
                        echo "   -> Calc recall: ${BASENAME}"
                        # Using the CPU Master Reference
                        python "$PYTHON_SCRIPT" "$REFERENCE_GT" "$INDEX_CSV" > "$RECALL_FILE"
                    done
                fi
            done
        done
        echo " ✓ Finished ${QUERY} index recalls."
    done
done

shopt -u nullglob

echo ""
echo "============================================================"
echo " All recall calculations completed successfully!"
echo "============================================================"