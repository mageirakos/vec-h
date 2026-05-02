#!/bin/bash
# quick_run_batch.sh — List of quick runs to execute back-to-back.
# Just uncomment/add lines and run: ./quick_run_batch.sh
# Runs sequentially to avoid OOM.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

COMMON_PARAMS="k=100,query_count=1,query_start=0,postfilter_ksearch=100,cagra_itopksize=128,ivf_nprobe=30,hnsw_efsearch=128,metric=IP"

echo "=== Quick Run Batch ==="
echo ""

# ============================================================================
# CAGRA variants (use_cuvs=1)
# ============================================================================

# # ---- Cagra: data=0, cache=0 (normal copyFrom_ex, data on host) ----
# "$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10--_data0_cache0 \
#   --queries q10_mid --index 'GPU,Cagra,64,32,NN_DESCENT' \
#   --storage_device cpu --index_storage_device cpu --device gpu \
#   --sf 1 --n_reps 4 \
#   --params "${COMMON_PARAMS},use_cuvs=1,index_data_on_gpu=0,cagra_cache_graph=0"

# # ---- Cagra(C+H): data=0, cache=1 (graph cache + host view — optimized) ----
# "$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10--_ch_data0_cache1 \
#   --queries q10_mid --index 'GPU,Cagra,64,32,NN_DESCENT' \
#   --storage_device cpu --index_storage_device cpu --device gpu \
#   --sf 1 --n_reps 4 \
#   --params "${COMMON_PARAMS},use_cuvs=1,index_data_on_gpu=0,cagra_cache_graph=1"

# # ---- Cagra: data=1, cache=1 (graph cache + GPU copy) ----
# "$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10--_data1_cache1 \
#   --queries q10_mid --index 'GPU,Cagra,64,32,NN_DESCENT' \
#   --storage_device cpu --index_storage_device cpu --device gpu \
#   --sf 1 --n_reps 4 \
#   --params "${COMMON_PARAMS},use_cuvs=1,index_data_on_gpu=1,cagra_cache_graph=1"

# # ---- Cagra: data=1, cache=0 (normal copyFrom_ex + GPU copy — default) ----
# "$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10--_data1_cache0 \
#   --queries q10_mid --index 'GPU,Cagra,64,32,NN_DESCENT' \
#   --storage_device cpu --index_storage_device cpu --device gpu \
#   --sf 1 --n_reps 4 \
#   --params "${COMMON_PARAMS},use_cuvs=1,index_data_on_gpu=1,cagra_cache_graph=0"

# # ---- Cagra: storage=gpu (everything on GPU, no movement) ----
# "$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10--_storagegpu \
#   --queries q10_mid --index 'GPU,Cagra,64,32,NN_DESCENT' \
#   --storage_device cpu --index_storage_device gpu --device gpu \
#   --sf 1 --n_reps 4 \
#   --params "${COMMON_PARAMS},use_cuvs=1,index_data_on_gpu=1,cagra_cache_graph=0"

# ============================================================================
# IVF variants (use_cuvs=0, required for non-interleaved layout)
# ============================================================================

# # ---- IVF: data=1 (normal copyInvertedListsFrom — default) ----
# "$SCRIPT_DIR/quick_run.sh" -l ivf_data1 \
#   --queries ann_reviews --index 'GPU,IVF1024,Flat' \
#   --storage_device cpu --index_storage_device cpu --device gpu \
#   --sf 1 --n_reps 4 \
#   --params "${COMMON_PARAMS},use_cuvs=0,index_data_on_gpu=1"

# # ---- IVF(H): data=0 (host view via referenceFrom — optimized) ----
# "$SCRIPT_DIR/quick_run.sh" -l ivf_host_view \
#   --queries ann_reviews --index 'GPU,IVF1024,Flat' \
#   --storage_device cpu --index_storage_device cpu --device gpu \
#   --sf 1 --n_reps 4 \
#   --params "${COMMON_PARAMS},use_cuvs=0,index_data_on_gpu=0"

# # ---- IVF: storage=gpu (everything on GPU, no movement) ----
# "$SCRIPT_DIR/quick_run.sh" -l ivf_storagegpu \
#   --queries ann_reviews --index 'GPU,IVF1024,Flat' \
#   --storage_device cpu --index_storage_device gpu --device gpu \
#   --sf 1 --n_reps 4 \
#   --params "${COMMON_PARAMS},use_cuvs=0,index_data_on_gpu=1"


# # ============================================================================
# # SF variant check (synthetic datasets)
# # ============================================================================
#
# # ---- SF-1 synthetic
# "$SCRIPT_DIR/quick_run.sh" -l sf1-q13-enn \
#   --queries q13_mid --index 'GPU,Flat' \
#   --storage_device gpu --index_storage_device gpu --device gpu \
#   --sf 1 --n_reps 10 --dataset "$DATASET" \
#   --params "${COMMON_PARAMS},use_cuvs=0,index_data_on_gpu=0"
#
# "$SCRIPT_DIR/quick_run.sh" -l sf1-q13-ivf \
#   --queries q13_mid --index 'GPU,IVF1024,Flat' \
#   --storage_device gpu --index_storage_device gpu --device gpu \
#   --sf 1 --n_reps 10 --dataset "$DATASET" \
#   --params "${COMMON_PARAMS},use_cuvs=0,index_data_on_gpu=1"
#
# "$SCRIPT_DIR/quick_run.sh" -l sf1-q13-cagra \
#   --queries q13_mid --index 'GPU,Cagra,64,32,NN_DESCENT' \
#   --storage_device gpu --index_storage_device gpu --device gpu \
#   --sf 1 --n_reps 10 --dataset "$DATASET" \
#   --params "${COMMON_PARAMS},use_cuvs=1,index_data_on_gpu=1"
#
# INDEX_CACHE="/tmp/maximus_faiss_cache"
#
# # ---- SF-10 synthetic
# "$SCRIPT_DIR/quick_run.sh" -l sf10-q13-enn \
#   --queries q13_mid --index 'GPU,Flat' \
#   --storage_device gpu --index_storage_device gpu --device gpu \
#   --sf 10 --n_reps 10 --dataset "$DATASET" \
#   --use_index_cache_dir "$INDEX_CACHE" \
#   --params "${COMMON_PARAMS},use_cuvs=0,index_data_on_gpu=0"
#
# "$SCRIPT_DIR/quick_run.sh" -l sf10-q13-ivf \
#   --queries q13_mid --index 'GPU,IVF1024,Flat' \
#   --storage_device gpu --index_storage_device gpu --device gpu \
#   --sf 10 --n_reps 10 --dataset "$DATASET" \
#   --use_index_cache_dir "$INDEX_CACHE" \
#   --params "${COMMON_PARAMS},use_cuvs=0,index_data_on_gpu=1"
#
# "$SCRIPT_DIR/quick_run.sh" -l sf10-q13-cagra \
#   --queries q13_mid --index 'GPU,Cagra,64,32,NN_DESCENT' \
#   --storage_device gpu --index_storage_device gpu --device gpu \
#   --sf 10 --n_reps 10 --dataset "$DATASET" \
#   --use_index_cache_dir "$INDEX_CACHE" \
#   --params "${COMMON_PARAMS},use_cuvs=1,index_data_on_gpu=1"
#
# # ---- SF-30 synthetic
# "$SCRIPT_DIR/quick_run.sh" -l sf30-q13-enn \
#   --queries q13_mid --index 'GPU,Flat' \
#   --storage_device gpu --index_storage_device gpu --device gpu \
#   --sf 30 --n_reps 10 --dataset "$DATASET" \
#   --use_index_cache_dir "$INDEX_CACHE" \
#   --params "${COMMON_PARAMS},use_cuvs=0,index_data_on_gpu=0"
#
# "$SCRIPT_DIR/quick_run.sh" -l sf30-q13-ivf \
#   --queries q13_mid --index 'GPU,IVF1024,Flat' \
#   --storage_device gpu --index_storage_device gpu --device gpu \
#   --sf 30 --n_reps 10 --dataset "$DATASET" \
#   --use_index_cache_dir "$INDEX_CACHE" \
#   --params "${COMMON_PARAMS},use_cuvs=0,index_data_on_gpu=1"
#
# "$SCRIPT_DIR/quick_run.sh" -l sf30-q13-cagra \
#   --queries q13_mid --index 'GPU,Cagra,64,32,NN_DESCENT' \
#   --storage_device gpu --index_storage_device gpu --device gpu \
#   --sf 30 --n_reps 10 --dataset "$DATASET" \
#   --use_index_cache_dir "$INDEX_CACHE" \
#   --params "${COMMON_PARAMS},use_cuvs=1,index_data_on_gpu=1"

# ============================================================================
# All VSDS queries — Cagra (Case 1: GPU device, CPU storage, CPU index storage)
#  VARY SF - synthetic quick tests 
# ============================================================================
INDEX_CACHE="/tmp/maximus_faiss_cache"
DATASET="synthetic"
# DATASET="industrial_and_scientific"
CAGRA_INDEX='GPU,Cagra,64,32,NN_DESCENT'
CAGRA_PARAMS="use_cuvs=1,index_data_on_gpu=0"
SF="10"
REPS="10"

# q2_start (use_post=1 for ANN)
"$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10-q2_start \
  --queries q2_start --index "$CAGRA_INDEX" \
  --device gpu --storage_device cpu --index_storage_device cpu \
  --sf "$SF" --n_reps "$REPS" --dataset "$DATASET" \
  --use_index_cache_dir "$INDEX_CACHE" \
  --params "k=100,query_count=1,query_start=0,postfilter_ksearch=1000,cagra_itopksize=1024,ivf_nprobe=30,hnsw_efsearch=128,${CAGRA_PARAMS},metric=IP,use_post=1"

# q10_mid
"$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10-q10_mid \
  --queries q10_mid --index "$CAGRA_INDEX" \
  --device gpu --storage_device cpu --index_storage_device cpu \
  --sf "$SF" --n_reps "$REPS" --dataset "$DATASET" \
  --use_index_cache_dir "$INDEX_CACHE" \
  --params "k=100,query_count=1,query_start=0,postfilter_ksearch=100,cagra_itopksize=128,ivf_nprobe=30,hnsw_efsearch=128,${CAGRA_PARAMS},metric=IP"

# q11_end (fixed k=1050 for sf>=1, use_limit_per_group=1 on GPU)
"$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10-q11_end \
  --queries q11_end --index "$CAGRA_INDEX" \
  --device gpu --storage_device cpu --index_storage_device cpu \
  --sf "$SF" --n_reps "$REPS" --dataset "$DATASET" \
  --use_index_cache_dir "$INDEX_CACHE" \
  --params "k=1050,query_count=1,query_start=0,postfilter_ksearch=500,cagra_itopksize=128,ivf_nprobe=30,hnsw_efsearch=128,${CAGRA_PARAMS},metric=IP,use_limit_per_group=1"

# q13_mid
"$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10-q13_mid \
  --queries q13_mid --index "$CAGRA_INDEX" \
  --device gpu --storage_device cpu --index_storage_device cpu \
  --sf "$SF" --n_reps "$REPS" --dataset "$DATASET" \
  --use_index_cache_dir "$INDEX_CACHE" \
  --params "k=100,query_count=1,query_start=0,postfilter_ksearch=100,cagra_itopksize=128,ivf_nprobe=30,hnsw_efsearch=128,${CAGRA_PARAMS},metric=IP"

# q15_end (use_post=1, use_cuvs=1 required on GPU)
"$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10-q15_end \
  --queries q15_end --index "$CAGRA_INDEX" \
  --device gpu --storage_device cpu --index_storage_device cpu \
  --sf "$SF" --n_reps "$REPS" --dataset "$DATASET" \
  --use_index_cache_dir "$INDEX_CACHE" \
  --params "k=100,query_count=1,query_start=0,postfilter_ksearch=1000,cagra_itopksize=1024,ivf_nprobe=30,hnsw_efsearch=128,${CAGRA_PARAMS},metric=IP,use_post=1"

# q16_start
"$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10-q16_start \
  --queries q16_start --index "$CAGRA_INDEX" \
  --device gpu --storage_device cpu --index_storage_device cpu \
  --sf "$SF" --n_reps "$REPS" --dataset "$DATASET" \
  --use_index_cache_dir "$INDEX_CACHE" \
  --params "k=100,query_count=1,query_start=0,postfilter_ksearch=100,cagra_itopksize=128,ivf_nprobe=30,hnsw_efsearch=128,${CAGRA_PARAMS},metric=IP"

# q18_mid (use_post=1 for ANN)
"$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10-q18_mid \
  --queries q18_mid --index "$CAGRA_INDEX" \
  --device gpu --storage_device cpu --index_storage_device cpu \
  --sf "$SF" --n_reps "$REPS" --dataset "$DATASET" \
  --use_index_cache_dir "$INDEX_CACHE" \
  --params "k=100,query_count=1,query_start=0,postfilter_ksearch=1000,cagra_itopksize=1024,ivf_nprobe=30,hnsw_efsearch=128,${CAGRA_PARAMS},metric=IP,use_post=1"

# q19_start (use_post=1 for ANN)
"$SCRIPT_DIR/quick_run.sh" -l cagra-syn-sf10-q19_start \
  --queries q19_start --index "$CAGRA_INDEX" \
  --device gpu --storage_device cpu --index_storage_device cpu \
  --sf "$SF" --n_reps "$REPS" --dataset "$DATASET" \
  --use_index_cache_dir "$INDEX_CACHE" \
  --params "k=100,query_count=1,query_start=0,postfilter_ksearch=100,cagra_itopksize=128,ivf_nprobe=30,hnsw_efsearch=128,${CAGRA_PARAMS},metric=IP,use_post=1"

echo ""
echo "=== All Quick Runs Complete ==="
echo "Results: results/0_quick_run/"
ls -1d "$SCRIPT_DIR/0_quick_run"/*/ 2>/dev/null || echo "(no results yet)"
