#!/bin/bash
#SBATCH --job-name=gh200-parallel-independent
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --output=DEBUG-%j.log
#SBATCH --error=DEBUG-%j.err
#SBATCH --partition=debug               # <- partition debug for quick runs;
#SBATCH --environment=maxvec
#SBATCH --container-writable
# NUMA note: CSCS SLURM auto-places tasks NUMA-locally with --exclusive --gpus-per-task.
# quick_run.sh prints taskset output to confirm correct core/GPU affinity.

ulimit -c 0
export MAXIMUS_NUM_OUTER_THREADS=1
export MAXIMUS_NUM_INNER_THREADS=72
export MAXIMUS_NUM_IO_THREADS=72
export MAXIMUS_OPERATORS_FUSION=0
export MAXIMUS_DATASET_API=0
export LD_LIBRARY_PATH="/opt/conda/envs/maximus_env/lib:${LD_LIBRARY_PATH}"


# --- Thread binding (72 cores per NUMA node) ---
# Each srun task gets --cpus-per-task=72 (one NUMA node).
# These exports ensure OpenMP/BLAS respect that boundary.
export OMP_NUM_THREADS=72
export OPENBLAS_NUM_THREADS=1


cd $SCRATCH/maxvec/Maximus

ninja -C ./build/Release

# ==============================================================================
# The Magic Formula for the srun lines:
# --exclusive        : Tells Slurm to carve out a dedicated chunk of the node.
# --ntasks=1         : This specific command is 1 task.
# --gpus-per-task=1  : Give this task 1 GPU.
# --cpus-per-task=72 : Give this task 72 Cores.
# &                  : (AT THE VERY END) Run this in the background!
# ==============================================================================

COMMON_PARAMS="k=100,query_count=1,query_start=0,postfilter_ksearch=100,cagra_itopksize=128,ivf_nprobe=30,hnsw_efsearch=128,use_cuvs=1,metric=IP"

# RUN 1
# #### QUICK RUN:
srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --output="IND-TEST-1.log" \
     $SCRATCH/maxvec/Maximus/results/quick_run.sh -l cagra_T1_data0_cache0 --nsys \
     --queries ann_reviews --index 'GPU,Cagra,64,32,NN_DESCENT' \
     --storage_device cpu --index_storage_device cpu --device gpu \
     --sf 1 --n_reps 10 \
     --params ${COMMON_PARAMS},index_data_on_gpu=0,cagra_cache_graph=0 &

# ### MAXBENCH DIRECT:
# srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --output="IND-TEST-1.log" \
#      $SCRATCH/maxvec/Maximus/build/Release/benchmarks/maxbench \
#      --benchmark vsds --queries ann_reviews \
#      --storage_device cpu --index_storage_device cpu --device gpu --flush_cache \
#      --n_reps 10 \
#      --index GPU,Flat \
#      --params k=100,query_count=1,query_start=0,use_cuvs=1,metric=IP \
#      --profile "nvtx,runtime-report(calc.inclusive=true,output=stdout)" \
#      --path $SCRATCH/maxvec/Maximus/tests/vsds/data-industrial_and_scientific-sf_1/ \
#      --using_large_list &

# RUN 2
# #### QUICK RUN:
srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --output="IND-TEST-2.log" \
     $SCRATCH/maxvec/Maximus/results/quick_run.sh -l cagra_T2_data0_cache1 --nsys \
     --queries ann_reviews --index 'GPU,Cagra,64,32,NN_DESCENT' \
     --storage_device cpu --index_storage_device cpu --device gpu \
     --sf 1 --n_reps 10 \
     --params ${COMMON_PARAMS},index_data_on_gpu=0,cagra_cache_graph=1 &

### MAXBENCH DIRECT:
# srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --output="IND-TEST-2.log" \
#      $SCRATCH/maxvec/Maximus/build/Release/benchmarks/maxbench \
#      --benchmark vsds --queries ann_reviews \
#      --storage_device cpu --index_storage_device cpu --device gpu --flush_cache \
#      --n_reps 10 \
#      --index GPU,Flat \
#      --params k=100,query_count=1,query_start=0,use_cuvs=1,metric=IP \
#      --profile "nvtx,runtime-report(calc.inclusive=true,output=stdout)" \
#      --path $SCRATCH/maxvec/Maximus/tests/vsds/data-industrial_and_scientific-sf_1/ \
#      --using_large_list &

# RUN 3
#### QUICK RUN:
srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --output="IND-TEST-3.log" \
     $SCRATCH/maxvec/Maximus/results/quick_run.sh -l cagra_T3_data1_cache1 --nsys \
     --queries ann_reviews --index 'GPU,Cagra,64,32,NN_DESCENT' \
     --storage_device cpu --index_storage_device cpu --device gpu \
     --sf 1 --n_reps 10 \
     --params ${COMMON_PARAMS},index_data_on_gpu=1,cagra_cache_graph=1 &

# ### MAXBENCH DIRECT:
# srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --output="IND-TEST-3.log" \
#      $SCRATCH/maxvec/Maximus/build/Release/benchmarks/maxbench \
#      --benchmark vsds --queries ann_reviews \
#      --storage_device cpu --index_storage_device cpu --device gpu --flush_cache \
#      --n_reps 10 \
#      --index GPU,Flat \
#      --params k=100,query_count=1,query_start=0,use_cuvs=1,metric=IP \
#      --profile "nvtx,runtime-report(calc.inclusive=true,output=stdout)" \
#      --path $SCRATCH/maxvec/Maximus/tests/vsds/data-industrial_and_scientific-sf_1/ \
#      --using_large_list &

# RUN 4
#### QUICK RUN:
srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --output="IND-TEST-4.log" \
     $SCRATCH/maxvec/Maximus/results/quick_run.sh -l cagra_T4_data1_cache0 --nsys \
     --queries ann_reviews --index 'GPU,Cagra,64,32,NN_DESCENT' \
     --storage_device cpu --index_storage_device cpu --device gpu \
     --sf 1 --n_reps 10 \
     --params ${COMMON_PARAMS},index_data_on_gpu=1,cagra_cache_graph=0 &

# ### MAXBENCH DIRECT:
# srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --output="IND-TEST-4.log" \
#      $SCRATCH/maxvec/Maximus/build/Release/benchmarks/maxbench \
#      --benchmark vsds --queries ann_reviews \
#      --storage_device cpu --index_storage_device cpu --device gpu --flush_cache \
#      --n_reps 10 \
#      --index GPU,Flat \
#      --params k=100,query_count=1,query_start=0,use_cuvs=1,metric=IP \
#      --profile "nvtx,runtime-report(calc.inclusive=true,output=stdout)" \
#      --path $SCRATCH/maxvec/Maximus/tests/vsds/data-industrial_and_scientific-sf_1/ \
#      --using_large_list &

# WAIT is strictly required. It tells the batch script not to exit 
# until all 4 background tasks have finished.
wait