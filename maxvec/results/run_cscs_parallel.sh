#!/bin/bash
#SBATCH --job-name=maxvec_parallel
#SBATCH --account=sm94
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# --- Container ---
#SBATCH --environment=maxvec
#SBATCH --container-writable

# --- Shared environment (inherited by all srun tasks) ---
ulimit -c 0
export MAXIMUS_NUM_OUTER_THREADS=1
export MAXIMUS_NUM_INNER_THREADS=72
export MAXIMUS_NUM_IO_THREADS=72
export MAXIMUS_OPERATORS_FUSION=0
export MAXIMUS_DATASET_API=0
export OMP_NUM_THREADS=72
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# --- Thread binding (72 cores per NUMA node) ---
# SLURM pins each task to its own 72-core NUMA node via --ntasks-per-node=4 --gpus-per-task=1.
# These exports ensure OpenMP/BLAS respect that boundary instead of spawning 288 threads.
# (Matches run_cscs.sh single-GPU settings.)
export OPENBLAS_NUM_THREADS=1          # BLAS calls single-threaded; OMP handles parallelism



# --- Configuration ---
# Positional: SF, PERSIST_CSV (stages are always 'all' for this 4-GPU driver).
export SYSTEM="cscs-gh200"
export SF="${1:-1}"
export PERSIST_CSV="${2:-true}"

FULL_RUN_FLAGS="--sf=${SF}"
[ "$PERSIST_CSV" = "false" ] && FULL_RUN_FLAGS="$FULL_RUN_FLAGS --no-csv"
export FULL_RUN_FLAGS

# --- Navigate to project root ---
export PROJECT_ROOT="${SCRATCH}/maxvec/Maximus"
cd "${PROJECT_ROOT}" || exit 1

# --- Build once (before srun, so tasks don't race) ---
ninja -C ./build/Release

# --- Auto-increment run ID (race-safe: single process before srun) ---
# Scans existing run directories to find the next available ID.
# full_run.sh creates its own directory via mkdir -p, so no pre-creation needed.
DATE_TAG=$(LC_TIME=C date +%d%b%y)
export RUN_DIR_BASE="results/runs/${DATE_TAG}-${SYSTEM}"
mkdir -p "$RUN_DIR_BASE"
MAX=$(ls -d "$RUN_DIR_BASE"/run* 2>/dev/null | grep -oP '(?<=/run)\d+' | sort -n | tail -1)
export RUN_ID=$(( ${MAX:-0} + 1 ))

echo "=========================================="
echo "PARALLEL RUN: $(date)"
echo "  SYSTEM:   $SYSTEM"
echo "  SF:       $SF"
echo "  TASKS:    4 (one per GPU)"
echo "  RUN_BASE: ${RUN_DIR_BASE}/run${RUN_ID}_gpu{0,1,2,3}"
echo "=========================================="

# --- Pre-flight: verify GPU/CPU affinity per task ---
echo "--- Expected NUMA topology ---"
nvidia-smi topo -m
echo "--- Actual task placement ---"
srun bash -c 'echo "Task ${SLURM_LOCALID} | GPU: ${CUDA_VISIBLE_DEVICES} | Cores: $(taskset -cp $$)"'

# --- Launch 4 full pipelines in parallel ---
# Each task picks its own RESULTS_BASE via SLURM_LOCALID.
# full_run.sh does: run_vsds.sh (all queries/cases) → parse → plot → recall.
# Its internal ninja call is a no-op since we already built above.
# NUMA note: CSCS SLURM auto-places tasks NUMA-locally with --ntasks-per-node + --gpus-per-task.
# The pre-flight srun above confirms correct core/GPU affinity.
srun bash -c '
    cd "${PROJECT_ROOT}" || exit 1
    RESULTS_BASE="${RUN_DIR_BASE}/run${RUN_ID}_gpu${SLURM_LOCALID}"
    ./results/full_run.sh "$RESULTS_BASE" "$SYSTEM" $FULL_RUN_FLAGS
'

# --- Summary ---
echo ""
echo "=========================================="
echo "ALL TASKS COMPLETE: $(date)"
for t in 0 1 2 3; do
    DIR="${RUN_DIR_BASE}/run${RUN_ID}_gpu${t}"
    echo "  GPU ${t}: ${DIR}/"
    echo "    Log:   ${DIR}/full_run.log"
    echo "    Plots: ${DIR}/plots/"
done
echo ""
echo "Re-analyze any run:"
echo "  ./results/full_run.sh ${RUN_DIR_BASE}/run${RUN_ID}_gpu0 $SYSTEM --sf=$SF --analyze-only"
echo "=========================================="
