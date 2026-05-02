#!/bin/bash
#SBATCH --job-name=maxvec_experiments
#SBATCH --account=sm94
#SBATCH --environment=maxvec
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_full_run_%x-%j.out
#SBATCH --error=slurm_full_run_%x-%j.err

# --- Usage ---
# Each sbatch = one node. Fan-out by submitting multiple jobs; each run writes
# its own run<N>_job<JOBID>/ dir with its own parse+plot outputs.
# Append --no-csv to any of these to skip CSV persistence + recall step.
# Pass -J <name> so slurm_full_run_<name>-<JOBID>.out reflects the stage.
#
#   sbatch                 run_cscs.sh                                   # all stages
#   sbatch -J vsds         run_cscs.sh --stages=vsds     --no-csv        # per-stage ...
#   sbatch -J varbatch     run_cscs.sh --stages=varbatch --no-csv        # ... in parallel
#   sbatch -J fullsweep    run_cscs.sh --stages=fullsweep                # (fullsweep never writes CSVs)
#   sbatch -J vsds+varb    run_cscs.sh --stages=vsds,varbatch --no-csv   # two stages together
#
# Re-parse/plot on the login node (fast, no sbatch):
#   ./results/full_run.sh results/runs/<date>-cscs-gh200/run<N>_job<M> cscs-gh200 \
#       --analyze-only --stages=vsds,varbatch


# --- cscs-gh200 (Alps / Daint) environment ---
#
# NUMA-aware CPU-GPU affinity is critical for throughput on multi-socket
# Grace Hopper systems. Cross-NUMA data movement through the host fabric
# adds latency and reduces bandwidth vs. NUMA-local access.
# Daint node topology (nvidia-smi topo -m):
#
#     GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity    GPU NUMA ID
# GPU0     X     NV6    NV6    NV6    0-71            0                4
# GPU1    NV6     X     NV6    NV6    72-143          1                12
# GPU2    NV6    NV6     X     NV6    144-215         2                20
# GPU3    NV6    NV6    NV6     X     216-287         3                28
#
# We use GPU0 (CUDA_VISIBLE_DEVICES=0), so pin threads to cores 0-71
# (NUMA node 0) for NUMA-local access.

# Thread counts — 72 threads for NUMA node 0.
# GPU0 + cores 0-71 enforced via CUDA_VISIBLE_DEVICES=0 + taskset -c 0-71.
export OMP_NUM_THREADS=72
export OPENBLAS_NUM_THREADS=1     # BLAS calls single-threaded; OMP handles parallelism
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export MAXIMUS_NUM_INNER_THREADS=72
export MAXIMUS_NUM_OUTER_THREADS=1
export MAXIMUS_OPERATORS_FUSION=0
export MAXIMUS_DATASET_API=0
export MAXIMUS_NUM_IO_THREADS=72
export CUDA_VISIBLE_DEVICES=0
ulimit -c 0

# --- Configuration ---
SYSTEM="cscs-gh200"
SF="1"
STAGES="vsds,varbatch,fullsweep"
ANALYZE_ONLY=""
PERSIST_CSV="true"

# Named-arg parsing (forwarded verbatim to full_run.sh).
for arg in "$@"; do
    case "$arg" in
        --sf=*)          SF="${arg#--sf=}" ;;
        --stages=*)      STAGES="${arg#--stages=}" ;;
        --analyze-only)  ANALYZE_ONLY="--analyze-only" ;;
        --no-csv)        PERSIST_CSV="false" ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

FULL_RUN_FLAGS="--sf=${SF} --stages=${STAGES}"
[ -n "$ANALYZE_ONLY" ] && FULL_RUN_FLAGS="$FULL_RUN_FLAGS $ANALYZE_ONLY"
[ "$PERSIST_CSV" = "false" ] && FULL_RUN_FLAGS="$FULL_RUN_FLAGS --no-csv"

# --- Auto-increment run ID (race-safe: unique SLURM_JOB_ID per sbatch) ---
DATE_TAG=$(LC_TIME=C date +%d%b%y)
RUN_DIR_BASE="results/runs/${DATE_TAG}-${SYSTEM}"
MAX=$(ls -d "$RUN_DIR_BASE"/run* 2>/dev/null | sed 's|.*/run\([0-9]*\).*|\1|' | sort -n | tail -1)
RUN_ID=$(( ${MAX:-0} + 1 ))
RESULTS_BASE="$RUN_DIR_BASE/run${RUN_ID}_job${SLURM_JOB_ID}"
# Append job name suffix if -J passed something other than the default.
if [ -n "$SLURM_JOB_NAME" ] && [ "$SLURM_JOB_NAME" != "maxvec_experiments" ]; then
    RESULTS_BASE="${RESULTS_BASE}-${SLURM_JOB_NAME}"
fi
mkdir -p "$RESULTS_BASE"

# --- Navigate to project root ---
cd "${SCRATCH}/maxvec/Maximus" || cd "$(dirname "$0")" || exit 1

# --- Summary (mirrors run_cscs_parallel.sh) ---
echo "=========================================="
echo "SINGLE-GPU RUN: $(date)"
echo "  SYSTEM:      $SYSTEM"
echo "  SF:          $SF"
echo "  STAGES:      $STAGES"
echo "  ANALYZE:     ${ANALYZE_ONLY:-no}"
echo "  PERSIST_CSV: $PERSIST_CSV"
echo "  FLAGS:       $FULL_RUN_FLAGS"
echo "  RESULTS:     $RESULTS_BASE"
echo "  JOB_ID:      ${SLURM_JOB_ID:-local}"
echo "=========================================="

# --- Run (full pipeline: build + bench + parse + plot) ---
# full_run.sh redirects its own output to full_run.log internally.
# Tail it in the background so SLURM .out also captures progress.
FULL_RUN_LOG="${RESULTS_BASE}/full_run.log"
touch "$FULL_RUN_LOG"
tail -f "$FULL_RUN_LOG" &
TAIL_PID=$!

taskset -c 0-71 ./results/full_run.sh "$RESULTS_BASE" "$SYSTEM" $FULL_RUN_FLAGS
EXIT_CODE=$?

# Stop tailing
sleep 1
kill "$TAIL_PID" 2>/dev/null || true

# --- Final status ---
echo ""
echo "=========================================="
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "RUN COMPLETE: $(date)"
else
    echo "RUN FAILED (exit code $EXIT_CODE): $(date)"
fi
echo "  Results: $RESULTS_BASE"
echo "  Log:     $RESULTS_BASE/full_run.log"
echo "  Plots:   $RESULTS_BASE/plots/"
echo ""
echo "Re-analyze:"
echo "  ./results/full_run.sh $RESULTS_BASE $SYSTEM --sf=$SF --analyze-only --stages=$STAGES"
echo "=========================================="

exit $EXIT_CODE

# --- Re-analyze an existing run (parse + plot only, no benchmarks) ---
# Run directly on the login node (no sbatch needed):
#   cd ${SCRATCH}/maxvec/Maximus
#   ./results/full_run.sh results/runs/<DATE>-cscs-gh200/run<N>_job<JOBID> cscs-gh200 \
#       --sf=<SF> --analyze-only --stages=vsds,varbatch
#
# Or batch all runs for a date:
#   for d in results/runs/07Apr26-cscs-gh200/run*_job*/; do
#       ./results/full_run.sh "$d" cscs-gh200 --sf=1 --analyze-only
#   done

# --- Kill all jobs on this system ---
# scancel -u $USER --name=maxvec_experiments
