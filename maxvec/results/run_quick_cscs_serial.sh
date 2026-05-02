#!/bin/bash
#SBATCH --job-name=maxvec_quick_serial
#SBATCH --account=sm94
#SBATCH --environment=maxvec
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_quick_serial_%x-%j.out
#SBATCH --error=slurm_quick_serial_%x-%j.err
#SBATCH --partition=debug

# --- run_quick_cscs_serial.sh ---
#
# Sbatch wrapper that runs a vsds cell N times back-to-back, pinned to
# GPU 0 / NUMA 0 on cscs-gh200. Each iteration is a fresh maxbench process
# so you can see whether per-process state drifts within one SLURM allocation.
#
# Two dispatch modes, picked from args:
#
# PRESET MODE (recommended for variance debug on known-problematic cells)
#   Delegates to run_vsds.sh with RUN_INDEX_FILTER + NSYS_OUT_DIR + REPS.
#   Uses run_vsds.sh's built-in per-query params (csv_batch_size max for q15,
#   use_post for q2/q15/q18/q19, q11 partitioning, case-4 MALLOC env, etc.).
#
#   Triggered by: --filter=qN_cM [--index_filter=TOKEN]
#
# CUSTOM MODE (for bespoke params / aliases not covered by a preset)
#   Delegates to quick_run.sh with passthrough args.
#   quick_run.sh now sources _vsds_transforms.sh, so aliases like IVFH4096
#   and CagraCH Just Work.
#
#   Triggered by: no --filter= passed; any --queries/--index/... goes to quick_run.sh.
#
# Wrapper-only flags (apply to both modes):
#   --n_runs=N     number of back-to-back iterations (default 3)
#   --label=NAME   base label (required); iteration label = NAME_iter<i>
#   --membind      numactl --cpunodebind=0 --membind=0 instead of taskset -c 0-71
#   --nsys         enable nsys profiling (one .nsys-rep per iteration)
#   --reps=N       override reps-per-iteration. Default 20 without --nsys, 4 with --nsys
#
# Preset-mode flags:
#   --filter=qN_cM       RUN_Q for run_vsds.sh (e.g. q10_c1, q15_c0, q11_c3, c1 for all case-1)
#   --index_filter=TOKEN run_vsds.sh RUN_INDEX_FILTER token (cagra, cagrach, ivf1024, ivfh1024, ivf4096, ivfh4096, flat)
#   --scenario=ann|enn|all   default ann
#   --k=VALUES             default 100
#   --sf=N                 default 1
#   --system=NAME          default cscs-gh200
#
# ==============================================================================
# Worked examples — the four cells worth debugging first (variance audit 2026-04-17/18)
# ==============================================================================
#
# All are case 1 (device=gpu, storage=cpu, index_storage=cpu), all with
# --n_runs=3, --n_reps=4, --nsys. The membind variants test whether hard NUMA 0
# memory pin collapses the drift.
#
#   # 1) q10_mid IVF4096, default pin (taskset -c 0-71)
#   sbatch run_quick_cscs_serial.sh \
#     --filter=q10_c1 --index_filter=ivf4096 \
#     --label=q10_ivf4096_base --n_runs=3 --nsys
#
#   # 2) q10_mid IVF4096, hard NUMA0 bind (numactl --cpunodebind=0 --membind=0)
#   sbatch run_quick_cscs_serial.sh \
#     --filter=q10_c1 --index_filter=ivf4096 \
#     --label=q10_ivf4096_membind --n_runs=3 --nsys --membind
#
#   # 3) q19_start IVF4096, default pin (cleanest between-date drift in audit)
#   sbatch run_quick_cscs_serial.sh \
#     --filter=q19_c1 --index_filter=ivf4096 \
#     --label=q19_ivf4096_base --n_runs=3 --nsys
#
#   # 4) q19_start IVF4096, hard NUMA0 bind
#   sbatch run_quick_cscs_serial.sh \
#     --filter=q19_c1 --index_filter=ivf4096 \
#     --label=q19_ivf4096_membind --n_runs=3 --nsys --membind
#
# Each sbatch produces 3 iterations x 1 .nsys-rep each = 3 traces per job.
# With 4 jobs = 12 total .nsys-rep files, grouped by label under
# results/0_quick_run/<label>_iter{1,2,3}/_data/nsys/.
#
# Custom-mode example (bespoke params, IVFH alias works via helper):
#
#   sbatch run_quick_cscs_serial.sh --label=custom_ivfh --n_runs=3 --nsys \
#     --queries q10_mid --index 'GPU,IVFH4096,Flat' \
#     --device gpu --storage_device cpu --index_storage_device cpu \
#     --sf 1 --n_reps 4 \
#     --params 'k=100,query_count=1,query_start=0,postfilter_ksearch=100,cagra_itopksize=128,ivf_nprobe=30,hnsw_efsearch=128,metric=IP' \
#     --use_index_cache_dir /tmp/maximus_faiss_cache
# ==============================================================================

# --- cscs-gh200 environment (mirror run_cscs.sh) ---
export OMP_NUM_THREADS=72
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export MAXIMUS_NUM_INNER_THREADS=72
export MAXIMUS_NUM_OUTER_THREADS=1
export MAXIMUS_OPERATORS_FUSION=0
export MAXIMUS_DATASET_API=0
export MAXIMUS_NUM_IO_THREADS=72
export CUDA_VISIBLE_DEVICES=0
ulimit -c 0
set -o pipefail

# --- Defaults ---
N_RUNS=3
LABEL=""
USE_MEMBIND="false"
USE_NSYS="false"
REPS=""                 # empty = auto: 4 if --nsys, else 20
FILTER=""               # if set -> preset mode
INDEX_FILTER=""         # preset mode only
SCENARIO="ann"
K="100"
SF="1"
SYSTEM="cscs-gh200"
PERSIST_CSV="true"
PASSTHROUGH=()          # collects args bound for quick_run.sh in custom mode

# --- Arg parsing ---
while [ $# -gt 0 ]; do
    case "$1" in
        --n_runs=*)        N_RUNS="${1#--n_runs=}" ;;
        --label=*)         LABEL="${1#--label=}" ;;
        --membind)         USE_MEMBIND="true" ;;
        --nsys)            USE_NSYS="true" ;;
        --reps=*)          REPS="${1#--reps=}" ;;
        --filter=*)        FILTER="${1#--filter=}" ;;
        --index_filter=*)  INDEX_FILTER="${1#--index_filter=}" ;;
        --scenario=*)      SCENARIO="${1#--scenario=}" ;;
        --k=*)             K="${1#--k=}" ;;
        --sf=*)            SF="${1#--sf=}" ;;
        --system=*)        SYSTEM="${1#--system=}" ;;
        *)                 PASSTHROUGH+=("$1") ;;
    esac
    shift
done

if [ -z "$LABEL" ]; then
    echo "Error: --label=<name> is required (iteration labels are <name>_iter<i>)" >&2
    exit 2
fi

# Mode dispatch
if [ -n "$FILTER" ]; then
    MODE="preset"
else
    MODE="custom"
fi

# Auto-pick REPS default if not given: 4 with --nsys (small traces), 20 otherwise.
if [ -z "$REPS" ]; then
    if [ "$USE_NSYS" = "true" ]; then
        REPS=4
    else
        REPS=20
    fi
fi

# --- Pin wrapper ---
if [ "$USE_MEMBIND" = "true" ]; then
    PIN=(numactl --cpunodebind=0 --membind=0 --)
else
    PIN=(taskset -c 0-71)
fi

# --- Topology log (once per SLURM job; fixed for the allocation) ---
echo "=== Topology at job start ==="
echo "-- nvidia-smi -L --"
nvidia-smi -L
echo "-- nvidia-smi topo -m --"
nvidia-smi topo -m
echo "-- numactl --show --"
numactl --show
echo "-- /proc/self/status (pre-pin) --"
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
echo "============================="
echo ""

cd "${SCRATCH}/maxvec/Maximus" || cd "$(dirname "$0")" || exit 1

# --- Summary ---
echo "=========================================="
echo "QUICK SERIAL RUN: $(date)"
echo "  MODE:         $MODE"
echo "  LABEL:        $LABEL"
echo "  N_RUNS:       $N_RUNS"
echo "  REPS/iter:    $REPS"
echo "  NSYS:         $USE_NSYS"
echo "  MEMBIND:      $USE_MEMBIND"
echo "  PIN:          ${PIN[*]}"
if [ "$MODE" = "preset" ]; then
    echo "  FILTER:       $FILTER"
    echo "  INDEX_FILTER: ${INDEX_FILTER:-<all>}"
    echo "  SCENARIO:     $SCENARIO"
    echo "  K:            $K"
    echo "  SF:           $SF"
    echo "  SYSTEM:       $SYSTEM"
else
    echo "  PASSTHROUGH:  ${PASSTHROUGH[*]}"
fi
echo "  JOB_ID:       ${SLURM_JOB_ID:-local}"
echo "=========================================="

OVERALL_RC=0

for i in $(seq 1 "$N_RUNS"); do
    ITER_LABEL="${LABEL}_iter${i}"
    ITER_DIR="results/0_quick_run/${ITER_LABEL}"
    DATA_DIR="${ITER_DIR}/_data"
    mkdir -p "$DATA_DIR"
    TOP_LOG="${ITER_DIR}/RUN.log"

    echo ""
    echo "--- Iteration $i / $N_RUNS: $(date)  label=$ITER_LABEL ---"

    if [ "$MODE" = "preset" ]; then
        # ============= PRESET MODE: delegate to run_vsds.sh =============
        export RUN_INDEX_FILTER="$INDEX_FILTER"
        export REPS="$REPS"
        if [ "$USE_NSYS" = "true" ]; then
            export NSYS_OUT_DIR="${DATA_DIR}/nsys"
        else
            unset NSYS_OUT_DIR
        fi

        # run_vsds.sh writes caliper logs to $DATA_DIR/vsds/gpu-<SYSTEM>/sf_<SF>/
        "${PIN[@]}" ./results/run_vsds.sh \
            "$FILTER" "$SF" "$SCENARIO" "$SYSTEM" "$K" "$DATA_DIR" "$PERSIST_CSV" \
            2>&1 | tee "$TOP_LOG"
        RC=${PIPESTATUS[0]}

        # Post-hoc parse + plot so preset-mode outputs match quick_run.sh's UX.
        # Best-effort; don't fail the iteration if parse/plot hiccups.
        if [ "$RC" -eq 0 ]; then
            (
                cd results || exit 0
                python parse_caliper.py \
                    --base_dir "0_quick_run/${ITER_LABEL}/_data" \
                    --sf "$SF" --system "$SYSTEM" --benchmark vsds || true
                python plot_caliper.py \
                    --in_dir "0_quick_run/${ITER_LABEL}/_data/parse_caliper" \
                    --out_dir "0_quick_run/${ITER_LABEL}/_data/plots" \
                    --sf "$SF" --system "$SYSTEM" --benchmark vsds --k "$K" \
                    --metric median min \
                    --quick --title --skip-cases "" || true
            )

            # Symlink top-level convenience pointers (mirrors quick_run.sh Step 4).
            for f in "${DATA_DIR}"/vsds/*/sf_*/*.log; do
                [ -f "$f" ] || continue
                ln -sf "$(realpath --relative-to="$ITER_DIR" "$f")" "${ITER_DIR}/$(basename "$f")" 2>/dev/null || true
            done
            for f in "${DATA_DIR}"/nsys/*.nsys-rep; do
                [ -f "$f" ] || continue
                ln -sf "$(realpath --relative-to="$ITER_DIR" "$f")" "${ITER_DIR}/$(basename "$f")" 2>/dev/null || true
            done
            for img in "${DATA_DIR}"/plots/vsds/sf_${SF}/${SYSTEM}/*/plot_1/median/per_query/*.jpeg \
                       "${DATA_DIR}"/plots/vsds/sf_${SF}/${SYSTEM}/*/plot_2/median/per_query/*.jpeg \
                       "${DATA_DIR}"/plots/vsds/sf_${SF}/${SYSTEM}/*/plot_1/min/per_query/*.jpeg \
                       "${DATA_DIR}"/plots/vsds/sf_${SF}/${SYSTEM}/*/plot_2/min/per_query/*.jpeg; do
                [ -f "$img" ] || continue
                [[ "$img" == *"coarse"* ]] && continue
                ln -sf "$(realpath --relative-to="$ITER_DIR" "$img")" "${ITER_DIR}/$(basename "$img")" 2>/dev/null || true
            done
        fi
    else
        # ============= CUSTOM MODE: delegate to quick_run.sh =============
        # quick_run.sh handles parse/plot/symlinks/nsys internally.
        QR_ARGS=(-l "$ITER_LABEL")
        [ "$USE_NSYS" = "true" ] && QR_ARGS+=(--nsys)
        # If the passthrough already has --n_reps, respect it; else inject.
        if ! printf '%s\n' "${PASSTHROUGH[@]}" | grep -q -- '--n_reps'; then
            QR_ARGS+=(--n_reps "$REPS")
        fi
        "${PIN[@]}" ./results/quick_run.sh "${QR_ARGS[@]}" "${PASSTHROUGH[@]}"
        RC=$?
    fi

    echo "  rc=$RC"
    [ "$RC" -ne 0 ] && OVERALL_RC="$RC"
done

echo ""
echo "=========================================="
if [ "$OVERALL_RC" -eq 0 ]; then
    echo "DONE (all $N_RUNS iterations OK): $(date)"
else
    echo "DONE with failures (last rc=$OVERALL_RC): $(date)"
fi
echo "Per-iteration output:"
for i in $(seq 1 "$N_RUNS"); do
    echo "  results/0_quick_run/${LABEL}_iter${i}/"
done
if [ "$USE_NSYS" = "true" ] && [ "$MODE" = "preset" ]; then
    echo "nsys traces (preset mode):"
    for i in $(seq 1 "$N_RUNS"); do
        echo "  results/0_quick_run/${LABEL}_iter${i}/_data/nsys/*.nsys-rep"
    done
fi
echo "=========================================="

exit $OVERALL_RC
