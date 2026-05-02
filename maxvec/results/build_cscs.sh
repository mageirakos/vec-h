#!/bin/bash
set -euo pipefail
srun --account=sm94 --environment=maxvec --container-writable \
    --partition=debug --nodes=1 --ntasks=1 --time=0:10:00 \
    bash -c "cd ${SCRATCH}/maxvec/Maximus && ninja -C build/Release"