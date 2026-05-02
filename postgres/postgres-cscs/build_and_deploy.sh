#!/bin/bash
# Build and deploy pgvector image to CSCS daint.
# Run this locally (not on the cluster).

set -euo pipefail

IMAGE="pgvector-cscs"
CONTEXT="../postgres-scripts"
CSCS_HOST="daint"  # adjust if your SSH config uses a different name

echo "=== Step 1: Build image ==="
docker build -t "$IMAGE" "$CONTEXT"

echo ""
echo "=== Step 2: Save image ==="
docker save "$IMAGE" | gzip > "${IMAGE}.tar.gz"
echo "Saved: ${IMAGE}.tar.gz ($(du -h ${IMAGE}.tar.gz | cut -f1))"

echo ""
echo "=== Step 3: Transfer to CSCS ==="
echo "Run: rsync -avP ${IMAGE}.tar.gz ${CSCS_HOST}:\$SCRATCH/"
echo ""

echo "=== Step 4: On CSCS (interactive session) ==="
cat << 'EOF'
srun -A sm94 --pty bash

# Load and convert image
mkdir -p $SCRATCH/ce-images
gunzip -c $SCRATCH/pgvector-cscs.tar.gz | podman load
enroot import -x mount \
    -o $SCRATCH/ce-images/pgvector.sqsh \
    podman://pgvector-cscs:latest

# Create EDF (coexists with any other ~/.edf/*.toml)
cat > ~/.edf/pgvector.toml << 'TOML'
image = "${SCRATCH}/ce-images/pgvector.sqsh"

mounts = [
    "/capstor",
    "/iopsstor",
    "/users",
]

workdir = "/users/${USER}"
TOML

# Create scratch directories
mkdir -p $SCRATCH/pg_data $SCRATCH/datasets

# Copy datasets to $SCRATCH/datasets/ if not already there
# Then submit setup job:
sbatch setup_pgvector_cscs.sh
EOF
