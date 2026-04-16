#!/bin/bash
# =============================================================================
# submit_euler.sh — SLURM job script for Psi0 finetuning on ETH Euler
#
# Usage:
#   sbatch scripts/submit_euler.sh <task> [exp_name]
#
# Prerequisites:
#   1. Build the Docker image locally:
#        docker build -t psi0-environment .
#
#   2. Convert to Apptainer sandbox and compress:
#      (rootless Docker: export first, then build from archive)
    #    docker save psi0-environment:latest -o psi0-environment.tar
    #    APPTAINER_NOHTTPS=1 apptainer build --sandbox --fakeroot \
    #        psi0-environment.sif docker-archive://psi0-environment.tar
    #    tar -czf psi0-environment.tar.gz psi0-environment.sif
    #    rm psi0-environment.tar
#
#   3. Transfer to Euler:
#        rsync -avP psi0-environment.tar.gz \
#            euler:/cluster/work/mrl/$USER/containers/
#
#   4. One-time setup on Euler:
#        mkdir -p /cluster/work/mrl/$USER/containers
#        mkdir -p /cluster/project/mrl/$USER/results/psi0
#        mkdir -p /cluster/scratch/$USER/hfm/{cache/checkpoints,data,runs}
#        cp .env.sample ~/Psi0/.env  # and edit with your tokens
# =============================================================================
#SBATCH --job-name=psi0-finetune
#SBATCH --output=logs/psi0-%j.out
#SBATCH --error=logs/psi0-%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --tmp=200G

set -euo pipefail

# --- Load modules ---
module load eth_proxy

# --- Parse arguments ---
TASK="${1:?Usage: sbatch submit_euler.sh <task> [exp_name]}"
EXP="${2:-}"

# --- Paths (adjust to match your Euler setup) ---
PROJECT_DIR="$HOME/Psi0"
CONTAINER_TAR="/cluster/work/mrl/$USER/containers/psi0-environment.tar.gz"
HFM_DIR="/cluster/scratch/$USER/hfm"
RESULTS_DIR="/cluster/project/mrl/$USER/results/psi0/$SLURM_JOB_ID"

# --- Job info ---
echo "=== Psi0 Finetune Job ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Host:      $(hostname)"
echo "Date:      $(date)"
echo "Task:      $TASK"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "TMPDIR:    $TMPDIR"
echo "========================="

# --- Create directories ---
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$HFM_DIR/cache/checkpoints"
mkdir -p "$HFM_DIR/cache/triton"
mkdir -p "$HFM_DIR/cache/torch_kernels"
mkdir -p "$HFM_DIR/data"
mkdir -p "$HFM_DIR/runs"
mkdir -p "$RESULTS_DIR"

# --- Extract container to local scratch (CRITICAL for performance) ---
echo "[euler] Extracting container to \$TMPDIR..."
time tar -xzf "$CONTAINER_TAR" -C "$TMPDIR"

# --- Launch with Singularity ---
# --nv:                 enable NVIDIA GPU support
# --bind /hfm:          scratch storage for checkpoints, data, caches (read-write)
# --bind .env:          mount env file with secrets (HF_TOKEN, WANDB, etc.)
# --bind .runs:         training outputs (checkpoints, logs) on scratch, NOT tmpfs
# --bind /results:      persistent results on project storage
# --env:                redirect triton/torch caches to writable scratch
echo "[euler] Starting container..."
time singularity exec --nv \
    --bind "$HFM_DIR:/hfm" \
    --bind "$PROJECT_DIR/.env:/app/.env" \
    --bind "$HFM_DIR/runs:/app/.runs" \
    --bind "$RESULTS_DIR:/results" \
    --bind "$PROJECT_DIR/scripts:/app/scripts" \
    --env TRITON_CACHE_DIR=/hfm/cache/triton \
    --env TORCH_EXTENSIONS_DIR=/hfm/cache/torch_kernels \
    --env XDG_CACHE_HOME=/hfm/cache \
    "$TMPDIR/psi0-environment.sif" \
    bash /app/scripts/setup_and_train.sh "$TASK" ${EXP:+"$EXP"}

echo "[euler] Job completed at $(date)"
