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
#        APPTAINER_NOHTTPS=1 apptainer build --sandbox --fakeroot \
#            psi0-environment.sif docker-daemon://psi0-environment:latest
#        tar -czf psi0-environment.tar.gz psi0-environment.sif
#
#   3. Transfer to Euler:
#        scp psi0-environment.tar.gz \
#            euler:/cluster/work/mrl/$USER/containers/
#
#   4. Place .env in your project directory on Euler
#      (copy .env.sample, fill in HF_TOKEN, WANDB_API_KEY, etc.)
#
#   5. Ensure scratch data dir exists:
#        ssh euler "mkdir -p /cluster/scratch/$USER/hfm"
# =============================================================================
#SBATCH --job-name=psi0-finetune
#SBATCH --output=logs/psi0-%j.out
#SBATCH --error=logs/psi0-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --tmp=200G

set -euo pipefail

# --- Load modules ---
# eth_proxy is required for outbound network (HuggingFace downloads, wandb)
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
mkdir -p "$HFM_DIR/data"
mkdir -p "$RESULTS_DIR"

# --- Extract container to fast local scratch (CRITICAL for performance) ---
echo "[euler] Extracting container to \$TMPDIR..."
time tar -xzf "$CONTAINER_TAR" -C "$TMPDIR"

# --- Launch with Singularity ---
# --nv:                    enable NVIDIA GPU support
# --bind HFM_DIR:/hfm:     mount scratch storage for checkpoints/data (read-write)
# --bind PROJECT_DIR:/app:  mount project dir for code, .env, and scripts
# --bind RESULTS_DIR:       mount persistent results directory
# --writable-tmpfs:         allow writes to container's overlay (e.g., /tmp)
echo "[euler] Starting container..."
time singularity exec --nv \
    --bind "$HFM_DIR:/hfm" \
    --bind "$PROJECT_DIR:/app" \
    --bind "$RESULTS_DIR:/results" \
    --writable-tmpfs \
    "$TMPDIR/psi0-environment.sif" \
    bash /app/scripts/setup_and_train.sh "$TASK" ${EXP:+"$EXP"}

echo "[euler] Job completed at $(date)"
