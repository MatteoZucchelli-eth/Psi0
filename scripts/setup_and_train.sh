#!/bin/bash
# =============================================================================
# setup_and_train.sh — One-shot script for Psi0 finetuning
# Works in both Docker and Apptainer (Euler cluster).
# Handles: .env loading, checkpoint download, data download, patching, training.
#
# Usage:
#   ./scripts/setup_and_train.sh <task> [exp_name]
#
# Example:
#   ./scripts/setup_and_train.sh Pick_bottle_and_turn_and_pour_into_cup
#
# Prerequisites:
#   - /hfm must be a writable directory (bind-mounted in Apptainer/Docker)
#   - .env must exist (or .env.sample copied and edited)
#   - HF_TOKEN must be set for downloading from HuggingFace
# =============================================================================
set -euo pipefail


TASK="Pick_bottle_and_turn_and_pour_into_cup"
EXP="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# --- 1. Source environment variables ---
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "[setup] Loading .env"
    set -a
    source "$PROJECT_DIR/.env"
    set +a
else
    echo "[ERROR] .env not found at $PROJECT_DIR/.env"
    echo "        Copy .env.sample to .env and fill in your tokens."
    exit 1
fi

# Verify critical variables
: "${PSI_HOME:?PSI_HOME not set in .env}"
: "${HF_TOKEN:?HF_TOKEN not set in .env}"

echo "[setup] PSI_HOME=$PSI_HOME"
echo "[setup] Task: $TASK"

# --- 2. Create directory structure ---
echo "[setup] Ensuring directory structure under $PSI_HOME ..."
mkdir -p "$PSI_HOME/cache/checkpoints/psi0"
mkdir -p "$PSI_HOME/data/real"

# --- 3. Download checkpoints if missing ---
VLM_CKPT="$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
ACTION_CKPT="$PSI_HOME/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"

if [ ! -d "$VLM_CKPT" ] || [ -z "$(ls -A "$VLM_CKPT" 2>/dev/null)" ]; then
    echo "[setup] Downloading VLM checkpoint..."
    hf download USC-PSI-Lab/psi-model \
        --include="psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/*" \
        --local-dir="$PSI_HOME/cache/checkpoints" \
        --repo-type=model
else
    echo "[setup] VLM checkpoint already exists, skipping download."
fi

if [ ! -d "$ACTION_CKPT" ] || [ -z "$(ls -A "$ACTION_CKPT" 2>/dev/null)" ]; then
    echo "[setup] Downloading Action Expert checkpoint..."
    hf download USC-PSI-Lab/psi-model \
        --include="psi0/postpre.1by1.pad36.2601131206.ckpt.he30k/*" \
        --local-dir="$PSI_HOME/cache/checkpoints" \
        --repo-type=model
else
    echo "[setup] Action Expert checkpoint already exists, skipping download."
fi

# --- 4. Download task data if missing ---
TASK_DATA_DIR="$PSI_HOME/data/real/$TASK"

if [ ! -d "$TASK_DATA_DIR" ] || [ -z "$(ls -A "$TASK_DATA_DIR" 2>/dev/null)" ]; then
    echo "[setup] Downloading task data for $TASK..."
    hf download USC-PSI-Lab/psi-data \
        "real/$TASK.zip" \
        --local-dir="$PSI_HOME/data" \
        --repo-type=dataset

    echo "[setup] Extracting task data..."
    unzip -o "$PSI_HOME/data/real/$TASK.zip" -d "$PSI_HOME/data/real"
else
    echo "[setup] Task data for $TASK already exists, skipping download."
fi

# --- 5. Apply the lerobot metadata patch (idempotent) ---
echo "[setup] Applying lerobot metadata patch..."
python3 scripts/data/patch_lerobot_meta.py "$TASK_DATA_DIR"

# --- 6. Launch training ---
echo "[setup] All prerequisites ready. Launching training..."

if [ -n "$EXP" ]; then
    exec scripts/train/psi0/finetune-real-psi0.sh "$TASK" "$EXP"
else
    exec scripts/train/psi0/finetune-real-psi0.sh "$TASK"
fi
