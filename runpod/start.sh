#!/usr/bin/env bash
# One-shot launcher for fine-tuning pi0.5 on alicheraghi/robot-arm-filtered
# (config: pi05_so101_lora) on a RunPod A6000 pod.
#
# Checkpoints are written to ./checkpoints/, which lives on RunPod's
# persistent /workspace volume — they survive pod restarts as long as you
# keep the volume.
#
# Invoke from the repo root, e.g.:
#   cd /workspace/openpi && bash runpod/start.sh
#
# Optional knobs come from runpod/runpod.env (see runpod.env.example).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[start] repo: $REPO_ROOT"

# ---------------- Env file (optional) ----------------
if [ -f runpod/runpod.env ]; then
  echo "[start] loading runpod/runpod.env"
  set -a
  # shellcheck disable=SC1091
  source runpod/runpod.env
  set +a
fi

EXP_NAME="${EXP_NAME:-robot_arm_v1}"
RUN_NORM_STATS="${RUN_NORM_STATS:-1}"

# ---------------- System deps ----------------
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v git-lfs >/dev/null 2>&1; then
  echo "[start] installing ffmpeg + git-lfs..."
  apt-get update && apt-get install -y --no-install-recommends ffmpeg git-lfs
fi
git lfs install --skip-repo 2>/dev/null || true

# ---------------- HF token (optional) ----------------
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  export HF_TOKEN
  echo "[start] HF token loaded"
fi

# ---------------- WandB ----------------
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
else
  export WANDB_MODE=offline
  echo "[start] WANDB_API_KEY empty -> WANDB_MODE=offline"
fi

# ---------------- Python env ----------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[start] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[start] uv sync (this is cached on subsequent runs)..."
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# ---------------- Cache hint ----------------
# Point HF caches into /workspace so they survive container restarts
# (instead of the ephemeral container FS).
if [ -d /workspace ]; then
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export LEROBOT_HOME="${LEROBOT_HOME:-/workspace/.cache/lerobot}"
  mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$LEROBOT_HOME"
  echo "[start] HF_HOME=$HF_HOME  LEROBOT_HOME=$LEROBOT_HOME"
fi

# ---------------- Norm stats ----------------
CONFIG_NAME="pi05_so101_lora"

NORM_STATS_PATH="assets/$CONFIG_NAME/alicheraghi/robot-arm-filtered/norm_stats.json"
if [ "$RUN_NORM_STATS" = "1" ] && [ ! -f "$NORM_STATS_PATH" ]; then
  echo "[start] computing norm stats..."
  uv run python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
else
  echo "[start] norm stats present or RUN_NORM_STATS=0 -> skipping"
fi

# ---------------- Train ----------------
echo "[start] launching training: config=$CONFIG_NAME exp=$EXP_NAME"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

# Resume automatically if a checkpoint dir for this exp already exists.
CKPT_DIR="./checkpoints/$CONFIG_NAME/$EXP_NAME"
RESUME_FLAG=""
if [ -d "$CKPT_DIR" ] && [ -n "$(ls -A "$CKPT_DIR" 2>/dev/null || true)" ]; then
  echo "[start] existing checkpoints found at $CKPT_DIR -> resuming"
  RESUME_FLAG="--resume"
fi

uv run python scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  $RESUME_FLAG

echo "[start] complete. Checkpoints at: $CKPT_DIR"

# ---------------- Auto-stop on finish (cost safety) ----------------
# Opt-in: set AUTO_STOP_ON_FINISH=1 and RUNPOD_API_KEY in the pod env to
# automatically stop this pod after training succeeds. Stopping (not
# terminating) preserves /workspace; you'll be billed only for volume
# storage (~$0.10/GB/mo) until you terminate the pod manually.
if [ "${AUTO_STOP_ON_FINISH:-0}" = "1" ]; then
  if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "[start] AUTO_STOP_ON_FINISH=1 but RUNPOD_POD_ID is unset — cannot auto-stop"
  elif ! command -v runpodctl >/dev/null 2>&1; then
    echo "[start] AUTO_STOP_ON_FINISH=1 but runpodctl not installed — cannot auto-stop"
  elif [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "[start] AUTO_STOP_ON_FINISH=1 but RUNPOD_API_KEY is unset — cannot auto-stop"
  else
    echo "[start] auto-stopping pod $RUNPOD_POD_ID (AUTO_STOP_ON_FINISH=1)"
    runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null
    runpodctl stop pod "$RUNPOD_POD_ID" || echo "[start] auto-stop failed; stop the pod manually"
  fi
fi
