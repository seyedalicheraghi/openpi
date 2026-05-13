# RunPod fine-tune: pi0.5 LoRA on SO-101 robot-arm

End-to-end recipe to LoRA-fine-tune the π₀.₅ base model on
[`alicheraghi/robot-arm-filtered`](https://huggingface.co/datasets/alicheraghi/robot-arm-filtered)
on a single RunPod **RTX A6000** (48 GB).

Training config: [`pi05_so101_lora`](../src/openpi/training/config.py) — LoRA
on both PaLI-Gemma and the action expert, 15 000 steps, batch 16, EMA off.

Checkpoints (and HF / LeRobot caches) live on `/workspace`, RunPod's
persistent volume — they survive pod restarts as long as you keep the
volume attached.

---

## 1. Create the pod

- **GPU:** 1× RTX A6000 (48 GB).
- **Template:** any PyTorch 2.x + CUDA 12.x community image. Recommended:
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
- **Volume:** **≥ 80 GB persistent volume mounted at `/workspace`**.
  Storage usage breakdown:
  - Repo + Python env: ~6 GB
  - Pi0.5 base checkpoint: ~13 GB
  - HF dataset cache (`robot-arm-filtered`): ~300 MB
  - Norm-stats + assets: a few MB
  - Training checkpoints (one every 1k steps, last 5k retained): ~25–35 GB

## 2. Pod env vars (optional — RunPod UI → "Environment Variables")

All optional. Defaults work fine.

| Key             | Default          | Purpose                                                  |
| --------------- | ---------------- | -------------------------------------------------------- |
| `EXP_NAME`      | `robot_arm_v1`   | Subdirectory under `checkpoints/pi05_so101_lora/`.       |
| `WANDB_API_KEY` | *(empty)*        | If set, training logs to W&B online; else offline mode. |
| `HF_TOKEN`      | *(empty)*        | Only needed if the dataset becomes private later.        |
| `RUN_NORM_STATS`| `1`              | Set `0` to skip norm-stats (use when resuming).          |

## 3. Pod startup command

Paste this into RunPod's "Container Start Command":

```bash
bash -lc 'set -e; \
  if [ ! -d /workspace/openpi ]; then \
    git clone --recurse-submodules https://github.com/seyedalicheraghi/openpi.git /workspace/openpi; \
  fi; \
  cd /workspace/openpi && git pull && git submodule update --init --recursive; \
  exec bash runpod/start.sh'
```

That's it. The pod will:

1. Clone the repo (first launch) or pull latest.
2. Install `ffmpeg`, `git-lfs`, `uv`, and sync Python deps.
3. Point HF and LeRobot caches at `/workspace/.cache/...` so they persist.
4. Compute norm-stats on `alicheraghi/robot-arm-filtered`.
5. Run `scripts/train.py pi05_so101_lora --exp-name $EXP_NAME`.
6. **Resume automatically** if `./checkpoints/pi05_so101_lora/$EXP_NAME` already
   exists, so restarting the pod picks up at the last checkpoint.

## 4. Watching progress

From the pod terminal:

```bash
# Training log (the openpi training script logs to stdout — usually tee'd by RunPod).
nvidia-smi -l 5

# List existing checkpoints.
ls -lh /workspace/openpi/checkpoints/pi05_so101_lora/$EXP_NAME/
```

Or attach a Web Terminal in the RunPod UI.

## 5. Expected resource use (A6000 48 GB)

| Phase             | VRAM       | Wall time          |
| ----------------- | ---------- | ------------------ |
| Base ckpt + LoRA  | ~22–28 GB  | n/a                |
| Norm stats        | < 2 GB     | ~3–5 min           |
| Training (15k)    | ~28 GB     | ~6–10 h            |

If you OOM, drop `batch_size` in the config from 16 → 8; if you have headroom,
push it to 24 or 32 (edit the `pi05_so101_lora` entry in
`src/openpi/training/config.py`).

## 6. Downloading a checkpoint to your laptop

```bash
# From your laptop — runpodctl is the official CLI (https://github.com/runpod/runpodctl)
runpodctl receive <pod-id>:/workspace/openpi/checkpoints/pi05_so101_lora/robot_arm_v1/15000 ./local_ckpt

# Or use `scp` if you've added an SSH key to the pod:
scp -r root@<pod-host>:/workspace/openpi/checkpoints/pi05_so101_lora/robot_arm_v1/15000 ./local_ckpt
```

## 7. Inference

Each checkpoint dir under `checkpoints/.../<step>/` contains:

```
params/        # JAX params for the LoRA-merged model
norm_stats.json
config.json
```

Serve:

```bash
uv run python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_so101_lora \
  --policy.dir ./local_ckpt
```

The inference client must pass the same flat-keyed dict that
[`so101_policy.py`](../src/openpi/policies/so101_policy.py) expects:

```python
{
  "observation/state":       np.float32[6],
  "observation/image_front": np.uint8[H, W, 3],
  "observation/image_top":   np.uint8[H, W, 3],
  "observation/image_wrist": np.uint8[H, W, 3],
  "prompt": "Pick up the orange ball and place it in the red bucket.",
}
```

## Troubleshooting

- **OOM during compile** — set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` in the pod
  env (`start.sh` defaults to 0.95).
- **Norm-stats step takes forever** — it streams every frame once through the
  video decoder. 5–10 min for this dataset is normal. If it's stuck
  > 20 min, check the network (HF download from the pod).
- **Want to skip norm-stats on relaunch** — once
  `assets/pi05_so101_lora/.../norm_stats.json` exists, set `RUN_NORM_STATS=0`
  to skip it.
- **Volume is filling up** — old checkpoints are kept according to
  `keep_period=5000`, so steps 5000 / 10000 / 15000 survive while intermediate
  ones are pruned. You can tighten this by editing the `pi05_so101_lora`
  config.
