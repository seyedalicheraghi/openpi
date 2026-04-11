# Running ALOHA Sim Inference with Live Visualization

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with ≥8GB VRAM (e.g. RTX 3060, 3090, 4090)
- A desktop/display environment (X11 or Wayland) for the live window
- `uv` package manager

### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

---

## Step 1 — Clone the repo

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi
```

---

## Step 2 — Set up the main environment

This installs the π₀ model and policy server dependencies:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

---

## Step 3 — Set up the simulation environment

The simulator needs its own Python 3.10 venv (separate from the main env):

```bash
uv venv --python 3.10 examples/aloha_sim/.venv
uv pip sync --python examples/aloha_sim/.venv/bin/python examples/aloha_sim/requirements.txt
uv pip install --python examples/aloha_sim/.venv/bin/python -e packages/openpi-client
```

Install PyQt5 and OpenCV (needed by the live viewer):
```bash
uv pip install --python examples/aloha_sim/.venv/bin/python pyqt5 opencv-python-headless
```

---

## Step 4 — Add the live viewer

The repo doesn't include a live viewer by default. Create `examples/aloha_sim/viewer.py`:

```python
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class LiveViewer(_subscriber.Subscriber):
    """Displays simulation frames in a live tkinter window (main-thread driven)."""

    def __init__(self, scale: int = 3) -> None:
        self._scale = scale
        self._root = None
        self._label = None
        self._tk_img = None

    @override
    def on_episode_start(self) -> None:
        w, h = 224 * self._scale, 224 * self._scale
        self._root = tk.Tk()
        self._root.title("ALOHA Sim — π₀ Policy Inference")
        self._root.resizable(False, False)
        blank = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
        self._tk_img = ImageTk.PhotoImage(blank)
        self._label = tk.Label(self._root, image=self._tk_img, bg="black")
        self._label.pack()
        self._root.update()

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        if self._root is None:
            return
        im = observation["images"]["cam_high"]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))        # [H, W, C]
        w, h = 224 * self._scale, 224 * self._scale
        pil = Image.fromarray(im).resize((w, h), Image.NEAREST)
        new_img = ImageTk.PhotoImage(pil)
        self._label.configure(image=new_img)
        self._label.image = new_img  # keep reference
        self._root.update()

    @override
    def on_episode_end(self) -> None:
        if self._root:
            self._root.destroy()
            self._root = None
```

Then edit `examples/aloha_sim/main.py` to import and register it. Find this block:

```python
import saver as _saver
import tyro
```

Change it to:

```python
import saver as _saver
import tyro
import viewer as _viewer
```

And find the `subscribers` list:

```python
        subscribers=[
            _saver.VideoSaver(args.out_dir),
        ],
```

Change it to:

```python
        subscribers=[
            _saver.VideoSaver(args.out_dir),
            _viewer.LiveViewer(),
        ],
```

---

## Step 5 — Run (two terminals)

The policy server and simulation client must run in separate terminals.

**Terminal 1 — start the policy server:**
```bash
uv run scripts/serve_policy.py --env ALOHA_SIM
```

This will download the `pi0_aloha_sim` checkpoint (~6GB) on first run and cache it at `~/.cache/openpi/`. Then it loads the model onto the GPU and starts listening on port 8000. Wait until you see:

```
INFO:websockets.server:server listening on 0.0.0.0:8000
```

**Terminal 2 — run the simulation:**
```bash
MUJOCO_GL=egl DISPLAY=:1 examples/aloha_sim/.venv/bin/python examples/aloha_sim/main.py
```

> **Note:** Replace `DISPLAY=:1` with your actual display (check with `echo $DISPLAY`). If you're running on a desktop normally, you may not need `DISPLAY=:1` at all.

---

## What you'll see

- A **672×672 tkinter window** pops up showing the MuJoCo ALOHA dual-arm robot
- The robot attempts the `AlohaTransferCube-v0` task: pick up the red cube with one arm and hand it to the other
- The episode lasts ~6 seconds (300 frames at 50Hz), then the window closes
- A video of the episode is saved to `data/aloha_sim/videos/out_N.mp4`

To run multiple episodes back-to-back, just re-run the Terminal 2 command. The server stays alive.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `EGL errors` on startup | `sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev` |
| Black window, no frames | Make sure you're on the correct `DISPLAY` — check `echo $DISPLAY` |
| Server OOM on 8GB GPU | The model needs ~8GB VRAM; close other GPU processes first |
| Checkpoint download fails | Check internet connection; downloads ~6GB from Google Cloud Storage |
