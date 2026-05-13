import tkinter as tk

import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from PIL import Image
from PIL import ImageTk
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
        self._root.resizable(width=False, height=False)
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
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]
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
