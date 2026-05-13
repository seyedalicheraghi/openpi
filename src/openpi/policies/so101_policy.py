"""Input/output transforms for the SO-101 6-DoF arm with three cameras (front/top/wrist).

Used by the `pi05_so101_lora` training config and matched at inference time by the
inference client (which must pass the same flat-keyed dict).
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so101_example() -> dict:
    """Random input example matching the SO-101 dataset schema."""
    return {
        "observation/state": np.random.rand(6).astype(np.float32),
        "observation/image_front": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_top": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Pick up the orange ball and place it in the red bucket.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Pack SO-101 raw inputs into the model's expected slots.

    Camera mapping:
      front  -> base_0_rgb        (third-person)
      wrist  -> left_wrist_0_rgb  (gripper-mounted)
      top    -> right_wrist_0_rgb (used as a second exterior view; all three masks True)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front = _parse_image(data["observation/image_front"])
        top = _parse_image(data["observation/image_top"])
        wrist = _parse_image(data["observation/image_wrist"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": top,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Trim model action chunk back to the 6-D SO-101 action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}
