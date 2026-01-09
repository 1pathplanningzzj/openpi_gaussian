import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robocasa_example() -> dict:
    """Creates a random input example for the Robocasa policy."""
    return {
        "observation/state": np.random.rand(9),  # 7 joints + 2 gripper?
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something in robocasa",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    """
    Transforms Robocasa data to the format expected by the model.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images
        # Robocasa typically provides "agentview" (base) and "robot0_eye_in_hand" (wrist)
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad right wrist with zeros as Robocasa (Panda) usually has 1 wrist cam
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Handle actions (only available during training)
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """
    Transforms model outputs back to Robocasa format.
    """
    
    # Robocasa action dim (e.g. 12 for mobile manipulation)
    action_dim: int = 12

    def __call__(self, data: dict) -> dict:
        # Slice the actions to the correct dimension
        return {"actions": data["actions"][:, : self.action_dim]}
