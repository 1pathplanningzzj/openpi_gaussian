import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ROBOCASA_ACTION_DIM = 12
ROBOCASA_ARM_GRIPPER_ACTION_DIM = 7
ROBOCASA_ACTION_TRAINING_MASK = (True,) * ROBOCASA_ARM_GRIPPER_ACTION_DIM + (False,) * (
    ROBOCASA_ACTION_DIM - ROBOCASA_ARM_GRIPPER_ACTION_DIM
)
ROBOCASA_MASKED_ACTION_DEFAULTS = (0.0, 0.0, 0.0, 0.0, -1.0)


@dataclasses.dataclass(frozen=True)
class MaskNonTrainingActions(transforms.DataTransformFn):
    action_mask: tuple[bool, ...] = ROBOCASA_ACTION_TRAINING_MASK

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            actions = np.asarray(data["actions"]).copy()
            mask = np.asarray(self.action_mask, dtype=bool)
            mask_len = min(actions.shape[-1], mask.shape[0])
            actions[..., :mask_len] *= mask[:mask_len].astype(actions.dtype)
            if actions.shape[-1] > mask_len:
                actions[..., mask_len:] = 0
            data["actions"] = actions
        return data


def _parse_image(image) -> np.ndarray:
    if isinstance(image, dict):
        if "bytes" in image:
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image["bytes"]))
            image = np.array(image)
        else:
            raise ValueError(f"Unexpected dict format for image: {image.keys()}")
    elif isinstance(image, (list, tuple)) and len(image) > 0 and isinstance(image[0], dict):
        from PIL import Image
        import io

        frames = []
        for img_dict in image:
            if "bytes" not in img_dict:
                raise ValueError(f"Unexpected dict format for image: {img_dict.keys()}")
            img = Image.open(io.BytesIO(img_dict["bytes"]))
            frames.append(np.array(img))
        image = np.stack(frames, axis=0)
        return image

    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.ndim == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


def _first_present(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Expected one of {keys}, found keys: {sorted(data)}")


def _parse_vector(value, *, expected_dims: tuple[int, ...] | None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {array.shape}")
    if expected_dims is not None and array.shape[0] not in expected_dims:
        raise ValueError(f"{name} must have dims {expected_dims}, got {array.shape[0]}")
    return array


def _axis_angle_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    axis = rotvec / angle
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    return np.array(
        [
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            np.cos(half_angle),
        ],
        dtype=np.float32,
    )


def _parse_rotation(value, *, name: str) -> np.ndarray:
    rotation = _parse_vector(value, expected_dims=(3, 4), name=name)
    if rotation.shape[0] == 3:
        rotation = _axis_angle_to_quaternion(rotation)
    return rotation


def _parse_prompt(data: dict) -> str | None:
    prompt = data.get("prompt", data.get("language.instruction"))
    if prompt is None:
        return None

    if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
        prompt = prompt[0]
    if isinstance(prompt, np.ndarray) and prompt.ndim == 1 and prompt.size == 1:
        prompt = prompt.item()
    return str(prompt)


def _parse_state(data: dict) -> np.ndarray:
    if "observation/state" in data:
        state = _parse_vector(data["observation/state"], expected_dims=None, name="observation/state")
        return state.astype(np.float32)

    compat_keys = (
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    )
    missing = [key for key in compat_keys if key not in data]
    if missing:
        raise KeyError(f"Missing RoboCasa state fields: {missing}")

    eef_position = _parse_vector(data["state.eef_position"], expected_dims=(3,), name="state.eef_position")
    eef_rotation = _parse_rotation(data["state.eef_rotation"], name="state.eef_rotation")
    gripper_qpos = _parse_vector(data["state.gripper_qpos"], expected_dims=(2,), name="state.gripper_qpos")
    base_position = _parse_vector(data["state.base_position"], expected_dims=(3,), name="state.base_position")
    base_rotation = _parse_rotation(data["state.base_rotation"], name="state.base_rotation")

    return np.concatenate([eef_position, eef_rotation, gripper_qpos, base_position, base_rotation], axis=0).astype(
        np.float32
    )


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(_first_present(data, "observation/image", "video.left_view"))
        wrist_image = _parse_image(_first_present(data, "observation/wrist_image", "video.wrist_view"))
        right_image = (
            _parse_image(_first_present(data, "observation/right_image", "video.right_view"))
            if "observation/right_image" in data or "video.right_view" in data
            else np.zeros_like(base_image)
        )

        inputs = {
            "state": _parse_state(data),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": (
                    np.True_
                    if "observation/right_image" in data or "video.right_view" in data or self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if (prompt := _parse_prompt(data)) is not None:
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    action_dim: int = ROBOCASA_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, : self.action_dim]).copy()
        actions[:, ROBOCASA_ARM_GRIPPER_ACTION_DIM :] = np.asarray(
            ROBOCASA_MASKED_ACTION_DEFAULTS,
            dtype=actions.dtype,
        )
        return {"actions": actions}
