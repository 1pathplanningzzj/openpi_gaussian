import numpy as np

from openpi.models import model as _model
from openpi.policies import robocasa_policy


def test_robocasa_inputs_accepts_beingh_eval_format():
    transform = robocasa_policy.RobocasaInputs(model_type=_model.ModelType.PI05)

    data = {
        "state.eef_position": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "state.eef_rotation": np.array([[0.0, 0.0, np.pi]], dtype=np.float32),
        "state.gripper_qpos": np.array([[0.1, 0.2]], dtype=np.float32),
        "state.base_position": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
        "state.base_rotation": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "video.left_view": np.zeros((1, 8, 8, 3), dtype=np.uint8),
        "video.wrist_view": np.ones((1, 8, 8, 3), dtype=np.uint8),
        "video.right_view": np.full((1, 8, 8, 3), 2, dtype=np.uint8),
        "language.instruction": ["open the drawer"],
    }

    output = transform(data)

    assert output["state"].shape == (16,)
    assert np.allclose(output["state"][:3], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(output["state"][3:7], np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(output["state"][7:9], np.array([0.1, 0.2], dtype=np.float32))
    assert np.allclose(output["state"][9:12], np.array([4.0, 5.0, 6.0], dtype=np.float32))
    assert np.allclose(output["state"][12:16], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    assert output["image"]["base_0_rgb"].shape == (8, 8, 3)
    assert output["image"]["left_wrist_0_rgb"].shape == (8, 8, 3)
    assert output["image"]["right_wrist_0_rgb"].shape == (8, 8, 3)
    assert bool(output["image_mask"]["right_wrist_0_rgb"])
    assert output["prompt"] == "open the drawer"
