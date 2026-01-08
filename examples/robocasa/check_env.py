import robosuite
import robocasa
from robosuite.controllers import load_composite_controller_config
import numpy as np

def main():
    config = {
        "env_name": "PnPCounterToCab",
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
    }

    env = robosuite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        reward_shaping=False,
        control_freq=20,
    )

    obs = env.reset()
    print("Observation keys:", obs.keys())
    
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {type(v)}")

if __name__ == "__main__":
    main()
