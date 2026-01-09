import dataclasses
import logging
import numpy as np
import imageio
import robosuite
import robocasa
from robosuite.controllers import load_composite_controller_config
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    env_name: str = "PnPCounterToCab"
    max_steps: int = 500
    # Add other arguments as needed

def main(args: Args):
    logging.basicConfig(level=logging.INFO)
    
    # Initialize Policy Client
    logging.info(f"Connecting to policy server at {args.host}:{args.port}")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Configure Robocasa Environment
    # Note: Ensure these configurations match what the model expects (camera names, resolution, etc.)
    config = {
        "env_name": args.env_name,
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
    }

    logging.info(f"Creating environment: {args.env_name}")
    env = robosuite.make(
        **config,
        has_renderer=False, # Set to True if you want to see the simulation window
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_center", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        reward_shaping=False,
        control_freq=20,
    )

    obs = env.reset()
    logging.info("Environment reset. Starting inference loop...")
    
    frames = []

    for step in range(args.max_steps):
        # 1. Process observation data to match model input
        # Note: OpenPI usually expects images to be uint8 [0, 255]
        # Robosuite returns images in [0, 255] uint8 usually, but check if they are flipped.
        # Robosuite images are often flipped vertically compared to standard CV2/PIL.
        
        agentview_img = obs["robot0_agentview_center_image"]
        wrist_img = obs["robot0_eye_in_hand_image"]
        
        if step % 2 == 0:
            # Robosuite images are upside down, verify if flipping is needed
            # Usually for saving to video we might want to flip them to look correct
            # if the raw output is inverted. Based on experience, robosuite offscreen
            # render might need `np.flipud`.
            # Let's try saving as is first, or flip if it looks upside down.
            # agentview_img is (H, W, 3)
            frames.append(np.flipud(agentview_img))

        # Flip images if necessary (Robosuite renders upside down by default in some versions)
        # agentview_img = np.flipud(agentview_img)
        # wrist_img = np.flipud(wrist_img)

        state = np.concatenate([obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]])
        
        # Construct request for Policy Server
        request = {
            "observation/image": agentview_img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            # Instruction should ideally come from the task definition or user input
            "prompt": "put the object in the cabinet" 
        }

        # 2. Get action from policy
        # The client handles serialization/deserialization
        # action is typically [chunk_size, action_dim], we need the first action [action_dim]
        response = client.infer(request)
        action = np.array(response["actions"][0])  # Take the first action and make writable
        # logging.info(f"Received action shape: {action.shape}")
        
        # 3. Execute action
        # Note: OpenPI output actions might need denormalization if the model outputs normalized actions.
        # However, if the policy server handles denormalization (which it often does if configured correctly),
        # we can use the action directly.
        # Also check if the action format (delta pos vs absolute pos) matches the controller config.
        
        obs, reward, done, info = env.step(action)
        
        if step % 10 == 0:
            logging.info(f"Step {step}: Reward={reward}")

        if done:
            logging.info("Episode finished.")
            break

    logging.info(f"Saving video with {len(frames)} frames to output.mp4...")
    try:
        imageio.mimsave("output.mp4", frames, fps=10)
        logging.info("Video saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save video: {e}")

    env.close()

if __name__ == "__main__":
    main(tyro.cli(Args))
