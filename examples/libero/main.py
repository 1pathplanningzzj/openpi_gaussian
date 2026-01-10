import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

import csv
import statistics
import os
import datetime

# [Hypothesis Analysis] Imports for 3D Guard Logic
import cv2
import robosuite.utils.camera_utils as camera_utils
import robosuite.utils.transform_utils as T

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

def project_point_to_image(sim, point_3d, camera_name, width, height, final_width, final_height):
    """Projects 3D point to the processed image coordinates (Flipped + Scaled)."""
    P = camera_utils.get_camera_transform_matrix(
        sim=sim, camera_name=camera_name, camera_height=height, camera_width=width
    )
    point_homogeneous = np.append(point_3d, 1.0)
    point_pixel_homogeneous = P @ point_homogeneous
    
    if point_pixel_homogeneous[2] == 0:
        return 0, 0
        
    u = point_pixel_homogeneous[0] / point_pixel_homogeneous[2]
    v_raw = point_pixel_homogeneous[1] / point_pixel_homogeneous[2]
    
    # Transformation: Raw(bottom-up) -> Flipped(H+V) -> Scaled
    # 1. Libero Main logic implies: Img_New[y, x] = Img_Raw[H-1-y, W-1-x]
    #    Since Raw is bottom-up (y=0 bottom), y_raw is from top? No, usually pinhole is top-down.
    #    Let's assume standard Pinhole (v_raw is top-down).
    #    Robosuite Raw Image corresponds to OpenGL (bottom-up). 
    #    So visual feature at v_raw (top-down) is at pixel y_gl = H-v_raw.
    #    Flipping V: y_new = H - 1 - y_gl = H - 1 - (H - v_raw) ~= v_raw.
    #    Flipping H: x_new = W - 1 - u.
    
    u_new = width - 1 - u
    v_new = v_raw
    
    # Scaling
    scale_x = final_width / width
    scale_y = final_height / height
    
    return int(u_new * scale_x), int(v_new * scale_y)

LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    
    # [Plan C] Modified for High Frequency Inference (Closed-Loop)
    # Reducing replan_steps from 2 (or 5) to 1 means we query the expert policy at every single step.
    # This maximizes the response frequency to ~20Hz (Libero native), allowing the robot to react 
    # immediately if the object enters the field of view or slips.
    replan_steps: int = 1 

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_spatial_vis_3d_aware/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    
    # [Hypothesis Analysis] Toggle for 3D Hybrid Policy
    use_3d_guard: bool = False # Enable to test Hypothesis 3 (3D Aware) - LOGGING ONLY if False
    active_3d_takeover: bool = False # Enable to ACTUALLY OVERRIDE policy with 3D logic

    # Experiment control
    max_tasks: int = 0  # 0 means all tasks, set to small number (e.g. 3) for quick experiments
    csv_filename: str = "eval_metrics.csv"  # per-run metrics filename


def _get_target_object_pos(env, task_description):

    """
    Heuristic to find the target object position from simulation ground truth.
    In a real system, this would come from a proper 3D perception module (Detect -> Depth).
    This serves as an 'Oracle' 3D perception for verification.
    """
    sim = env.sim
    # Mapping task description keywords to likely object names in Sim
    # This is a simplification for the verification experiment.
    # Libero Spatial Task 0: "pick up the black bowl..." -> "akita_black_bowl_1_main"
    target_name = None
    task_lower = task_description.lower()
    
    # 扩展对象名称映射
    if "black bowl" in task_lower:
        target_name = "akita_black_bowl_1_main"
    elif "plate" in task_lower:
        target_name = "plate_1_main"
    elif "ramekin" in task_lower:
        target_name = "ramekin_1_main"
    elif "cookie box" in task_lower:
        target_name = "cookie_box_1_main"
    elif "alphabet soup" in task_lower:
        target_name = "alphabet_soup_1_main"
    elif "cream cheese" in task_lower:
        target_name = "cream_cheese_1_main"
    elif "salad dressing" in task_lower:
        target_name = "salad_dressing_1_main"
    elif "bbq sauce" in task_lower:
        target_name = "bbq_sauce_1_main"
    elif "ketchup" in task_lower:
        target_name = "ketchup_1_main"
    elif "tomato sauce" in task_lower:
        target_name = "tomato_sauce_1_main"
    elif "butter" in task_lower:
        target_name = "butter_1_main"
    elif "milk" in task_lower:
        target_name = "milk_1_main"
    elif "chocolate pudding" in task_lower:
        target_name = "chocolate_pudding_1_main"
    elif "orange juice" in task_lower:
        target_name = "orange_juice_1_main"
    elif "wine bottle" in task_lower:
        target_name = "wine_bottle_1_main"
    elif "bowl" in task_lower and "black" not in task_lower:
        # 通用 bowl，尝试查找
        target_name = "bowl_1_main"
    
    if target_name is None:
        # 如果找不到，尝试查找所有 body 名称中包含关键词的
        try:
            keywords = task_lower.split()
            for body_name in sim.model.body_names:
                if any(keyword in body_name.lower() for keyword in keywords if len(keyword) > 3):
                    try:
                        obj_id = sim.model.body_name2id(body_name)
                        return sim.data.body_xpos[obj_id]
                    except:
                        continue
        except:
            pass
        return None

    try:
        obj_id = sim.model.body_name2id(target_name)
        return sim.data.body_xpos[obj_id]
    except ValueError:
        return None

def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.max_tasks > 0:
        num_tasks = min(num_tasks_in_suite, args.max_tasks)
    else:
        num_tasks = num_tasks_in_suite
    logging.info(f"Task suite: {args.task_suite_name} (using {num_tasks} tasks)")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            # Per-episode analytics
            num_takeovers = 0
            misalignment_warnings = 0
            alignment_list = []
            policy_action_norms = []
            takeover_action_norms = []
            
            # [Enhanced Analysis] Initialize per-episode metrics
            angle_errors = []
            distance_errors = []
            target_in_view_list = []
            true_distances = []
            predicted_distances = []
            alignment_at_distance = []

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            
            # [Plan C Analysis] Metrics for High Frequency Utility
            prev_full_chunk = None
            corrections = []
            
            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        
                        # [Enhanced Analysis] 2D-3D Projection Error Analysis
                        # [Hypothesis Analysis] 3D Guard Logic
                        if args.use_3d_guard:
                            target_pos_3d = _get_target_object_pos(env, task_description)
                            
                            if target_pos_3d is not None:
                                current_eef_pos = obs["robot0_eef_pos"]
                                
                                # 1. 计算 policy 预测的轨迹终点（累积 action_chunk）
                                predicted_end_pos = current_eef_pos.copy()
                                for action in action_chunk[:10]:  # 只看前 10 步
                                    predicted_end_pos = predicted_end_pos + action[:3]
                                
                                # 2. 计算真实目标方向
                                true_direction = target_pos_3d - current_eef_pos
                                true_distance = np.linalg.norm(true_direction)
                                
                                # 3. 计算 policy 预测方向
                                predicted_direction = predicted_end_pos - current_eef_pos
                                predicted_distance = np.linalg.norm(predicted_direction)
                                
                                # 4. 计算方向误差（角度，度）
                                if true_distance > 1e-6 and predicted_distance > 1e-6:
                                    cos_sim = np.dot(true_direction, predicted_direction) / (true_distance * predicted_distance)
                                    cos_sim = np.clip(cos_sim, -1, 1)
                                    angle_error = np.arccos(cos_sim) * 180 / np.pi
                                else:
                                    angle_error = 0.0
                                
                                # 5. 计算距离误差
                                distance_error = abs(predicted_distance - true_distance)
                                
                                # 6. 将 3D 目标投影到 2D 图像，看是否在视野内
                                u_target, v_target = project_point_to_image(
                                    env.sim, target_pos_3d, "agentview",
                                    LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION,
                                    args.resize_size, args.resize_size
                                )
                                in_view = (0 <= u_target < args.resize_size and 0 <= v_target < args.resize_size)
                                
                                # 7. 记录指标
                                angle_errors.append(float(angle_error))
                                distance_errors.append(float(distance_error))
                                target_in_view_list.append(int(in_view))
                                true_distances.append(float(true_distance))
                                predicted_distances.append(float(predicted_distance))
                                
                                # Visualization Logic
                                if len(replay_images) > 0:
                                    # Get the current frame (mutable copy reference from list? No, numpy array in list is mutable)
                                    # We modify the last image added to replay_images directly
                                    vis_img = replay_images[-1].copy() # Copy to avoid messing up history if ref used elsewhere?
                                    # Actually replay_images IS the history. We want the video to have arrows.
                                    # So we modify replay_images[-1] in place or replace it.
                                    
                                    # Visualization - "Red Tentacle" (Predicted Trajectory)
                                    # We simulate the future path the policy wants to take
                                    predicted_path_pixels = []
                                    sim_eef_pos = current_eef_pos.copy()
                                    
                                    # Start point
                                    u_start, v_start = project_point_to_image(env.sim, sim_eef_pos, "agentview", 
                                                                  LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 
                                                                  args.resize_size, args.resize_size)
                                    predicted_path_pixels.append((u_start, v_start))

                                    # Project future N steps (e.g. 20 steps to avoid clutter, or all)
                                    # Assuming standard Libero action space (x,y,z delta in world frame)
                                    # And assuming scale=1.0 for simplicity (Libero env actions usually roughly correspond to units)
                                    # If actions are normalized, this might look short, but let's try.
                                    vis_steps = min(len(action_chunk), 20) 
                                    vis_scale_factor = 10.0 # Exaggerate the movement for visualization clarity if needed
                                    
                                    for i in range(vis_steps):
                                        action_vec = action_chunk[i][:3]
                                        # Update sim_eef_pos cumulatively
                                        # Note: This ignores physics/collisions, just visualizing 'intent'
                                        sim_eef_pos = sim_eef_pos + action_vec * vis_scale_factor 
                                        
                                        u_next, v_next = project_point_to_image(env.sim, sim_eef_pos, "agentview", 
                                                                      LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 
                                                                      args.resize_size, args.resize_size)
                                        predicted_path_pixels.append((u_next, v_next))
                                    
                                    # Draw Red Trajectory (Policy Intent)
                                    for i in range(len(predicted_path_pixels) - 1):
                                        pt1 = predicted_path_pixels[i]
                                        pt2 = predicted_path_pixels[i+1]
                                        cv2.line(vis_img, pt1, pt2, (255, 0, 0), 1) # Red line, thickness 1
                                    
                                    # Draw Green Arrow (True Target Direction) - Keep this as reference
                                    vec_to_obj = target_pos_3d - current_eef_pos
                                    if np.linalg.norm(vec_to_obj) > 1e-6:
                                        # Draw a simpler green line to object center
                                        u_obj, v_obj = project_point_to_image(env.sim, target_pos_3d, "agentview",
                                                                            LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION,
                                                                            args.resize_size, args.resize_size)
                                        cv2.arrowedLine(vis_img, (u_start, v_start), (u_obj, v_obj), (0, 255, 0), 2, tipLength=0.1)

                                        # Add Text Info
                                        dist_val = np.linalg.norm(vec_to_obj)
                                        cv2.putText(vis_img, f"Dist: {dist_val:.3f}", (10, 210), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                                        
                                    
                                    # Replace the image in replay buffer
                                    replay_images[-1] = vis_img

                                # Guard Logic
                                # Heuristic: If we are close to the object (xy plane), try to align
                                # Calculate simple vector to object
                                vec_to_obj = target_pos_3d - current_eef_pos
                                dist_to_obj = np.linalg.norm(vec_to_obj)
                                
                                # Policy proposed action (delta pos)
                                policy_action = action_chunk[0][:3]
                                
                                if np.linalg.norm(policy_action) > 1e-3:
                                    # record policy action magnitude
                                    policy_action_norms.append(np.linalg.norm(policy_action))
                                    policy_dir = policy_action / np.linalg.norm(policy_action)
                                    perfect_dir = vec_to_obj / (dist_to_obj + 1e-6)
                                    
                                    # Cosine similarity
                                    alignment = np.dot(policy_dir, perfect_dir)
                                    alignment_list.append(float(alignment))
                                    
                                    # [Enhanced Analysis] 记录对齐度和距离的关系
                                    alignment_at_distance.append({
                                        'distance': float(dist_to_obj),
                                        'alignment': float(alignment)
                                    })
                                    
                                    # Condition for Takeover: Close enough (<10cm) AND Misaligned (<0.5 cos sim)
                                    if alignment < 0.5 and dist_to_obj < 0.10: 
                                         msg = f"3D GUARD: Misalignment detected! Dist={dist_to_obj:.3f}, CosSim={alignment:.3f}. Policy might miss."
                                         logging.warning(msg)
                                         misalignment_warnings += 1
                                         
                                         if args.active_3d_takeover:
                                             logging.info(">>> ACTIVATING 3D TAKEOVER >>> Overriding Action!")
                                             
                                             # Construct the Perfect Action (P-Controller)
                                             # Move towards object with a safe speed (similar to policy's usual magnitude)
                                             # Usually around 0.05 speed? Let's take norm of policy action to match speed
                                             speed = np.linalg.norm(policy_action)
                                             # Or fixed small speed to be safe
                                             speed = max(speed, 0.02) 
                                             
                                             new_action_vec = perfect_dir * speed
                                             takeover_action_norms.append(np.linalg.norm(new_action_vec))
                                             num_takeovers += 1
                                             
                                             # Overwrite the first step of the chunk (since we replan often)
                                             # Keep the gripper action (index 6, which is -1 for open usually)
                                             original_gripper = action_chunk[0][6]
                                             
                                             # We need to construct a full 7-dim action
                                             # Rotation: Keep policy's rotation (indices 3,4,5)? Or zero it?
                                             # Let's trust policy for rotation, only fix translation.
                                             fix_action = np.concatenate([new_action_vec, action_chunk[0][3:6], [original_gripper]])
                                             
                                             # OVERRIDE
                                             # action_chunk[0] might be immutable (tuple/read-only numpy), convert to list of numpy arrays if needed
                                             # or just update the list element.
                                             # The error "assignment destination is read-only" suggests action_chunk is a read-only numpy array.
                                             # Make it writable copy first.
                                             if not action_chunk.flags.writeable:
                                                action_chunk = action_chunk.copy()
                                             
                                             action_chunk[0] = fix_action
                                             
                                             # Visual confirmation in video
                                             cv2.putText(replay_images[-1], "3D TAKEOVER ACTIVE", (50, 50), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                        
                        # [Plan C Analysis] Compute Correction Magnitude
                        # We compare the action we are about to take (action_chunk[0]) 
                        # with what we *planned* to take at this timestamp in the previous step (prev_full_chunk[1])
                        # This tells us if replanning actually changed the decision.
                        if prev_full_chunk is not None and len(prev_full_chunk) > 1:
                            # We compare only the first 6 dimensions (pose), ignoring gripper for the norm
                            planned_action = np.array(prev_full_chunk[1][:6])
                            new_action = np.array(action_chunk[0][:6])
                            diff = np.linalg.norm(new_action - planned_action)
                            corrections.append(diff)
                        
                        prev_full_chunk = action_chunk

                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Save per-episode metrics to CSV
            avg_alignment = float(np.mean(alignment_list)) if alignment_list else float('nan')
            avg_correction = float(np.mean(corrections)) if corrections else float('nan')
            avg_policy_action_norm = float(np.mean(policy_action_norms)) if policy_action_norms else float('nan')
            avg_takeover_action_norm = float(np.mean(takeover_action_norms)) if takeover_action_norms else float('nan')
            
            # [Enhanced Analysis] 计算新增指标的平均值
            avg_angle_error = float(np.mean(angle_errors)) if angle_errors else float('nan')
            avg_distance_error = float(np.mean(distance_errors)) if distance_errors else float('nan')
            target_in_view_ratio = float(np.mean(target_in_view_list)) if target_in_view_list else float('nan')
            avg_true_distance = float(np.mean(true_distances)) if true_distances else float('nan')
            avg_predicted_distance = float(np.mean(predicted_distances)) if predicted_distances else float('nan')
            
            # 计算接近目标时的对齐度（距离 < 10cm）
            close_alignments = [a['alignment'] for a in alignment_at_distance 
                              if a['distance'] < 0.10 and not np.isnan(a['alignment'])]
            avg_alignment_close = float(np.mean(close_alignments)) if close_alignments else float('nan')

            episode_metrics = {
                "timestamp": datetime.datetime.now().isoformat(),
                "task_id": int(task_id),
                "task_description": task_description,
                "seed": int(args.seed),
                "use_3d_guard": int(args.use_3d_guard),
                "active_3d_takeover": int(args.active_3d_takeover),
                "success": int(done),
                "steps": int(t),
                "num_takeovers": int(num_takeovers),
                "num_misalignment_warnings": int(misalignment_warnings),
                "avg_alignment": avg_alignment,
                "avg_alignment_close": avg_alignment_close,  # 新增：接近目标时的对齐度
                "avg_correction": avg_correction,
                "avg_policy_action_norm": avg_policy_action_norm,
                "avg_takeover_action_norm": avg_takeover_action_norm,
                "avg_angle_error_deg": avg_angle_error,  # 新增：平均角度误差（度）
                "avg_distance_error": avg_distance_error,  # 新增：平均距离误差
                "target_in_view_ratio": target_in_view_ratio,  # 新增：目标在视野内的比例
                "avg_true_distance": avg_true_distance,  # 新增：平均真实距离
                "avg_predicted_distance": avg_predicted_distance,  # 新增：平均预测距离
            }
            csvpath = pathlib.Path(args.video_out_path) / args.csv_filename
            write_header = not csvpath.exists()
            with open(csvpath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(episode_metrics.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(episode_metrics)

            # Log current results
            if corrections:
                avg_correction = np.mean(corrections)
                max_correction = np.max(corrections)
                logging.info(f"Episode Analysis - Avg Correction: {avg_correction:.4f}, Max Correction: {max_correction:.4f}")
                if avg_correction > 0.1: # Threshold depends on action scale, typically actions are normalized or small deltas
                    logging.info("=> HIGH FREQUENCY IMPACT: Large corrections detected. The policy is actively reacting to deviations.")
                else:
                    logging.info("=> LOW IMPACT: Corrections are small. The trajectory is stable.")
            
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    # env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    try:
        if hasattr(env, "seed") and callable(env.seed):
            env.seed(seed)
    except Exception as e:
        logging.warning(f"Could not seed environment: {e}")

    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
