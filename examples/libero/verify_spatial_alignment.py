import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.camera_utils as camera_utils
import robosuite.utils.transform_utils as T

# 设置中文字体 (可选，防止乱码)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def project_point_to_image(sim, point_3d, camera_name, width, height):
    """
    将 3D 点投影到相机图像平面。
    """
    # robosuite 的 camera_utils.get_camera_transform_matrix 返回的是 (World -> Pixel) 的变换矩阵
    P = camera_utils.get_camera_transform_matrix(
        sim=sim, 
        camera_name=camera_name, 
        camera_height=height, 
        camera_width=width
    )
    
    # 构建齐次坐标 [x, y, z, 1]
    point_homogeneous = np.append(point_3d, 1.0)
    
    # 投影: [u*w, v*w, w, w] (或其他形式，关键是前三维)
    point_pixel_homogeneous = P @ point_homogeneous
    
    # 归一化 (除以 w)
    u = point_pixel_homogeneous[0] / point_pixel_homogeneous[2]
    v = point_pixel_homogeneous[1] / point_pixel_homogeneous[2]
    
    return int(u), int(v)

def main():
    # 1. 初始化 Libero 环境
    # 使用 libero_spatial 任务作为测试
    TASK_SUITE_NAME = "libero_spatial"
    TASK_ID = 0
    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256

    print(f"Initializing {TASK_SUITE_NAME}, Task {TASK_ID}...")
    
    # 获取 Benchmark 和 Task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE_NAME]()
    task = task_suite.get_task(TASK_ID)
    init_states = task_suite.get_task_init_states(TASK_ID)

    print(f"Task Name: {task.name}")
    print(f"Task Description: {task.language}")

    # 创建OffScreen环境
    bddl_path = os.path.join(
        get_libero_path("bddl_files"), 
        task.problem_folder, 
        task.bddl_file
    )
    env_args = {
        "bddl_file_name": bddl_path,
        "render_gpu_device_id": 0,
    }

    env = OffScreenRenderEnv(**env_args)
    env.reset()

    # 统计数据
    NUM_EPISODES = 20
    stats = {
        "agentview": {"visible": 0, "total": 0, "out_of_bounds": []},
        "robot0_eye_in_hand": {"visible": 0, "total": 0, "out_of_bounds": []}
    }

    print(f"\nRunning verification for {NUM_EPISODES} episodes...")

    for ep in range(NUM_EPISODES):
        # 随机选择一个初始状态
        init_state_idx = ep % len(init_states)
        env.set_init_state(init_states[init_state_idx])

        # 执行一步
        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])

        # 查找目标物体 (Black Bowl)
        target_object_name = "akita_black_bowl_1_main"
        try:
            obj_body_id = env.sim.model.body_name2id(target_object_name)
            target_pos_3d = env.sim.data.body_xpos[obj_body_id]
        except ValueError:
            print(f"Episode {ep}: Object '{target_object_name}' not found.")
            continue

        cameras_to_verify = ["agentview", "robot0_eye_in_hand"]
        save_this_episode = False
        
        for camera_name in cameras_to_verify:
            image_key = f"{camera_name}_image"
            if image_key in obs:
                image = obs[image_key]
                real_height, real_width = image.shape[:2]

                # 投影
                u, v_raw = project_point_to_image(env.sim, target_pos_3d, camera_name, real_width, real_height)
                v = real_height - v_raw

                is_visible = (0 <= u < real_width) and (0 <= v < real_height)
                
                stats[camera_name]["total"] += 1
                if is_visible:
                    stats[camera_name]["visible"] += 1
                else:
                    stats[camera_name]["out_of_bounds"].append((ep, u, v))
                    save_this_episode = True

        # Save visualization for first episode OR bad cases
        if ep == 0 or save_this_episode:
            print(f"Generating visualization for Episode {ep} (Bad Case: {save_this_episode})...")
            fig, axes = plt.subplots(1, len(cameras_to_verify), figsize=(8 * len(cameras_to_verify), 8))
            if len(cameras_to_verify) == 1:
                axes = [axes]

            for i, camera_name in enumerate(cameras_to_verify):
                image_key = f"{camera_name}_image"
                image = obs[image_key]
                
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = (image * 255).astype(np.uint8)
                real_height, real_width = image.shape[:2]
                
                u, v_raw = project_point_to_image(env.sim, target_pos_3d, camera_name, real_width, real_height)
                v = real_height - v_raw
                
                axes[i].imshow(image)
                axes[i].scatter(u, v, c='yellow', marker='*', s=200, label='Target Object Center')
                axes[i].set_title(f"{camera_name}\nTarget: {target_object_name}\n(u={u}, v={v})")
                axes[i].legend()


if __name__ == "__main__":
    main()
