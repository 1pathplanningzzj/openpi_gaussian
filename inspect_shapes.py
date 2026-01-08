import h5py
import sys

file_path = "/data/zijianzhang/robocasa/demo_gentex_im128_randcams.hdf5"

try:
    with h5py.File(file_path, "r") as f:
        demo = f['data']['demo_1']
        obs = demo['obs']
        print(f"robot0_joint_pos shape: {obs['robot0_joint_pos'].shape}")
        print(f"robot0_gripper_qpos shape: {obs['robot0_gripper_qpos'].shape}")
        print(f"robot0_agentview_left_image shape: {obs['robot0_agentview_left_image'].shape}")
        print(f"actions shape: {demo['actions'].shape}")

except Exception as e:
    print(f"Error reading file: {e}")
