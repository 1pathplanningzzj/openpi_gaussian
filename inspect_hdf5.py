import h5py
import sys

file_path = "/data/zijianzhang/robocasa/demo_gentex_im128_randcams.hdf5"

try:
    with h5py.File(file_path, "r") as f:
        print(f"Keys in root: {list(f.keys())}")
        if "data" in f:
            print(f"Keys in 'data': {list(f['data'].keys())[:5]}")
            first_demo = list(f['data'].keys())[0]
            print(f"Keys in first demo ({first_demo}): {list(f['data'][first_demo].keys())}")
            if "obs" in f['data'][first_demo]:
                print(f"Keys in obs: {list(f['data'][first_demo]['obs'].keys())}")
        elif "demo" in f: # Just guessing other structures
             print(f"Keys in 'demo': {list(f['demo'].keys())}")
        
        # Check attrs
        print(f"Attributes: {list(f.attrs.keys())}")
        if "env_name" in f.attrs:
            print(f"Env name: {f.attrs['env_name']}")

except Exception as e:
    print(f"Error reading file: {e}")
