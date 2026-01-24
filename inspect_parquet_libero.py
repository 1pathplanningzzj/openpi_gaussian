import pandas as pd
import numpy as np

try:
    parquet_path = "/data/zijianzhang/LIBERA/data/chunk-000/episode_000001.parquet"
    df = pd.read_parquet(parquet_path)
    
    print("Columns:", df.columns.tolist())
    
    if 'observation.state' in df.columns:
        state = df['observation.state'].iloc[0]
        print("observation.state type:", type(state))
        if isinstance(state, (np.ndarray, list)):
             print("observation.state shape/len:", len(state))
             print("observation.state sample:", state)
        # Check dictionary columns usually found in LeRobot formatting
        for col in df.columns:
             if 'state' in col:
                  print(f"Column {col} sample: {df[col].iloc[0]}")
    
    # Check image columns
    image_cols = [c for c in df.columns if 'image' in c]
    print("Image columns:", image_cols)
    
except Exception as e:
    print(f"Error: {e}")
