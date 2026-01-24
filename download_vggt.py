
import os
import requests
from tqdm import tqdm

def download_file(url, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")
    
    local_filename = os.path.join(folder, filename)
    print(f"Downloading {url} to {local_filename}...")
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024 # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            
            with open(local_filename, 'wb') as f:
                for data in r.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
            
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading file: {e}")

if __name__ == "__main__":
    # check if HF_ENDPOINT is set, otherwise default to hf-mirror.com for speed in China
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    url = f"{hf_endpoint}/facebook/VGGT-1B/resolve/main/model.pt"
    
    target_folder = "/data/zijianzhang/official_ckpts/"
    filename = "VGGT-1B_model.pt" # Renaming it to be specific just in case, or keep model.pt? 
    # The original code looks for 'model.pt' inside a VGGT-1B folder or similar. 
    # Let's keep the name simple or match what the code expects if I change the code.
    # I will save it as 'vggt_1b_model.pt' to be clear, and update the code to point to it.
    filename = "vggt_1b_model.pt"
    
    download_file(url, target_folder, filename)
