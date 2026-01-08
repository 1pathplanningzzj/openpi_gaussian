#!/usr/bin/env python3
"""预下载数据集到本地缓存"""

import os
import time
import sys
from pathlib import Path

def setup_environment():
    """设置优化环境"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DOWNLOAD_MAX_WORKERS'] = '1'
    os.environ['HF_DATASETS_DOWNLOAD_MAX_WORKERS'] = '1'
    os.environ['HF_HUB_DISABLE_TQDM'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
    os.environ['HF_HUB_DOWNLOAD_RETRY'] = '10'
    return True

def download_with_retry(repo_id, max_episodes=50):
    """带重试的下载"""
    from huggingface_hub import HfApi, hf_hub_download
    import requests
    
    api = HfApi()
    
    # 获取文件列表
    print(f"获取文件列表: {repo_id}")
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        parquet_files = [f for f in files if f.endswith('.parquet')]
        print(f"找到 {len(parquet_files)} 个parquet文件")
        
        # 只下载前 max_episodes 个文件
        files_to_download = parquet_files[:max_episodes]
        print(f"将下载前 {len(files_to_download)} 个文件")
        
    except Exception as e:
        print(f"无法获取文件列表: {e}")
        return False
    
    # 逐个下载文件
    success_count = 0
    for i, filename in enumerate(files_to_download):
        try:
            print(f"\n[{i+1}/{len(files_to_download)}] 下载: {filename}")
            start_time = time.time()
            
            # 下载到缓存
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=None,  # 使用默认缓存
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            elapsed = time.time() - start_time
            print(f"    完成 ({elapsed:.1f}s)")
            success_count += 1
            
            # 添加延迟，避免请求过快
            if i < len(files_to_download) - 1:
                time.sleep(1)
                
        except Exception as e:
            print(f"    失败: {e}")
            # 继续下一个文件
    
    print(f"\n✅ 下载完成: {success_count}/{len(files_to_download)} 个文件")
    return success_count > 0

def main(repo_id, max_episodes=50):
    """主函数"""
    print("=" * 60)
    print(f"预下载数据集: {repo_id}")
    print(f"最大episodes: {max_episodes}")
    print("=" * 60)
    
    setup_environment()
    
    # 清理可能的老缓存
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        print(f"缓存目录: {cache_dir}")
    
    success = download_with_retry(repo_id, max_episodes)
    
    if success:
        print("\n🎉 预下载完成！现在可以运行计算脚本了。")
        print("运行命令:")
        print(f"  python compute_norm_stats.py --config-name your_config --max-frames 10000")
    else:
        print("\n❌ 预下载失败")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="数据集ID，如 physical-intelligence/libero")
    parser.add_argument("--max-episodes", type=int, default=50, 
                       help="最大下载的episode数量")
    args = parser.parse_args()
    
    main(args.repo_id, args.max_episodes)