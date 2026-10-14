"""
Parallel Downloader for Mip-NeRF 360 `bicycle` Dataset (HuggingFace Mirror)
========================================================================
Downloads:
1. poses_bounds.npy
2. sparse/0/ COLMAP camera parameters (cameras.bin, images.bin, points3D.bin)
3. images_8/ (44 MB - fast iteration) and images_4/ (162 MB - high-res 1.2K)
Total footprint: ~220 MB (safe for local disk space).
"""

import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_HF_URL = "https://huggingface.co/datasets/alexmkwizu/gaussian_training_datasets/resolve/main/mipnerf360/bicycle"
OUTPUT_DIR = "/root/.gemini/antigravity-cli/scratch/wave_engine/data/bicycle"


def download_single_file(rel_path: str, retries: int = 3) -> bool:
    """Downloads a single file from the HF repository."""
    url = f"{BASE_HF_URL}/{rel_path}"
    dest_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
                f.write(resp.read())
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed to download {rel_path}: {e}")
                return False
            time.sleep(1.0)
    return False


def get_hf_file_list(subfolder: str) -> list:
    """Fetches list of file names in a subfolder from HF API."""
    import json
    api_url = f"https://huggingface.co/api/datasets/alexmkwizu/gaussian_training_datasets/tree/main/mipnerf360/bicycle/{subfolder}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
            return [os.path.basename(item["path"]) for item in data]
    except Exception as e:
        print(f"Error listing {subfolder}: {e}")
        return []


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Starting download of Mip-NeRF 360 `bicycle` benchmark dataset")
    print(f"Target directory: {OUTPUT_DIR}")
    print("=" * 70)

    # 1. Download metadata and camera parameters
    critical_files = [
        "poses_bounds.npy",
        "sparse/0/cameras.bin",
        "sparse/0/images.bin",
        "sparse/0/points3D.bin"
    ]
    print("\n[1/3] Downloading COLMAP camera poses and metadata...")
    for f in critical_files:
        print(f"  Fetching {f}...")
        success = download_single_file(f)
        if not success:
            print(f"  Warning: Could not fetch {f}")

    # 2. Query file lists for images_8 and images_4
    print("\n[2/3] Querying image manifests from HuggingFace...")
    files_8 = get_hf_file_list("images_8")
    files_4 = get_hf_file_list("images_4")
    print(f"  Found {len(files_8)} files in images_8")
    print(f"  Found {len(files_4)} files in images_4")

    # 3. Parallel download of images_8 (fast prototyping)
    print("\n[3/3] Downloading images_8 dataset (~44 MB)...")
    download_tasks = [f"images_8/{fn}" for fn in files_8]
    
    start_t = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_single_file, path): path for path in download_tasks}
        for future in as_completed(futures):
            res = future.result()
            if res:
                completed += 1
                if completed % 40 == 0 or completed == len(download_tasks):
                    print(f"  Progress: {completed}/{len(download_tasks)} images downloaded...")

    elapsed = time.time() - start_t
    print(f"\nImages_8 downloaded successfully in {elapsed:.1f}s ({completed} files).")

    # Also download images_4 for high-resolution evaluation
    print("\nDownloading high-res images_4 dataset (~162 MB)...")
    download_tasks_4 = [f"images_4/{fn}" for fn in files_4]
    start_t = time.time()
    completed_4 = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_single_file, path): path for path in download_tasks_4}
        for future in as_completed(futures):
            res = future.result()
            if res:
                completed_4 += 1
                if completed_4 % 40 == 0 or completed_4 == len(download_tasks_4):
                    print(f"  Progress: {completed_4}/{len(download_tasks_4)} images downloaded...")

    elapsed_4 = time.time() - start_t
    print(f"Images_4 downloaded successfully in {elapsed_4:.1f}s ({completed_4} files).")

    # Check disk usage
    total_bytes = sum(
        os.path.getsize(os.path.join(root, file))
        for root, dirs, files in os.walk(OUTPUT_DIR)
        for file in files
    )
    print("\n" + "=" * 70)
    print(f"[COMPLETED] Total dataset size on disk: {total_bytes / (1024 * 1024):.2f} MB")
    print("Dataset structure is ready for Phase 2 Autoencoder training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
