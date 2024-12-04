# scripts/download_dataset.py
import os
import sys
import json
import requests
from tqdm import tqdm
from ..data.processing.data_retrieval import (
    load_annotations, load_objects, _load_object_paths,
    BASE_PATH
)

if __name__ == "__main__":
    # Create base directory
    os.makedirs(BASE_PATH, exist_ok=True)
    
    # Load object paths
    print("Loading object paths...")
    object_paths = _load_object_paths()
    
    # Select first 50k models
    uids = []
    for i in range(10):
        folder = f"glbs/000-{str(i).zfill(3)}"
        folder_uids = [k for k, v in object_paths.items() if v.startswith(folder)]
        uids.extend(folder_uids)
        print(f"Added {len(folder_uids)} UIDs from folder {folder}")
    
    print(f"\nTotal UIDs selected: {len(uids)}")
    
    # Estimate size
    print("\nEstimating total download size...")
    sample_size = 10
    total_size = 0
    
    for uid in tqdm(uids[:sample_size], desc="Sampling files"):
        url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths[uid]}"
        try:
            response = requests.head(url, allow_redirects=True)
            if 'content-length' in response.headers:
                total_size += int(response.headers['content-length'])
        except:
            continue
    
    estimated_gb = (total_size / sample_size * len(uids)) / (1024**3)
    print(f"Estimated total download size: {estimated_gb:.2f} GB")
    
    # Confirm download
    if input("\nProceed with download? (y/n): ").lower() != 'y':
        print("Download cancelled")
        sys.exit(0)
    
    # Download
    print("\nLoading annotations...")
    annotations = load_annotations(uids)
    print(f"Loaded annotations for {len(annotations)} models")
    
    print("\nDownloading models...")
    objects = load_objects(uids, download_processes=12)  # Using 12 cores
    
    # Save annotations
    annotations_file = os.path.join(BASE_PATH, "annotations.json")
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nDownload complete:")
    print(f"- Successfully downloaded: {len(objects)} objects")
    print(f"- Failed downloads: {len([uid for uid in uids if uid not in objects])}")
    print(f"- Files saved to: {BASE_PATH}")
    print(f"- Annotations saved to: {annotations_file}")