# scripts/download_dataset.py
import os
import sys
import json
import requests
from tqdm import tqdm
from blockgen.data.processing.data_retrieval import (
    load_annotations, load_objects, _load_object_paths,
    BASE_PATH
)

if __name__ == "__main__":
    # Create base directory
    os.makedirs(BASE_PATH, exist_ok=True)
    
    # Load object paths
    print("Loading object paths...")
    object_paths = _load_object_paths()

    # Count total UIDs across all folders
    folder_counts = {}
    total_uids = 0
    
    for uid, path in object_paths.items():
        if path.startswith('glbs/'):
            folder = path.split('/')[1][:7]  # Get XXX-YYY part
            folder_counts[folder] = folder_counts.get(folder, 0) + 1
            total_uids += 1
    
    print(f"\nTotal UIDs in dataset: {total_uids}")
    print(f"Number of folders with models: {len(folder_counts)}")
    
    # Print some statistics
    print("\nFolder statistics:")
    print(f"- Average models per folder: {total_uids/len(folder_counts):.1f}")
    print(f"- Folders with most models:")
    for folder, count in sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {folder}: {count} models")
    
    # Select 200k models sequentially
    uids = []
    target_count = 200_000
    current_folder_num = 0  # This will go from 0 to 159 (inclusive)
    
    while len(uids) < target_count and current_folder_num < len(folder_counts):
        # Create folder path like "glbs/000-000" TODO change if needed if new glbs are added for now hardcoded
        folder = f"glbs/000-{str(current_folder_num).zfill(3)}"
        
        # Get UIDs from this folder
        folder_uids = [k for k, v in object_paths.items() if v.startswith(folder)]
        
        if folder_uids:  # Only add and print if we found UIDs in this folder
            uids.extend(folder_uids)
            print(f"Added {len(folder_uids)} UIDs from folder {folder}")
            print(f"Total UIDs so far: {len(uids)}")
        
        current_folder_num += 1  # Move to next folder number
    
    # Trim to exactly 200k if we went over
    uids = uids[:target_count]
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