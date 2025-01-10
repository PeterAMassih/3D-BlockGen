# blockgen/data/processing/data_retrieval.py
# This file was mostly taken from the objaverse github repository. It contains functions to load the metadata and download the objects from the dataset.

import gzip
import json
import multiprocessing
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import time

# Save to current directory
BASE_PATH = os.path.join(os.getcwd(), "objaverse_data")
_VERSIONED_PATH = os.path.join(BASE_PATH, "hf-objaverse-v1")
DOWNLOADED_FILES = os.path.join(BASE_PATH, "downloaded_files.json")
FAILED_FILES = os.path.join(BASE_PATH, "failed_files.json")

def load_annotations(uids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load the full metadata of all objects in the dataset."""
    metadata_path = os.path.join(_VERSIONED_PATH, "metadata")
    object_paths = _load_object_paths()
    dir_ids = (
        set([object_paths[uid].split("/")[1] for uid in uids])
        if uids is not None
        else [f"{i // 1000:03d}-{i % 1000:03d}" for i in range(160)]
    )
    
    out = {}
    for i_id in tqdm(dir_ids, desc="Loading annotations"):
        json_file = f"{i_id}.json.gz"
        local_path = os.path.join(metadata_path, json_file)
        if not os.path.exists(local_path):
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            data = json.load(f)
            if uids is not None:
                data = {uid: data[uid] for uid in uids if uid in data}
            out.update(data)
            
    return out

def _load_object_paths() -> Dict[str, str]:
    """Load the object paths from the dataset."""
    object_paths_file = "object-paths.json.gz"
    local_path = os.path.join(_VERSIONED_PATH, object_paths_file)
    if not os.path.exists(local_path):
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        return json.load(f)

def _download_with_retry(url: str, tmp_path: str, max_retries: int = 3, delay: int = 5) -> bool:
    """Download with retries."""
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, tmp_path)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 504 and attempt < max_retries - 1:  # Gateway Timeout
                time.sleep(delay)
                continue
            return False
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            return False
    return False

def _download_object(args: Tuple) -> Tuple[str, Optional[str]]:
    """Download a single object with retries."""
    uid, object_path = args
    
    try:
        local_path = os.path.join(_VERSIONED_PATH, object_path)
        tmp_local_path = os.path.join(_VERSIONED_PATH, object_path + ".tmp")
        
        # Skip if already downloaded
        if os.path.exists(local_path):
            return uid, local_path
            
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_path}"
        
        os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
        
        # Try download with retries
        if _download_with_retry(hf_url, tmp_local_path):
            os.rename(tmp_local_path, local_path) # This is atomic like this we do write corrupted files
            return uid, local_path
        else:
            if os.path.exists(tmp_local_path):
                os.remove(tmp_local_path) # remove if not downloaded correctly
            return uid, None
            
    except Exception as e:
        print(f"Error downloading {uid}: {str(e)}")
        if os.path.exists(tmp_local_path):
            os.remove(tmp_local_path)
        return uid, None

def load_objects(uids: List[str], download_processes: int = 1) -> Dict[str, str]:
    """Download objects and return their paths."""
    object_paths = _load_object_paths()
    downloaded = {}
    failed = {}
    
    # Load previous state
    if os.path.exists(DOWNLOADED_FILES):
        try:
            with open(DOWNLOADED_FILES, 'r') as f:
                downloaded = json.load(f)
            print(f"Found {len(downloaded)} previously downloaded files")
        except:
            print("Starting fresh download")
            
    if os.path.exists(FAILED_FILES):
        try:
            with open(FAILED_FILES, 'r') as f:
                failed = json.load(f)
            print(f"Found {len(failed)} previously failed downloads")
        except:
            pass
    
    # Prepare download list
    to_download = []
    for uid in uids:
        if uid.endswith(".glb"):
            uid = uid[:-4]
        if uid not in object_paths:
            continue
            
        if uid in downloaded and os.path.exists(downloaded[uid]):
            continue
            
        to_download.append((uid, object_paths[uid]))
    
    if not to_download:
        return downloaded
    
    print(f"Downloading {len(to_download)} files with {download_processes} processes")
    
    # Download files
    with multiprocessing.Pool(download_processes) as pool:
        for i, result in enumerate(tqdm(
            pool.imap_unordered(_download_object, to_download),
            total=len(to_download),
            desc="Downloading"
        )):
            uid, path = result
            if path is not None:
                downloaded[uid] = path
            else:
                failed[uid] = time.time()
                
            # Save progress periodically
            if i % 100 == 0:
                with open(DOWNLOADED_FILES, 'w') as f:
                    json.dump(downloaded, f)
                with open(FAILED_FILES, 'w') as f:
                    json.dump(failed, f)
    
    # Save final state
    with open(DOWNLOADED_FILES, 'w') as f:
        json.dump(downloaded, f)
    with open(FAILED_FILES, 'w') as f:
        json.dump(failed, f)
        
    print(f"\nDownload Summary:")
    print(f"- Successfully downloaded: {len(downloaded)}")
    print(f"- Failed downloads: {len(failed)}")
    
    return downloaded