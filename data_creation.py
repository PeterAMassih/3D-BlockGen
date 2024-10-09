import os
import json
import trimesh
import numpy as np
import torch
import pandas as pd
import logging
from sklearn.decomposition import PCA
from scipy import ndimage
import objaverse.xl as oxl
import glob
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def improve_voxelization(voxels: np.ndarray) -> np.ndarray:
    filled = ndimage.binary_fill_holes(voxels)
    kernel = np.ones((3, 3, 3))
    closed = ndimage.binary_closing(filled, structure=kernel)
    return closed

def standardize_orientation(voxels: np.ndarray, use_pca: bool = True) -> np.ndarray:
    if not use_pca:
        return voxels
    filled_coords = np.array(np.where(voxels)).T
    pca = PCA(n_components=3)
    pca.fit(filled_coords)
    rotation_matrix = pca.components_
    rotated_coords = np.dot(filled_coords - np.mean(filled_coords, axis=0), rotation_matrix.T)
    rotated_coords -= rotated_coords.min(axis=0)
    rotated_coords /= rotated_coords.max()
    rotated_coords *= (np.array(voxels.shape) - 1)
    new_voxels = np.zeros_like(voxels)
    new_voxels[rotated_coords[:, 0].round().astype(int),
               rotated_coords[:, 1].round().astype(int),
               rotated_coords[:, 2].round().astype(int)] = 1
    return new_voxels

def load_and_voxelize_model(file_path: str, resolution: int = 32) -> np.ndarray:
    try:
        scene = trimesh.load(file_path)
        if isinstance(scene, trimesh.Scene):
            mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                                             for g in scene.geometry.values()])
        else:
            mesh = scene
        voxel_grid = mesh.voxelized(pitch=mesh.bounding_box.extents.max() / resolution)
        voxels = voxel_grid.matrix.astype(bool).astype(int)
        if voxels.shape != (resolution, resolution, resolution):
            pad_width = [(0, max(0, resolution - s)) for s in voxels.shape]
            voxels = np.pad(voxels, pad_width)
            voxels = voxels[:resolution, :resolution, :resolution]
        return voxels
    except Exception as e:
        logging.error(f"Error loading and voxelizing model {file_path}: {str(e)}")
        return None

def save_as_tensor(voxel_grid: np.ndarray, original_filename: str, output_dir: str) -> str:
    tensor = torch.from_numpy(voxel_grid).bool().unsqueeze(0).unsqueeze(0)
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    new_filename = f"{base_name}_voxel_grid.pt"
    full_path = os.path.join(output_dir, new_filename)
    torch.save(tensor, full_path)
    return full_path

def process_single_object(obj, original_dir, processed_dir, resolution, use_pca, timeout=30):
    file_identifier = obj['fileIdentifier']
    start_time = time.time()

    try:
        local_paths = oxl.download_objects(
            objects=pd.DataFrame([obj]),
            download_dir=original_dir
        )
        if not local_paths:
            logging.warning(f"Failed to download {file_identifier}. Skipping.")
            return None

        glb_files = glob.glob(os.path.join(original_dir, '**', '*.glb'), recursive=True)
        if not glb_files:
            logging.warning(f"No GLB files found for {file_identifier}")
            return None

        for glb_file in glb_files:
            if time.time() - start_time > timeout:
                logging.warning(f"Processing timed out for {file_identifier}")
                return None

            voxel_grid = load_and_voxelize_model(glb_file, resolution)
            if voxel_grid is None:
                continue

            voxel_grid = standardize_orientation(voxel_grid, use_pca)
            voxel_grid = improve_voxelization(voxel_grid)

            # Use the sha256 hash from the metadata
            sha256 = obj['sha256']
            new_filename = f"{os.path.splitext(os.path.basename(glb_file))[0]}_{sha256[:8]}_voxel_grid.pt"
            new_file_path = os.path.join(processed_dir, new_filename)

            tensor = torch.from_numpy(voxel_grid).bool().unsqueeze(0).unsqueeze(0)
            torch.save(tensor, new_file_path)

            relative_path = os.path.relpath(glb_file, original_dir)
            processed_metadata = {
                "new_file_path": new_file_path,
                "original_file_path": relative_path,
                "original_url": file_identifier,
                "metadata": obj.to_dict()
            }
            logging.info(f"Successfully processed: {glb_file}")
            return processed_metadata

    except Exception as e:
        logging.error(f"Error processing {file_identifier}: {str(e)}")

    return None

def process_objaverse_dataset(num_objects: int = 10, resolution: int = 32, base_dir: str = 'objaverse_processed', use_pca: bool = True):
    original_dir = os.path.join(base_dir, 'original_models')
    processed_dir = os.path.join(base_dir, 'processed_models')

    for dir_path in [original_dir, processed_dir]:
        os.makedirs(dir_path, exist_ok=True)

    logging.info("Fetching alignment annotations from Objaverse...")
    alignment_annotations = oxl.get_alignment_annotations()
    logging.info(f"Found {len(alignment_annotations)} objects in alignment annotations.")

    non_github_glb_objects = alignment_annotations[
        (alignment_annotations['fileType'] == 'glb') & 
        (alignment_annotations['source'] != 'github')
    ]
    logging.info(f"Found {len(non_github_glb_objects)} non-GitHub GLB objects.")

    processed_objects = []
    attempted_objects = 0
    total_attempts = 0

    pbar = tqdm(total=num_objects, desc="Processing objects")

    while len(processed_objects) < num_objects and total_attempts < len(non_github_glb_objects):
        obj = non_github_glb_objects.iloc[total_attempts]
        total_attempts += 1
        result = process_single_object(obj, original_dir, processed_dir, resolution, use_pca)
        if result:
            processed_objects.append(result)
            pbar.update(1)
        else:
            logging.warning(f"Skipping object {obj['fileIdentifier']} due to errors")

    pbar.close()

    logging.info(f"Successfully processed {len(processed_objects)} objects after {total_attempts} attempts.")

    json_path = os.path.join(base_dir, 'processed_objects_metadata.json')
    with open(json_path, 'w') as f:
        json.dump(processed_objects, f, indent=2)

    logging.info(f"Processing complete. Metadata saved to {json_path}")
    return processed_objects

if __name__ == "__main__":
    processed_objects = process_objaverse_dataset(num_objects=1000, resolution=32, base_dir='objaverse_processed', use_pca=False)
    print(f"Processed {len(processed_objects)} objects successfully.")
