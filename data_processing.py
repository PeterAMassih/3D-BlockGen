import numpy as np
import torch
import matplotlib.pyplot as plt
import trimesh
import os
from sklearn.decomposition import PCA
from typing import Tuple, List
import shutil
from scipy import ndimage

def improve_voxelization(voxels: np.ndarray) -> np.ndarray:
    """
    Improve voxelization by filling holes and smoothing.
    
    Args:
        voxels (np.ndarray): Binary 3D numpy array representing the voxelized model.
    
    Returns:
        np.ndarray: Improved voxelized model.
    """
    # Fill internal holes
    filled = ndimage.binary_fill_holes(voxels)
    
    # Apply morphological closing to connect nearby voxels
    kernel = np.ones((3,3,3))
    closed = ndimage.binary_closing(filled, structure=kernel)
    
    return closed

def standardize_orientation(voxels: np.ndarray) -> np.ndarray:
    """
    Standardize the orientation of a voxelized 3D model using PCA.

    Args:
        voxels (np.ndarray): Binary 3D numpy array representing the voxelized model.

    Returns:
        np.ndarray: Reoriented binary 3D numpy array.
    """
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
    """
    Load a 3D model file and voxelize it to the specified resolution.

    Args:
        file_path (str): Path to the 3D model file.
        resolution (int, optional): Desired resolution of the voxel grid. Defaults to 32.

    Returns:
        np.ndarray: Binary 3D numpy array representing the voxelized model.

    Raises:
        NotImplementedError: If USDZ format is detected (not implemented).
        ValueError: If the file format is not supported.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.usdz':
        raise NotImplementedError("USDZ conversion not implemented")
    elif file_extension in ['.obj', '.stl', '.ply', '.glb', '.gltf']:
        scene = trimesh.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                                         for g in scene.geometry.values()])
    else:
        mesh = scene

    voxel_grid = mesh.voxelized(pitch=mesh.bounding_box.extents.max() / resolution)
    voxels = voxel_grid.matrix.astype(bool).astype(int) # check matching_cube TODO Remove the int at the end
    
    if voxels.shape != (resolution, resolution, resolution):
        pad_width = [(0, max(0, resolution - s)) for s in voxels.shape]
        voxels = np.pad(voxels, pad_width)
        voxels = voxels[:resolution, :resolution, :resolution]
    
    return voxels

def save_as_tensor(voxel_grid: np.ndarray, original_filename: str, output_dir: str) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor and save with a name based on the original file.

    Args:
        voxel_grid (np.ndarray): Binary 3D numpy array representing the voxelized model.
        original_filename (str): Original filename of the 3D model.
        output_dir (str): Directory to save the tensor.

    Returns:
        torch.Tensor: PyTorch tensor representation of the voxel grid.
    """
    tensor = torch.from_numpy(voxel_grid).float().unsqueeze(0).unsqueeze(0) # Make sure to save it as a bool for low memory usage
    
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    new_filename = f"{base_name}_voxel_grid.pt"
    full_path = os.path.join(output_dir, new_filename)
    
    torch.save(tensor, full_path)
    print(f"Tensor saved as: {full_path}")
    return tensor

def visualize_voxels(voxel_grid: np.ndarray) -> None:
    """
    Visualize the voxel grid using matplotlib.

    Args:
        voxel_grid (np.ndarray): Binary 3D numpy array representing the voxelized model.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('32x32x32 Binary Voxel Grid Visualization')
    plt.show()

def process_model(file_path: str, resolution: int = 32, output_dir: str = 'output') -> Tuple[np.ndarray, torch.Tensor]:
    """
    Process a single 3D model: load, voxelize, standardize orientation, improve voxelization, and save as tensor.

    Args:
        file_path (str): Path to the 3D model file.
        resolution (int, optional): Desired resolution of the voxel grid. Defaults to 32.
        output_dir (str, optional): Directory to save the processed files. Defaults to 'output'.

    Returns:
        Tuple[np.ndarray, torch.Tensor]: Voxel grid and PyTorch tensor of the processed model.
    """
    voxel_grid = load_and_voxelize_model(file_path, resolution)
    voxel_grid = standardize_orientation(voxel_grid)
    voxel_grid = improve_voxelization(voxel_grid)
    
    os.makedirs(output_dir, exist_ok=True)
    tensor = save_as_tensor(voxel_grid, file_path, output_dir)
    
    return voxel_grid, tensor

def process_single_model(file_path: str, resolution: int = 32, output_dir: str = 'output'):
    """
    Process a single 3D model file and visualize the result.

    Args:
        file_path (str): Path to the 3D model file.
        resolution (int, optional): Desired resolution of the voxel grid. Defaults to 32.
        output_dir (str, optional): Directory to save the processed files. Defaults to 'output'.
    """
    supported_extensions = ['.obj', '.stl', '.ply', '.glb', '.gltf']
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension not in supported_extensions:
        raise ValueError(f"Unsupported file format. Supported formats are: {', '.join(supported_extensions)}")

    print(f"Processing file: {file_path}")
    voxel_grid, tensor = process_model(file_path, resolution, output_dir)
    
    print(f"\nModel: {os.path.basename(file_path)}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Unique values in tensor: {torch.unique(tensor)}")
    visualize_voxels(voxel_grid)

# -----------------------------------------------
# Function to process all 3D models in a directory and its subdirectories
# -----------------------------------------------
def process_directory_recursive(directory: str, resolution: int = 32, output_dir: str = 'output'):
    """
    Process all 3D model files in a directory and its subdirectories.

    Args:
        directory (str): Directory path containing 3D model files and subdirectories.
        resolution (int, optional): Desired resolution of the voxel grid. Defaults to 32.
        output_dir (str, optional): Directory to save the processed files. Defaults to 'output'.
    """
    # Supported file extensions
    supported_extensions = ['.obj', '.stl', '.ply', '.glb', '.gltf']

    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_extension = os.path.splitext(filename)[1].lower()

            # Check if the file is a supported 3D model format
            if file_extension in supported_extensions:
                try:
                    print(f"\nProcessing file: {file_path}")
                    # Call process_model for each valid file
                    process_model(file_path, resolution, output_dir)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            else:
                print(f"Skipping unsupported file: {filename}")

# -----------------------------------------------
# Main script entry point
# -----------------------------------------------
if __name__ == "__main__":
    process_single_model("/Users/PeterAM/Desktop/Research_Project/banana_low-poly.glb", resolution=32, output_dir='output')