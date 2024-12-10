import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_voxel_tensor(tensor_path: str, threshold: float = 0.5):
    """
    Load and visualize a single .pt tensor file containing voxel data.
    
    Args:
        tensor_path: Path to the .pt file
        threshold: Threshold for binary occupancy
    """
    # Load tensor
    tensor = torch.load(tensor_path)
    
    # Convert to numpy
    voxel_data = tensor.cpu().numpy()
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(12, 5))
    
    # Determine if RGB or occupancy only
    is_rgba = voxel_data.shape[0] == 4
    
    if is_rgba:
        # Print basic stats
        print(f"RGB range: [{voxel_data[:3].min():.3f}, {voxel_data[:3].max():.3f}]")
        print(f"Alpha range: [{voxel_data[3].min():.3f}, {voxel_data[3].max():.3f}]")
        
        # Get occupancy and RGB
        occupancy = (voxel_data[3] > threshold).astype(bool)
        rgb = voxel_data[:3]
        print(f"Occupied voxels: {np.sum(occupancy)} ({(np.sum(occupancy)/occupancy.size)*100:.2f}% of volume)")
        
        # 2D slice visualization
        ax1 = fig.add_subplot(121)
        mid_slice_idx = voxel_data.shape[2]//2
        rgb_slice = np.moveaxis(rgb[:, :, mid_slice_idx], 0, -1)
        alpha_slice = occupancy[:, :, mid_slice_idx]
        slice_img = np.zeros((*rgb_slice.shape[:-1], 4))
        slice_img[alpha_slice] = np.concatenate([rgb_slice[alpha_slice], np.ones((np.sum(alpha_slice), 1))], axis=1)
        ax1.imshow(slice_img)
        
        # 3D visualization
        ax2 = fig.add_subplot(122, projection='3d')
        rgba = np.zeros((*occupancy.shape, 4))
        for c in range(3):
            rgba[..., c] = np.where(occupancy, rgb[c], 0)
        rgba[..., 3] = occupancy.astype(float)
        
        ax2.voxels(occupancy, facecolors=rgba, edgecolor='k')
    else:
        # Handle occupancy-only case
        occupancy = (voxel_data[0] > threshold).astype(bool)
        print(f"Occupied voxels: {np.sum(occupancy)} ({(np.sum(occupancy)/occupancy.size)*100:.2f}% of volume)")
        
        # 2D slice
        ax1 = fig.add_subplot(121)
        ax1.imshow(occupancy[:, :, voxel_data.shape[2]//2], cmap="gray")
        
        # 3D view
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.voxels(occupancy, edgecolor='k')
    
    ax1.set_title("Center Slice")
    ax2.set_title("3D View")
    ax2.view_init(elev=30, azim=45)
    
    fname = Path(tensor_path).stem
    plt.suptitle(f"Voxel Visualization: {fname}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize a voxel tensor file')
    parser.add_argument('tensor_path', type=str, help='Path to the .pt tensor file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary occupancy')
    args = parser.parse_args()
    
    plot_voxel_tensor(args.tensor_path, args.threshold)