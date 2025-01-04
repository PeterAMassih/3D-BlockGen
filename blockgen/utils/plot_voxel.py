def plot_voxel_tensor(tensor: torch.Tensor, threshold: float = 0.5):
    """Create matplotlib visualization of voxel tensor."""
    # Convert tensor to numpy and handle RGBA vs occupancy
    tensor = tensor.cpu()
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(12, 5))
    
    if tensor.shape[0] == 4:  # RGBA format
        occupancy = tensor[3] > threshold  # Use alpha channel for occupancy
        colors = tensor[:3]  # RGB channels
        
        # Print stats
        print(f"RGB range: [{colors.min():.3f}, {colors.max():.3f}]")
        print(f"Alpha range: [{tensor[3].min():.3f}, {tensor[3].max():.3f}]")
        print(f"Occupied voxels: {torch.sum(occupancy)} ({(torch.sum(occupancy)/occupancy.numel())*100:.2f}% of volume)")
        
        # 2D slice visualization
        ax1 = fig.add_subplot(121)
        mid_slice_idx = tensor.shape[2]//2
        rgb_slice = colors[:, :, mid_slice_idx].permute(1, 2, 0).numpy()
        alpha_slice = occupancy[:, :, mid_slice_idx].numpy()
        
        slice_img = np.zeros((*rgb_slice.shape[:-1], 4))
        slice_img[alpha_slice] = np.concatenate([rgb_slice[alpha_slice], np.ones((np.sum(alpha_slice), 1))], axis=1)
        ax1.imshow(slice_img)
        ax1.set_title("Center Slice")
        ax1.axis('equal')
        
        # 3D visualization
        ax2 = fig.add_subplot(122, projection='3d')
        rgba = np.zeros((*occupancy.shape, 4))
        for c in range(3):
            rgba[..., c] = np.where(occupancy, colors[c], 0)
        rgba[..., 3] = occupancy.numpy().astype(float)
        
        ax2.voxels(occupancy, facecolors=rgba, edgecolor='k', alpha=0.8)
        
    else:  # Occupancy only
        occupancy = tensor[0] > threshold
        print(f"Occupied voxels: {torch.sum(occupancy)} ({(torch.sum(occupancy)/occupancy.numel())*100:.2f}% of volume)")
        
        # 2D slice
        ax1 = fig.add_subplot(121)
        mid_slice = occupancy[:, :, tensor.shape[2]//2].numpy()
        ax1.imshow(mid_slice, cmap='gray')
        ax1.set_title("Center Slice")
        ax1.axis('equal')
        
        # 3D view
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.voxels(occupancy.numpy(), edgecolor='k', alpha=0.3)
    
    # Adjust 3D view
    ax2.set_title("3D View")
    ax2.view_init(elev=30, azim=45)
    ax2.set_box_aspect([1, 1, 1])
    
    # Add grid and axis labels
    ax2.grid(True)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    return fig

# For standalone use with file input
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize a voxel tensor file')
    parser.add_argument('tensor_path', type=str, help='Path to the .pt tensor file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary occupancy')
    args = parser.parse_args()
    
    # Load tensor
    tensor = torch.load(args.tensor_path)
    plot_voxel_tensor(tensor, args.threshold)
    plt.show()