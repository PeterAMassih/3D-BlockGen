import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def analyze_voxel_dataset(data_dir: str, save_dir: str = "analysis_results"):
    """Analyze key dataset statistics including occupancy patterns and colors."""
    data_dir, save_dir = Path(data_dir), Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    voxel_files = list(data_dir.rglob("*.pt"))
    total_files = len(voxel_files)

    # Initialize trackers
    has_color = 0
    occupancy_ratios = []
    red_vals, green_vals, blue_vals = [], [], []
    occupancy_heatmap = torch.zeros((32, 32, 32))

    for file_path in tqdm(voxel_files):
        try:
            tensor = torch.load(file_path)

            if tensor.shape[0] == 4:  # RGBA format
                has_color += 1
                occupancy = tensor[3] > 0.5
                if occupancy.any():
                    red_vals.append(tensor[0, occupancy].mean().item())
                    green_vals.append(tensor[1, occupancy].mean().item())
                    blue_vals.append(tensor[2, occupancy].mean().item())
            else:  # Binary occupancy
                occupancy = tensor[0] > 0.5

            occupancy_ratios.append(occupancy.float().mean().item() * 100)
            occupancy_heatmap += occupancy.float()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Compute statistics
    avg_occupancy = np.mean(occupancy_ratios)
    color_ratio = (has_color / total_files) * 100

    # Create visualization figure
    fig = plt.figure(figsize=(18, 6))  # Wider layout for better spacing

    # 1. Occupancy histogram
    ax1 = plt.subplot(131)
    ax1.hist(occupancy_ratios, bins=30, color='blue', alpha=0.7)
    ax1.axvline(avg_occupancy, color='red', linestyle='--', label=f'Mean: {avg_occupancy:.1f}%')
    ax1.set_title('Occupancy Distribution')
    ax1.set_xlabel('Occupancy %')
    ax1.set_ylabel('Count')
    ax1.set_yscale('log')  # Logarithmic scale for better visualization
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x / 1000)}k'))
    ax1.legend()

    # 2. RGB distribution
    ax2 = plt.subplot(132)
    if len(red_vals) > 0:
        labels = ['Red', 'Green', 'Blue']
        values = [np.mean(red_vals), np.mean(green_vals), np.mean(blue_vals)]
        ax2.bar(labels, values, color=['red', 'green', 'blue'])
        ax2.set_title('Average RGB Values')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Intensity (0 to 1)')

    # 3. Occupancy heatmap (middle slice)
    ax3 = plt.subplot(133)
    mid_slice = occupancy_heatmap[:, :, 16] / total_files
    heatmap = ax3.imshow(mid_slice, cmap='viridis', origin='lower')
    ax3.set_title('Average Occupancy (Middle Slice)')
    ax3.set_xlabel('X-axis (Voxel Index)')
    ax3.set_ylabel('Y-axis (Voxel Index)')
    cbar = plt.colorbar(heatmap, ax=ax3)
    cbar.set_label('Occupancy Probability')

    # Adjust spacing
    plt.tight_layout(w_pad=3.0)  # Increase spacing between plots
    plt.savefig(save_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nDataset Statistics:")
    print(f"Total models: {total_files:,}")
    print(f"Models with color: {has_color:,} ({color_ratio:.1f}%)")
    print(f"Average occupancy: {avg_occupancy:.1f}%")

    if len(red_vals) > 0:
        print("\nAverage RGB values:")
        print(f"Red: {np.mean(red_vals):.3f}")
        print(f"Green: {np.mean(green_vals):.3f}")
        print(f"Blue: {np.mean(blue_vals):.3f}")


if __name__ == "__main__":
    analyze_voxel_dataset("objaverse_data_voxelized")
