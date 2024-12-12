import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import Counter

@dataclass
class LegoBrick:
    size: Tuple[int, int, int]  # width, length, height
    position: Tuple[int, int, int]  # x, y, z
    color: Tuple[float, float, float, float]  # RGBA

class LegoConverter:
    def __init__(self):
        # Define common LEGO brick sizes in terms of 1x1x1 voxels
        # TODO Add the rest of the brick sizes,
        # Must be sorted descending by volume (w * l * h)
        self.brick_sizes = [
            (2, 4, 1),  # 2x4 brick
            (2, 3, 1),  # 2x3 brick
            (2, 2, 1),  # 2x2 brick
            (2, 1, 1),  # 1x2 brick
            (1, 4, 1),  # 1x4 brick
            (1, 3, 1),  # 1x3 brick
            (1, 2, 1),  # 1x2 brick
            (1, 1, 1)   # 1x1 brick
        ]
        
        # Dictionary to map sizes to LEGO brick names
        self.brick_names = {
            (2, 4, 1): "2x4 brick",
            (2, 3, 1): "2x3 brick",
            (2, 2, 1): "2x2 brick",
            (2, 1, 1): "2x1 brick",
            (1, 4, 1): "1x4 brick",
            (1, 3, 1): "1x3 brick",
            (1, 2, 1): "1x2 brick",
            (1, 1, 1): "1x1 brick"
        }
        
        # Default color for occupancy-only models (gray)
        self.default_color = (0.5, 0.5, 0.5, 1.0)

    def preprocess_voxels(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the voxel tensor, handling both RGBA and occupancy formats
        """
        # TODO Choose between this and the inference script preprocessing
        # Create a copy to avoid modifying the input
        processed = voxels.clone()
        
        # Handle different input formats
        if processed.shape[0] == 4:  # RGBA format
            # Clip RGB values to [0, 1]
            processed[:3] = torch.clamp(processed[:3], 0, 1)
            # Discretize alpha channel
            processed[3] = (processed[3] > 0.5).float()
        else:  # Occupancy format
            # Convert to RGBA format with default gray color
            occupancy = (processed > 0.5).float()
            rgba = torch.zeros((4, *processed.shape[1:]), device=processed.device)
            rgba[0] = occupancy * self.default_color[0]  # R
            rgba[1] = occupancy * self.default_color[1]  # G
            rgba[2] = occupancy * self.default_color[2]  # B
            rgba[3] = occupancy  # A
            processed = rgba
            
        return processed

    def _check_space_available(self, occupied: torch.Tensor, pos: Tuple[int, int, int], 
                             size: Tuple[int, int, int]) -> bool:
        """Check if space is available for placing a brick."""
        x, y, z = pos
        w, l, h = size
        
        # Check bounds
        if (x + w > occupied.shape[1] or y + l > occupied.shape[2] or 
            z + h > occupied.shape[3]):
            return False
            
        # Check if space is already occupied
        return not occupied[:, x:x+w, y:y+l, z:z+h].any()

    def _get_average_color(self, voxels: torch.Tensor, pos: Tuple[int, int, int], 
                          size: Tuple[int, int, int]) -> Tuple[float, float, float, float]:
        """Get average color for the brick."""
        x, y, z = pos
        w, l, h = size
        region = voxels[:, x:x+w, y:y+l, z:z+h]
        # Only average colors where alpha > 0
        mask = (region[3] > 0).float()
        if mask.sum() == 0:
            return (0, 0, 0, 0)
        
        rgb = region[:3] * mask
        avg_rgb = (rgb.sum(dim=(1,2,3)) / mask.sum()).tolist()
        return (*avg_rgb, 1.0)  # Always use full alpha for placed bricks

    def _check_support(self, occupied: torch.Tensor, pos: Tuple[int, int, int], 
                      size: Tuple[int, int, int]) -> bool:
        """Check if brick has adequate support from below."""
        x, y, z = pos
        w, l, h = size
        
        if z == 0:  # On the ground
            return True
            
        # Check if at least 25% of the brick is supported from below
        support_area = occupied[0, x:x+w, y:y+l, z-1].sum()
        total_area = w * l
        return support_area >= total_area * 0.25

    def convert_to_lego(self, voxels: torch.Tensor) -> List[LegoBrick]:
        """Convert voxel tensor to LEGO bricks."""

        if voxels.shape[0] not in [1, 4]:
            raise ValueError("Input must be either RGBA (4 channels) or occupancy (1 channel)")
        
        voxels = self.preprocess_voxels(voxels)
        
        bricks = []
        occupied = torch.zeros((1, *voxels.shape[1:]))  # Track placed bricks
        
        for z in range(voxels.shape[3]):
            for x in range(0, voxels.shape[1]):
                for y in range(0, voxels.shape[2]):
                    if voxels[3, x, y, z] > 0 and not occupied[0, x, y, z]:
                        best_size = None
                        
                        # Try each brick size
                        for size in self.brick_sizes:
                            if self._check_space_available(occupied, (x, y, z), size):
                                region = voxels[3, x:x+size[0], y:y+size[1], z:z+size[2]]
                                if region.sum() > 0 and self._check_support(occupied, (x, y, z), size):
                                    best_size = size
                                    break
                        
                        if best_size is not None:
                            w, l, h = best_size
                            color = self._get_average_color(voxels, (x, y, z), best_size)
                            bricks.append(LegoBrick(best_size, (x, y, z), color))
                            occupied[0, x:x+w, y:y+l, z:z+h] = 1
        
        return bricks

    def plot_3d_bricks(self, bricks: List[LegoBrick], save_path: str = None, show: bool = True):
        """Create a 3D visualization of the LEGO bricks using matplotlib."""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        def create_brick_vertices(position, size):
            x, y, z = position
            w, l, h = size
            vertices = np.array([
                [[x, y, z], [x+w, y, z], [x+w, y+l, z], [x, y+l, z]],
                [[x, y, z+h], [x+w, y, z+h], [x+w, y+l, z+h], [x, y+l, z+h]],
                [[x, y, z], [x, y, z+h], [x, y+l, z+h], [x, y+l, z]],
                [[x+w, y, z], [x+w, y, z+h], [x+w, y+l, z+h], [x+w, y+l, z]],
                [[x, y, z], [x+w, y, z], [x+w, y, z+h], [x, y, z+h]],
                [[x, y+l, z], [x+w, y+l, z], [x+w, y+l, z+h], [x, y+l, z+h]]
            ])
            return vertices

        for brick in bricks:
            vertices = create_brick_vertices(brick.position, brick.size)
            poly3d = Poly3DCollection(vertices, alpha=0.9)
            rgb_color = brick.color[:3]
            poly3d.set_facecolor(rgb_color)
            poly3d.set_edgecolor('black')
            ax.add_collection3d(poly3d)

        # Set fixed axis limits to 32x32x32
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_zlim(0, 32)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('LEGO Brick Visualization (32x32x32 voxels)')
        ax.view_init(elev=20, azim=45)

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def get_brick_statistics(self, bricks: List[LegoBrick]) -> Dict[str, int]:
        """Count the number of bricks of each type used."""
        size_counter = Counter(brick.size for brick in bricks)
        stats = {self.brick_names[size]: count for size, count in size_counter.items()}
        return stats

if __name__ == "__main__":
    # Load voxel data (works with both RGBA and occupancy formats)
    voxel_data = torch.load('0002c6eafa154e8bb08ebafb715a8d46.pt', weights_only=True)
    
    # Create converter instance
    converter = LegoConverter()
    
    # Convert to LEGO bricks
    lego_bricks = converter.convert_to_lego(voxel_data)
    
    # Get and print brick statistics
    brick_stats = converter.get_brick_statistics(lego_bricks)
    print("\nLEGO Bricks needed:")
    print("-----------------")
    for brick_name, count in brick_stats.items():
        print(f"{brick_name}: {count} pieces")
    
    # Plot the 3D visualization
    converter.plot_3d_bricks(lego_bricks)
    
    print(f"\nTotal number of bricks: {len(lego_bricks)}")