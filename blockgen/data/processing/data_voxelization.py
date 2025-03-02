"""
This script contains utilities for processing 3D meshes and voxelization,
including color-aware voxelization and data augmentation.

The `voxelize_with_color` function is adapted from the `subdivide_to_size`
algorithm in the trimesh repository. It integrates color handling and
voxelization tailored for use in the BlockGen project.

File: blockgen/data/processing/data_voxelization.py
"""

import numpy as np
import trimesh
import torch
from trimesh import transformations as tr
from trimesh import grouping
from trimesh import remesh
from trimesh.voxel import encoding as enc
from trimesh.voxel import base
from scipy.ndimage import rotate
from typing import List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os
import json
import gc

def voxelize_with_color(mesh: trimesh.Trimesh,
                       pitch: float,
                       resolution: int = 32,
                       max_iter: int = 10,
                       edge_factor: float = 2.0) -> Tuple[base.VoxelGrid, np.ndarray, Optional[np.ndarray]]:
    """
    Voxelize a surface and detect if it has colors.
    Returns None for colors if no color information is available.
    Uses subdivide as the remeshing method.
    """
    max_edge = pitch / edge_factor
    
    # Try to get colors
    try:
        if hasattr(mesh.visual.to_color(), 'vertex_colors'):
            colors = mesh.visual.to_color().vertex_colors
            # Add debug info
            # print(f"Vertex count: {len(mesh.vertices)}")
            # print(f"Color array shape: {colors.shape if colors is not None else None}")
            if not isinstance(colors, np.ndarray) or len(colors.shape) != 2:
                colors = None
        elif hasattr(mesh.visual.material, 'main_color'):
            main_color = mesh.visual.material.main_color
            if isinstance(main_color, (list, np.ndarray)):
                main_color = np.asarray(main_color).flatten()
                colors = np.tile(main_color, (len(mesh.vertices), 1))
            else:
                colors = None
        else:
            colors = None
            
    except Exception as e:
        # logging.info(f"No color information available: {str(e)}")
        colors = None

    # Process colors if they exist and are valid
    if colors is not None: # colors (num_vert, RGB)
        colors = np.asarray(colors)
        if len(colors.shape) != 2:  # Ensure 2D array
            colors = None
        elif colors.shape[1] == 3:  # RGB -> RGBA
            alpha = np.full((len(colors), 1), 255)
            colors = np.hstack([colors, alpha])
        elif colors.shape[1] != 4:  # Invalid color format
            colors = None

    # Combine vertices with colors if they exist
    if colors is not None:
        vertices_with_color = np.hstack([mesh.vertices, colors])
        v_color, f = remesh.subdivide_to_size(vertices_with_color,
                                           mesh.faces,
                                           max_edge=max_edge,
                                           max_iter=max_iter)
        v, colors = v_color[:, :3], v_color[:, 3:]
    else:
        v, f = remesh.subdivide_to_size(mesh.vertices,
                                      mesh.faces,
                                      max_edge=max_edge,
                                      max_iter=max_iter)

    # Convert vertices to voxel grid positions
    hit = v / pitch
    hit = np.round(hit).astype(int)
    
    # Remove duplicates
    unique, inverse = grouping.unique_rows(hit)
    occupied_index = hit[unique]
    if colors is not None:
        unique_colors = colors[unique]
    else:
        unique_colors = None
    
    # Center the voxels in the grid
    origin_index = occupied_index.min(axis=0)
    centered_indices = occupied_index - origin_index
    
    # Ensure voxels fit within resolution
    mask = np.all((centered_indices >= 0) & (centered_indices < resolution), axis=1)
    centered_indices = centered_indices[mask]
    if unique_colors is not None:
        unique_colors = unique_colors[mask]
    
    # Calculate origin position in model space
    origin_position = origin_index * pitch
    
    # Create binary voxel grid
    voxel_grid = base.VoxelGrid(
        enc.SparseBinaryEncoding(centered_indices),
        transform=tr.scale_and_translate(
            scale=pitch, translate=origin_position))
    
    return voxel_grid, centered_indices, unique_colors

class VoxelizerWithAugmentation:
    def __init__(self, resolution: int = 32):
        self.resolution = resolution
    
    def _load_and_normalize_mesh(self, mesh_path: str) -> Optional[trimesh.Trimesh]:
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
    
        # Handle case where mesh is a list
        if isinstance(mesh, list):
            meshes = [m for m in mesh if isinstance(m, trimesh.Trimesh)]
            if not meshes:
                return None
            if len(meshes) > 1:
                mesh = trimesh.util.concatenate(meshes)
            else:
                mesh = meshes[0]
        
        # Handle case where mesh is a scene
        elif isinstance(mesh, trimesh.Scene):
            meshes = []
            for g in mesh.geometry.values():
                if isinstance(g, trimesh.Trimesh):
                    meshes.append(g)
            
            if not meshes:
                return None
                
            if len(meshes) > 1:
                mesh = trimesh.util.concatenate(meshes)
            else:
                mesh = meshes[0]
    
        # Verify we have a valid mesh
        if not isinstance(mesh, trimesh.Trimesh):
            return None
    
        try:
            mesh.fill_holes()
        except Exception as e:
            logging.warning(f"Could not fill holes in mesh: {str(e)}")

        # Calculate scaling to fit in grid with margin
        margin = 1  # Reduce margin to 1 voxel
        available_size = self.resolution - 2 * margin
        
        # Center mesh at origin first
        mesh.vertices -= mesh.bounds.mean(axis=0)
        
        # Calculate scale to fit largest dimension
        max_extent = mesh.bounding_box.extents.max()
        scale = available_size / max_extent
        
        # Apply scaling
        mesh.vertices *= scale
        
        # Center in grid
        offset = self.resolution / 2 - mesh.bounds.mean(axis=0)
        mesh.vertices += offset
        
        # Verify bounds
        min_coords = mesh.vertices.min(axis=0)
        max_coords = mesh.vertices.max(axis=0)
        
        # # Print debug info
        # print(f"Mesh bounds after normalization:")
        # print(f"Min: {min_coords}")
        # print(f"Max: {max_coords}")
        # print(f"Extents: {max_coords - min_coords}")
        
        # Final safety check - just force it to fit if still out of bounds
        if np.any(min_coords < margin) or np.any(max_coords > (self.resolution - margin)):
            # Scale slightly more if needed
            current_extent = max_coords - min_coords
            max_extent = current_extent.max()
            if max_extent > available_size:
                extra_scale = available_size / max_extent
                mesh.vertices = (mesh.vertices - self.resolution/2) * extra_scale + self.resolution/2
            
            # Ensure centered
            min_coords = mesh.vertices.min(axis=0)
            max_coords = mesh.vertices.max(axis=0)
            center = (min_coords + max_coords) / 2
            mesh.vertices += (self.resolution/2 - center)
            
            # Final verification
            min_coords = mesh.vertices.min(axis=0)
            max_coords = mesh.vertices.max(axis=0)
            # print(f"After final adjustment:")
            # print(f"Min: {min_coords}")
            # print(f"Max: {max_coords}")
        
        # Ensure all vertices are within bounds
        mesh.vertices = np.clip(mesh.vertices, margin, self.resolution - margin)
        
        return mesh
            
    def process_mesh(self, mesh_path: str, visualize: bool = False) -> List[torch.Tensor]:
        try:

            mesh = self._load_and_normalize_mesh(mesh_path)
            if mesh is None:
                return []
            
            voxel_grid, indices, colors = voxelize_with_color(
                mesh=mesh,
                pitch=1.0,  # Keep pitch at 1.0 for integer voxel coordinates
                resolution=self.resolution,
                max_iter=10,
                edge_factor=2.0  # Slightly larger edge factor for better voxelization
            )
            
            # Create the base tensor based on colors existence
            base_tensor = self._create_color_tensor(indices, colors)
            augmented = self._create_augmentations(base_tensor)
            results = [base_tensor] # + augmented ## TODO Instead of commenting add boolean
            
            if visualize:
                self.visualize_results(results, mesh_path)
            
            return results
        
        except Exception as e:
            logging.error(f"Error processing {mesh_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_color_tensor(self, indices: np.ndarray, colors: Optional[np.ndarray]) -> torch.Tensor:
        """Create tensor with either 1 channel (occupancy) or 4 channels (RGB + occupancy)."""
        if colors is not None:
            # Create 4-channel tensor (RGB + occupancy)
            tensor = torch.zeros((4, self.resolution, self.resolution, self.resolution), 
                            dtype=torch.float32)
            
            # Normalize colors to [0, 1]
            colors = colors.astype(np.float32) / 255.0
            
            # Fill RGB and occupancy using alpha channel
            for idx, color in zip(indices, colors):
                # Convert color to tensor before assignment
                color_tensor = torch.from_numpy(color.copy())  # Need .copy() to ensure contiguous array
                tensor[0:3, idx[0], idx[1], idx[2]] = color_tensor[0:3]  # RGB
                tensor[3, idx[0], idx[1], idx[2]] = float(color_tensor[3])  # Alpha for occupancy
        else:
            # Create single-channel tensor (just occupancy)
            tensor = torch.zeros((1, self.resolution, self.resolution, self.resolution), 
                            dtype=torch.float32)
            
            # Mark occupied voxels
            for idx in indices:
                tensor[0, idx[0], idx[1], idx[2]] = 1.0
        
        return tensor
    
    def _create_augmentations(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Create exactly 3 rotated versions of the input tensor."""
        augmentations = []
        rotations = [
            (90, (1, 2)),   # Around X-axis
            (90, (0, 2)),   # Around Y-axis
            (90, (0, 1))    # Around Z-axis
        ]
        
        for angle, axes in rotations:
            rotated = tensor.clone()
            num_channels = rotated.shape[0]  # Works for both 1 and 4 channels
            for c in range(num_channels):
                channel = rotated[c].numpy()
                rotated[c] = torch.from_numpy(rotate(channel, angle, axes=axes, reshape=False, order=0))
            augmentations.append(rotated)
        
        return augmentations
        
    def visualize_results(self, tensors: List[torch.Tensor], title: str):
        """Visualize original and augmented versions (should be 4 total)."""
        n = len(tensors)
        fig = plt.figure(figsize=(5*n, 5))  # Single row of plots
        
        for i, tensor in enumerate(tensors):
            ax = fig.add_subplot(1, n, i+1, projection='3d')
            
            if tensor.shape[0] == 4:  # Has RGB + occupancy
                occupancy = tensor[3].numpy()
                rgb = tensor[:3].numpy()
            else:  # Only occupancy
                occupancy = tensor[0].numpy()
                rgb = np.stack([np.full_like(occupancy, 0.5) for _ in range(3)])
            
            self._plot_colored_voxels(occupancy, rgb, ax)
            ax.view_init(elev=30, azim=45)
            ax.set_title(f"{'Original' if i==0 else f'Aug {i}'}\nVoxels: {np.sum(occupancy > 0)}")
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def _plot_colored_voxels(self, occupancy: np.ndarray, rgb: np.ndarray, ax):
        rgba = np.zeros((*occupancy.shape, 4))
        for i in range(3):
            rgba[..., i] = np.where(occupancy, rgb[i], 0)
        rgba[..., 3] = occupancy.astype(float)
        ax.voxels(occupancy, facecolors=rgba)