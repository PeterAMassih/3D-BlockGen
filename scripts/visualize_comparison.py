# scripts/visualize_comparison.py

import torch
import trimesh
import numpy as np
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import json
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    look_at_view_transform,
    TexturesVertex
)


def plot_voxel_tensor(tensor: torch.Tensor, threshold: float = 0.5):
    """Create matplotlib visualization of voxel tensor."""
    # Convert tensor to numpy and handle RGBA vs occupancy
    tensor = tensor.cpu()

    if tensor.shape[0] == 4:  # RGBA format
        occupancy = tensor[3] > threshold  # Use alpha channel for occupancy
        colors = tensor[:3]  # RGB channels

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        rgba = np.zeros((*occupancy.shape, 4))
        for c in range(3):
            rgba[..., c] = np.where(occupancy, colors[c], 0)
        rgba[..., 3] = occupancy.numpy().astype(float)

        ax.voxels(occupancy, facecolors=rgba)

    else:  # Occupancy only
        occupancy = tensor[0] > threshold

        # Create 3D visualization
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(occupancy, edgecolor='k', alpha=0.3)

    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect([1, 1, 1])

    return fig


def get_mesh_colors(mesh, device):
    """Extract color information from mesh.
    
    In GLB/GLTF files, colors can come from several sources:
    1. Texture maps with UV coordinates (most detailed)
    2. Per-vertex colors
    3. Material colors (simplest)
    
    Returns:
        torch.Tensor: Colors tensor of shape (N, 3) where N is number of vertices
                     Values are in range [0, 1] for RGB
    """
    print("\nExtracting color information...")

    # Check if mesh has a texture map and UV coordinates
    if (hasattr(mesh.visual, 'material') and
            hasattr(mesh.visual.material, 'baseColorTexture') and
            mesh.visual.material.baseColorTexture is not None):

        print("Found baseColorTexture, computing vertex colors from texture...")

        # Convert texture image to numpy array and normalize to [0, 1]
        texture = np.array(mesh.visual.material.baseColorTexture).astype(np.float32) / 255.0
        print(f"Texture shape: {texture.shape}, dtype: {texture.dtype}")

        # Handle different texture formats (RGB or RGBA)
        if texture.shape[-1] == 4:  # RGBA format
            texture = texture[..., :3]  # Keep only RGB channels
        elif texture.shape[-1] != 3:  # Invalid format
            print(f"Unexpected texture format with {texture.shape[-1]} channels, using default gray")
            return torch.ones((len(mesh.vertices), 3), device=device, dtype=torch.float32) * 0.7

        # UV coordinates are [N, 2] array where N is number of vertices
        uv = mesh.visual.uv

        # For each vertex, sample its color from the texture using UV mapping
        vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
        for i, uv_coord in enumerate(uv):
            # Convert UV coordinates (range [0,1]) to pixel coordinates
            u, v = uv_coord
            x = int(u * (texture.shape[1] - 1))  # width
            y = int((1 - v) * (texture.shape[0] - 1))  # height (flip v since image origin is top-left)
            vertex_colors[i] = texture[y, x]  # Sample RGB color at pixel position

        colors = torch.tensor(vertex_colors, device=device, dtype=torch.float32)
        print(f"Sampled colors shape: {colors.shape}")
        return colors

    # If no texture, try using material's main color
    elif hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'main_color'):
        print("Using main_color...")
        main_color = mesh.visual.material.main_color[:3].astype(np.float32) / 255.0
        colors = torch.tensor(main_color, device=device, dtype=torch.float32)
        return colors.expand(len(mesh.vertices), 3)

    # Default fallback to gray
    print("No color information found, using default gray...")
    return torch.ones((len(mesh.vertices), 3), device=device, dtype=torch.float32) * 0.7


def create_spinning_gif(glb_path: str, output_path: str, num_frames: int = 36):
    """Create a GIF of a GLB model spinning around its z-axis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GLB file
    print("Loading GLB file...")
    mesh = trimesh.load(str(glb_path), force='mesh')
    # Handle scene containing multiple meshes
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(meshes) > 1:
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = meshes[0]

    # Convert vertices [N, 3] and faces [F, 3] to PyTorch tensors
    # N = number of vertices, F = number of faces
    verts = torch.tensor(mesh.vertices, device=device, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, device=device, dtype=torch.int64)

    # Center and normalize vertices to unit cube
    verts = verts - verts.mean(dim=0, keepdim=True)
    scale = verts.abs().max()
    verts = verts / scale

    # Get per-vertex colors [N, 3] in range [0, 1]
    colors = get_mesh_colors(mesh, device)

    # Create mesh with vertex colors
    # colors needs to be [1, N, 3] for TexturesVertex
    textures = TexturesVertex(verts_features=colors.unsqueeze(0))
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    # Setup renderer with 256x256 output resolution
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0.0,
        faces_per_pixel=1
    )

    # Setup lighting for better color visualization
    # All light colors should be in range [0, 1]
    lights = PointLights(
        device=device,
        location=[[2.0, 2.0, 2.0]],  # Light position in 3D space
        ambient_color=[[0.7, 0.7, 0.7]],  # Ambient light color
        diffuse_color=[[0.8, 0.8, 0.8]],  # Diffuse light color
        specular_color=[[0.0, 0.0, 0.0]]  # No specular highlights
    )

    # Create renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=FoVPerspectiveCameras(device=device), raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights)
    )

    # Create frames by rotating camera around z-axis
    print("\nRendering frames...")
    frames = []
    for i in range(num_frames):
        angle = i * 360.0 / num_frames
        R, T = look_at_view_transform(dist=3.0, elev=30.0, azim=angle)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        # Render frame: output is [B, H, W, 4] RGBA image
        image = renderer(mesh, cameras=cameras)
        # Convert to uint8 RGB: [H, W, 3] with values in [0, 255]
        image = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        frames.append(image)

    # Save as GIF
    print("Saving GIF...")
    imageio.mimsave(output_path, frames, duration=int(1000 / 12))
    print(f"Saved to {output_path}")


def create_augmentation_grid(voxel_path: str):
    """Create a 2x2 grid visualization of original and augmented voxels."""
    # Load original and augmented versions
    base_path = Path(voxel_path)
    paths = [
        base_path,  # Original
        base_path.parent / f"{base_path.stem}_aug1.pt",  # 90° X-axis
        base_path.parent / f"{base_path.stem}_aug2.pt",  # 90° Y-axis
        base_path.parent / f"{base_path.stem}_aug3.pt"  # 90° Z-axis
    ]

    # Labels for each subplot matching the actual rotations in your augmentation code
    labels = ["Original", "90° X-axis", "90° Y-axis", "90° Z-axis"]

    fig = plt.figure(figsize=(12, 12))

    for idx, (path, label) in enumerate(zip(paths, labels), 1):
        tensor = torch.load(path)
        ax = fig.add_subplot(2, 2, idx, projection='3d')

        if tensor.shape[0] == 4:  # RGBA format
            occupancy = tensor[3] > 0.5
            colors = tensor[:3]
            rgba = np.zeros((*occupancy.shape, 4))
            for c in range(3):
                rgba[..., c] = np.where(occupancy, colors[c], 0)
            rgba[..., 3] = occupancy.numpy().astype(float)
            ax.voxels(occupancy, facecolors=rgba, edgecolor='k', alpha=0.8)
        else:
            occupancy = tensor[0] > 0.5
            ax.voxels(occupancy, edgecolor='k', alpha=0.3)

        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f"Rotation: {label}", pad=20)
        ax.grid(True)

    plt.tight_layout()
    return fig


def create_comparison_visualization(glb_path: str, voxel_path: str, annotation_file: str, output_path: str,
                                    show_augmentations: bool = False):
    """Create a visualization comparing animated GLB and voxelized versions."""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    model_id = Path(voxel_path).stem.split('_aug')[0]
    prompt = ""
    if model_id in annotations:
        name = annotations[model_id].get('name', 'an object')
        categories = [cat['name'] for cat in annotations[model_id].get('categories', [])]
        tags = [tag['name'] for tag in annotations[model_id].get('tags', [])]

        prompt_parts = [
            f"Object: {name}",
            f"Category: {', '.join(categories)}" if categories else None,
            f"Tags: {', '.join(tags)}" if tags else None
        ]
        prompt = "\n".join([p for p in prompt_parts if p is not None])

    # Create GLB spinning animation
    print("Creating GLB animation...")
    temp_gif = "temp_spin.gif"
    create_spinning_gif(glb_path, temp_gif)

    # Load voxel data and create static plot
    print("Creating voxel visualization...")
    voxel_data = torch.load(voxel_path)
    fig = plot_voxel_tensor(voxel_data)
    temp_voxel = "temp_voxel.png"
    plt.savefig(temp_voxel, bbox_inches='tight', dpi=300)
    plt.close()

    # Set up initial sizes
    target_height = 400

    # Load and process GLB frames
    glb_gif = Image.open(temp_gif)
    glb_frames = []
    try:
        while True:
            glb_frames.append(glb_gif.copy())
            glb_gif.seek(glb_gif.tell() + 1)
    except EOFError:
        pass

    # Load and process voxel image
    voxel_img = Image.open(temp_voxel)

    # Calculate sizes
    aspect = glb_frames[0].size[0] / glb_frames[0].size[1]
    glb_size = (int(target_height * aspect), target_height)
    voxel_size = (int(target_height * aspect), target_height)

    # Resize base images
    glb_frames = [f.resize(glb_size, Image.Resampling.LANCZOS) for f in glb_frames]
    voxel_img = voxel_img.resize(voxel_size, Image.Resampling.LANCZOS)

    # Create and add augmentation grid if requested
    if show_augmentations:
        print("Creating augmentation grid...")
        aug_fig = create_augmentation_grid(voxel_path)
        temp_aug = "temp_augmentations.png"
        aug_fig.savefig(temp_aug, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(aug_fig)

        aug_img = Image.open(temp_aug)
        aug_width = int(target_height * 1.5)
        aug_aspect = aug_img.size[0] / aug_img.size[1]
        aug_size = (aug_width, int(aug_width / aug_aspect))
        aug_img = aug_img.resize(aug_size, Image.Resampling.LANCZOS)

    # Calculate layout
    padding = 80
    spacing = 150
    text_height = 100

    # Adjust width based on whether augmentations are included
    if show_augmentations:
        width = glb_size[0] + voxel_size[0] + aug_size[0] + spacing * 3 + 2 * padding
        height = max(glb_size[1], voxel_size[1], aug_size[1]) + 2 * padding + text_height
    else:
        width = glb_size[0] + voxel_size[0] + spacing + 2 * padding
        height = max(glb_size[1], voxel_size[1]) + 2 * padding + text_height

    # Load fonts
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()

    # Create frames
    print(f"Creating {len(glb_frames)} comparison frames...")
    output_frames = []
    for glb_frame in glb_frames:
        frame = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(frame)

        # Add title
        y_offset = padding // 2
        for line in prompt.split('\n'):
            bbox = draw.textbbox((0, 0), line, font=font_title)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            draw.text((text_x, y_offset), line, fill='black', font=font_title)
            y_offset += 30

        # Paste base images
        frame.paste(glb_frame, (padding, text_height + padding))
        frame.paste(voxel_img, (padding + glb_size[0] + spacing, text_height + padding))

        # Add voxelization arrow
        arrow_x1 = padding + glb_size[0] + spacing // 4
        arrow_x2 = padding + glb_size[0] + spacing * 3 // 4
        arrow_y = text_height + padding + target_height // 2

        draw.line([(arrow_x1, arrow_y), (arrow_x2, arrow_y)], fill='black', width=3)
        draw.polygon([(arrow_x2, arrow_y), (arrow_x2 - 15, arrow_y - 10), (arrow_x2 - 15, arrow_y + 10)], fill='black')

        # Add voxelization label
        label = "Voxelization"
        bbox = draw.textbbox((0, 0), label, font=font_label)
        label_width = bbox[2] - bbox[0]
        label_x = padding + glb_size[0] + (spacing - label_width) // 2
        label_y = arrow_y - 30
        draw.text((label_x, label_y), label, fill='black', font=font_label)

        # Add augmentations if requested
        if show_augmentations:
            # Add data augmentation arrow
            aug_arrow_x1 = padding + glb_size[0] + voxel_size[0] + spacing * 1.5
            aug_arrow_x2 = aug_arrow_x1 + spacing
            aug_arrow_y = arrow_y

            draw.line([(aug_arrow_x1, aug_arrow_y), (aug_arrow_x2, aug_arrow_y)], fill='black', width=3)
            draw.polygon([(aug_arrow_x2, aug_arrow_y), (aug_arrow_x2 - 15, aug_arrow_y - 10),
                          (aug_arrow_x2 - 15, aug_arrow_y + 10)], fill='black')

            # Add augmentation label
            aug_label = "Data Augmentation"
            bbox = draw.textbbox((0, 0), aug_label, font=font_label)
            aug_label_width = bbox[2] - bbox[0]
            aug_label_x = aug_arrow_x1 + (spacing - aug_label_width) // 2
            aug_label_y = aug_arrow_y - 30
            draw.text((aug_label_x, aug_label_y), aug_label, fill='black', font=font_label)

            # Paste augmentation grid
            frame.paste(aug_img, (int(aug_arrow_x2 + spacing / 2), text_height + padding))

        # Add base image labels
        draw.text((padding + glb_size[0] // 2 - 40, text_height + padding - 25),
                  "Original GLB", fill='black', font=font_label)
        draw.text((padding + glb_size[0] + spacing + voxel_size[0] // 2 - 60,
                   text_height + padding - 25), "Voxelized Model", fill='black', font=font_label)

        output_frames.append(frame)

    # Save final animation
    print(f"Saving final animation...")
    output_frames[0].save(
        output_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=1000 / 12,
        loop=0
    )

    # Cleanup
    Path(temp_gif).unlink()
    Path(temp_voxel).unlink()
    if show_augmentations:
        Path(temp_aug).unlink()

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    glb_path = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data/hf-objaverse-v1/glbs/000-035/0708b47ff5cb43ebbd35e24509771cc2.glb"
    voxel_path = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data_voxelized/hf-objaverse-v1/glbs/000-035/0708b47ff5cb43ebbd35e24509771cc2.pt"
    annotation_file = "objaverse_data/annotations.json"
    output_path = "comparison.gif"

    create_comparison_visualization(glb_path, voxel_path, annotation_file, output_path, show_augmentations=True)
