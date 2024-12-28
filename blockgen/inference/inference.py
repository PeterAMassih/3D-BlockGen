import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.io import save_obj
from pytorch3d.transforms import random_rotation
from pytorch3d.renderer import look_at_view_transform
import matplotlib.pyplot as plt
from pytorch3d.renderer import PointLights
from pytorch3d.io.ply_io import _save_ply, _open_file
import torch
from typing import Optional
from iopath.common.file_io import PathManager

def save_ply_with_colors(
    f,
    verts: torch.Tensor,
    faces: Optional[torch.LongTensor] = None,
    verts_normals: Optional[torch.Tensor] = None,
    verts_colors: Optional[torch.Tensor] = None,
    ascii: bool = False,
    decimal_places: Optional[int] = None,
    path_manager: Optional[PathManager] = None,
    colors_as_uint8=False
) -> None:
    """
    Save a mesh to a .ply file with support for vertex colors.

    Args:
        f: File (or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        verts_colors: FloatTensor of shape (V, 3) giving vertex colors in range [0, 1].
        ascii: (bool) whether to use the ascii ply format.
        decimal_places: Number of decimal places for saving if ascii=True.
        path_manager: PathManager for interpreting f if it is a str.
    """
    if len(verts) and not (verts.dim() == 2 and verts.size(1) == 3):
        message = "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if faces is not None and len(faces) and not (faces.dim() == 2 and faces.size(1) == 3):
        message = "Argument 'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if (
        verts_normals is not None
        and len(verts_normals)
        and not (
            verts_normals.dim() == 2
            and verts_normals.size(1) == 3
            and verts_normals.size(0) == verts.size(0)
        )
    ):
        message = "Argument 'verts_normals' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if (
        verts_colors is not None
        and len(verts_colors)
        and not (
            verts_colors.dim() == 2
            and verts_colors.size(1) == 3
            and verts_colors.size(0) == verts.size(0)
        )
    ):
        message = "Argument 'verts_colors' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if path_manager is None:
        path_manager = PathManager()

    with _open_file(f, path_manager, "wb") as f:
        _save_ply(
            f,
            verts=verts,
            faces=faces,
            verts_normals=verts_normals,
            verts_colors=verts_colors,
            ascii=ascii,
            decimal_places=decimal_places,
            colors_as_uint8=colors_as_uint8,
        )

class DiffusionInference3D:
    def __init__(self, model, noise_scheduler, config, device='cuda'):
        self.model = model.to(device)
        self.noise_scheduler = noise_scheduler
        self.config = config  # VoxelConfig object
        self.device = device
        # Add text encoder for conditioning
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder.eval()

    def encode_prompt(self, prompt):
        # Tokenize and encode text
        text_inputs = self.tokenizer(
            prompt,
            padding=True,
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(**text_inputs)[0]
        return encoder_hidden_states

    def sample(self, prompt, num_samples=8, image_size=(32, 32, 32), show_intermediate=False, guidance_scale=7.0, use_mean_init=False, py3d=True, use_rotations=True):
        with torch.no_grad():
            do_class_guidance = guidance_scale > 1.0
            
            # Initialize noise with correct shape: [B, C, H, W, D]
            num_channels = self.config.in_channels
            noise = torch.randn(num_samples, num_channels, *image_size).to(self.device)
            
            # Encode prompts
            encoder_hidden_states = self.encode_prompt([prompt] * num_samples)
            if do_class_guidance:
                encoder_hidden_states_uncond = self.encode_prompt([""] * num_samples)
    
            timesteps = self.noise_scheduler.timesteps.to(self.device)
            # print(timesteps)
    
            # timesteps = timesteps[300:]
            
            # Initialize starting point
            if use_mean_init:
                mean_data = 0.5 * torch.ones_like(noise)
                sample = self.noise_scheduler.add_noise(
                    mean_data, 
                    noise, 
                    torch.tensor([timesteps[0]]).to(self.device)
                )
            else:
                sample = noise
            
            for t in tqdm(timesteps, desc="Sampling Steps", total=len(timesteps)):
                # Function to get prediction with rotations
                def get_prediction_with_rotations(input_sample, states):
                    # Original orientation
                    pred1 = self.model(input_sample, t, encoder_hidden_states=states).sample
    
                    # First rotation: [B, C, H, W, D] -> [B, C, D, H, W]
                    sample_rot1 = input_sample.permute(0, 1, 4, 2, 3)
                    pred2 = self.model(sample_rot1, t, encoder_hidden_states=states).sample
                    pred2 = pred2.permute(0, 1, 3, 4, 2)  # Back to [B, C, H, W, D]
    
                    # Second rotation: [B, C, H, W, D] -> [B, C, W, D, H]
                    sample_rot2 = input_sample.permute(0, 1, 3, 4, 2)
                    pred3 = self.model(sample_rot2, t, encoder_hidden_states=states).sample
                    pred3 = pred3.permute(0, 1, 4, 2, 3)  # Back to [B, C, H, W, D]
    
                    # Average predictions
                    return (pred1 + pred2 + pred3) / 3.0
    
                # Get conditioned prediction
                if use_rotations:
                    residual = get_prediction_with_rotations(sample, encoder_hidden_states)
                else:
                    residual = self.model(sample, t, encoder_hidden_states=encoder_hidden_states).sample
    
                # Apply classifier guidance if needed
                if do_class_guidance:
                    if use_rotations:
                        residual_uncond = get_prediction_with_rotations(sample, encoder_hidden_states_uncond)
                    else:
                        residual_uncond = self.model(
                            sample,
                            t,
                            encoder_hidden_states=encoder_hidden_states_uncond
                        ).sample
                    
                    # Apply classifier guidance
                    residual = residual_uncond + guidance_scale * (residual - residual_uncond)
    
                # Compute predicted original sample
                alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (sample - beta_prod_t**0.5 * residual) / (alpha_prod_t ** 0.5)
                # print(t.item(), (alpha_prod_t**0.5).item(), (beta_prod_t**0.5).item())
    
                # pred_original_sample = pred_original_sample.clamp(0, 1)
    
                if show_intermediate and t % 50 == 49:
                    print(f"timestep: {t}")
                    if not py3d:
                        self.visualize_samples(sample, threshold=0.5)
                        self.visualize_samples(pred_original_sample, threshold=0.5)
                    else:
                        self.visualize_samples_p3d(pred_original_sample[0], threshold=0.5)
    
                # Get next sample
                sample = self.noise_scheduler.step(residual, t, sample).prev_sample
                
            return sample

    def sample_ddim(self, prompt, num_samples=8, image_size=(32, 32, 32), num_inference_steps=50, show_intermediate=False, guidance_scale=7.0):
        with torch.no_grad():
            do_class_guidance = guidance_scale > 1.0
            
            num_channels = self.config.in_channels
            sample = torch.randn(num_samples, num_channels, *image_size).to(self.device)
            
            encoder_hidden_states = self.encode_prompt([prompt] * num_samples)
            if do_class_guidance:
                encoder_hidden_states_uncond = self.encode_prompt([""] * num_samples)
            
            self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.noise_scheduler.timesteps

            for t in tqdm(timesteps, desc="Sampling Steps", total=len(timesteps)):
                residual = self.model(
                    sample, 
                    t,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                if do_class_guidance:
                    residual_uncond = self.model(
                        sample, 
                        t,
                        encoder_hidden_states=encoder_hidden_states_uncond
                    ).sample
                    
                    residual = residual_uncond + guidance_scale * (residual - residual_uncond)

                alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (sample - beta_prod_t**0.5 * residual) / (alpha_prod_t ** 0.5)
                
                if show_intermediate:
                    print(f"timestep: {t}")
                    self.visualize_samples(sample, threshold=0.5)
                    self.visualize_samples(pred_original_sample, threshold=0.5)

                sample = self.noise_scheduler.step(residual, t, sample).prev_sample

            return sample

    def visualize_samples(self, samples, threshold=None):
        """
        Visualize generated samples with both 2D slices and 3D rendering.
        Args:
            samples: Tensor of shape [B, C, H, W, D] where:
                    B = batch size
                    C = channels (1 for occupancy, 4 for RGBA)
                    H, W, D = spatial dimensions (e.g., 32, 32, 32)
            threshold: Optional threshold value. If None, will be determined by distribution
        """
        # Convert to numpy: maintains same dimensions
        samples_ = samples.cpu().numpy()  # [B, C, H, W, D]
        
        # Setup plots
        fig, axs = plt.subplots(2, len(samples_), figsize=(4*len(samples_), 8))
        if len(samples_) == 1:  # Handle single sample case
            axs = axs.reshape(-1, 1)
        
        for i, sample in enumerate(samples_):  # sample shape: [C, H, W, D]
            if self.config.in_channels > 1:
                # Print statistics
                if threshold is None:
                    adaptive_threshold = np.percentile(sample[3], 90)
                    print(f"Using adaptive threshold: {adaptive_threshold:.3f}")
                else:
                    adaptive_threshold = threshold
                
                print(f"RGB range: [{sample[:3].min():.3f}, {sample[:3].max():.3f}]")
                print(f"Alpha range: [{sample[3].min():.3f}, {sample[3].max():.3f}]")
                print(f"Red: [{sample[0].min():.3f}, {sample[0].max():.3f}]")
                print(f"Green: [{sample[1].min():.3f}, {sample[1].max():.3f}]")
                print(f"Blue: [{sample[2].min():.3f}, {sample[2].max():.3f}]")
                
                # Get binary occupancy from alpha channel [H, W, D]
                occupancy = (sample[3] > adaptive_threshold).astype(bool)
                # Clip RGB values to [0,1] range [3, H, W, D]
                rgb = np.clip(sample[:3], 0, 1)
                
                print(f"Occupied voxels: {np.sum(occupancy)} ({(np.sum(occupancy)/occupancy.size)*100:.2f}% of volume)")
                
                # Mid-depth slice visualization
                mid_depth = sample.shape[3] // 2  # Get middle of depth dimension
                
                # Get RGB and alpha for middle slice
                rgb_slice = rgb[:, :, :, mid_depth]  # [3, H, W]
                alpha_slice = occupancy[:, :, mid_depth]  # [H, W]
                
                # Create RGBA slice image [H, W, 4]
                slice_img = np.zeros((*rgb_slice.shape[1:], 4))  # [H, W, 4]
                if np.any(alpha_slice):
                    # Move RGB channels to last dimension for occupied voxels
                    rgb_slice_hwc = np.moveaxis(rgb_slice, 0, -1)  # [H, W, 3]
                    slice_img[alpha_slice] = np.concatenate([rgb_slice_hwc[alpha_slice], np.ones((np.sum(alpha_slice), 1))], axis=1)
                
                # Show 2D slice
                axs[0, i].imshow(slice_img)
                axs[0, i].set_title("Center Slice")
                axs[0, i].axis("off")
                
                # 3D visualization
                ax = axs[1, i]
                ax.remove()
                ax = fig.add_subplot(2, len(samples_), len(samples_) + i + 1, projection='3d')
                
                # Create colors for voxels [H, W, D, 4]
                colors = np.zeros((*occupancy.shape, 4))
                if np.any(occupancy):
                    # Convert RGB from [3, H, W, D] to [H, W, D, 3]
                    rgb_hwdc = np.moveaxis(rgb, 0, -1)
                    colors[occupancy, :3] = rgb_hwdc[occupancy]
                    colors[occupancy, 3] = 1.0
                
                ax.voxels(occupancy, facecolors=colors, edgecolor='k', alpha=0.8)
                
            else:
                # Single channel case
                binary_sample = (sample[0] > threshold).astype(bool)  # [H, W, D]
                
                # Show middle slice
                axs[0, i].imshow(binary_sample[:, :, binary_sample.shape[2]//2], cmap="gray")
                axs[0, i].set_title("Center Slice")
                axs[0, i].axis("off")
                
                # 3D visualization
                ax = axs[1, i]
                ax.remove()
                ax = fig.add_subplot(2, len(samples_), len(samples_) + i + 1, projection='3d')
                ax.voxels(binary_sample, edgecolor='k')
            
            # Set 3D plot properties
            ax.view_init(elev=30, azim=45)
            ax.set_title("3D View")
            ax.set_box_aspect([1, 1, 1])
            shape = occupancy.shape if self.config.in_channels > 1 else binary_sample.shape
            ax.set_xlim(0, shape[0])
            ax.set_ylim(0, shape[1])
            ax.set_zlim(0, shape[2])
        
        plt.tight_layout()
        plt.show()
    
    def visualize_samples_p3d(self, sample, threshold=0.5):
        device = "cuda"
        # Handle both single-channel and RGBA inputs
        if sample.shape[0] == 1:  # Single channel case
            # Create an RGBA tensor with default gray color
            rgba_sample = torch.zeros((4, *sample.shape[1:]), device=sample.device)
            rgba_sample[0] = 0  # R
            rgba_sample[1] = 0  # G
            rgba_sample[2] = 0  # B
            rgba_sample[3] = sample[0]  # Use input as alpha/occupancy
            voxel_grid = rgba_sample
        else:  # RGBA case (4 channels)
            voxel_grid = sample
            
        sample_ = voxel_grid.cpu().numpy()
        
        # Extract non-zero voxels using alpha/occupancy channel
        binary_sample = (voxel_grid[3] > threshold)
        indices = torch.nonzero(binary_sample, as_tuple=False)
        vertices = indices.float()  # Convert voxel indices to vertices
        
        cube_faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # Bottom face
            [4, 5, 6], [4, 6, 7], [4, 6, 5], [4, 7, 6],  # Top face
            [0, 1, 5], [0, 5, 4], [0, 5, 1], [0, 4, 5],  # Side faces
            [2, 3, 7], [2, 7, 6], [2, 7, 3], [2, 6, 7],
            [1, 2, 6], [1, 6, 5], [1, 6, 2], [1, 5, 6],
            [0, 3, 7], [0, 7, 4], [0, 7, 3], [0, 4, 7]
        ], dtype=torch.int64, device=device)
        
        # Generate vertices for each voxel cube
        cube_vertices = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
        ], dtype=torch.float32, device=device)
    
        cube_vertices = 0.95 * cube_vertices
        
        all_vertices = []
        all_colors = []
        all_faces = []
        
        for idx, voxel in enumerate(indices):
            # Get color from voxel grid
            color = torch.stack([voxel_grid[0:3, voxel[0], voxel[1], voxel[2]]]*8)
            voxel_vertices = cube_vertices + voxel  # Add voxel position to cube vertices
            all_vertices.append(voxel_vertices)
            all_colors.append(color)
            
            # Offset cube faces by the current number of vertices
            offset = idx * 8  # 8 vertices per voxel
            all_faces.append(cube_faces + offset)
        
        if not all_vertices:  # Handle empty case
            print("No vertices found - model may be empty or threshold too high")
            return
        
        # Combine all vertices and faces
        vertices = torch.cat(all_vertices, dim=0)
        faces = torch.cat(all_faces, dim=0)
        colors = torch.cat(all_colors, dim=0)
        
        # Use vertices directly without deduplication for simplicity
        unique_vertices = vertices
        
        # Debugging info
        print(f"Number of vertices: {len(unique_vertices)}")
        print(f"Max face index: {faces.max()}")
        print(f"Number of faces: {faces.size(0)}")
        
        # Ensure face indices are within bounds
        assert faces.max() < len(unique_vertices), "Face indices exceed number of vertices!"
        
        # Create the mesh object
        mesh = Meshes(verts=[unique_vertices], faces=[faces])
        
        # Define the object's center
        object_center = unique_vertices.mean(dim=0)
        
        # Set camera position (using fixed values for consistency)
        radius = 60.0  # Distance from object
        theta, phi = torch.tensor([5.6699]), torch.tensor([1.5873])  # Fixed angles
        
        # Convert spherical coordinates to Cartesian
        camera_position = torch.tensor([
            radius * torch.sin(phi) * torch.cos(theta),
            radius * torch.sin(phi) * torch.sin(theta),
            radius * torch.cos(phi)
        ], device=device)
        
        # Compute look-at rotation and translation
        R, T = look_at_view_transform(eye=camera_position.unsqueeze(0), at=object_center.unsqueeze(0), up=((0, 1, 0),))
        
        # Update the camera
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        
        # Create vertex textures
        textures = TexturesVertex(verts_features=[colors])
        
        # Create the mesh object with textures
        mesh = Meshes(verts=[unique_vertices], faces=[faces], textures=textures)
        
        # Renderer settings
        raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1)
        
        # Lighting setup
        lights = PointLights(device=device, location=[[0.0, 100.0, 100.0]])
        
        # Create renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
        )
        
        # Render the image
        images = renderer(mesh)
        image = images[0, ..., :3].cpu().numpy()  # Get the RGB image
        
        # Display the rendered image
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
        # Save the mesh as PLY
        save_ply_with_colors(
            "generated.ply",
            verts=unique_vertices,
            faces=faces,
            verts_colors=colors,
            colors_as_uint8=False  # Colors will be saved as floats in [0,1]
        )