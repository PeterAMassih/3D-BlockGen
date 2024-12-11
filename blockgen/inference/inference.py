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

    def sample(self, prompt, num_samples=8, image_size=(32, 32, 32), show_intermediate=False, guidance_scale=7.0, use_mean_init=False, py3d=True):
        with torch.no_grad():
            do_class_guidance = guidance_scale > 1.0
            
            num_channels = 4 if self.config.use_rgb else 1
            noise = torch.randn(num_samples, num_channels, *image_size).to(self.device)
            
            
            encoder_hidden_states = self.encode_prompt([prompt] * num_samples)
            if do_class_guidance:
                encoder_hidden_states_uncond = self.encode_prompt([""] * num_samples)
    
            timesteps = self.noise_scheduler.timesteps.to(self.device)
            # print(timesteps)

            # timesteps = timesteps[300:]

            
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
                    
                    residual = residual_uncond + guidance_scale*(residual - residual_uncond)
    
                alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (sample - beta_prod_t**0.5 * residual) / (alpha_prod_t ** 0.5)
                # print(t.item(), (alpha_prod_t**0.5).item(), (beta_prod_t**0.5).item())

                # pred_original_sample = pred_original_sample.clamp(0, 1)
    
                if show_intermediate and t % 50 == 49:
                    print(f"timestep: {t}")
                    if not py3d:
                        self.visualize_samples(sample, threshold=0.25) 
                        self.visualize_samples(pred_original_sample, threshold=0.25)
                    else:
                        #self.visualize_samples_p3d(sample[0], threshold=0.5)
                        self.visualize_samples_p3d(pred_original_sample[0], threshold=0.25)
                
                # prev_t = self.noise_scheduler.previous_timestep(t)
                # alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.noise_scheduler.one
                # beta_prod_t_prev = 1 - alpha_prod_t_prev
                # current_alpha_t = alpha_prod_t / alpha_prod_t_prev
                # current_beta_t = 1 - current_alpha_t
                # pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
                # current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
                # sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
                
                sample = self.noise_scheduler.step(residual, t, sample).prev_sample
                
            return sample

    def sample_ddim(self, prompt, num_samples=8, image_size=(32, 32, 32), num_inference_steps=50, show_intermediate=False, guidance_scale=7.0):
        with torch.no_grad():
            do_class_guidance = guidance_scale > 1.0
            
            num_channels = 4 if self.config.use_rgb else 1
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
            samples: Tensor of shape [B, C, H, W, D] where C is 1 (occupancy) or 4 (RGBA)
            threshold: Optional threshold value. If None, will be determined by distribution
        """
        samples_ = samples.cpu().numpy()
        fig, axs = plt.subplots(2, len(samples_), figsize=(4*len(samples_), 8))
        
        for i, sample in enumerate(samples_):
            if self.config.use_rgb:
                # Print distributions and set threshold
                if threshold is None:
                    adaptive_threshold = np.percentile(sample[3], 90)
                    print(f"Using adaptive threshold: {adaptive_threshold:.3f}")
                else:
                    adaptive_threshold = threshold
                
                print(f"RGB range: [{sample[:3].min():.3f}, {sample[:3].max():.3f}]")
                print(f"Alpha range: [{sample[3].min():.3f}, {sample[3].max():.3f}]")
                print(f"Red:[{sample[0].min():.3f}, {sample[0].max():.3f}]" )
                print(f"Green: [[{sample[1].min():.3f}, {sample[1].max():.3f}]]")
                print(f"Blue: [[{sample[1].min():.3f}, {sample[1].max():.3f}]]")
                
                # Get occupancy from alpha channel
                occupancy = (sample[3] > adaptive_threshold).astype(bool)
                # Ensure RGB is in [0,1] range
                rgb = np.clip(sample[:3], 0, 1)
                
                print(f"Occupied voxels: {np.sum(occupancy)} ({(np.sum(occupancy)/occupancy.size)*100:.2f}% of volume)")
                
                # 2D slice visualization
                mid_slice_idx = sample.shape[2]//2
                # Create RGBA slice image
                rgb_slice = rgb[:, :, mid_slice_idx]  # Shape: [3, H, W]
                alpha_slice = occupancy[:, :, mid_slice_idx]  # Shape: [H, W]
                
                # Create empty RGBA image
                slice_img = np.zeros((*rgb_slice.shape[1:], 4))  # Shape: [H, W, 4]
                if np.any(alpha_slice):
                    # For occupied voxels, set RGB and alpha
                    rgb_occupied = np.moveaxis(rgb_slice[:, alpha_slice], 0, -1)  # Move channels to end
                    slice_img[alpha_slice] = np.concatenate([rgb_occupied, np.ones((np.sum(alpha_slice), 1))], axis=1)
                
                # Show 2D slice
                axs[0, i].imshow(slice_img)
                axs[0, i].set_title(f"Center Slice")
                axs[0, i].axis("off")
                
                # 3D visualization
                ax = axs[1, i]
                ax.remove()
                ax = fig.add_subplot(2, len(samples_), len(samples_) + i + 1, projection='3d')
                
                # Create RGBA colors for voxels
                colors = np.zeros((*occupancy.shape, 4))
                if np.any(occupancy):
                    # Set RGB values only for occupied voxels
                    for c in range(3):
                        colors[occupancy, c] = rgb[c, occupancy]
                    # Set alpha to 1 for occupied voxels
                    colors[occupancy, 3] = 1.0
                
                ax.voxels(occupancy, facecolors=colors, edgecolor='k', alpha=0.8)
                ax.view_init(elev=30, azim=45)
                ax.set_title("3D View")
                
            else:
                # Single channel visualization
                binary_sample = (sample[0] > threshold).astype(bool)
                axs[0, i].imshow(binary_sample[:, :, sample.shape[2]//2], cmap="gray")
                axs[0, i].set_title(f"Center Slice")
                axs[0, i].axis("off")
                
                ax = axs[1, i]
                ax.remove()
                ax = fig.add_subplot(2, len(samples_), len(samples_) + i + 1, projection='3d')
                ax.voxels(binary_sample, edgecolor='k')
                ax.view_init(elev=30, azim=45)
                ax.set_title("3D View")
            
            # Set axis limits for 3D plot
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(0, occupancy.shape[0])
            ax.set_ylim(0, occupancy.shape[1])
            ax.set_zlim(0, occupancy.shape[2])
        
        plt.tight_layout()
        plt.show()
    
    def visualize_samples_p3d(self, sample, threshold=0.5):
        device = "cuda"
        # Voxel grid parameters
        voxel_grid = sample
        sample_ = sample.cpu().numpy()

        # hist, bins = np.histogram(sample_[3], bins=10)
        # print(f"\nSample {0} distribution:")
        # print(f"Histogram bins: {bins}")
        # print(f"Histogram counts: {hist}")
        # print(f"Percentiles: 10%: {np.percentile(sample_[3], 10):.3f}, "
        #       f"50%: {np.percentile(sample_[3], 50):.3f}, "
        #       f"90%: {np.percentile(sample_[3], 90):.3f}")
        
        # print(f"RG range: [{sample_[0:3].min():.3f}, {sample_[0:3].max():.3f}]")
        # print(f"Alpha range: [{sample_[3].min():.3f}, {sample_[3].max():.3f}]")
        
        
        # Extract non-zero voxels
        binary_sample = (sample[3] > threshold)
        indices = torch.nonzero(binary_sample, as_tuple=False)
        vertices = indices.float()  # Convert voxel indices to vertices
        
        cube_faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # Bottom face
            [4, 5, 6], [4, 6, 7],  # Top face
            [0, 1, 5], [0, 5, 4],  # Side faces
            [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5],
            [0, 3, 7], [0, 7, 4]
        ], dtype=torch.int64, device=device)
        
        # Generate vertices for each voxel cube
        cube_vertices = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
        ], dtype=torch.float32, device=device)
        
        all_vertices = []
        all_colors = []
        all_faces = []
        
        for idx, voxel in enumerate(indices):
            # Translate cube vertices to voxel position
        
            color = torch.stack([voxel_grid[0:3, voxel[0], voxel[1], voxel[2]]]*8)
            voxel_vertices = cube_vertices + voxel  # Add voxel position to cube vertices
            all_vertices.append(voxel_vertices)
            all_colors.append(color)
        
            # Offset cube faces by the current number of vertices
            offset = idx * 8  # 8 vertices per voxel
            all_faces.append(cube_faces + offset)
        
        # Combine all vertices and faces
        vertices = torch.cat(all_vertices, dim=0)
        faces = torch.cat(all_faces, dim=0)
        colors = torch.cat(all_colors, dim=0)
        
        # Deduplicate vertices and remap faces
        #unique_vertices, inverse_indices = torch.unique(vertices, dim=0, return_inverse=True)
        #faces = inverse_indices[faces]
        unique_vertices = vertices
        
        # Debugging step
        print(f"Number of vertices: {len(unique_vertices)}")
        print(f"Max face index: {faces.max()}")
        print(f"Number of faces: {faces.size(0)}")
        
        # Ensure face indices are within bounds
        assert faces.max() < len(unique_vertices), "Face indices exceed number of vertices!"
        
        # Create the mesh object
        mesh = Meshes(verts=[unique_vertices], faces=[faces])
        
        # Define the object's center
        object_center = unique_vertices.mean(dim=0)
        
        # Generate a random camera position in spherical coordinates
        radius = 60.0  # Distance from the object
        theta = torch.rand(1) * 2 * torch.pi  # Azimuthal angle
        phi =  torch.rand(1) * torch.pi  # Polar angle
        #print(theta, phi)
        theta, phi = torch.tensor([5.6699]), torch.tensor([1.5873])
        
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
        
        # Define a color for each vertex (e.g., all white or random colors)
        vertex_colors = colors#torch.rand_like(unique_vertices)
        # Alternatively, use random colors:
        # vertex_colors = torch.rand_like(unique_vertices)
        
        # Create vertex-based textures
        textures = TexturesVertex(verts_features=[vertex_colors])
        
        # Create the mesh object with textures
        mesh = Meshes(verts=[unique_vertices], faces=[faces], textures=textures)
        
        # Renderer settings (unchanged)
        raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1)
        
        
        lights = PointLights(device=device, location=[[0.0, 100.0, 100.0]])  # Place a light in front of the object
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

        save_ply_with_colors(
            "generated.ply",
            verts=unique_vertices,
            faces=faces,
            verts_colors=vertex_colors,
            colors_as_uint8=False  # Colors will be saved as floats in [0,1]
        )
            

def visualize_tensor(tensor: torch.Tensor, config, threshold=0.5) -> None:
    if config.use_rgb and (tensor.dim() != 5 or tensor.shape[0] != 1 or tensor.shape[1] != 4):
        raise ValueError("Expected RGBA tensor shape: (1, 4, resolution, resolution, resolution)")
    elif not config.use_rgb and (tensor.dim() != 5 or tensor.shape[0] != 1 or tensor.shape[1] != 1):
        raise ValueError("Expected occupancy tensor shape: (1, 1, resolution, resolution, resolution)")

    voxel_grid = tensor.squeeze().cpu().numpy()
    
    if config.use_rgb:
        binary_grid = (voxel_grid[3] > threshold).astype(bool)
        colors = np.zeros((*binary_grid.shape, 4))
        colors[binary_grid] = np.array([*voxel_grid[:3, binary_grid], 1]).T
    else:
        binary_grid = (voxel_grid > threshold).astype(np.float32)
        colors = None

    fig = plt.figure(figsize=(12, 4))
    
    ax1 = fig.add_subplot(121)
    if config.use_rgb:
        rgb_slice = np.moveaxis(voxel_grid[:3, :, :, voxel_grid.shape[2]//2], 0, -1)
        alpha_slice = voxel_grid[3, :, :, voxel_grid.shape[2]//2] > threshold
        slice_img = np.zeros((*rgb_slice.shape[:-1], 4))
        slice_img[alpha_slice] = np.array([*rgb_slice[alpha_slice].T, 1]).T
        ax1.imshow(slice_img)
    else:
        ax1.imshow(binary_grid[:, :, voxel_grid.shape[2]//2], cmap="gray")
    ax1.set_title("Center Slice")
    ax1.axis("off")

    ax2 = fig.add_subplot(122, projection='3d')
    if config.use_rgb:
        ax2.voxels(binary_grid, facecolors=colors, edgecolor='k')
    else:
        ax2.voxels(binary_grid, edgecolor='k')
    ax2.set_title("3D Visualization")

    plt.tight_layout()
    plt.show()