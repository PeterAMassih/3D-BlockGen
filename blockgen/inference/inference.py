# blockgen/inference/inference.py
# Code for sampling from a trained diffusion model and visualizing the results.
# The save_ply_with_colors is taken from the pytorch3d library and customized for our use case with colors

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

from typing import Optional
from diffusers import DDPMScheduler, DDIMScheduler
from blockgen.models.diffusion import DiffusionModel3D
import os


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
    def __init__(self,
                 model: DiffusionModel3D,
                 noise_scheduler: DDPMScheduler,
                 device: str = 'cuda',
                 color_model: Optional[DiffusionModel3D] = None,
                 color_noise_scheduler: Optional[DDPMScheduler] = None):
        """Initialize with optional color model for two-stage generation.
        
        Args:
            model: Base diffusion model (shape model in two-stage case)
            noise_scheduler: Noise scheduler for base model
            device: Device to run inference on
            color_model: Optional color model for two-stage generation
            color_noise_scheduler: Optional noise scheduler for color model
        """
        self.model = model.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device

        # Add color model components (optional)
        self.color_model = color_model.to(device) if color_model is not None else None
        self.color_noise_scheduler = color_noise_scheduler

        # Initialize text encoder (used for both stages)
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

    def visualize_generation_pipeline(
            self,
            prompt: str,
            shape_sample: torch.Tensor,
            colored_sample: torch.Tensor,
            save_path: Optional[str] = None
    ):
        """
        Create a visualization showing the progression: Text -> Shape -> Colored Model
        
        Args:
            prompt: The text prompt used for generation
            shape_sample: The binary shape tensor [1, H, W, D] or [B, 1, H, W, D]
            colored_sample: The final RGBA tensor [4, H, W, D] or [B, 4, H, W, D]
            save_path: Optional path to save the visualization
        """

        # Ensure we have the right shape
        if shape_sample.dim() == 5:
            shape_sample = shape_sample[0]
        if colored_sample.dim() == 5:
            colored_sample = colored_sample[0]

        # Create figure with extra space between subplots
        fig = plt.figure(figsize=(16, 5))
        plt.subplots_adjust(wspace=0.6, left=0.05, right=0.95, top=0.85, bottom=0.15)

        # 1) Text prompt
        ax1 = fig.add_subplot(131)
        ax1.text(
            0.5, 0.5, f'"{prompt}"',
            horizontalalignment='center',
            verticalalignment='center',
            wrap=True,
            fontsize=14
        )
        ax1.axis('off')

        # 2) Shape visualization
        ax2 = fig.add_subplot(132, projection='3d')
        shape_occupancy = (shape_sample[0] > 0.5).cpu().numpy()
        ax2.voxels(shape_occupancy, edgecolor='k', alpha=0.5)
        ax2.view_init(elev=30, azim=45)
        ax2.set_title("Generated Shape")

        # 3) Color visualization
        ax3 = fig.add_subplot(133, projection='3d')
        occupancy = (colored_sample[3] > 0.5).cpu().numpy()
        rgb = colored_sample[:3].cpu().numpy()

        # Create RGBA values for voxels
        colors = np.zeros((*occupancy.shape, 4))
        rgb_clipped = np.clip(rgb, 0, 1)  # Ensure RGB values are in [0,1]
        rgb_hwdc = np.moveaxis(rgb_clipped, 0, -1)  # [H, W, D, 3]
        colors[..., :3] = rgb_hwdc
        colors[..., 3] = occupancy.astype(float)

        ax3.voxels(occupancy, facecolors=colors, edgecolor='k')
        ax3.view_init(elev=30, azim=45)
        ax3.set_title("Colored Model")

        # Arrow between "Text" and "Generated Shape" (lowered from y=0.50 to y=0.40)
        plt.annotate(
            '', xy=(0.31, 0.40), xytext=(0.26, 0.40),
            xycoords='figure fraction', textcoords='figure fraction',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
        )
        # Arrow between "Generated Shape" and "Colored Model" (same y shift)
        plt.annotate(
            '', xy=(0.66, 0.40), xytext=(0.61, 0.40),
            xycoords='figure fraction', textcoords='figure fraction',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
        )

        # Stage labels
        plt.figtext(0.32, 0.92, 'Stage 1', ha='center', va='top', fontsize=12)
        plt.figtext(0.62, 0.92, 'Stage 2', ha='center', va='top', fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_two_stage_diffusion(
            self,
            shape_intermediates: list[torch.Tensor],
            shape_timesteps: list[int],
            color_intermediates: list[torch.Tensor],
            color_timesteps: list[int],
            save_path: Optional[str] = None
    ):
        """Create a visualization showing both shape and color diffusion processes."""
        num_steps = len(shape_timesteps)
        fig = plt.figure(figsize=(4 * num_steps, 8))

        # Create subplots with proper spacing
        gs = plt.GridSpec(2, num_steps)
        gs.update(wspace=0.3, hspace=0.4)  # Add space between plots

        # Plot shape process
        for idx, (sample, t) in enumerate(zip(shape_intermediates, shape_timesteps)):
            ax = fig.add_subplot(gs[0, idx], projection='3d')
            occupancy = (sample[0, 0] > 0.5).cpu().numpy()

            # Use light blue color for shape visualization
            colors = np.zeros((*occupancy.shape, 4))
            colors[occupancy, :] = [0.5, 0.7, 1.0, 1.0]  # Light blue with alpha

            ax.voxels(occupancy, facecolors=colors, edgecolor='k', alpha=0.8)
            ax.view_init(elev=30, azim=45)
            ax.set_title(f'Shape t={t}')

            # Set proper aspect ratio and limits
            shape = occupancy.shape
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(0, shape[0])
            ax.set_ylim(0, shape[1])
            ax.set_zlim(0, shape[2])

            # Remove axis labels for cleaner look
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        # Plot color process
        for idx, (sample, t) in enumerate(zip(color_intermediates, color_timesteps)):
            ax = fig.add_subplot(gs[1, idx], projection='3d')
            occupancy = (sample[0, 3] > 0.5).cpu().numpy()

            # Important: Clip RGB values to [0, 1] range
            rgb = np.clip(sample[0, :3].cpu().numpy(), 0, 1)

            # Create colors for occupied voxels
            colors = np.zeros((*occupancy.shape, 4))
            rgb_hwdc = np.moveaxis(rgb, 0, -1)  # [H, W, D, 3]
            colors[occupancy, :3] = rgb_hwdc[occupancy]
            colors[occupancy, 3] = 1.0

            ax.voxels(occupancy, facecolors=colors, edgecolor='k', alpha=0.8)
            ax.view_init(elev=30, azim=45)
            ax.set_title(f'Color t={t}')

            # Set proper aspect ratio and limits
            shape = occupancy.shape
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(0, shape[0])
            ax.set_ylim(0, shape[1])
            ax.set_zlim(0, shape[2])

            # Remove axis labels for cleaner look
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        # Add stage labels
        plt.figtext(0.02, 0.75, 'Shape\nDiffusion', ha='left', va='center', fontsize=12)
        plt.figtext(0.02, 0.25, 'Color\nDiffusion', ha='left', va='center', fontsize=12)

        if save_path:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            # Save with high quality
            plt.savefig(save_path, bbox_inches='tight', dpi=300, format='png')
            plt.close()
        else:
            plt.show()

    def sample(self, prompt, num_samples=8, image_size=(32, 32, 32), show_intermediate=False, guidance_scale=7.0,
               use_mean_init=False, py3d=True, use_rotations=True, save_diffusion_process: bool = False,
               diffusion_steps_to_save: list[int] = None):

        with torch.no_grad():
            do_class_guidance = guidance_scale > 1.0

            # Initialize noise with correct shape: [B, C, H, W, D]
            num_channels = self.model.in_channels
            noise = torch.randn(num_samples, num_channels, *image_size).to(self.device)

            # Encode prompts
            encoder_hidden_states = self.encode_prompt([prompt] * num_samples)
            if do_class_guidance:
                encoder_hidden_states_uncond = self.encode_prompt([""] * num_samples)

            timesteps = self.noise_scheduler.timesteps.to(self.device)

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

            intermediates = []
            saved_timesteps = []

            for t in tqdm(timesteps, desc="Sampling Steps", total=len(timesteps)):
                # Get conditioned prediction
                if use_rotations:
                    residual = self._get_prediction_with_rotations(sample, t, encoder_hidden_states, self.model)
                else:
                    residual = self.model(sample, t, encoder_hidden_states=encoder_hidden_states).sample

                # Apply classifier guidance if needed
                if do_class_guidance:
                    if use_rotations:
                        residual_uncond = self._get_prediction_with_rotations(sample, t, encoder_hidden_states_uncond,
                                                                              self.model)
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
                pred_original_sample = (sample - beta_prod_t ** 0.5 * residual) / (alpha_prod_t ** 0.5)

                if show_intermediate and t % 50 == 49:
                    print(f"timestep: {t}")
                    if not py3d:
                        self.visualize_samples(sample, threshold=0.5)
                        self.visualize_samples(pred_original_sample, threshold=0.5)
                    else:
                        self.visualize_samples_p3d(pred_original_sample[0], threshold=0.5)

                # Get next sample
                sample = self.noise_scheduler.step(residual, t, sample).prev_sample

                # Save intermediates if requested
                if save_diffusion_process:
                    if diffusion_steps_to_save is None or t.item() in diffusion_steps_to_save:
                        intermediates.append(sample.detach().clone())
                        saved_timesteps.append(t.item())

            # Return based on whether we're saving the diffusion process
            if save_diffusion_process:
                return sample, intermediates, saved_timesteps

            return sample

    def sample_two_stage(self, prompt, num_samples=8, image_size=(32, 32, 32),
                         show_intermediate=False, guidance_scale=7.0, show_after_shape=False,
                         color_guidance_scale=7.0, use_rotations=True, use_mean_init=False,
                         save_pipeline_viz: Optional[str] = None, save_diffusion_process: bool = False,
                         diffusion_steps_to_save: list[int] = None):
        """
        Two-stage generation process: first shape, then color.
        
        This follows our training setup:
        1. Shape stage generates binary occupancy mask (1 channel)
        2. Color stage uses generated shape as occupancy mask and adds RGB colors (4 channels total)
        
        Args:
            prompt: Text prompt for conditioning
            num_samples: Number of samples to generate
            image_size: Size of voxel grid (H, W, D)
            show_intermediate: Show denoising steps
            guidance_scale: Classifier-free guidance scale for shape generation
            color_guidance_scale: Classifier-free guidance scale for color generation
            use_rotations: Use rotational augmentation during sampling
            use_mean_init: Initialize with mean values instead of pure noise
            save_pipeline_viz: Optional path to save visualization of the generation pipeline
        """
        if self.color_model is None:
            raise ValueError("Color model and scheduler required for two-stage sampling")

        with torch.no_grad():
            # Stage 1: Shape Generation
            print("Stage 1: Generating shapes...")

            if save_diffusion_process:
                shape_samples, shape_intermediates, shape_timesteps = self.sample(
                    prompt=prompt,
                    num_samples=num_samples,
                    image_size=image_size,
                    show_intermediate=show_intermediate,
                    guidance_scale=guidance_scale,
                    use_rotations=use_rotations,
                    use_mean_init=use_mean_init,
                    save_diffusion_process=True,
                    diffusion_steps_to_save=diffusion_steps_to_save
                )  # Output: [B, 1, H, W, D]
            else:
                shape_samples = self.sample(
                    prompt=prompt,
                    num_samples=num_samples,
                    image_size=image_size,
                    show_intermediate=show_intermediate,
                    guidance_scale=guidance_scale,
                    use_rotations=use_rotations,
                    use_mean_init=use_mean_init
                )

            # Convert to binary occupancy
            shape_occupancy = (shape_samples > 0.5).float()  # [B, 1, H, W, D]

            if show_after_shape:
                print("\nShape stage completed. Visualization:")
                self.visualize_samples(shape_occupancy, threshold=0.5)

            # Stage 2: Color Generation
            color_intermediates = []
            color_timesteps = []
            print("\nStage 2: Adding colors...")

            # Initialize noise for color stage
            # We want RGB noise only where we have shape, and clean alpha mask
            # Initialize noise for color stage
            noise = torch.randn(num_samples, 4, *image_size).to(self.device)  # [B, 4, H, W, D]
            # Properly broadcast shape occupancy for RGB channels
            noise[:, :3] = noise[:, :3] * shape_occupancy.repeat(1, 3, 1, 1, 1)  # [B, 3, H, W, D]
            noise[:, 3:] = shape_occupancy  # Set alpha to binary shape

            if use_mean_init:
                # For mean init, we still respect the shape mask
                mean_data = torch.zeros_like(noise)
                mean_data[:, :3] = 0.5 * shape_occupancy  # RGB = 0.5 where shape exists
                mean_data[:, 3:] = shape_occupancy
                noise = mean_data

            # Text embeddings for color stage
            encoder_hidden_states = self.encode_prompt([prompt] * num_samples)
            if color_guidance_scale > 1.0:
                encoder_hidden_states_uncond = self.encode_prompt([""] * num_samples)

            # Color denoising loop
            timesteps = self.color_noise_scheduler.timesteps.to(self.device)
            sample = noise

            for t in tqdm(timesteps, desc="Color Sampling"):
                # Get color prediction (model outputs RGB noise)
                if use_rotations:
                    residual = self._get_prediction_with_rotations(
                        sample=sample,  # Full RGBA input [B, 4, H, W, D]
                        t=t,
                        encoder_hidden_states=encoder_hidden_states,
                        model=self.color_model
                    )  # RGB prediction [B, 3, H, W, D]

                    if color_guidance_scale > 1.0:
                        residual_uncond = self._get_prediction_with_rotations(
                            sample=sample,
                            t=t,
                            encoder_hidden_states=encoder_hidden_states_uncond,
                            model=self.color_model
                        )
                else:
                    residual = self.color_model(
                        sample, t, encoder_hidden_states=encoder_hidden_states
                    ).sample

                    if color_guidance_scale > 1.0:
                        residual_uncond = self.color_model(
                            sample, t, encoder_hidden_states=encoder_hidden_states_uncond
                        ).sample

                # Apply guidance scale if needed
                if color_guidance_scale > 1.0:
                    residual = residual_uncond + color_guidance_scale * (residual - residual_uncond)

                # Mask residual to shape
                residual = residual * shape_occupancy

                # Create full RGBA residual with clean alpha
                full_residual = torch.zeros_like(sample)  # [B, 4, H, W, D]
                full_residual[:, :3] = residual  # RGB noise
                # Alpha residual stays zero

                # Denoising step
                sample = self.color_noise_scheduler.step(full_residual, t, sample).prev_sample

                # Ensure RGB exists only where we have shape
                sample = torch.cat([
                    sample[:, :3] * shape_occupancy,  # RGB masked by shape
                    shape_occupancy  # Keep original binary mask
                ], dim=1)

                if show_intermediate and t % 50 == 49:
                    print(f"\nColor timestep: {t}")
                    self.visualize_samples(sample, threshold=0.5)

                if save_diffusion_process:
                    if diffusion_steps_to_save is None or t in diffusion_steps_to_save:
                        color_intermediates.append(sample.detach().clone())
                        color_timesteps.append(t.item())

            # Create pipeline visualization if requested
            if save_pipeline_viz:
                self.visualize_generation_pipeline(
                    prompt=prompt,
                    shape_sample=shape_occupancy,
                    colored_sample=sample,
                    save_path=save_pipeline_viz
                )

            if save_diffusion_process:
                self.visualize_two_stage_diffusion(
                    shape_intermediates=shape_intermediates,
                    shape_timesteps=shape_timesteps,
                    color_intermediates=color_intermediates,
                    color_timesteps=color_timesteps,
                    save_path='plots/two_stage_diffusion_process.png'
                )

            return sample

    def _get_prediction_with_rotations(self, sample, t, encoder_hidden_states, model):
        """Helper for rotation augmented prediction with specified model."""
        # Original orientation
        pred1 = model(sample, t, encoder_hidden_states=encoder_hidden_states).sample

        # First rotation: [B, C, H, W, D] -> [B, C, D, H, W]
        sample_rot1 = sample.permute(0, 1, 4, 2, 3)
        pred2 = model(sample_rot1, t, encoder_hidden_states=encoder_hidden_states).sample
        pred2 = pred2.permute(0, 1, 3, 4, 2)  # Back to [B, C, H, W, D]

        # Second rotation: [B, C, H, W, D] -> [B, C, W, D, H]
        sample_rot2 = sample.permute(0, 1, 3, 4, 2)
        pred3 = model(sample_rot2, t, encoder_hidden_states=encoder_hidden_states).sample
        pred3 = pred3.permute(0, 1, 4, 2, 3)  # Back to [B, C, H, W, D]

        # Average predictions
        return (pred1 + pred2 + pred3) / 3.0

    def visualize_samples(self, samples, prompt="3D view", threshold=0.5, save_path=None):
        """
        Visualize samples with 2D and 3D views.
        
        Args:
            samples: Tensor [B, C, H, W, D]
                    C=1 for shape stage (occupancy)
                    C=4 for color stage (RGB + alpha/occupancy)
            threshold: Binary occupancy threshold (default 0.5)
        """
        # Convert to numpy and handle single sample case
        samples_ = samples.cpu().numpy()
        if len(samples_.shape) == 4:  # Single sample [C, H, W, D]
            samples_ = samples_[np.newaxis, ...]  # Add batch dim [1, C, H, W, D]

        # Setup plots
        num_samples = len(samples_)
        if num_samples == 1:
            fig, axs = plt.subplots(2, 1, figsize=(6, 10))
            axs = axs.reshape(-1, 1)  # Reshape to 2x1 for consistent indexing
        else:
            fig, axs = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

        for i, sample in enumerate(samples_):  # sample shape: [C, H, W, D]
            if sample.shape[0] > 1:  # RGBA case [4, 32, 32, 32]
                # Get binary occupancy from alpha channel
                occupancy = (sample[3] > threshold).astype(bool)  # [32, 32, 32]
                rgb = np.clip(sample[:3], 0, 1)  # [3, 32, 32, 32]

                # Print basic stats
                print("\nSample Statistics:")
                print("Occupancy:")
                print(f"- Total voxels: {occupancy.size}")
                print(f"- Occupied voxels: {np.sum(occupancy)} ({(np.sum(occupancy) / occupancy.size) * 100:.2f}%)")

                # Color dominance analysis
                # Reshape RGB to [3, -1] and occupancy to [-1] for proper indexing
                rgb_reshaped = rgb.reshape(3, -1)  # [3, 32*32*32]
                occupancy_flat = occupancy.flatten()  # [32*32*32]
                rgb_occupied = rgb_reshaped[:, occupancy_flat]  # Get colors of occupied voxels [3, N]

                # Calculate color percentages
                channel_means = rgb_occupied.mean(axis=1)  # Average per channel [3]
                total = channel_means.sum()
                if total > 0:  # Avoid division by zero
                    color_percentages = (channel_means / total) * 100
                    print("\nColor Dominance:")
                    print(f"Red: {color_percentages[0]:.1f}%")
                    print(f"Green: {color_percentages[1]:.1f}%")
                    print(f"Blue: {color_percentages[2]:.1f}%")

                # 2D slice visualization
                mid_depth = sample.shape[3] // 2
                slice_img = np.zeros((*rgb[:, :, :, mid_depth].shape[1:], 4))

                # Get middle slices
                rgb_slice = rgb[:, :, :, mid_depth]  # [3, H, W]
                alpha_slice = occupancy[:, :, mid_depth]  # [H, W]

                if np.any(alpha_slice):
                    rgb_slice_hwc = np.moveaxis(rgb_slice, 0, -1)  # [H, W, 3]
                    slice_img[alpha_slice] = np.concatenate([
                        rgb_slice_hwc[alpha_slice],
                        np.ones((np.sum(alpha_slice), 1))
                    ], axis=1)

                # Show slices
                axs[0, i].imshow(slice_img)
                axs[0, i].set_title(f"Center Slice (d={mid_depth})")
                axs[0, i].axis("off")

                # 3D visualization
                ax = axs[1, i]
                ax.remove()
                ax = fig.add_subplot(2, num_samples, num_samples + i + 1, projection='3d')

                # Create colors for occupied voxels
                colors = np.zeros((*occupancy.shape, 4))
                if np.any(occupancy):
                    rgb_hwdc = np.moveaxis(rgb, 0, -1)  # [H, W, D, 3]
                    colors[occupancy, :3] = rgb_hwdc[occupancy]
                    colors[occupancy, 3] = 1.0

                ax.voxels(occupancy, facecolors=colors, edgecolor='k', alpha=0.8)
                shape = occupancy.shape

            else:  # Shape stage (single channel)
                binary_sample = (sample[0] > threshold).astype(bool)
                print(
                    f"\nShape occupancy: {np.sum(binary_sample)} voxels ({(np.sum(binary_sample) / binary_sample.size) * 100:.2f}%)")

                # 2D slice
                axs[0, i].imshow(binary_sample[:, :, binary_sample.shape[2] // 2], cmap="gray")
                axs[0, i].set_title("Center Slice")
                axs[0, i].axis("off")

                # 3D view
                ax = axs[1, i]
                ax.remove()
                ax = fig.add_subplot(2, num_samples, num_samples + i + 1, projection='3d')
                ax.voxels(binary_sample, edgecolor='k')
                shape = binary_sample.shape

            # Plot settings
            ax.view_init(elev=30, azim=45)
            ax.set_title(prompt)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(0, shape[0])
            ax.set_ylim(0, shape[1])
            ax.set_zlim(0, shape[2])

        plt.tight_layout()

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()
        else:
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
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]  # Top face
        ], dtype=torch.float32, device=device)

        cube_vertices = 0.95 * cube_vertices

        all_vertices = []
        all_colors = []
        all_faces = []

        for idx, voxel in enumerate(indices):
            # Get color from voxel grid
            color = torch.stack([voxel_grid[0:3, voxel[0], voxel[1], voxel[2]]] * 8)
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
