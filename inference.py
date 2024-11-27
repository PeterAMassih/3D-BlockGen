import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

class DiffusionInference3D:
    def __init__(self, model, noise_scheduler, config, device='cuda'):
        self.model = model
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

    def sample(self, prompt, num_samples=8, image_size=(32, 32, 32), show_intermediate=False):
        with torch.no_grad():
            num_channels = 4 if self.config.use_rgb else 1
            sample = torch.randn(num_samples, num_channels, *image_size).to(self.device)
            
            encoder_hidden_states = self.encode_prompt([prompt] * num_samples)
    
            timesteps = self.noise_scheduler.timesteps
            for t in tqdm(timesteps, desc="Sampling Steps", total=len(timesteps)):
                residual = self.model(
                    sample, 
                    t,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
    
                alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (sample - beta_prod_t**0.5 * residual) / (alpha_prod_t ** 0.5)
                
                if show_intermediate:
                    print(f"timestep: {t}")
                    self.visualize_samples(sample, threshold=0.5) 
                    self.visualize_samples(pred_original_sample, threshold=0.5)  
    
                sample = self.noise_scheduler.step(residual, t, sample).prev_sample
    
                # if t.item() < 500:
                #     return torch.sigmoid(pred_original_sample)  # Apply sigmoid
    
            return sample  #

    def sample_ddim(self, prompt, num_samples=8, image_size=(32, 32, 32), num_inference_steps=50, show_intermediate=False):
        num_channels = 4 if self.config.use_rgb else 1
        sample = torch.randn(num_samples, num_channels, *image_size).to(self.device)
        
        # Encode prompt
        encoder_hidden_states = self.encode_prompt(prompt)
        
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        for t in tqdm(timesteps, desc="Sampling Steps", total=len(timesteps)):
            with torch.no_grad():
                residual = self.model(
                    sample, 
                    t,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

            alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (sample - beta_prod_t**0.5 * residual) / (alpha_prod_t ** 0.5)
            
            if show_intermediate:
                print(f"timestep: {t}")
                self.visualize_samples(sample, threshold=0.5)
                self.visualize_samples(pred_original_sample, threshold=0.5)

            sample = self.noise_scheduler.step(residual, t, sample).prev_sample

        return sample

    def visualize_samples(self, samples, threshold=0.5):
        samples_ = samples.cpu().numpy()
        print(samples_.shape)

        fig, axs = plt.subplots(2, len(samples_), figsize=(4*len(samples_), 8))
        for i, sample in enumerate(samples_):
            if self.config.use_rgb:
                # For RGBA, use alpha channel for occupancy and RGB for colors
                binary_sample = (sample[3] > threshold).astype(bool)
                colors = np.zeros((*binary_sample.shape, 4))
                colors[binary_sample] = np.array([*sample[:3, binary_sample], 1]).T
            else:
                binary_sample = (sample[0] > threshold).astype(bool)
                colors = None
            
            # Visualize center slice
            if self.config.use_rgb:
                # Get middle slice of RGB channels (3, H, W) - RGB channels for middle depth slice
                # # Shape: (H, W, 3) - Format that matplotlib expects for RGB images
                rgb_slice = np.moveaxis(sample[:3, :, :, sample.shape[3]//2], 0, -1)
                # Get alpha mask for same slice Shape: (H, W) - Binary mask where alpha > threshold of the slice
                alpha_slice = sample[3, :, :, sample.shape[3]//2] > threshold
                # Shape: (H, W, 4) - Empty RGBA image
                slice_img = np.zeros((*rgb_slice.shape[:-1], 4))
                # Fill RGB + alpha values where mask is True
                # after transpose shape is (3, N) after alpha (4, N)
                # each row is RGBA value for each pixel
                slice_img[alpha_slice] = np.array([*rgb_slice[alpha_slice].T, 1]).T # * needed to unpack RGB values rgb_slice[alpha_slice] shape is (N, 3)
                axs[0, i].imshow(slice_img)
            else:
                axs[0, i].imshow(binary_sample[:, :, sample.shape[3]//2], cmap="gray")
            axs[0, i].set_title("Center Slice")
            axs[0, i].axis("off")

            # Visualize 3D plot
            ax = axs[1, i]
            ax.remove()
            ax = fig.add_subplot(2, len(samples_), len(samples_) + i + 1, projection='3d')
            if self.config.use_rgb:
                ax.voxels(binary_sample, facecolors=colors, edgecolor='k')
            else:
                ax.voxels(binary_sample, edgecolor='k')
            ax.set_title("3D Visualization")

        plt.tight_layout()
        plt.show()

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