import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class DiffusionInference3D:
    def __init__(self, model, noise_scheduler, device='cuda'):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device

    def sample(self, num_samples=8, image_size=(32, 32, 32), intermediate_visualisation=False):
        # Initialize random noise as the starting point
        sample = torch.randn(num_samples, 1, *image_size).to(self.device)

        # Reverse denoising process, step by step, with progress bar
        timesteps = self.noise_scheduler.timesteps
        for t in tqdm(timesteps, desc="Sampling Steps", total=len(timesteps)):
            with torch.no_grad():
                residual = self.model(sample, t).sample

            alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (sample - beta_prod_t**0.5 * residual) / (alpha_prod_t ** 0.5)
            
            if intermediate_visualisation:
                print(f"timestep:{t}")
                self.visualize_samples(sample, threshold=0.5)
                self.visualize_samples(pred_original_sample, threshold=0.5)
                
            # Update the sample using the scheduler
            sample = self.noise_scheduler.step(residual, t, sample).prev_sample
            

        return sample

    
    def sample_ddim(self, num_samples=8, image_size=(32, 32, 32), num_inference_steps=50, intermediate_visualisation=False):
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        
        # Initialize random noise as the starting point
        sample = torch.randn(num_samples, 1, *image_size).to(self.device)

        # Reverse denoising process, step by step, with progress bar
        timesteps = self.noise_scheduler.timesteps
        for t in tqdm(timesteps, desc="Sampling Steps", total=len(timesteps)):
            with torch.no_grad():
                residual = self.model(sample, t).sample

            alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (sample - beta_prod_t**0.5 * residual) / (alpha_prod_t ** 0.5)
            
            if intermediate_visualisation:
                print(f"timestep:{t}")
                self.visualize_samples(sample, threshold=0.5)
                self.visualize_samples(pred_original_sample, threshold=0.5)
            
            # Update the sample using the scheduler
            sample = self.noise_scheduler.step(residual, t, sample).prev_sample
            

        return sample

    def visualize_samples(self, samples, threshold=0.5):
        # Convert the tensor to numpy for visualization
        samples_ = samples.cpu().numpy()
        print(samples_.shape)

        fig, axs = plt.subplots(1, len(samples_), figsize=(4*len(samples_), 8))
        for i, sample in enumerate(samples_):
            binary_sample = (sample > threshold).astype(np.bool)
            # Visualize center slice
            axs[i].imshow(binary_sample[0, :, :, sample.shape[3]//2], cmap="gray")
            axs[i].set_title("Center Slice")
            axs[i].axis("off")

        plt.tight_layout()
        plt.show()

        
        # fig, axs = plt.subplots(1, len(samples_), figsize=(len(samples_), 4))
        # for i, sample in enumerate(samples_):
        #     # Visualize 3D plot
        #     binary_sample = (sample > threshold).astype(np.bool) # TODO change to bin
        #     ax = axs[i]
        #     ax.remove()
        #     ax = fig.add_subplot(1, len(samples), len(samples) + i + 1, projection='3d')
        #     ax.voxels(binary_sample[0], edgecolor='k')
        #     ax.set_title("3D Visualization")

        # plt.tight_layout()
        # plt.show()

def visualize_tensor(tensor: torch.Tensor, threshold=0.5) -> None:
    """
    Visualize a PyTorch tensor representing a voxel grid.

    Args:
        tensor (torch.Tensor): 4D PyTorch tensor with shape (1, 1, resolution, resolution, resolution).
        threshold (float): Threshold value for binarizing the tensor.
    """
    # Check if the tensor has the expected shape
    if tensor.dim() != 5 or tensor.shape[0] != 1 or tensor.shape[1] != 1:
        raise ValueError("Expected tensor shape: (1, 1, resolution, resolution, resolution)")

    # Convert tensor to numpy array and remove singleton dimensions
    voxel_grid = tensor.squeeze().cpu().numpy()

    # Binarize the voxel grid
    binary_grid = (voxel_grid > threshold).astype(np.float32)

    fig = plt.figure(figsize=(12, 4))
    
    # Plot center slice
    ax1 = fig.add_subplot(121)
    ax1.imshow(voxel_grid[:, :, voxel_grid.shape[2]//2], cmap="gray")
    ax1.set_title("Center Slice")
    ax1.axis("off")

    # Plot 3D visualization
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.voxels(binary_grid, edgecolor='k')
    ax2.set_title("3D Visualization")

    plt.tight_layout()
    plt.show()