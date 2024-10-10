import torch
import matplotlib.pyplot as plt

class DiffusionInference3D:
    def __init__(self, model, noise_scheduler, device='cuda'):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device

    def sample(self, num_samples=8, image_size=(32, 32, 32)):
        # Initialize random noise as the starting point
        sample = torch.randn(num_samples, 1, *image_size).to(self.device)

        # Reverse denoising process, step by step
        for t in self.noise_scheduler.timesteps:
            with torch.no_grad():
                residual = self.model(sample, t).sample
            # Update the sample using the scheduler
            sample = self.noise_scheduler.step(residual, t, sample).prev_sample

        return sample

    def visualize_samples(self, samples):
        # Convert the tensor to numpy for visualization
        samples = samples.cpu().numpy()

        fig, axs = plt.subplots(1, len(samples), figsize=(12, 4))
        for i, sample in enumerate(samples):
            axs[i].imshow(sample[0, :, :, 16], cmap="gray")  # Visualize the center slice along Z-axis
            axs[i].axis("off")
        plt.show()

