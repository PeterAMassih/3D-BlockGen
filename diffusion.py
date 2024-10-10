import torch.nn as nn
from diffusers import DDPMScheduler

class DiffusionModel3D(nn.Module):
    def __init__(self, model, num_timesteps=1000):
        super(DiffusionModel3D, self).__init__()
        self.model = model  # Your custom UNet3D model
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)

    def forward(self, x, t):
        return self.model(x, t)

    def add_noise(self, clean_images, noise, timesteps):
        # Add noise to the clean images according to the timestep
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        return noisy_images

    def get_noise_prediction(self, noisy_images, timesteps):
        # Get the model's prediction of the noise
        return self.forward(noisy_images, timesteps).sample

