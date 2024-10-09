import torch.nn as nn
from diffusers import DDPMScheduler

class CustomUNetDiffusionPipeline(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

    def forward(self, noisy_images, timesteps):
        return self.unet(noisy_images)
