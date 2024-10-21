import torch
import torch.nn as nn
from diffusers import DDPMScheduler

class DiffusionModel3D(nn.Module):
    def __init__(self, model, num_timesteps=1000):
        super(DiffusionModel3D, self).__init__()
        self.model = model 
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
        
        # Zero vector for dummy encoder_hidden_states
        self.dummy_encoder = nn.Parameter(torch.zeros(1, 1, model.cross_attention_dim), requires_grad=False)
        # used nn.Parameter to make it a part of the model's parameters, + no grad required for this tensor because it's just a dummy tensor

    def forward(self, x, t):
        # Use the dummy encoder_hidden_states
        batch_size = x.shape[0]
        encoder_hidden_states = self.dummy_encoder.repeat(batch_size, 1, 1) # 1, 1 just to repeat on the batch dimension only
        return self.model(x, t, encoder_hidden_states=encoder_hidden_states)

    def add_noise(self, clean_images, noise, timesteps):
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        return noisy_images

    def get_noise_prediction(self, noisy_images, timesteps):
        return self.forward(noisy_images, timesteps).sample