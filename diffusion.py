import torch
import torch.nn as nn
from diffusers import DDPMScheduler

class DiffusionModel3D(nn.Module):
    def __init__(self, model, num_timesteps=1000):
        super(DiffusionModel3D, self).__init__()
        self.model = model
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)

    def forward(self, x, timesteps, encoder_hidden_states=None, return_dict=True):
        return self.model(
            x, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict
        )

    def add_noise(self, clean_images, noise, timesteps):
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        return noisy_images
        
    def predict_original_sample(self, noisy_sample, noise_pred, timesteps):
        """Predict the original sample from a noisy sample and predicted noise"""
        alpha_prod = self.noise_scheduler.alphas_cumprod[timesteps] # dim = (, bs)
        beta_prod = 1 - alpha_prod
        
        # use broadcasting because we have different alphas and betas for each sample in the batch
        alpha_prod = alpha_prod.view(-1, 1, 1, 1, 1) # dim = (bs, 1, 1, 1, 1)
        beta_prod = beta_prod.view(-1, 1, 1, 1, 1) # dim = (bs, 1, 1, 1, 1)
        
        # Predict original sample
        pred_original = (noisy_sample - (beta_prod ** 0.5) * noise_pred) / (alpha_prod ** 0.5)
        return pred_original
