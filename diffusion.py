import torch
import torch.nn as nn
from diffusers import DDPMScheduler, EMAModel
import numpy as np
import random

class DiffusionConfig:
    """Configuration for diffusion model training"""
    def __init__(self,
                 num_timesteps=1000,
                 use_ema=False,
                 ema_decay=0.9999,
                 ema_update_after_step=0,
                 ema_device='cuda',
                 seed=None):
        self.num_timesteps = num_timesteps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_after_step = ema_update_after_step
        self.ema_device = ema_device
        self.seed = seed
        
        if seed is not None:
            # Set main seeds
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            # Basic CUDA settings for reproducibility
            torch.backends.cudnn.deterministic = True
            # Create generator for DataLoader
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = None


class DiffusionModel3D(nn.Module):
    def __init__(self, model, config: DiffusionConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_timesteps)
        
        # Initialize EMA if configured
        self.ema_model = None
        if config.use_ema:
            self.ema_model = EMAModel(
                model.parameters(),
                decay=config.ema_decay,
                model_cls=type(model),  # Use same class as base model
                model_config=model.config,
            )
            self.ema_model.to(config.ema_device)
        
    def update_ema(self, step=None):
        """Update EMA model if enabled"""
        if self.ema_model is not None:
            if step is None or step >= self.config.ema_update_after_step:
                self.ema_model.step(self.model.parameters())

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

    def save_pretrained(self, save_path):
        """Save both main model and EMA model if enabled"""
        # Save main model
        torch.save(self.model.state_dict(), f"{save_path}_main.pth")
        
        # Save EMA model if enabled
        if self.ema_model is not None:
            ema_state = self.model.state_dict()  # Get a copy of current model state
            self.ema_model.copy_to(ema_state)  # Copy EMA weights into it
            torch.save(ema_state, f"{save_path}_ema.pth")
    
    def load_pretrained(self, load_path, load_ema=False):
        """Load model weights, optionally from EMA checkpoint"""
        if load_ema and self.ema_model is not None:
            state_dict = torch.load(f"{load_path}_ema.pth")
        else:
            state_dict = torch.load(f"{load_path}_main.pth")
        self.model.load_state_dict(state_dict)
