import torch.nn as nn
from diffusers import DDPMScheduler, EMAModel, DDIMScheduler
from ..configs.diffusion_config import DiffusionConfig
import torch

# TODO FOR COMPLETNESS OVERRIDE THE .to() method of nn.Module to like this we load also in here to the device !
class DiffusionModel3D(nn.Module):
    def __init__(self, model, config: DiffusionConfig, use_ddim=False, model_color=None):
        super().__init__()
        self.model = model
        self.config = config
        self.model_color = model_color
        self.two_stage = model_color is not None
        self.training_stage = 'shape' if self.two_stage else 'combined'
        
        # Initialize schedulers
        if use_ddim:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=config.num_timesteps,
                beta_schedule="linear" 
            )
            if self.two_stage:
                self.color_scheduler = DDIMScheduler(
                    num_train_timesteps=config.num_timesteps,
                    beta_schedule="linear"
                )
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=config.num_timesteps
            )
            if self.two_stage:
                self.color_scheduler = DDPMScheduler(
                    num_train_timesteps=config.num_timesteps
                )

        # Initialize EMA if configured
        self.ema_model = None
        self.ema_color = None
        if config.use_ema:
            self.ema_model = EMAModel(
                parameters=model.parameters(),
                decay=config.ema_decay,
                update_after_step=config.ema_update_after_step,
                model_cls=type(model),
                model_config=model.config
            )
            self.ema_model.to(config.ema_device)
            
            if self.two_stage:
                self.ema_color = EMAModel(
                    parameters=model_color.parameters(),
                    decay=config.ema_decay,
                    update_after_step=config.ema_update_after_step,
                    model_cls=type(model_color),
                    model_config=model_color.config
                )
                self.ema_color.to(config.ema_device)

    def set_stage(self, stage: str):
        """Set current training/inference stage for two-stage mode"""
        if not self.two_stage:
            return
        if stage not in ['shape', 'color']:
            raise ValueError("Stage must be 'shape' or 'color' in two-stage mode")
        self.training_stage = stage

    def update_ema(self, step=None):
        """Update EMA model if enabled."""
        if step is None or step >= self.config.ema_update_after_step:
            if self.ema_model is not None:
                self.ema_model.step(self.model.parameters())
            if self.ema_color is not None and self.training_stage == 'color':
                self.ema_color.step(self.model_color.parameters())

    def forward(self, x, timesteps, encoder_hidden_states=None, return_dict=True):
        if not self.two_stage or self.training_stage == 'shape':
            return self.model(
                x, 
                timesteps, 
                encoder_hidden_states=encoder_hidden_states,
                return_dict=return_dict
            )
        else:
            return self.model_color(
                x,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=return_dict
            )

    def add_noise(self, clean_images, noise, timesteps):
        if not self.two_stage:
            return self.noise_scheduler.add_noise(clean_images, noise, timesteps)
            
        if self.training_stage == 'shape':
            # For shape stage, use only alpha channel if RGBA input
            if clean_images.shape[1] == 4:
                clean_images = clean_images[:, 3:4]
                noise = noise[:, 3:4]
            return self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        else:
            # For color stage, only noise RGB channels and keep alpha
            alpha = clean_images[:, 3:4]
            rgb = clean_images[:, :3]
            noisy_rgb = self.color_scheduler.add_noise(rgb, noise[:, :3], timesteps)
            return torch.cat([noisy_rgb, alpha], dim=1)

    def predict_original_sample(self, noisy_sample, noise_pred, timesteps):
        """Predict the original sample from a noisy sample and predicted noise"""
        scheduler = self.color_scheduler if self.two_stage and self.training_stage == 'color' else self.noise_scheduler
        alpha_prod = scheduler.alphas_cumprod[timesteps]
        beta_prod = 1 - alpha_prod
        
        # Use broadcasting because we have different alphas and betas for each sample in the batch
        alpha_prod = alpha_prod.view(-1, 1, 1, 1, 1)  # dim = (bs, 1, 1, 1, 1)
        beta_prod = beta_prod.view(-1, 1, 1, 1, 1)    # dim = (bs, 1, 1, 1, 1)
        
        # Predict original sample
        pred_original = (noisy_sample - (beta_prod ** 0.5) * noise_pred) / (alpha_prod ** 0.5)
        return pred_original

    def save_pretrained(self, save_path):
        """Save model(s) and EMA model(s) if enabled."""
        if not self.two_stage:
            # Original behavior
            self.model.save_pretrained(f"{save_path}_main")
            if self.ema_model is not None:
                self.ema_model.save_pretrained(f"{save_path}_ema")
        else:
            # Two-stage saving
            self.model.save_pretrained(f"{save_path}_shape")
            self.model_color.save_pretrained(f"{save_path}_color")
            if self.ema_model is not None:
                self.ema_model.save_pretrained(f"{save_path}_shape_ema")
            if self.ema_color is not None:
                self.ema_color.save_pretrained(f"{save_path}_color_ema")

    def load_pretrained(self, save_path, load_ema=False):
        """Load model weights, optionally from EMA checkpoint."""
        if not self.two_stage:
            # Original behavior
            if load_ema and self.ema_model is not None:
                self.ema_model = EMAModel.from_pretrained(f"{save_path}_ema", model_cls=type(self.model))
                self.ema_model.copy_to(self.model.parameters())
            else:
                main_path = f"{save_path}_main"
                self.model = self.model.from_pretrained(main_path)
        else:
            # Two-stage loading
            if load_ema:
                if self.ema_model is not None:
                    self.ema_model = EMAModel.from_pretrained(f"{save_path}_shape_ema", model_cls=type(self.model))
                    self.ema_model.copy_to(self.model.parameters())
                if self.ema_color is not None:
                    self.ema_color = EMAModel.from_pretrained(f"{save_path}_color_ema", model_cls=type(self.model_color))
                    self.ema_color.copy_to(self.model_color.parameters())
            else:
                self.model = self.model.from_pretrained(f"{save_path}_shape")
                self.model_color = self.model_color.from_pretrained(f"{save_path}_color")