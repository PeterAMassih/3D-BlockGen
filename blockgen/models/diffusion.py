import torch.nn as nn
from diffusers import DDPMScheduler, EMAModel, DDIMScheduler
from ..configs.diffusion_config import DiffusionConfig
import torch

class DiffusionModel3D(nn.Module):
    def __init__(self, model: nn.Module, config: DiffusionConfig, mode: str = 'combined', stage: str = None):
        """
        Args:
            model: UNet3DConditionModel instance
            config: DiffusionConfig instance
            mode: 'shape', 'combined', or 'two_stage'
            stage: For two_stage mode, either 'shape' or 'color'
        """
        super().__init__()
        self.model = model
        self.config = config
        self.mode = mode
        self.stage = stage

        if mode == 'two_stage' and stage not in ['shape', 'color']:
            raise ValueError("Stage must be 'shape' or 'color' in two_stage mode")

        # Initialize noise scheduler
        self.noise_scheduler = (
            DDIMScheduler(num_train_timesteps=config.num_timesteps, beta_schedule="linear")
            if config.use_ddim else
            DDPMScheduler(num_train_timesteps=config.num_timesteps)
        )

        # Initialize EMA if configured
        self.ema_model = None
        if config.use_ema:
            self.ema_model = EMAModel(
                parameters=model.parameters(),
                decay=config.ema_decay,
                update_after_step=config.ema_update_after_step,
                model_cls=type(model),
                model_config=model.config
            )
            self.ema_model.to(config.ema_device)

    def update_ema(self, step: int = None) -> None:
        """Update EMA model if enabled."""
        if self.ema_model is not None and (step is None or step >= self.config.ema_update_after_step):
            self.ema_model.step(self.model.parameters())

    def add_noise(self, clean_images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to images based on mode and stage
        
        Args:
            clean_images: Input tensor to add noise to
                shape: Either [B, 1, H, W, D] for occupancy or [B, 4, H, W, D] for RGBA
            noise: Random noise tensor matching input shape
            timesteps: Timesteps for noise addition
        
        Returns:
            Noisy tensor with same shape as input
        """
        if self.mode == 'two_stage':
            if self.stage == 'shape':
                # Shape stage: Add noise to occupancy channel
                return self.noise_scheduler.add_noise(clean_images, noise, timesteps)
            else:  # color stage
                # Color stage: Add noise only to RGB, keep alpha clean
                rgb = clean_images[:, :3]  # [B, 3, H, W, D]
                alpha = clean_images[:, 3:4]  # [B, 1, H, W, D]
                noisy_rgb = self.noise_scheduler.add_noise(rgb, noise[:, :3], timesteps)
                return torch.cat([noisy_rgb, alpha], dim=1)  # [B, 4, H, W, D]
        else:
            # Shape or combined mode: Add noise to all channels
            return self.noise_scheduler.add_noise(clean_images, noise, timesteps)

    def predict_original_sample(self, noisy_sample: torch.Tensor, noise_pred: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict original sample from noisy sample and predicted noise.
        
        Note: In color stage, this should be called only with RGB channels.
        """
        alpha_prod = self.noise_scheduler.alphas_cumprod[timesteps]  # [B]
        beta_prod = 1 - alpha_prod
        
        # Use broadcasting: [B] -> [B, 1, 1, 1, 1]
        alpha_prod = alpha_prod.view(-1, 1, 1, 1, 1)
        beta_prod = beta_prod.view(-1, 1, 1, 1, 1)
        
        return (noisy_sample - (beta_prod ** 0.5) * noise_pred) / (alpha_prod ** 0.5)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states: torch.Tensor = None, return_dict: bool = True):
        """Model forward pass"""
        return self.model(
            x,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict
        )

    def save_pretrained(self, save_path: str) -> None:
        """Save model and EMA model if enabled.
        
        Args:
            save_path: Base path for saving models. Will be appended with appropriate suffixes.
                      e.g., 'path/to/model' becomes 'path/to/model_main' and 'path/to/model_ema'
        """
        # Just append _main and _ema without stage (stage is handled in directory structure)
        self.model.save_pretrained(f"{save_path}_main")
        if self.ema_model is not None:
            self.ema_model.save_pretrained(f"{save_path}_ema")

    def load_pretrained(self, save_path: str, load_ema: bool = False) -> None:
        """Load model weights, optionally from EMA checkpoint.
        
        Args:
            save_path: Base path for loading models. Will be appended with appropriate suffixes.
            load_ema: Whether to load EMA weights.
        """
        if load_ema and self.ema_model is not None:
            self.ema_model = EMAModel.from_pretrained(
                f"{save_path}_ema",
                model_cls=type(self.model)
            )
            self.ema_model.copy_to(self.model.parameters())
        else:
            self.model = self.model.from_pretrained(f"{save_path}_main")
    
    @property
    def in_channels(self):
        return self.model.config.in_channels