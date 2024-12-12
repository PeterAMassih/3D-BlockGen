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
        if self.ema_model is not None:
            if step is None or step >= self.config.ema_update_after_step:
                self.ema_model.step(self.model.parameters())

    def add_noise(self, clean_images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to images based on mode and stage
        
        In two_stage color mode:
        - Takes full RGBA input
        - Only adds noise to RGB channels
        - Keeps alpha channel clean as a mask
        """
        if self.mode == 'two_stage' and self.stage == 'color':
            # For color stage, add noise only to RGB channels
            rgb = clean_images[:, :3]  # RGB channels
            alpha = clean_images[:, 3:4]  # Keep alpha as mask
            noisy_rgb = self.noise_scheduler.add_noise(rgb, noise[:, :3], timesteps)
            return noisy_rgb
        else:
            # For all other modes, add noise to all channels
            return self.noise_scheduler.add_noise(clean_images, noise, timesteps)

    def predict_original_sample(self, noisy_sample: torch.Tensor, noise_pred: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict original sample from noisy sample and predicted noise"""
        alpha_prod = self.noise_scheduler.alphas_cumprod[timesteps] # dim = [B, T, 1, 1, 1]
        beta_prod = 1 - alpha_prod
        
        # Use broadcasting for different samples in batch
        alpha_prod = alpha_prod.view(-1, 1, 1, 1, 1) # dim = [B*T, 1, 1, 1, 1]
        beta_prod = beta_prod.view(-1, 1, 1, 1, 1) # dim = [B*T, 1, 1, 1, 1]
        
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
        """Save model and EMA model if enabled"""
        # Save main model with stage suffix in two_stage mode
        suffix = f"_{self.stage}" if self.mode == 'two_stage' else ""
        self.model.save_pretrained(f"{save_path}{suffix}_main")
        
        if self.ema_model is not None:
            self.ema_model.save_pretrained(f"{save_path}{suffix}_ema")

    def load_pretrained(self, save_path: str, load_ema: bool = False) -> None:
        """Load model weights, optionally from EMA checkpoint"""
        suffix = f"_{self.stage}" if self.mode == 'two_stage' else ""
        
        if load_ema and self.ema_model is not None:
            self.ema_model = EMAModel.from_pretrained(
                f"{save_path}{suffix}_ema", 
                model_cls=type(self.model)
            )
            self.ema_model.copy_to(self.model.parameters())
        else:
            main_path = f"{save_path}{suffix}_main"
            self.model = self.model.from_pretrained(main_path)