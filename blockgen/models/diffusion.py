import torch.nn as nn
from diffusers import DDPMScheduler, EMAModel, DDIMScheduler
from ..configs.diffusion_config import DiffusionConfig

# TODO FOR COMPLETNESS OVERRIDE THE .to() method of nn.Module to like this we load also in here to the device !
class DiffusionModel3D(nn.Module):
    def __init__(self, model, config: DiffusionConfig, use_ddim=False):
        super().__init__()
        self.model = model
        self.config = config
        if use_ddim:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=config.num_timesteps,
                beta_schedule="linear" 
            )
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=config.num_timesteps
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

    def update_ema(self, step=None):
        """Update EMA model if enabled."""
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
        """Save both main model and EMA model if enabled."""
        self.model.save_pretrained(f"{save_path}_main")

        if self.ema_model is not None:
            self.ema_model.save_pretrained(f"{save_path}_ema")

    def load_pretrained(self, save_path, load_ema=False):
        """Load model weights, optionally from EMA checkpoint."""
        if load_ema and self.ema_model is not None:
            # Keep EMA loading as is since it works
            self.ema_model = EMAModel.from_pretrained(f"{save_path}_ema", model_cls=type(self.model))
            self.ema_model.copy_to(self.model.parameters())
        else:
            # Load main model following diffusers convention
            main_path = f"{save_path}_main"
            self.model = self.model.from_pretrained(main_path)