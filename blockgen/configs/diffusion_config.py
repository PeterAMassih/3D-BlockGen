import torch
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