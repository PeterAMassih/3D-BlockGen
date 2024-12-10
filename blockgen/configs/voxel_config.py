import torch
import torch.nn as nn
from ..models.losses import RGBALoss

class VoxelConfig:
    """Configuration for voxel processing"""
    def __init__(self, 
                 use_rgb=False,
                 default_color=[0.5, 0.5, 0.5],  # Default gray
                 alpha_weight=1.0,
                 rgb_weight=1.0,
                 use_simple_mse=False):
        """
        Args:
            use_rgb: Whether to use RGB+alpha channels
            default_color: Default RGB values for occupancy-only data
            alpha_weight: Weight for occupancy/alpha loss
            rgb_weight: Weight for RGB loss when using colors
            use_simple_mse: Use simple MSE loss for RGBA
        """
        self.use_rgb = use_rgb
        self.default_color = torch.tensor(default_color)
        self.in_channels = 4 if use_rgb else 1
        self.alpha_weight = alpha_weight
        self.rgb_weight = rgb_weight
        self.use_simple_mse = use_simple_mse
    
    def get_loss_fn(self, device):
        """Returns appropriate loss function"""
        if self.use_rgb:
            return RGBALoss(
                alpha_weight=self.alpha_weight,
                rgb_weight=self.rgb_weight,
                use_simple_mse=self.use_simple_mse
            ).to(device)
        return nn.MSELoss().to(device)