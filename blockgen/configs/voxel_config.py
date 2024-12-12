import torch
import torch.nn as nn
from ..models.losses import RGBALoss, TwoStageLoss

class VoxelConfig:
    """Configuration for voxel processing"""
    def __init__(self, 
                 mode='occupancy_only',  # 'occupancy_only', 'rgba_combined', or 'two_stage'
                 default_color=[0.5, 0.5, 0.5],  # Default gray
                 alpha_weight=1.0,
                 rgb_weight=1.0,
                 use_simple_mse=False):
        """
        Args:
            mode: Training mode
                - 'occupancy_only': Original single channel mode
                - 'rgba_combined': Current RGBA mode
                - 'two_stage': Separate shape and color models
        """
        if mode not in ['occupancy_only', 'rgba_combined', 'two_stage']:
            raise ValueError("Mode must be one of: 'occupancy_only', 'rgba_combined', 'two_stage'")
            
        self.mode = mode
        self.default_color = torch.tensor(default_color)
        
        # Set channels based on mode
        self.in_channels = 1 if mode == 'occupancy_only' else 4
        self.use_rgb = mode in ['rgba_combined', 'two_stage']
        self.use_two_stage = mode == 'two_stage'
        
        # Loss weights
        self.alpha_weight = alpha_weight
        self.rgb_weight = rgb_weight
        self.use_simple_mse = use_simple_mse
    
    def get_loss_fn(self, device):
        """Returns appropriate loss function based on mode"""
        if self.mode == 'occupancy_only':
            return nn.MSELoss().to(device)
        elif self.mode == 'rgba_combined':
            return RGBALoss(
                alpha_weight=self.alpha_weight,
                rgb_weight=self.rgb_weight,
                use_simple_mse=self.use_simple_mse
            ).to(device)
        else:  # two_stage
            return TwoStageLoss(
                alpha_weight=self.alpha_weight,
                rgb_weight=self.rgb_weight
            ).to(device)