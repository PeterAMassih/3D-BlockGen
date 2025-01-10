import torch
import torch.nn as nn
from ..models.losses import RGBALoss, ColorStageLoss


class VoxelConfig:
    """Configuration for voxel processing"""

    def __init__(self,
                 mode: str = 'shape',  # 'shape', 'combined', or 'two_stage'
                 stage: str = None,  # For two_stage: 'shape' or 'color'
                 default_color: list = [0.5, 0.5, 0.5],
                 alpha_weight: float = 1.0,
                 rgb_weight: float = 1.0,
                 use_simple_mse: bool = False):
        """
        Args:
            mode: Training mode ('shape', 'combined', 'two_stage')
            stage: Only used in two_stage mode ('shape' or 'color')
            default_color: Default RGB values for non-colored voxels
            alpha_weight: Weight for occupancy/alpha loss
            rgb_weight: Weight for RGB loss when using colors
            use_simple_mse: Use simple MSE loss for RGBA
        """
        if mode not in ['shape', 'combined', 'two_stage']:
            raise ValueError("Mode must be 'shape', 'combined', or 'two_stage'")

        if mode == 'two_stage' and stage not in ['shape', 'color']:
            raise ValueError("Stage must be 'shape' or 'color' in two_stage mode")

        self.mode = mode
        self.stage = stage
        self.default_color = torch.tensor(default_color)

        # Determine channels based on mode/stage
        if mode == 'shape':
            self.in_channels = 1
        elif mode == 'combined':
            self.in_channels = 4
        elif mode == 'two_stage':
            if stage == 'shape':
                self.in_channels = 1
            else:  # color stage
                self.in_channels = 4  # RGB + alpha mask
                self.out_channels = 3  # Only predict RGB noise

        self.alpha_weight = alpha_weight
        self.rgb_weight = rgb_weight
        self.use_simple_mse = use_simple_mse

    def get_loss_fn(self, device: str) -> nn.Module:
        """Returns appropriate loss function based on mode"""
        if self.mode == 'shape':
            # Shape mode only needs basic MSE
            return nn.MSELoss().to(device)
        elif self.mode == 'combined':
            # Combined mode needs all RGBA loss parameters
            return RGBALoss(
                alpha_weight=self.alpha_weight,
                rgb_weight=self.rgb_weight,
                use_simple_mse=self.use_simple_mse
            ).to(device)
        else:  # two_stage
            if self.stage == 'shape':
                # Shape stage only needs basic MSE
                return nn.MSELoss().to(device)
            else:  # color stage
                # Color stage only needs RGB-related parameters
                return ColorStageLoss(
                    rgb_weight=self.rgb_weight,
                    # no need for alpha_weight since we don't predict alpha
                ).to(device)

    def get_stage(self):
        """Determine the correct stage for logging based on mode and stage."""
        if self.mode == 'two_stage':
            return self.stage  # Either 'shape' or 'color'
        return self.mode  # 'shape' or 'combined' for other modes
