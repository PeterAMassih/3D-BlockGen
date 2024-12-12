import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorStageLoss(nn.Module):
    """Loss function for color stage of two-stage training"""
    def __init__(self, rgb_weight: float = 1.0,):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.rgb_loss = nn.MSELoss()
    
    def forward(self, model_output, noisy_sample, timesteps, target, diffusion_model):
        """
        Args:
            model_output: Predicted noise for RGB channels
            noisy_sample: Current noisy input (RGBA)
            timesteps: Current timesteps
            target: Original clean sample [B, 4, H, W, D] (R, G, B, alpha)
            diffusion_model: DiffusionModel3D instance for scheduler access
        """
        # Predict original RGB values
        pred_original = diffusion_model.predict_original_sample(
            noisy_sample[:, :3],  # Only RGB channels
            model_output,  # Predicted RGB noise
            timesteps
        )
        
        true_rgb = target[:, :3]  # RGB channels
        true_alpha = target[:, 3:4]  # Alpha channel as mask
        
        # MSE for RGB, weighted by alpha mask
        rgb_loss = F.mse_loss(
            true_alpha * pred_original,
            true_alpha * true_rgb
        )
        
        return self.rgb_weight * rgb_loss


class RGBALoss(nn.Module):
    """Combined loss for RGBA voxels - used in combined mode"""
    def __init__(self, alpha_weight: float = 1.0, rgb_weight: float = 1.0, use_simple_mse: bool = False):
        super().__init__()
        self.alpha_weight = alpha_weight
        self.rgb_weight = rgb_weight
        self.alpha_loss = nn.MSELoss()
        self.use_simple_mse = use_simple_mse
    
    def forward(self, model_output, noisy_sample, timesteps, target, diffusion_model):
        # Predict original sample
        pred_original = diffusion_model.predict_original_sample(
            noisy_sample, model_output, timesteps
        )

        if self.use_simple_mse:
            return F.mse_loss(pred_original, target)
        
        # Split channels - RGBA format
        pred_rgb = pred_original[:, :3]  # RGB channels
        pred_alpha = pred_original[:, 3]  # Alpha channel
        
        true_rgb = target[:, :3]
        true_alpha = target[:, 3]
        
        # Binary Cross Entropy for alpha/occupancy
        alpha_loss = self.alpha_loss(pred_alpha, true_alpha)
        
        # MSE for RGB weighted by true occupancy
        true_alpha_expanded = true_alpha.unsqueeze(1)
        rgb_loss = F.mse_loss(
            true_alpha_expanded * pred_rgb,
            true_alpha_expanded * true_rgb
        )
        
        return self.alpha_weight * alpha_loss + self.rgb_weight * rgb_loss