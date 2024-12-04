import torch
import torch.nn as nn
import torch.nn.functional as F

class SharpBCEWithLogitsLoss(nn.Module):
    def __init__(self, sharpness_weight=0.6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.sharpness_weight = sharpness_weight
        
    def forward(self, pred, target):
        # Standard BCE loss with the logits
        bce_loss = self.bce(pred, target)
        
        probs = torch.sigmoid(pred)
        # Now we add loss when close to 0.5 instead of subtracting when far from 0.5
        sharpness_loss = torch.mean(1.0 - torch.abs(probs - 0.5) * 2)
        
        return bce_loss + self.sharpness_weight * sharpness_loss
    
class RGBALoss(nn.Module):
    """Combined loss for RGBA voxels"""
    def __init__(self, alpha_weight=1.0, rgb_weight=1.0):
        super().__init__()
        self.alpha_weight = alpha_weight
        self.rgb_weight = rgb_weight
        self.alpha_loss = nn.MSELoss()  # Better for binary classification, because we need class probabilities
    
    def forward(self, model_output, noisy_sample, timesteps, target, diffusion_model):
        """
        Args:
            model_output: Predicted noise
            noise: Target noise
            noisy_sample: Current noisy input
            timesteps: Current timesteps
            target: Original clean sample [B, 4, H, W, D] (R, G, B, alpha)
            diffusion_model: DiffusionModel3D instance for scheduler access
        """
        # Predict original sample
        pred_original = diffusion_model.predict_original_sample(
            noisy_sample, model_output, timesteps
        )
        
        # Split channels - RGBA format
        pred_rgb = pred_original[:, :3]  # First 3 channels are RGB
        pred_alpha = pred_original[:, 3]  # Last channel is alpha
        
        true_rgb = target[:, :3]  # RGB channels
        true_alpha = target[:, 3]  # Alpha channel
        
        # Binary Cross Entropy for alpha/occupancy
        alpha_loss = self.alpha_loss(pred_alpha, true_alpha)
        
        # MSE for RGB weighted by true occupancy
        # IF add RGBAO occupancy change to pred_rgba
        true_alpha_expanded = true_alpha.unsqueeze(1)  # [B, 1, H, W, D]
        rgb_loss = F.mse_loss(
            true_alpha_expanded * pred_rgb,
            true_alpha_expanded * true_rgb
        )
        
        return self.alpha_weight * alpha_loss + self.rgb_weight * rgb_loss