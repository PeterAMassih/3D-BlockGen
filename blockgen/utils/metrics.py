# blockgen/utils/metrics.py
import torch
from typing import Dict, Optional
from pathlib import Path
from tqdm import tqdm
import json

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute both shape and color metrics between prediction and target.
    
    Args:
        pred: Predicted tensor [B, C, H, W, D] 
        target: Ground truth tensor [B, C, H, W, D]
        threshold: Threshold for occupancy/alpha channel
    """
    # Ensure we have batch dimension
    if pred.dim() == 4:
        pred = pred.unsqueeze(0)
    if target.dim() == 4:
        target = target.unsqueeze(0)
    # Ensure both tensors are on the same device
    target = target.to(pred.device)
    # Get occupancy from last channel
    pred_occ = (pred[:, -1] > threshold).float()  # Shape: [B, H, W, D]
    target_occ = (target[:, -1] > threshold).float()  # Shape: [B, H, W, D]
    
    # Compute intersection and union for IoU
    intersection = torch.sum(pred_occ * target_occ, dim=(1, 2, 3))
    union = torch.sum((pred_occ + target_occ) > 0, dim=(1, 2, 3))
    
    # Calculate IoU
    iou = (intersection / (union + 1e-6)).mean().item()
    
    # Calculate F1 Score - new addition
    true_positives = intersection
    false_positives = torch.sum(pred_occ * (1 - target_occ), dim=(1, 2, 3))
    false_negatives = torch.sum((1 - pred_occ) * target_occ, dim=(1, 2, 3))
    
    precision = (true_positives / (true_positives + false_positives + 1e-6)).mean().item()
    recall = (true_positives / (true_positives + false_negatives + 1e-6)).mean().item()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    metrics = {'iou': iou, 'f1': f1}
    
    # If we have RGBA tensors, compute color metrics
    if pred.shape[1] == 4:  # Check for RGBA format
        common_mask = (pred_occ > 0) & (target_occ > 0)  # Shape: [B, H, W, D]
        if torch.any(common_mask):
            # Extract RGB channels
            pred_rgb = pred[:, :3]  # Shape: [B, 3, H, W, D]
            target_rgb = target[:, :3]  # Shape: [B, 3, H, W, D]
            
            # Apply the common_mask to the spatial dimensions
            common_mask = common_mask.unsqueeze(1)  # Add channel dimension: [B, 1, H, W, D]
            pred_rgb_masked = pred_rgb * common_mask  # Masked RGB values
            target_rgb_masked = target_rgb * common_mask  # Masked RGB values
            
            # Compute squared difference where the mask is true
            squared_diff = (pred_rgb_masked - target_rgb_masked) ** 2
            
            # Compute mean squared error (MSE) over valid entries
            mse = torch.sum(squared_diff) / torch.sum(common_mask).item()
            color_score = 1.0 / (1.0 + mse)  # Normalize to a score
            
            # Add color score to metrics
            metrics['color_score'] = color_score
            
            # Combined score (weighted average of IoU, F1, and color score)
            metrics['combined_score'] = 0.25 * iou + 0.25 * f1 + 0.5 * color_score
    else:
        metrics['combined_score'] = 0.5 * (iou + f1)  # Equal weighting for shape-only
    return metrics