# blockgen/utils/evaluate.py

import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

from ..inference.inference import DiffusionInference3D
from .metrics import compute_metrics
from ..models.diffusion import DiffusionModel3D

def evaluate_generation(
    shape_model: DiffusionModel3D,
    color_model: Optional[DiffusionModel3D],
    test_data_dir: str,
    annotation_file: str,
    num_eval_samples: int = 100,
    guidance_scale: float = 7.0,
    color_guidance_scale: float = 7.0,
    use_rotations: bool = True,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate model generation quality using best-known parameters.
    Matches VoxelDataset prompt creation exactly.
    """
    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Get test files (excluding augmentations)
    test_files = [p for p in Path(test_data_dir).rglob("*.pt") if "_aug" not in p.stem][:num_eval_samples]
    print(f"Evaluating {len(test_files)} samples")
    
    # Create inferencer
    inferencer = DiffusionInference3D(
        model=shape_model,
        noise_scheduler=shape_model.noise_scheduler,
        color_model=color_model,
        color_noise_scheduler=color_model.noise_scheduler if color_model else None,
        device=device
    )
    
    metrics_list = []
    progress_bar = tqdm(test_files, desc="Generating samples", leave=True)
    
    for test_file in progress_bar:
        model_id = test_file.stem
        
        # Create detailed prompt - exactly matching dataset
        if model_id in annotations:
            # Get base name
            name = annotations[model_id].get('name', 'an object')
            
            # Get categories and tags safely
            categories = annotations[model_id].get('categories', [])
            category_names = [cat.get('name', '') for cat in categories if isinstance(cat, dict)]
            
            tags = annotations[model_id].get('tags', [])
            tag_names = [tag.get('name', '') for tag in tags if isinstance(tag, dict)]
            
            # Build prompt parts
            prompt_parts = [name]
            if category_names:
                prompt_parts.append(', '.join(category_names))
            if tag_names:
                prompt_parts.append(', '.join(tag_names))
            
            # Join all parts
            prompt = ' '.join(prompt_parts)
        else:
            prompt = "an object"  # Default prompt
        
        # Print current prompt
        print(f"\nCurrent prompt: {prompt}")
        
        # Generate sample
        if color_model:
            samples = inferencer.sample_two_stage(
                prompt=prompt,
                num_samples=1,
                image_size=(32, 32, 32),
                guidance_scale=guidance_scale,
                color_guidance_scale=color_guidance_scale,
                show_intermediate=False,
                use_rotations=use_rotations
            )
        else:
            samples = inferencer.sample(
                prompt=prompt,
                num_samples=1,
                image_size=(32, 32, 32),
                guidance_scale=guidance_scale,
                show_intermediate=False,
                use_rotations=use_rotations
            )
        
        # Load target and compute metrics
        target = torch.load(test_file)
        metrics = compute_metrics(samples[0], target)
        metrics_list.append(metrics)
        
        # Update progress bar description with metrics only
        progress_bar.set_description(
            " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        )
    
    # Compute average metrics
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
    
    return avg_metrics