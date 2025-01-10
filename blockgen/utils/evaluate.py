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
        model: DiffusionModel3D,  # Primary model (either combined RGBA or shape model)
        model_type: str,  # 'combined' or 'two_stage'
        test_data_dir: str,
        annotation_file: str,
        color_model: Optional[DiffusionModel3D] = None,  # Only for two_stage
        num_eval_samples: int = 100,
        guidance_scale: float = 7.0,
        color_guidance_scale: float = 7.0,
        use_rotations: bool = False,
        use_mean_init: bool = False,
        device: str = 'cuda'
) -> Dict:
    """
    Evaluate model generation quality.
    
    Args:
        model: Primary model - either combined RGBA model or shape model for two-stage
        model_type: Either 'combined' or 'two_stage'
        test_data_dir: Directory containing evaluation samples
        annotation_file: Path to annotations file
        color_model: Color stage model (only used if model_type='two_stage')
        num_eval_samples: Number of samples to evaluate
        guidance_scale: Guidance scale for shape/combined model
        color_guidance_scale: Guidance scale for color model (two_stage only)
        use_rotations: Whether to use rotation augmentation during sampling
        device: Device to run inference on
    """
    if model_type not in ['combined', 'two_stage']:
        raise ValueError("model_type must be either 'combined' or 'two_stage'")

    if model_type == 'two_stage' and color_model is None:
        raise ValueError("color_model must be provided when model_type is 'two_stage'")

    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Get test files (excluding augmentations)
    test_files = [p for p in Path(test_data_dir).rglob("*.pt") if "_aug" not in p.stem][:num_eval_samples]
    print(f"Evaluating {len(test_files)} samples")

    # Create inferencer based on model type
    if model_type == 'combined':
        inferencer = DiffusionInference3D(
            model=model,
            noise_scheduler=model.noise_scheduler,
            device=device
        )
    else:  # two_stage
        inferencer = DiffusionInference3D(
            model=model,  # Shape model
            noise_scheduler=model.noise_scheduler,
            color_model=color_model,
            color_noise_scheduler=color_model.noise_scheduler,
            device=device
        )

    # Initialize metrics tracking
    per_metric_values = {
        'iou': [],
        'f1': [],
        'color_score': [],  # Always track color for both types
        'combined_score': []
    }

    progress_bar = tqdm(test_files, desc="Generating samples", leave=True)

    for test_file in progress_bar:
        model_id = test_file.stem

        # Create prompt matching dataset format
        if model_id in annotations:
            name = annotations[model_id].get('name', 'an object')
            categories = [cat.get('name', '') for cat in annotations[model_id].get('categories', [])
                          if isinstance(cat, dict)]
            tags = [tag.get('name', '') for tag in annotations[model_id].get('tags', [])
                    if isinstance(tag, dict)]

            prompt_parts = [name]
            if categories:
                prompt_parts.append(', '.join(categories))
            if tags:
                prompt_parts.append(', '.join(tags))
            prompt = ' '.join(prompt_parts)
        else:
            prompt = "an object"

        print(f"\nCurrent prompt: {prompt}")

        try:
            # Generate sample based on model type
            if model_type == 'combined':
                samples = inferencer.sample(
                    prompt=prompt,
                    num_samples=1,
                    image_size=(32, 32, 32),
                    guidance_scale=guidance_scale,
                    show_intermediate=False,
                    use_rotations=use_rotations,
                    use_mean_init=use_mean_init
                )
            else:  # two_stage
                samples = inferencer.sample_two_stage(
                    prompt=prompt,
                    num_samples=1,
                    image_size=(32, 32, 32),
                    guidance_scale=guidance_scale,
                    color_guidance_scale=color_guidance_scale,
                    show_intermediate=False,
                    use_rotations=use_rotations,
                    use_mean_init=use_mean_init
                )

            # Compute metrics
            target = torch.load(test_file)
            metrics = compute_metrics(samples[0], target)

            # Store metrics
            for key, value in metrics.items():
                if key in per_metric_values:
                    per_metric_values[key].append(value)

            # Update progress display
            progress_bar.set_description(
                " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            )

        except Exception as e:
            print(f"\nError processing {test_file}: {str(e)}")
            continue

    # Compute averages
    avg_metrics = {
        key: sum(values) / len(values)
        for key, values in per_metric_values.items()
        if values
    }

    return avg_metrics
