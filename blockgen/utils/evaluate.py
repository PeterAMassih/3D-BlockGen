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
        model: DiffusionModel3D,
        model_type: str,
        test_data_dir: str,
        annotation_file: str,
        color_model: Optional[DiffusionModel3D] = None,
        num_eval_samples: int = 100,
        guidance_scale: float = 7.0,
        color_guidance_scale: float = 7.0,
        use_rotations: bool = False,
        use_mean_init: bool = False,
        device: str = 'cuda'
) -> Dict:
    """Evaluate model generation quality."""
    if model_type not in ['combined', 'two_stage']:
        raise ValueError("model_type must be either 'combined' or 'two_stage'")

    if model_type == 'two_stage' and color_model is None:
        raise ValueError("color_model must be provided when model_type is 'two_stage'")

    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Get test files (excluding augmentations cause there is none anyways in the eval dataset)
    test_files = [p for p in Path(test_data_dir).rglob("*.pt") if "_aug" not in p.stem][:num_eval_samples]
    print(f"\nStarting evaluation of {len(test_files)} samples...")
    print(f"Model type: {model_type}")
    print(f"Guidance scale: {guidance_scale}" + (f", Color guidance: {color_guidance_scale}" if model_type == 'two_stage' else ""))
    print("-" * 50)

    if model_type == 'combined':
        inferencer = DiffusionInference3D(
            model=model,
            noise_scheduler=model.noise_scheduler,
            device=device
        )
    else:  # two_stage
        inferencer = DiffusionInference3D(
            model=model,
            noise_scheduler=model.noise_scheduler,
            color_model=color_model,
            color_noise_scheduler=color_model.noise_scheduler,
            device=device
        )

    # Initialize metrics tracking
    per_metric_values = {
        'iou': [],
        'f1': [],
        'color_score': [],
        'combined_score': []
    }

    # Set up progress bars with clear positions
    main_pbar = tqdm(total=len(test_files), 
                    desc="Overall Progress",
                    position=0, 
                    leave=True,
                    ncols=100)
    
    metrics_pbar = tqdm(bar_format='{desc}',
                       position=1, 
                       leave=True,
                       desc="Metrics | Waiting for first sample...")

    try:
        for i, test_file in enumerate(test_files, 1):
            model_id = test_file.stem

            # Create prompt
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

            try:
                # Generate samples
                if model_type == 'combined':
                    samples = inferencer.sample(
                        prompt=prompt,
                        num_samples=1,
                        image_size=(32, 32, 32),
                        guidance_scale=guidance_scale,
                        show_intermediate=False,
                        use_rotations=use_rotations,
                        use_mean_init=use_mean_init,
                        disable_progress_bar=True
                    )
                else:
                    samples = inferencer.sample_two_stage(
                        prompt=prompt,
                        num_samples=1,
                        image_size=(32, 32, 32),
                        guidance_scale=guidance_scale,
                        color_guidance_scale=color_guidance_scale,
                        show_intermediate=False,
                        use_rotations=use_rotations,
                        use_mean_init=use_mean_init,
                        disable_progress_bar=True
                    )

                # Compute metrics
                target = torch.load(test_file)
                metrics = compute_metrics(samples[0], target)

                # Store metrics
                for key, value in metrics.items():
                    if key in per_metric_values:
                        per_metric_values[key].append(value)

                print(f"\nSample {i}/{len(test_files)} - {model_id}")
                print(f"Prompt: {prompt}")
                print("Metrics:")
                for metric, value in metrics.items():
                    print(f"  {metric:15s}: {value:.4f}")
                print("-" * 50)

                # Update metrics display
                if per_metric_values['iou']:
                    current_averages = {
                        k: sum(v)/len(v) 
                        for k, v in per_metric_values.items()
                    }
                    metrics_str = " | ".join(
                        [f"{k}: {v:.4f}" for k, v in current_averages.items()]
                    )
                    metrics_pbar.set_description_str(
                        f"Running Averages | {metrics_str}"
                    )

                # Update progress
                main_pbar.update(1)
                main_pbar.set_description(
                    f"Processing ({i}/{len(test_files)})"
                )

            except Exception as e:
                print(f"\nError processing {test_file}: {str(e)}")
                continue

    finally:
        main_pbar.close()
        metrics_pbar.close()
        print("\n" + "=" * 50)

    avg_metrics = {
        key: sum(values) / len(values)
        for key, values in per_metric_values.items()
        if values
    }

    print("\nFinal Results:")
    print("-" * 20)
    for metric, value in avg_metrics.items():
        print(f"{metric:15s}: {value:.4f}")
    print("=" * 50)

    return avg_metrics
