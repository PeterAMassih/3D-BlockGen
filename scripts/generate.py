from pathlib import Path
from diffusers import UNet3DConditionModel
from blockgen.configs.voxel_config import VoxelConfig
from blockgen.configs.diffusion_config import DiffusionConfig
from blockgen.models.diffusion import DiffusionModel3D
from blockgen.inference.inference import DiffusionInference3D
import argparse


def load_model_for_inference(model_path: str, voxel_config: VoxelConfig, diffusion_config: DiffusionConfig,
                             device='cuda', ema=True, inference_steps: int = 50):
    """Load model for inference."""

    out_channels = getattr(voxel_config, 'out_channels', voxel_config.in_channels)
    print(f"The number of input channels is: {voxel_config.in_channels}",
          f"The number of out channels is{out_channels}")
    model = UNet3DConditionModel(
        sample_size=32,
        in_channels=voxel_config.in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types=(
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        cross_attention_dim=512,
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    diffusion_model = DiffusionModel3D(model, config=diffusion_config)

    if ema:
        diffusion_model.load_pretrained(model_path, load_ema=True)
        print("Loaded EMA model weights")
    else:
        diffusion_model.load_pretrained(model_path, load_ema=False)
        print("Loaded main model weights")

    if diffusion_config.use_ddim:
        diffusion_model.noise_scheduler.set_timesteps(inference_steps)
        print(f"Set DDIM inference steps to {inference_steps}")

    return diffusion_model


def parse_args():
    parser = argparse.ArgumentParser(description='Generate 3D models using trained diffusion models')

    # Model configuration
    parser.add_argument('--mode', type=str, choices=['shape', 'combined', 'two_stage'],
                        default='two_stage', help='Generation mode')
    parser.add_argument('--shape_model_path', type=str,
                        default='runs/experiment_two_stage/shape/models/final_model',
                        help='Path to shape model checkpoint')
    parser.add_argument('--color_model_path', type=str,
                        default='runs/experiment_two_stage/color/models/final_model',
                        help='Path to color model checkpoint (for two_stage mode)')

    # Generation parameters
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for generation')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate')
    parser.add_argument('--guidance_scale', type=float, default=20.0,
                        help='Classifier-free guidance scale for shape generation')
    parser.add_argument('--color_guidance_scale', type=float, default=20.0,
                        help='Classifier-free guidance scale for color generation')

    # Inference settings
    parser.add_argument('--use_ddim', action='store_true',
                        help='Use DDIM sampling instead of DDPM')
    parser.add_argument('--inference_steps', type=int, default=50,
                        help='Number of inference steps for DDIM')
    parser.add_argument('--use_rotations', action='store_true',
                        help='Use rotation augmentation during inference')
    parser.add_argument('--use_mean_init', action='store_true',
                        help='Initialize with mean values instead of noise')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA weights for inference')

    # Output settings
    parser.add_argument('--save_dir', type=str, default='generated',
                        help='Directory to save generated samples')
    parser.add_argument('--show_intermediate', action='store_true',
                        help='Show intermediate generation steps')

    return parser.parse_args()


def main():
    args = parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Configure diffusion settings
    diffusion_config = DiffusionConfig(
        num_timesteps=1000,
        use_ema=args.use_ema,
        ema_decay=0.9999,
        ema_update_after_step=0,
        ema_device=args.device,
        use_ddim=args.use_ddim,
        seed=42
    )

    if args.mode == 'two_stage':
        # Shape stage
        shape_voxel_config = VoxelConfig(
            mode='two_stage',
            stage='shape',
            default_color=[0.5, 0.5, 0.5],
            alpha_weight=1.0,
            rgb_weight=1.0
        )

        shape_model = load_model_for_inference(
            model_path=args.shape_model_path,
            voxel_config=shape_voxel_config,
            diffusion_config=diffusion_config,
            device=args.device,
            ema=args.use_ema,
            inference_steps=args.inference_steps
        )

        # Color stage
        color_voxel_config = VoxelConfig(
            mode='two_stage',
            stage='color',
            default_color=[0.5, 0.5, 0.5],
            alpha_weight=1.0,
            rgb_weight=1.0
        )

        color_model = load_model_for_inference(
            model_path=args.color_model_path,
            voxel_config=color_voxel_config,
            diffusion_config=diffusion_config,
            device=args.device,
            ema=args.use_ema,
            inference_steps=args.inference_steps
        )

        # Create inference handler
        inferencer = DiffusionInference3D(
            model=shape_model.model,
            noise_scheduler=shape_model.noise_scheduler,
            config=shape_voxel_config,
            device=args.device,
            color_model=color_model.model,
            color_noise_scheduler=color_model.noise_scheduler
        )

        # Generate samples
        samples = inferencer.sample_two_stage(
            prompt=args.prompt,
            num_samples=args.num_samples,
            image_size=(32, 32, 32),
            show_intermediate=args.show_intermediate,
            guidance_scale=args.guidance_scale,
            color_guidance_scale=args.color_guidance_scale,
            use_rotations=args.use_rotations,
            use_mean_init=args.use_mean_init
        )

    else:  # shape or combined mode
        voxel_config = VoxelConfig(
            mode=args.mode,
            default_color=[0.5, 0.5, 0.5],
            alpha_weight=1.0,
            rgb_weight=1.0
        )

        model = load_model_for_inference(
            model_path=args.shape_model_path,  # Use shape_model_path for both modes
            voxel_config=voxel_config,
            diffusion_config=diffusion_config,
            device=args.device,
            ema=args.use_ema,
            inference_steps=args.inference_steps
        )

        inferencer = DiffusionInference3D(
            model=model.model,
            noise_scheduler=model.noise_scheduler,
            config=voxel_config,
            device=args.device
        )

        samples = inferencer.sample(
            prompt=args.prompt,
            num_samples=args.num_samples,
            image_size=(32, 32, 32),
            show_intermediate=args.show_intermediate,
            guidance_scale=args.guidance_scale,
            use_rotations=args.use_rotations,
            use_mean_init=args.use_mean_init
        )

    # Visualize and save samples
    inferencer.visualize_samples(samples, save_path=save_dir)


if __name__ == "__main__":
    main()

# Example usage:
# Basic two-stage generation
# python generate.py --prompt "A red apple" --num_samples 2 --guidance_scale 20.0 --color_guidance_scale 20.0

# # Two-stage with DDIM sampling
# python generate.py --prompt "A blue car" --use_ddim --inference_steps 50 --guidance_scale 20.0

# # Shape-only generation
# python generate.py --mode shape --prompt "A chair" --guidance_scale 20.0

# # Combined mode with rotations and EMA
# python generate.py --mode combined --prompt "A table" --use_rotations --use_ema
