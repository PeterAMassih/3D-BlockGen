import torch
from pathlib import Path
from diffusers import UNet3DConditionModel, DDPMScheduler
from blockgen.configs.voxel_config import VoxelConfig
from blockgen.configs.diffusion_config import DiffusionConfig
from blockgen.models.diffusion import DiffusionModel3D
from blockgen.inference.inference import DiffusionInference3D

def load_model_for_inference(model_path: str, voxel_config: VoxelConfig, diffusion_config: DiffusionConfig, device='cuda', ema=True, use_ddim=False):
    """Load model for inference."""
    model = UNet3DConditionModel(
        sample_size=32,
        in_channels=voxel_config.in_channels,
        out_channels=voxel_config.in_channels,
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

    diffusion_model = DiffusionModel3D(model, config=diffusion_config, use_ddim=use_ddim)
    
    if ema:
        diffusion_model.load_pretrained(model_path, load_ema=True)
        print("Loaded EMA model weights")
    else:
        diffusion_model.load_pretrained(model_path, load_ema=False)
        print("Loaded main model weights")
    
    return diffusion_model

if __name__ == "__main__":
    voxel_config = VoxelConfig(
        use_rgb=True,  # Set to True if using RGBA data
        default_color=[0.5, 0.5, 0.5],
        alpha_weight=1.0,
        rgb_weight=1.0
    )
    
    diffusion_config = DiffusionConfig(
        num_timesteps=1000,
        use_ema=True,
        ema_decay=0.9999,
        ema_update_after_step=0,
        ema_device='cuda'
    )

    model_path = "runs/experiment_color/best_model"
    diffusion_model = load_model_for_inference(
        model_path=model_path,
        voxel_config=voxel_config,
        diffusion_config=diffusion_config
    )

    inferencer = DiffusionInference3D(
        model=diffusion_model.model,
        noise_scheduler=diffusion_model.noise_scheduler,
        config=voxel_config,
        device='cuda'
    )
 
    samples = inferencer.sample(
        prompt="A stone statue",
        num_samples=2,
        image_size=(32, 32, 32),
        show_intermediate=False,
        guidance_scale=7.0
    )

    inferencer.visualize_samples(samples)