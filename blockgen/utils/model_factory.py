from diffusers import UNet3DConditionModel
from ..models.diffusion import DiffusionModel3D
from ..trainers.trainer import DiffusionTrainer
from ..configs import VoxelConfig, DiffusionConfig
from typing import Tuple


def create_model_and_trainer(
        voxel_config: VoxelConfig,
        diffusion_config: DiffusionConfig,
        resolution: int = 32,
        device: str = 'cuda',
        wandb_key: str = None,
        project_name: str = "3D-BlockGen",
        run_name: str = None
) -> Tuple[DiffusionTrainer, DiffusionModel3D]:
    """Creates model and trainer with specified configuration."""

    # Create UNet model with channels from voxel config
    out_channels = getattr(voxel_config, 'out_channels', voxel_config.in_channels)
    print(f"The number of input channels is: {voxel_config.in_channels}",
          f"The number of out channels is{out_channels}")
    model = UNet3DConditionModel(
        sample_size=resolution,
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
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Create diffusion model with mode and stage from config
    diffusion_model = DiffusionModel3D(
        model=model,
        config=diffusion_config,
        mode=voxel_config.mode,
        stage=voxel_config.stage
    ).to(device)

    # Create trainer with stage-specific project name if needed
    trainer = DiffusionTrainer(
        diffusion_model,
        voxel_config,
        device=device,
        wandb_key=wandb_key,
        project_name=f"{project_name}",
        run_name=run_name
    )

    return trainer, diffusion_model
