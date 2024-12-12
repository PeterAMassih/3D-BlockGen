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
    project_name: str = "3D-BlockGen"
) -> Tuple[DiffusionTrainer, DiffusionModel3D]:
    """Creates model and trainer with specified configuration."""
    
    if voxel_config.mode == 'two_stage':
        # Create shape model
        shape_model = UNet3DConditionModel(
            sample_size=resolution,
            in_channels=1,
            out_channels=1,
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
        
        # Create color model
        color_model = UNet3DConditionModel(
            sample_size=resolution,
            in_channels=4,  # RGB + shape mask
            out_channels=3,  # RGB only
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
        
        print(f"Shape model parameters: {sum(p.numel() for p in shape_model.parameters())}")
        print(f"Color model parameters: {sum(p.numel() for p in color_model.parameters())}")
        
        diffusion_model = DiffusionModel3D(
            model=shape_model,
            model_color=color_model,
            config=diffusion_config
        ).to(device)
        
    else:
        # for occupancy_only or rgba_combined
        model = UNet3DConditionModel(
            sample_size=resolution,
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
        
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        
        diffusion_model = DiffusionModel3D(
            model=model,
            config=diffusion_config
        ).to(device)
    
    trainer = DiffusionTrainer(
        diffusion_model, 
        voxel_config, 
        device=device,
        wandb_key=wandb_key,
        project_name=project_name,
    )
    
    return trainer, diffusion_model