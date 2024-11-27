import torch
from pathlib import Path

# Import your classes here (assuming they're in a separate file)
from voxel_diffusion import (
    VoxelConfig,
    create_dataloaders,
    create_model_and_trainer
)

from diffusion import DiffusionConfig

def main():
    # Set up configs
    voxel_config = VoxelConfig(
        use_rgb=False,  # Set to True if using RGBA data
        default_color=[0.5, 0.5, 0.5],
        alpha_weight=1.0,
        rgb_weight=1.0
    )
    
    # Training parameters
    train_params = {
        'batch_size': 2,
        'test_split': 0.05,
        'total_steps': 100_000,
        'save_every': 10_000,
        'eval_every': 5_000,
        'initial_lr': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    diffusion_config = DiffusionConfig(
        num_timesteps=1000,
        use_ema=True,  # Enable EMA
        ema_decay=0.9999,
        ema_update_after_step=0,
        ema_device=train_params['device'],
        seed=42  # Set seed for reproducibility
    )
    # Paths
    data_dir = Path('objaverse_data_voxelized')
    annotation_file = Path('objaverse_data/annotations.json')
    save_dir = Path('runs/experiment_5')
    checkpoint_path = None  # Set to path if resuming training
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        voxel_dir=data_dir,
        annotation_file=annotation_file,
        config=diffusion_config,
        batch_size=train_params['batch_size'],
        test_split=train_params['test_split']
    )
    
    # Create model and trainer
    # Create model and trainer with both configs
    trainer, diffusion_model = create_model_and_trainer(
        voxel_config=voxel_config,
        diffusion_config=diffusion_config,
        resolution=32,
        device=train_params['device']
    )
    
    # Start training
    training_metrics = trainer.train(
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        total_steps=train_params['total_steps'],
        save_every=train_params['save_every'],
        eval_every=train_params['eval_every'],
        save_dir=save_dir,
        checkpoint_path=checkpoint_path
    )
    
    print("Training completed!")
    print(f"Best test loss: {training_metrics['best_test_loss']:.4f}")
    print(f"Final step: {training_metrics['final_step']}")
    print(f"Models and metrics saved in: {save_dir}")

if __name__ == "__main__":
    main()