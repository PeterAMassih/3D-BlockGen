import torch
import argparse
from pathlib import Path

from blockgen.configs.voxel_config import VoxelConfig
from blockgen.configs.diffusion_config import DiffusionConfig
from blockgen.utils.dataloaders import create_dataloaders
from blockgen.utils.model_factory import create_model_and_trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D-BlockGen model')
    parser.add_argument('--wandb_key', type=str, help='Weights & Biases API key')
    parser.add_argument('--project_name', type=str, default="3D-Blockgen", help='W&B project name')
    parser.add_argument('--data_dir', type=str, default='objaverse_data_voxelized', help='Directory containing voxel data')
    parser.add_argument('--annotation_file', type=str, default='objaverse_data/annotations.json', 
                       help='Path to annotations file')
    parser.add_argument('--save_dir', type=str, default='runs/experiment_two_stage_shape', help='Directory to save outputs')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--mode', type=str, choices=['occupancy_only', 'rgba_combined', 'two_stage'], default='two_stage',
                       help='Training mode')
    parser.add_argument('--stage', type=str, choices=['shape', 'color'], default='shape',
                       help='Training stage for two-stage mode')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up configs
    voxel_config = VoxelConfig(
        mode=args.mode,  # Using two-stage mode
        stage=args.stage,  # Start with shape stage
        default_color=[0.5, 0.5, 0.5],
        alpha_weight=1.0,
        rgb_weight=1.0,
        use_simple_mse=False  # If use simple MSE loss for RGBALoss only in the combined mode
    )
    
    # Training parameters
    train_params = {
        'batch_size': 4,
        'test_split': 0.0001, # We have 135560 3d objects * 4 augmentation 0.05 is 6778 so 128782 normal steps we use 129k
        'total_steps': 100,
        'save_every': 20,
        'eval_every': 20,
        'initial_lr': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    diffusion_config = DiffusionConfig(
        num_timesteps=1000,
        use_ema=True,  # Enable EMA
        ema_decay=0.9999,
        ema_update_after_step=0,
        ema_device=train_params['device'],
        use_ddim=False, 
        seed=42  # Set seed for reproducibility
    )
        
    # Paths - add stage to save directory
    data_dir = Path(args.data_dir)
    annotation_file = Path(args.annotation_file)
    save_dir = Path(args.save_dir) # Add shape subdirectory
    checkpoint_path = args.checkpoint_path
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        voxel_dir=data_dir,
        annotation_file=annotation_file,
        config=diffusion_config,
        config_voxel=voxel_config,
        batch_size=train_params['batch_size'],
        test_split=train_params['test_split']
    )
    
    # Create model and trainer with wandb config
    trainer, diffusion_model = create_model_and_trainer(
        voxel_config=voxel_config,
        diffusion_config=diffusion_config,
        resolution=32,
        device=train_params['device'],
        wandb_key=args.wandb_key,
        project_name=args.project_name,
        run_name = f"{args.mode}_{args.stage}" if args.mode == 'two_stage' else f"{args.mode}"
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
    
    print("Shape stage training completed!")
    print(f"Best test loss: {training_metrics['best_test_loss']:.4f}")
    print(f"Final step: {training_metrics['final_step']}")
    print(f"Models and metrics saved in: {save_dir}")
    print("\nNow you can proceed with color stage training using the shape model from:")
    print(f"  Best model: {save_dir}/{args.stage}/best_model/best_model")  
    print(f"  Final model: {save_dir}/{args.stage}/models/final_model")   

if __name__ == "__main__":
    main()