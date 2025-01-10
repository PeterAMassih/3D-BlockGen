# finetune.py
import torch
import argparse
from pathlib import Path

from blockgen.configs.voxel_config import VoxelConfig
from blockgen.configs.diffusion_config import DiffusionConfig
from blockgen.utils.dataloaders import create_dataloaders
from blockgen.utils.model_factory import create_model_and_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune 3D-BlockGen model')
    parser.add_argument('--wandb_key', type=str, help='Weights & Biases API key')
    parser.add_argument('--project_name', type=str, default="3D-Blockgen-Finetune", help='W&B project name')
    parser.add_argument('--data_dir', type=str, default='downloaded_objects_voxelized',
                        help='Directory containing voxelized files for finetuning')
    parser.add_argument('--label_mapping', type=str, default='objaverse_finetune/file_to_label_map.json',
                        help='Path to label mapping file')
    parser.add_argument('--save_dir', type=str, default='runs/finetune',
                        help='Directory to save outputs')
    parser.add_argument('--checkpoint_path', type=str, required=True,  # Made required since we need it for finetuning
                        help='Path to checkpoint for resuming training (e.g., runs/experiment_two_stage/shape/checkpoints/checkpoint_step_10000.pth)')
    parser.add_argument('--mode', type=str, choices=['occupancy_only', 'rgba_combined', 'two_stage'],
                        default='two_stage', help='Training mode')
    parser.add_argument('--stage', type=str, choices=['shape', 'color'], default='shape',
                        help='Training stage for two-stage mode')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up configs
    voxel_config = VoxelConfig(
        mode=args.mode,
        stage=args.stage,
        default_color=[0.5, 0.5, 0.5],
        alpha_weight=1.0,
        rgb_weight=1.0,
        use_simple_mse=False
    )

    # Training parameters - adjusted for finetuning
    train_params = {
        'batch_size': 4,
        'test_split': 0.1,  # Larger test split for finetuning
        'total_steps': 135_000,  # Fewer steps for finetuning
        'save_every': 1_000,
        'eval_every': 1_000,
        'initial_lr': 1e-5,  # Lower learning rate for finetuning
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    diffusion_config = DiffusionConfig(
        num_timesteps=1000,
        use_ema=True,
        ema_decay=0.9999,
        ema_update_after_step=0,
        ema_device=train_params['device'],
        use_ddim=False,
        seed=42
    )

    # Paths
    data_dir = Path(args.data_dir)
    label_mapping = Path(args.label_mapping)
    save_dir = Path(args.save_dir)

    # Create dataloaders with use_label_mapping=True
    train_loader, test_loader = create_dataloaders(
        voxel_dir=data_dir,
        annotation_file=label_mapping,  # Now using label mapping file
        config=diffusion_config,
        config_voxel=voxel_config,
        batch_size=train_params['batch_size'],
        test_split=train_params['test_split'],
        use_label_mapping=True  # Enable label mapping mode
    )

    # Create model and trainer
    trainer, diffusion_model = create_model_and_trainer(
        voxel_config=voxel_config,
        diffusion_config=diffusion_config,
        resolution=32,
        device=train_params['device'],
        wandb_key=args.wandb_key,
        project_name=args.project_name,
        run_name=f"finetune_{args.mode}_{args.stage}" if args.mode == 'two_stage' else f"finetune_{args.mode}"
    )

    # Start finetuning with checkpoint
    training_metrics = trainer.train(
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        total_steps=train_params['total_steps'],
        save_every=train_params['save_every'],
        eval_every=train_params['eval_every'],
        save_dir=save_dir,
        checkpoint_path=args.checkpoint_path  # Checkpoint handles model loading
    )

    print("Finetuning completed!")
    print(f"Best test loss: {training_metrics['best_test_loss']:.4f}")
    print(f"Final step: {training_metrics['final_step']}")
    print(f"Models and metrics saved in: {save_dir}")


if __name__ == "__main__":
    main()
