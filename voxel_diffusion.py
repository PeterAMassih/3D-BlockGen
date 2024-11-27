import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CLIPTokenizer, CLIPTextModel
from pathlib import Path
from tqdm import tqdm
import json
import time
from diffusers import UNet3DConditionModel
from diffusion import DiffusionModel3D, DiffusionConfig
import random
import numpy as np


## ADD SEED: IMPORTANT FOR REPRO 
## USE EMA MAYBE

class SharpBCEWithLogitsLoss(nn.Module):
    def __init__(self, sharpness_weight=0.6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.sharpness_weight = sharpness_weight
        
    def forward(self, pred, target):
        # Standard BCE loss with the logits
        bce_loss = self.bce(pred, target)
        
        probs = torch.sigmoid(pred)
        # Now we add loss when close to 0.5 instead of subtracting when far from 0.5
        sharpness_loss = torch.mean(1.0 - torch.abs(probs - 0.5) * 2)
        
        return bce_loss + self.sharpness_weight * sharpness_loss

class VoxelConfig:
    """Configuration for voxel processing"""
    def __init__(self, 
                 use_rgb=False,
                 default_color=[0.5, 0.5, 0.5],  # Default gray
                 alpha_weight=1.0,
                 rgb_weight=1.0):
        """
        Args:
            use_rgb: Whether to use RGB+alpha channels
            default_color: Default RGB values for occupancy-only data
            alpha_weight: Weight for occupancy/alpha loss
            rgb_weight: Weight for RGB loss when using colors
        """
        self.use_rgb = use_rgb
        self.default_color = torch.tensor(default_color)
        self.in_channels = 4 if use_rgb else 1
        self.alpha_weight = alpha_weight
        self.rgb_weight = rgb_weight
    
    def get_loss_fn(self, device):
        """Returns appropriate loss function"""
        if self.use_rgb:
            return RGBALoss(
                alpha_weight=self.alpha_weight,
                rgb_weight=self.rgb_weight
            ).to(device)
        return nn.MSELoss().to(device)

class RGBALoss(nn.Module):
    """Combined loss for RGBA voxels"""
    def __init__(self, alpha_weight=1.0, rgb_weight=1.0):
        super().__init__()
        self.alpha_weight = alpha_weight
        self.rgb_weight = rgb_weight
        self.bce_loss = nn.BCEWithLogitsLoss()  # Better for binary classification, because we need class probabilities
    
    def forward(self, model_output, noisy_sample, timesteps, target, diffusion_model):
        """
        Args:
            model_output: Predicted noise
            noise: Target noise
            noisy_sample: Current noisy input
            timesteps: Current timesteps
            target: Original clean sample [B, 4, H, W, D] (R, G, B, alpha)
            diffusion_model: DiffusionModel3D instance for scheduler access
        """
        # Predict original sample
        pred_original = diffusion_model.predict_original_sample(
            noisy_sample, model_output, timesteps
        )
        
        # Split channels - RGBA format
        pred_rgb = pred_original[:, :3]  # First 3 channels are RGB
        pred_alpha = pred_original[:, 3]  # Last channel is alpha
        
        true_rgb = target[:, :3]  # RGB channels
        true_alpha = target[:, 3]  # Alpha channel
        
        # Binary Cross Entropy for alpha/occupancy
        alpha_loss = self.bce_loss(pred_alpha, true_alpha)
        
        # MSE for RGB weighted by true occupancy
        # IF add RGBAO occupancy change to pred_rgba
        true_alpha_expanded = true_alpha.unsqueeze(1)  # [B, 1, H, W, D]
        rgb_loss = F.mse_loss(
            true_alpha_expanded * pred_rgb,
            true_alpha_expanded * true_rgb
        )
        
        return self.alpha_weight * alpha_loss + self.rgb_weight * rgb_loss

class VoxelTextDataset(Dataset):
    """Dataset for text-conditioned voxel generation"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32") # Now static to all instances

    def __init__(self, voxel_dir, annotation_file, config: VoxelConfig):
        self.voxel_dir = Path(voxel_dir)
        self.config = config
        
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Get all files recursively
        self.files = []
        for pt_file in tqdm(self.voxel_dir.rglob("*.pt"), desc="Finding voxel files"):
            if "_aug" not in pt_file.name:
                self.files.append(pt_file)
                for i in range(1, 4):
                    aug_file = pt_file.with_name(f"{pt_file.stem}_aug{i}.pt")
                    if aug_file.exists():
                        self.files.append(aug_file)
        
        print(f"\nFound {len(self.files)} files (including augmentations)")
        
    def _process_voxel_data(self, voxel_data):
        """Process voxel data according to configuration"""
        if self.config.use_rgb:
            if voxel_data.shape[0] == 4:
                # Already in RGBA format
                alpha = (voxel_data[3:4] > 0).float()
                rgb = voxel_data[:3]
                return torch.cat([alpha, rgb], dim=0)
            else:
                # Convert occupancy to RGBA
                alpha = (voxel_data > 0).float()  # [1, H, W, D]
                rgb = self.config.default_color.view(3, 1, 1, 1).to(voxel_data.device)
                rgb = rgb.repeat(1, *voxel_data.shape[1:])  # [3, H, W, D]
                return torch.cat([alpha, rgb], dim=0)  # [4, H, W, D]
        else:
            if voxel_data.shape[0] > 1:
                voxel_data = voxel_data[-1:] # take the last one which is occupancy TODO IF RGBAO change

            return (voxel_data > 0).float()
    
    def _create_simple_prompt(self, model_id):
        if model_id in self.annotations:
            name = self.annotations[model_id].get('name', 'an object')
            return f"{name}" # remove a 3d model of
        return ""
    
    def _create_detailed_prompt(self, model_id):
        if model_id in self.annotations:
            name = self.annotations[model_id].get('name', 'an object')
            categories = [cat['name'] for cat in self.annotations[model_id].get('categories', [])]
            tags = [tag['name'] for tag in self.annotations[model_id].get('tags', [])]
            
            prompt_parts = [f"{name}"] # REMOVE a 3d model of
            if categories:
                prompt_parts.append(f"in category {', '.join(categories)}")
            if tags:
                prompt_parts.append(f"with traits: {', '.join(tags)}")
            
            return ' '.join(prompt_parts)
        return ""
    
    def _get_random_prompt(self, model_id):
        rand = torch.rand(1).item()
        if rand < 0.10:
            #print("")
            return ""
        elif rand < 0.55:
            #print(self._create_detailed_prompt(model_id))
            return self._create_detailed_prompt(model_id)
        else:
            #print(self._create_simple_prompt(model_id))
            return self._create_simple_prompt(model_id)
    
    def __len__(self):
        return len(self.files)
    
    @staticmethod
    def collate_fn(batch):
        voxels = torch.stack([item['voxels'] for item in batch])
        prompts = [item['text'] for item in batch]
        
        
        tokens = VoxelTextDataset.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
            # clean_up_tokenization_spaces=True  # Not available in current version Sometimes fires a warning
        )
        
        return {
            'voxels': voxels,
            **tokens
        }
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        model_id = file_path.stem.split('_aug')[0]
        
        voxel_data = torch.load(file_path)
        voxel_data = self._process_voxel_data(voxel_data)
        
        return {
            'voxels': voxel_data,
            'text': self._get_random_prompt(model_id)
        }

class DiffusionTrainer:
    def __init__(self, model, config: VoxelConfig, device='cuda', initial_lr=1e-4):
        self.model = model
        self.model.to(device) # DO NOT FORGET
        self.config = config
        self.device = device
        self.initial_lr = initial_lr
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.text_encoder.eval()
        self.loss_fn = config.get_loss_fn(device)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                voxels = batch['voxels'].to(self.device)
                text_inputs = {
                    k: v.to(self.device) 
                    for k, v in batch.items() 
                    if k not in ['voxels']
                }
                
                encoder_outputs = self.text_encoder(**text_inputs)
                encoder_hidden_states = encoder_outputs.last_hidden_state
                
                noise = torch.randn_like(voxels)
                timesteps = torch.randint(
                    0, self.model.noise_scheduler.config.num_train_timesteps,
                    (voxels.shape[0],), device=self.device
                ).long()
                
                noisy_voxels = self.model.add_noise(voxels, noise, timesteps)
                predicted_noise = self.model(
                    noisy_voxels,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=True
                ).sample
                
                pred_original = self.model.predict_original_sample(
                    noisy_voxels, predicted_noise, timesteps
                )
                
                if self.config.use_rgb:
                    loss = self.loss_fn(
                        model_output=predicted_noise,
                        noisy_sample=noisy_voxels,
                        timesteps=timesteps,
                        target=voxels,
                        diffusion_model=self.model
                    )
                else:
                    loss = self.loss_fn(pred_original, voxels) # Because shape is [B, 1, H, W, D] maybe need to squeeze(1) to check
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train_step(self, batch, optimizer, current_step):
        """Performs a single training step"""
        voxels = batch['voxels'].to(self.device)
        text_inputs = {
            k: v.to(self.device) 
            for k, v in batch.items() 
            if k not in ['voxels']
        }
        
        with torch.no_grad():
            encoder_outputs = self.text_encoder(**text_inputs)
            encoder_hidden_states = encoder_outputs.last_hidden_state
        
        noise = torch.randn_like(voxels)
        timesteps = torch.randint(
            0, self.model.noise_scheduler.config.num_train_timesteps,
            (voxels.shape[0],), device=self.device
        ).long()
        
        noisy_voxels = self.model.add_noise(voxels, noise, timesteps)
        predicted_noise = self.model(
            noisy_voxels,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        ).sample
        
        pred_original = self.model.predict_original_sample(
            noisy_voxels, predicted_noise, timesteps
        )
        
        if self.config.use_rgb:
            loss = self.loss_fn(
                model_output=predicted_noise, 
                noisy_sample=noisy_voxels,
                timesteps=timesteps,
                target=voxels,
                diffusion_model=self.model
            )
        else:
            loss = self.loss_fn(pred_original, voxels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Update EMA model if enabled
        self.model.update_ema(current_step)
        
        return loss.item()
        
    def train(self, train_dataloader, test_dataloader, 
        total_steps=100_000, 
        save_every=20_000,
        eval_every=20_000,
        save_dir='training_runs',
        checkpoint_path=None):
        """
        Train the diffusion model with organized checkpointing.
        Continues training seamlessly from checkpoint if provided.
        """
        # Create directory structure
        save_dir = Path(save_dir)
        checkpoints_dir = save_dir / "checkpoints"
        models_dir = save_dir / "models"
        best_model_dir = save_dir / "best_model"
        
        for dir_path in [checkpoints_dir, models_dir, best_model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.initial_lr, 
            weight_decay=0.01
        )
        
        # Initialize training state
        best_test_loss = float('inf')
        losses = []
        test_losses = []
        lr_history = []
        current_step = 0
        starting_step = 0
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # Load model states
            self.model.load_pretrained(checkpoint_path)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            current_step = checkpoint['step']
            starting_step = current_step
            best_test_loss = checkpoint.get('best_test_loss', float('inf'))
            losses = checkpoint.get('training_losses', [])
            test_losses = checkpoint.get('test_losses', [])
            lr_history = checkpoint.get('lr_history', [])
            
            print(f"Resumed at step {current_step} with best test loss: {best_test_loss:.4f}")
        
        # Create/update scheduler
        remaining_steps = total_steps - current_step
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=1e-6)
        if checkpoint_path is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Training configuration
        training_config = {
            'total_steps': total_steps,
            'save_every': save_every,
            'eval_every': eval_every,
            'initial_lr': self.initial_lr,
            'device': str(self.device),
            'start_time': time.strftime("%Y%m%d-%H%M%S"),
            'resume_checkpoint': checkpoint_path,
            'starting_step': starting_step
        }
        
        with open(save_dir / "training_config.json", 'w') as f:
            json.dump(training_config, f, indent=2)
        
        train_iter = iter(train_dataloader)
        
        # Main training loop
        steps_to_run = total_steps - current_step
        with tqdm(total=steps_to_run, initial=current_step, desc="Training") as pbar:
            while current_step < total_steps:
                self.model.train()
                
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)

                # Do a training step
                loss = self.train_step(batch, optimizer, current_step)
                
                # Update learning rate
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                
                # Update tracking
                current_step += 1
                losses.append(loss)
                lr_history.append(current_lr)
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
                
                # Evaluation
                if current_step % eval_every == 0:
                    test_loss = self.evaluate(test_dataloader)
                    test_losses.append(test_loss)
                    
                    # Save metrics
                    metrics = {
                        'training_losses': losses,
                        'test_losses': test_losses,
                        'lr_history': lr_history,
                        'test_steps': list(range(0, len(test_losses) * eval_every, eval_every)),
                        'current_step': current_step,
                        'best_test_loss': best_test_loss
                    }
                    
                    with open(save_dir / "metrics.json", 'w') as f:
                        json.dump(metrics, f)
                    
                    # Save best model
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        self.model.save_pretrained(str(best_model_dir / f"best_model"))
                        print(f"\nNew best model saved with Test Loss: {test_loss:.4f}")
                
                # Regular checkpoints
                if current_step % save_every == 0:
                    # Save model and checkpoint
                    self.model.save_pretrained(str(models_dir / f"model_step_{current_step}"))
                    
                    checkpoint_data = {
                        'step': current_step,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'test_loss': test_losses[-1] if test_losses else None,
                        'best_test_loss': best_test_loss,
                        'training_losses': losses,
                        'test_losses': test_losses,
                        'lr_history': lr_history
                    }
                    
                    torch.save(checkpoint_data, checkpoints_dir / f"checkpoint_step_{current_step}.pth")
                    
                    # Keep only last 5 checkpoints
                    checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_step_*.pth"))
                    if len(checkpoint_files) > 5:
                        for old_ckpt in checkpoint_files[:-5]:
                            old_ckpt.unlink()
        
        # Final saves
        self.model.save_pretrained(str(models_dir / "final_model"))
        metrics = {
            'training_losses': losses,
            'test_losses': test_losses,
            'lr_history': lr_history,
            'test_steps': list(range(0, len(test_losses) * eval_every, eval_every)),
            'final_step': current_step,
            'best_test_loss': best_test_loss
        }
        
        with open(save_dir / "final_metrics.json", 'w') as f:
            json.dump(metrics, f)
        
        return metrics

def create_dataloaders(voxel_dir, annotation_file, config: DiffusionConfig, config_voxel: VoxelConfig, 
                      batch_size=32, test_split=0.05):
    """Create train and test dataloaders"""
    dataset = VoxelTextDataset(voxel_dir, annotation_file, config_voxel)
    
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    
    # Use config's generator if seed was set
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=config.generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=VoxelTextDataset.collate_fn,
        generator=config.generator  # Use same generator for shuffling
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=VoxelTextDataset.collate_fn
    )
    
    return train_loader, test_loader

def create_model_and_trainer(voxel_config: VoxelConfig, diffusion_config: DiffusionConfig, 
                           resolution=32, device='cuda'):
    """Creates model and trainer with specified configuration
    
    Args:
        voxel_config: VoxelConfig object specifying model configuration
        diffusion_config: DiffusionConfig object specifying diffusion parameters
        resolution: Size of voxel grid (default: 32)
        device: Device to put model on (default: 'cuda')
    """
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
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Use diffusion_config here
    diffusion_model = DiffusionModel3D(
        model, 
        config=diffusion_config
    )
    
    trainer = DiffusionTrainer(diffusion_model, voxel_config, device=device)
    
    return trainer, diffusion_model