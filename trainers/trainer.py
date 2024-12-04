import torch
from transformers import CLIPTextModel
from tqdm import tqdm
import json
import time
from pathlib import Path
from ..configs.voxel_config import VoxelConfig
from torch.optim.lr_scheduler import CosineAnnealingLR

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
                
                # Evaluation TODO bug here with the best_loss
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