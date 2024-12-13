import torch
import wandb
from transformers import CLIPTextModel
from tqdm import tqdm
import json
import time
from pathlib import Path
from ..configs.voxel_config import VoxelConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
from ..models.diffusion import DiffusionModel3D
from ..models.losses import ColorStageLoss, RGBALoss

class DiffusionTrainer:
    def __init__(self, 
                 model: DiffusionModel3D, 
                 config: VoxelConfig, 
                 device: str = 'cuda', 
                 initial_lr: float = 1e-4, 
                 wandb_key: str = None, 
                 project_name: str = "3D-Blockgen",
                 run_name: str = None):
        self.model = model
        self.model.to(device)  # Explicit device placement
        self.config = config
        self.device = device
        self.initial_lr = initial_lr
        self.run_name = run_name
        
        # Initialize text encoder
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.text_encoder.eval()
        
        # Get appropriate loss function from config
        self.loss_fn = config.get_loss_fn(device)
        
        # Initialize wandb
        self.wandb_key = wandb_key
        self.project_name = project_name
        self.wandb_run = None
        if self.wandb_key:
            wandb.login(key=self.wandb_key)

    def compute_loss(self, model_output, noisy_sample, timesteps, target):
        """
        Computes loss based on the current mode and stage.
        Args:
            model_output: Predicted noise
            noisy_sample: Current noisy input
            timesteps: Current timesteps
            target: Original clean sample
        Returns:
            Computed loss
        """
        assert model_output.shape[1] == 3, "Color stage should predict RGB noise only"
        assert noisy_sample.shape[1] == 4, "Color stage input should be RGBA"
        assert target.shape[1] == 4, "Color stage target should be RGBA"

        if isinstance(self.loss_fn, (ColorStageLoss, RGBALoss)):
            # For custom loss functions that require the diffusion model
            return self.loss_fn(
                model_output=model_output,
                noisy_sample=noisy_sample,
                timesteps=timesteps,
                target=target,
                diffusion_model=self.model
            )
        else:
            # For simple loss functions like MSELoss
            return self.loss_fn(model_output, target)
    
    def _get_model_save_path(self, base_dir: str, step: int = None) -> str:
        """Constructs save path for model checkpoint directories."""
        suffix = f"_{self.model.stage}" if self.model.mode == 'two_stage' else ""
        step_suffix = f"_step_{step}" if step is not None else ""
        return f"{base_dir}{suffix}{step_suffix}"
    
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
                
                loss = self.compute_loss(
                    model_output=predicted_noise,
                    noisy_sample=noisy_voxels,
                    timesteps=timesteps,
                    target=voxels
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Test Loss: {avg_loss:.4f}")
        
        if self.wandb_run:
            self.wandb_run.log({
                'test_loss': avg_loss,
                'stage': self.config.get_stage()  # Use helper method
            })
            
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
        
        loss = self.compute_loss(
                model_output=predicted_noise,
                noisy_sample=noisy_voxels,
                timesteps=timesteps,
                target=voxels
            )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        self.model.update_ema(current_step)
        
        if self.wandb_run:
            self.wandb_run.log({
                'train_loss': loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'step': current_step,
                'stage': self.config.get_stage()  # Use helper method
            })
        
        return loss.item()
        
    def train(self, train_dataloader, test_dataloader, 
              total_steps: int = 100_000, 
              save_every: int = 20_000,
              eval_every: int = 20_000,
              save_dir: str = 'training_runs',
              checkpoint_path: str = None):
        """Train the diffusion model with organized checkpointing."""
        
        # Create directory structure
        save_dir = Path(save_dir)
        if self.config.mode == 'two_stage':
            save_dir = save_dir / self.config.stage
            
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

        train_iter = iter(train_dataloader)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            checkpoint = torch.load(checkpoint_path)
            checkpoint_dir = Path(checkpoint_path).parent.parent
            last_save_step = (checkpoint['step'] // save_every) * save_every
            model_path = checkpoint_dir / 'models' / f"model_step_{last_save_step}"
            
            self.model.load_pretrained(str(model_path))
            self.model.to(self.device)
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
            'starting_step': starting_step,
            'mode': self.config.mode,
            'stage': self.config.get_stage(),  # Use helper method
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'batch_size': train_dataloader.batch_size
        }
        
        # Save config and initialize wandb
        with open(save_dir / "training_config.json", 'w') as f:
            json.dump(training_config, f, indent=2)
            
        if self.wandb_key and not self.wandb_run:
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.run_name, 
                config=training_config,
                resume=True if checkpoint_path else False
            )
        
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

                loss = self.train_step(batch, optimizer, current_step)
                
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                
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
                    
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        self.model.save_pretrained(str(best_model_dir / "best_model"))
                        if self.wandb_run:
                            self.wandb_run.summary['best_test_loss'] = best_test_loss
                            self.wandb_run.summary['best_model_step'] = current_step
                        print(f"\nNew best model saved with Test Loss: {test_loss:.4f}")
                
                # Regular checkpoints
                if current_step % save_every == 0:
                    # Determine the base save path for the step
                    save_path = self._get_model_save_path(str(models_dir), step=current_step)
                    
                    # Save the model and optionally the EMA model
                    self.model.save_pretrained(save_path)
                    
                    # Save checkpoint data
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
                    
                    checkpoint_path = checkpoints_dir / f"checkpoint_step_{current_step}.pth"
                    torch.save(checkpoint_data, checkpoint_path)

                    # Log artifacts to wandb
                    if self.wandb_run:
                        artifact = wandb.Artifact(f'checkpoint-{current_step}', type='model')
                        artifact.add_file(str(checkpoint_path))
                        artifact.add_file(save_path + "_main")
                        if self.model.ema_model is not None:
                            artifact.add_file(save_path + "_ema")
                        self.wandb_run.log_artifact(artifact)
                    
                    # Keep only last 5 checkpoints
                    checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_step_*.pth"))
                    if len(checkpoint_files) > 5:
                        for old_ckpt in checkpoint_files[:-5]:
                            old_ckpt.unlink()
        
        # Final saves
        final_save_path = str(models_dir / "final_model")  # Ensure "final_model" is appended to the directory
        self.model.save_pretrained(final_save_path)

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
            
        if self.wandb_run:
            self.wandb_run.summary.update({
                'final_test_loss': test_losses[-1] if test_losses else None,
                'final_train_loss': losses[-1] if losses else None,
                'best_test_loss_overall': best_test_loss,
                'total_training_time': time.time() - time.mktime(time.strptime(training_config['start_time'], "%Y%m%d-%H%M%S"))
            })
            
            final_artifact = wandb.Artifact('final_model', type='model')
            final_artifact.add_file(str(final_save_path) + "_main")
            if self.model.ema_model is not None:
                final_artifact.add_file(str(final_save_path) + "_ema")
            self.wandb_run.log_artifact(final_artifact)
            self.wandb_run.finish()
        
        return metrics