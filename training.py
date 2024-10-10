import torch
import torch.nn.functional as F
from tqdm import tqdm

use_cuda = torch.cuda.is_available()

if use_cuda:
    from torch.cuda.amp import autocast, GradScaler
else:
    class autocast:
        def __init__(self, enabled=True):
            self.enabled = enabled
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
    
    class GradScaler:
        def scale(self, loss):
            return loss
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def train_diffusion_model(diffusion_model, train_dataloader, test_dataloader, epochs=30, device='cpu', model_save_path='best_model.pth'):
    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=4e-4)
    scaler = GradScaler()
    best_test_loss = float('inf')
    losses = []
    test_losses = []
    
    for epoch in range(epochs):
        diffusion_model.train()
        running_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            clean_images = batch.to(device)
            
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, diffusion_model.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

            with autocast(enabled=use_cuda):
                noisy_images = diffusion_model.add_noise(clean_images, noise, timesteps)
                noise_pred = diffusion_model.get_noise_prediction(noisy_images, timesteps)
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

        avg_loss = running_loss / len(train_dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss}")

        test_loss = evaluate_diffusion_model(diffusion_model, test_dataloader, device)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(diffusion_model.state_dict(), model_save_path)
            print(f"Saved best model at epoch {epoch+1} with Test Loss: {test_loss:.4f}")

    return losses, test_losses

def evaluate_diffusion_model(diffusion_model, test_dataloader, device='cpu'):
    diffusion_model.eval()
    test_loss = 0.0
    
    with torch.no_grad(), autocast(enabled=use_cuda):
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            clean_images = batch.to(device)
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, diffusion_model.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

            noisy_images = diffusion_model.add_noise(clean_images, noise, timesteps)
            noise_pred = diffusion_model.get_noise_prediction(noisy_images, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss