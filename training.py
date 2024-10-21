import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_diffusion_model(diffusion_model, train_dataloader, test_dataloader, epochs=100, device='cpu', model_save_path='best_model.pth'):
    initial_lr = 1e-4
    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=initial_lr, weight_decay=0.01)
    
    # TODO Cosine Annealing scheduler maybe to change if not working well
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
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

            # Add noise and get model prediction
            noisy_images = diffusion_model.add_noise(clean_images, noise, timesteps)
            noise_pred = diffusion_model.get_noise_prediction(noisy_images, timesteps)

            # Calculate loss and perform backprop
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(train_dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss}, LR: {scheduler.get_last_lr()[0]:.6f}")

        test_loss = evaluate_diffusion_model(diffusion_model, test_dataloader, device)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(diffusion_model.state_dict(), model_save_path)
            print(f"Saved best model at epoch {epoch+1} with Test Loss: {test_loss:.4f}")

        torch.save(diffusion_model.state_dict(), "latest_model.pth")
        print(f"Saved latest model at epoch {epoch+1}")

    return losses, test_losses

def evaluate_diffusion_model(diffusion_model, test_dataloader, device='cpu'):
    diffusion_model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            clean_images = batch.to(device)
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, diffusion_model.noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

            noisy_images = diffusion_model.add_noise(clean_images, noise, timesteps)
            noise_pred = diffusion_model.get_noise_prediction(noisy_images, timesteps)

            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss
