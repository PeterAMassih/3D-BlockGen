import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from UNet3D import UNet3DWithAttention
from diffusion import CustomUNetDiffusionPipeline
from data_loader import create_dataloader
from torch.utils.data import random_split

def train_model(data_path, num_epochs=50, batch_size=8, learning_rate=1e-4, model_save_path='best_model.pth', test_split_ratio=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = create_dataloader(data_path, batch_size=batch_size, return_dataset=True)  # Modified to return dataset
    train_size = int((1 - test_split_ratio) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for both train and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model and optimizer
    unet3d = UNet3DWithAttention().to(device)
    pipeline = CustomUNetDiffusionPipeline(unet3d).to(device)
    # TODO scheduler on learning rate 
    optimizer = optim.Adam(pipeline.parameters(), lr=learning_rate)

    scheduler = pipeline.scheduler

    best_loss = float('inf')

    # Training Loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pipeline.train()
        
        # Training phase
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            # Prepare data
            batch = batch.to(device)
            timesteps = torch.randint(0, 1000, (batch.size(0),), device=device).long()
            noise = torch.randn_like(batch)
            noisy_batch = scheduler.add_noise(batch, noise, timesteps)

            # Forward pass
            predicted_noise = pipeline(noisy_batch, timesteps)
            loss = nn.MSELoss()(predicted_noise, noise)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Testing phase
        test_loss = 0.0
        pipeline.eval()
        with torch.no_grad():
            for i, test_batch in enumerate(test_loader):
                test_batch = test_batch.to(device)
                timesteps = torch.randint(0, 1000, (test_batch.size(0),), device=device).long()
                noise = torch.randn_like(test_batch)
                noisy_test_batch = scheduler.add_noise(test_batch, noise, timesteps)

                predicted_test_noise = pipeline(noisy_test_batch, timesteps)
                test_loss += nn.MSELoss()(predicted_test_noise, noise).item()

        avg_test_loss = test_loss / len(test_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}')

        # Save model if test loss is the best so far
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(pipeline.state_dict(), model_save_path)

    print(f"Training complete. Best test loss: {best_loss:.4f}")
    return best_loss
