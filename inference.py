import torch
from diffusion import CustomUNetDiffusionPipeline
from UNet3D import UNet3DWithAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the best model
unet3d = UNet3DWithAttention().to(device)
pipeline = CustomUNetDiffusionPipeline(unet=unet3d).to(device)
pipeline.load_state_dict(torch.load('best_model.pth'))

# Sampling loop
num_samples = 4
shape = (num_samples, 1, 32, 32, 32)  # Example: 3D shape

# Start with random noise
samples = torch.randn(shape).to(device)

for t in reversed(range(1000)):
    with torch.no_grad():
        predicted_noise = pipeline(samples, torch.tensor([t]).to(device))
    # Update samples with noise removal based on predicted noise
    samples = pipeline.scheduler.step(predicted_noise, t, samples).prev_sample
    print("Sampling step:", t)

# Save the generated samples
torch.save(samples, 'generated_samples.pt')

print("Generated samples:", samples.shape)
