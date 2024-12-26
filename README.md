# 3D-BlockGen

This project explores the use of diffusion models to automatically generate 3D building block designs, such as LEGO®️, conditioned on textual descriptions. Traditional tools for creating 3D brick models are manual and time-consuming, and this project aims to automate the process by leveraging deep learning generative models.

## Features

- **Two-Stage Diffusion Model**: Separate models for shape and color generation
- **Text-Conditioned Generation**: Generate 3D models from textual descriptions using CLIP embeddings
- **Voxel-Based Representation**: 32³ voxel grid with RGBA channels
- **LEGO Conversion**: Automatic conversion of voxel models to LEGO brick designs
- **Data Augmentation**: Rotational augmentations for improved training
- **Experiment Tracking**: Integration with Weights & Biases for experiment monitoring

## Architecture

The project implements a two-stage diffusion model (and other architecture but this is the one with the better results):

1. **Shape Stage**: 
   - Input: Text prompt
   - Output: Binary occupancy grid (1 channel)
   - Purpose: Generate the basic 3D shape

2. **Color Stage**:
   - Input: Text prompt + shape from first stage
   - Output: RGB colors for occupied voxels
   - Purpose: Add colors to the generated shape

### Core Components

- **Diffusion Model**: Modified UNet3D architecture with cross-attention for text conditioning
- **Text Encoder**: CLIP text encoder for converting prompts to embeddings
- **Voxel Format**: RGBA format [4, 32, 32, 32] where:
  - First 3 channels: RGB colors
  - Last channel: Binary occupancy mask

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/3D-BlockGen.git
cd 3D-BlockGen

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install PyTorch3D (for visualization)
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html
```

## Data Pipeline

1. **Data Download**:
```bash
python scripts/download_dataset.py
```

2. **Data Processing**:
```bash
python scripts/process_dataset.py
```

This creates:
- Voxelized 3D models (32³ resolution)
- 4 versions per model (original + 3 rotated augmentations)
- RGBA format with binary occupancy mask

## Training

The training process consists of two stages:

1. **Shape Stage**:
```bash
python scripts/train.py \
    --mode two_stage \
    --stage shape \
    --wandb_key YOUR_WANDB_KEY
```

2. **Color Stage**:
```bash
python scripts/train.py \
    --mode two_stage \
    --stage color \
    --wandb_key YOUR_WANDB_KEY
```

## Inference

Generate 3D models from text:
```bash
python scripts/generate.py \
    --prompt "A castle tower" \
    --num_samples 4
```

Convert to LEGO:
```bash
python scripts/legolize.py \
    --input generated.pt \
    --output lego_model.png
```

## Project Structure

```
3D-BlockGen/
├── blockgen/
│   ├── configs/
│   │   ├── diffusion_config.py    # Training configuration
│   │   └── voxel_config.py        # Voxel processing settings
│   ├── data/
│   │   ├── dataset.py             # Dataset implementation
│   │   └── processing/            # Data processing utilities
│   ├── models/
│   │   ├── diffusion.py           # Diffusion model implementation
│   │   └── losses.py              # Loss functions
│   ├── trainers/
│   │   └── trainer.py             # Training loop implementation
│   ├── inference/
│   │   └── inference.py           # Generation and visualization
│   └── utils/                     # Utility functions
├── scripts/                       # Training and processing scripts
└── tests/                         # Unit tests
```
