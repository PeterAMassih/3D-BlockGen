# 3D-BlockGen: Text-to-3D Generation Using Diffusion Models

3D-BlockGen is a deep learning framework that generates LEGO®-compatible 3D models from text descriptions. The project addresses the time-consuming nature of manual brick design by automating the generation process using a two-stage diffusion model pipeline.

## Report

report of the project with all the findings are in report.pdf in the current directory

## Features

- Text-to-3D generation using CLIP embeddings
- Two-stage diffusion pipeline (shape then color)
- RGBA voxel representation (32³ resolution)
- Automated LEGO® brick conversion
- Rotational data augmentation
- Comprehensive experiment tracking using Weights & Biases

## Installation

### Option 1: Docker (Recommended)

```bash
docker pull peteram/blockgen:cuda11.7v3
docker run --gpus all -it peteram/blockgen:cuda11.7v3
```

OR use this image from creating a running pod

### Option 2: Manual Installation

```bash
git clone https://github.com/PeterAMassih/3D-BlockGen
cd 3D-BlockGen

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install PyTorch3D
pip install iopath
pip install --no-deps --no-index --no-cache-dir pytorch3d -f \
    https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html
```

## Pretrained Models

Download our pretrained models from [Google Drive](https://drive.google.com/drive/folders/1BN4F55P6b_sMKyuQeRK4L0LTI1HoIfS_?usp=sharing) and place them in their respective directories (this contains the best models for shape/color):

- Base Shape Model → `runs/experiment_two_stage/shape/best_model/`
- Base Color Model → `runs/experiment_two_stage/color/best_model/`
- Finetuned Shape Model → `runs/finetune/shape/best_model/`

```bash
# Create directories for models
mkdir -p runs/experiment_two_stage/shape/best_model/
mkdir -p runs/experiment_two_stage/color/best_model/
mkdir -p runs/finetune/shape/best_model/
```

## Usage

### Data

Due to the big size of the data, the finetuning and base dataset are not available as is but through huggingface (which has a different format due to not having the choice for a same column with different datatypes in the dataset). Example of scripts are presents on huggingface to compensate and know how to load the data and use it in our code (requires minor tweaking). The evaluation dataset on the other hand is made available from [Google Drive](https://drive.google.com/drive/folders/1K7vYEpENUHsa1vttXRD8q2S5Sr-5TYFy?usp=sharing) due to its relatively small size (8 3D objects). Another way would be to process your own data but that would require 2.1T of disk storage for the main data not voxelized (vs 4.34 GB on huggingface after voxelization).

#### Pre-processed Datasets
Available on Hugging Face:
- [3D-BlockGen Base Dataset](https://huggingface.co/datasets/peteram/3d-blockgen) - 542,292 processed models from Objaverse
- [3D-BlockGen Finetuning Dataset](https://huggingface.co/datasets/peteram/3d-blockgen-finetune) - 11,464 processed from Objaverse non augmented

Each dataset contains:
- Voxelized models (32³ resolution)
- Shape tensors [1, 32, 32, 32] for occupancy 
- RGB tensors [3, 32, 32, 32] for colors
- 4 versions per model (original + 3 rotated augmentations) - Only Base dataset

#### Process Your Own Data

1. Base Dataset Creation:
```bash
# Download raw models from Objaverse
python scripts/download_dataset.py

# Voxelize and process models
python scripts/process_dataset.py \
    --input_dir objaverse_data \ # annotations.json is created here for prompts
    --output_dir objaverse_data_voxelized \
    --resolution 32
```

2. Finetuning Dataset Creation:
```bash
# Download specific classes from Objaverse-LVIS
python scripts/download_dataset_finetune.py

# Process finetuning dataset
python scripts/process_dataset.py \
    --input_dir objaverse_finetune \ # file_to_label_map.json will be created here for prompts
    --output_dir objaverse_finetune_voxelized \
    --resolution 32
```

#### Dataset Format
The voxelized data follows this structure:
```
dataset_dir/
├── hf-objaverse-v1/
│   └── glbs/
│       ├── 000-000/
│       │   ├── model_id.pt         # Original model
│       │   ├── model_id_aug1.pt    # 90° X-axis rotation
│       │   ├── model_id_aug2.pt    # 90° Y-axis rotation
│       │   └── model_id_aug3.pt    # 90° Z-axis rotation
│       └── ...
└── annotations.json                # Model metadata and prompts
```

For uploading to Hugging Face:
```bash
python scripts/huggingface_dataset.py \
    --dataset_path path/to/voxelized/data \
    --annotation_path path/to/annotations.json \
    --repo_id your-username/dataset-name \
    --token YOUR_HF_TOKEN
```

### Training

1. Shape Stage:
```bash
bash scripts/train.sh
```

2. Color Stage:
```bash
python scripts/train.py \
    --mode two_stage \
    --stage color \
    --wandb_key YOUR_WANDB_KEY
```

3. Optional: Finetuning
```bash
bash scripts/finetune.sh
```

### Generation

#### Interactive Notebook (Recommended)
Use `run.ipynb` for an interactive generation experience with visualizations and generation options.

`results.ipynb` contains the code used for the ablations study, some visualization and results for the report

#### Inference Parameters

1. **Guidance Scale Control**:
   - Shape guidance (--guidance_scale): Controls shape fidelity
   - Color guidance (--color_guidance_scale): Controls color fidelity
   - Range: 10.0 to 30.0 (default: 20.0)
   - Higher values: More prompt-faithful generations
   - Lower values: More diverse generations

2. **Generation Strategies**:
   - Rotation-augmented sampling (--use_rotations): Averages predictions across rotations
   - Mean initialization (--use_mean_init): Starts from 0.5 instead of random noise
3. **EMA**:
    - Exponential moving average of the weights for weight stability. (--use_ema)
4. **DDIM**:
    --use_ddim with --inference_step

If lost use help of python/scripts/generate.py for all args possible

#### Command Line Examples

1. Basic Generation:
```bash
python scripts/generate.py \
    --mode two_stage \
    --shape_model_path runs/experiment_two_stage/shape/best_model/model \ # This should be the model path up to the directory with the ema and main models, our script will handle adding _ema and _main for the respective ema and main models
    --color_model_path runs/experiment_two_stage/color/best_model/model \
    --prompt "a red apple" \
    --guidance_scale 20.0 \
    --color_guidance_scale 20.0

# Convert to LEGO design
python scripts/legolize.py \
    --input generated.pt \
    --output lego_design.png
```

## Architecture

The framework uses a two-stage diffusion model:

1. **Shape Stage**
   - Input: Text prompt → CLIP embedding
   - Output: Binary occupancy grid [1, 32, 32, 32]
   - Purpose: Basic 3D shape generation

2. **Color Stage**
   - Input: Text prompt + shape mask
   - Output: RGB colors for occupied voxels
   - Final output: RGBA tensor [4, 32, 32, 32]

Core components:
- Modified UNet3D with cross-attention
- CLIP text encoder for conditioning
- Custom voxelization pipeline
- LEGO brick conversion algorithm

## Project Structure

```
3D-BlockGen/
├── blockgen/
│   ├── configs/           # Configuration classes
│   ├── data/              # Dataset and processing
│   ├── models/            # Model implementations
│   ├── trainers/          # Training logic
│   ├── inference/         # Generation pipeline
│   └── utils/             # Utility functions
├── scripts/               # Training & processing scripts
├── requirements.txt       # Dependencies
├── run.ipynb              # Main running script
└── report.pdf             # Report on findings and viz
```

## Citation

```bibtex
@misc{massih2025blockgen,
    title={3D-BlockGen: Generating Brick-Compatible 3D Models Using Diffusion Models},
    author={Peter A. Massih},
    year={2025},
    howpublished={EPFL Image and Visual Representation Lab Master Project}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- EPFL Image and Visual Representation Lab
- Supervisors: Martin Nicolas Everaert, Eric Bezzam
- Professor: Sabine Süsstrunk
