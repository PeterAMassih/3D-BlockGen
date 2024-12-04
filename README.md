# 3D-BlockGen
This project explores the use of diffusion models to automatically generate 3D building block designs, such as LEGO®️, conditioned on textual descriptions or images. Traditional tools for creating 3D bricks models are manual and time-consuming, and this project aims to automate the process by leveraging deep learning generative models.

3D-BlockGen/
├── configs/
│   ├── __init__.py
│   ├── diffusion_config.py      # DiffusionConfig
│   └── voxel_config.py          # VoxelConfig
│
├── models/
│   ├── __init__.py
│   ├── losses.py                # SharpBCEWithLogitsLoss, RGBALoss
│   └── diffusion.py             # DiffusionModel3D
│
├── data/
│   ├── __init__.py
│   ├── dataset.py               # VoxelTextDataset
│   └── processing/
│       ├── __init__.py
│       ├── data_retrieval.py    # download data code
│       └── data_voxelization.py # voxelization code
│
├── trainers/
│   ├── __init__.py
│   └── trainer.py               # DiffusionTrainer
│
└── inference/
    ├── __init__.py
    └── inference.py             # Inference code

scripts/
├── download_dataset.py          # download script calling data/data_retrieval.py
├── process_dataset.py           # voxelization calling data/voxelization.py
├── train.py                     # Main training script
└── generate.py                  # Main inference script
