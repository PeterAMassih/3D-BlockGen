#!/bin/bash

echo "Starting training process..."

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
pip install -e .
pip install iopath
pip install --no-deps --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html


echo "Running train.py..."
python scripts/train.py --wandb_key 67c58b45890c274b89d401e46da195e6071b6872 --project_name "3D-Blockgen" --data_dir objaverse_data_voxelized --annotation_file objaverse_data/annotations.json --save_dir runs/experiment_color_simple_mse

echo "Training process completed successfully."
