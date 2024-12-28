#!/bin/bash

echo "Starting finetuning process..."

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
pip install -e .
pip install iopath
pip install --no-deps --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html

echo "Running finetune.py..."
python scripts/finetune.py \
    --wandb_key 67c58b45890c274b89d401e46da195e6071b6872 \
    --project_name "3D-Blockgen" \
    --data_dir objaverse_finetune_voxelized \
    --label_mapping objaverse_finetune/file_to_label_map.json \
    --save_dir runs/finetune \
    --checkpoint_path /scratch/students/2024-fall-sp-pabdel/3D-BlockGen/runs/experiment_two_stage/shape/checkpoints/checkpoint_step_120000.pth \
    --mode two_stage \
    --stage shape

echo "Finetuning process completed successfully."