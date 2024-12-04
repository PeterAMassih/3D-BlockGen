#!/bin/bash

echo "Starting training process..."

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Running train.py..."
python scripts/train.py

echo "Training process completed successfully."
