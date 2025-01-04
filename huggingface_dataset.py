import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, Features, Array4D, Value, Sequence, DatasetDict
from tqdm import tqdm
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockGenDatasetCreator:
    """Create HuggingFace dataset from BlockGen voxel files."""
    
    def __init__(self, 
                 voxel_dir: Union[str, Path], 
                 annotation_file: Union[str, Path],
                 test_split: float = 0.05,
                 seed: int = 42):
        """Initialize BlockGen dataset creator.
        
        Args:
            voxel_dir: Directory containing .pt voxel files
            annotation_file: Path to annotations JSON file
            test_split: Fraction of data to use for test set (default: 0.05)
            seed: Random seed for reproducible splits (default: 42)
        """
        self.voxel_dir = Path(voxel_dir)
        self.annotation_file = Path(annotation_file)
        self.test_split = test_split
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load annotations
        logger.info("Loading annotations...")
        with open(self.annotation_file, 'r') as f:
            self.annotations = json.load(f)
        logger.info("Done loading annotations")
        
        # Define features
        self.features = Features({
            'voxels_occupancy': Array4D(shape=(1, 32, 32, 32), dtype='float32'),  # Alpha channel
            'voxels_colors': Array4D(shape=(3, 32, 32, 32), dtype='float32'),  # RGB channels
            'model_id': Value('string'),
            'name': Value('string'),
            'categories': Sequence(Value('string')),
            'tags': Sequence(Value('string')),
            'metadata': {
                'is_augmented': Value('bool'),
                'original_file': Value('string'),
                'num_occupied': Value('int32'),
                'split': Value('string')
            }
        })

    def _get_all_files(self) -> List[Path]:
        """Get all .pt files from the voxel directory."""
        files = sorted(self.voxel_dir.rglob("*.pt"))
        logger.info(f"Found {len(files)} voxel files")
        return files

    def _process_voxel_file(self, file_path: Path, split: str) -> Optional[Dict]:
        """Process a single voxel file."""
        try:
            voxel_data = torch.load(file_path)
            model_id = file_path.stem.split('_aug')[0]
            
            data_dict = {
                'voxels_occupancy': None,
                'voxels_colors': None,
                'model_id': model_id,
            }
            
            # Process occupancy (Alpha channel) and colors (RGB channels)
            if voxel_data.shape[0] == 1:
                data_dict['voxels_occupancy'] = voxel_data.numpy().reshape(1, 32, 32, 32)
                occupancy = voxel_data[0]
            elif voxel_data.shape[0] == 4:
                data_dict['voxels_occupancy'] = voxel_data[3].unsqueeze(0).numpy()  # Alpha channel as (1, 32, 32, 32)
                data_dict['voxels_colors'] = voxel_data[:3].numpy()  # RGB channels as (3, 32, 32, 32)
                occupancy = voxel_data[3]
            else:
                raise ValueError(f"Unexpected channel count: {voxel_data.shape[0]} in {file_path}")
            
            # Add annotations and metadata
            annotation = self.annotations.get(model_id, {})
            data_dict.update({
                'name': annotation.get('name', ''),
                'categories': [cat['name'] for cat in annotation.get('categories', [])],
                'tags': [tag['name'] for tag in annotation.get('tags', [])],
                'metadata': {
                    'is_augmented': '_aug' in file_path.stem,
                    'original_file': str(file_path.relative_to(self.voxel_dir)),
                    'num_occupied': torch.sum(occupancy > 0).item(),
                    'split': split
                }
            })
            return data_dict
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            return None

    def create_dataset(self, batch_size: int = 10000, save_path: str = "blockgen_dataset") -> DatasetDict:
        """Create HuggingFace dataset with train/test splits incrementally and save to disk."""
        logger.info("Getting the files")
        files = self._get_all_files()
        
        # Split files into train/test
        train_size = int(len(files) * (1 - self.test_split))
        test_size = len(files) - train_size
        generator = torch.Generator().manual_seed(self.seed)
        train_indices, test_indices = torch.utils.data.random_split(
            range(len(files)), [train_size, test_size], generator=generator
        )
        train_files = [files[i] for i in train_indices]
        test_files = [files[i] for i in test_indices]
        
        # Ensure save_path exists
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize counters for incremental saving
        train_shard_idx, test_shard_idx = 0, 0
        
        # Create and save train dataset incrementally
        logger.info("Processing and saving training files...")
        train_examples = []
        for i in tqdm(range(0, len(train_files), batch_size)):
            batch_files = train_files[i:i + batch_size]
            batch_examples = [
                self._process_voxel_file(file, split="train") for file in batch_files if file
            ]
            train_examples.extend(filter(None, batch_examples))
            
            # Save shard
            train_dataset = Dataset.from_list(train_examples, features=self.features)
            shard_path = save_path / f"train-{train_shard_idx:05d}.parquet"
            train_dataset.to_parquet(str(shard_path))
            train_examples = []  # Clear examples to save memory
            train_shard_idx += 1
    
        # Create and save test dataset incrementally
        logger.info("Processing and saving test files...")
        test_examples = []
        for i in tqdm(range(0, len(test_files), batch_size)):
            batch_files = test_files[i:i + batch_size]
            batch_examples = [
                self._process_voxel_file(file, split="test") for file in batch_files if file
            ]
            test_examples.extend(filter(None, batch_examples))
            
            # Save shard
            test_dataset = Dataset.from_list(test_examples, features=self.features)
            shard_path = save_path / f"test-{test_shard_idx:05d}.parquet"
            test_dataset.to_parquet(str(shard_path))
            test_examples = []  # Clear examples to save memory
            test_shard_idx += 1
    
        # Return paths to the dataset shards for verification
        logger.info(f"Datasets saved at {save_path}")
        logger.info(f"Train shards created: {train_shard_idx}")
        logger.info(f"Test shards created: {test_shard_idx}")
        return {"train": train_shard_idx, "test": test_shard_idx}


def push_to_huggingface(dataset_path: str,
                        annotation_path: str,
                        repo_id: str,
                        token: str,
                        test_split: float = 0.05,
                        seed: int = 42,
                        save_path: str = "blockgen_dataset",
                        private: bool = False):
    """Create and push BlockGen dataset to HuggingFace."""
    # Create the dataset creator object
    creator = BlockGenDatasetCreator(
        voxel_dir=dataset_path,
        annotation_file=annotation_path,
        test_split=test_split,
        seed=seed
    )

    #dataset_info = creator.create_dataset(save_path=save_path)

    #logger.info(f"Dataset created with {dataset_info['train']} train shards and {dataset_info['test']} test shards.")
    
    save_path = Path(save_path)
    logger.info("Identifying dataset shards...")
    train_files = sorted(save_path.glob("train-*.parquet"))
    test_files = sorted(save_path.glob("test-*.parquet"))

    if not train_files or not test_files:
        logger.error("No train or test shards found in the specified save path!")
        return
    
    logger.info("Loading datasets from shards...")
    train_dataset = Dataset.from_parquet([str(file) for file in train_files])
    test_dataset = Dataset.from_parquet([str(file) for file in test_files])
    
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })
    
    logger.info(f"Private: {private}")
    logger.info(f"Pushing dataset to HuggingFace: {repo_id}")
    dataset.push_to_hub(repo_id, token=token, private=private)
    logger.info("Dataset successfully pushed to HuggingFace!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Push BlockGen dataset to HuggingFace")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to voxel files directory")
    parser.add_argument("--annotation_path", type=str, required=True, help="Path to annotations file")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repository ID")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace API token")
    parser.add_argument("--test_split", type=float, default=0.05, help="Fraction for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    
    args = parser.parse_args()
    
    push_to_huggingface(
        dataset_path=args.dataset_path,
        annotation_path=args.annotation_path,
        repo_id=args.repo_id,
        token=args.token,
        test_split=args.test_split,
        seed=args.seed,
        save_path="blockgen_dataset",
        private=args.private
    )
