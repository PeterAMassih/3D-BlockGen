from torch.utils.data import DataLoader, random_split
from ..data.dataset import VoxelTextDataset
from ..configs import DiffusionConfig, VoxelConfig
from typing import Tuple

def create_dataloaders(
    voxel_dir: str,
    annotation_file: str,
    config: DiffusionConfig, 
    config_voxel: VoxelConfig,
    batch_size: int = 32,
    test_split: float = 0.05,
    num_workers: int = 4,
    use_label_mapping: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders"""
    dataset = VoxelTextDataset(voxel_dir, annotation_file, config_voxel, use_label_mapping)
    
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=config.generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=VoxelTextDataset.collate_fn,
        generator=config.generator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=VoxelTextDataset.collate_fn
    )
    
    return train_loader, test_loader