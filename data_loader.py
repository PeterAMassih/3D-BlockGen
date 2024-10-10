import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os

class ObjaverseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(self.data_dir)
        print("Found", len(self.file_list), "files in", data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_path)  # Load the tensor
        data = data.squeeze(0)  # Squeeze the first dimension if it's 1
        return data.float()  # Convert to float32 for training

def create_dataloader(data_dir, batch_size=8, test_split=0.1, return_dataset=False):
    """
    Create train and test DataLoaders.
    
    Args:
    - data_dir: Directory where data is located.
    - batch_size: Batch size for the DataLoaders.
    - test_split: Fraction of data to be used for testing.
    - return_dataset: Whether to return the raw datasets.
    
    Returns:
    - train_dataloader: DataLoader for training.
    - test_dataloader: DataLoader for testing.
    - Optionally, the raw datasets as well.
    """
    print(f"Loading data from {data_dir}...")
    dataset = ObjaverseDataset(data_dir)
    
    # Calculate the sizes for training and testing datasets
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for both sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if return_dataset:
        return train_dataloader, test_dataloader, train_dataset, test_dataset
    return train_dataloader, test_dataloader
