import torch
from torch.utils.data import Dataset, DataLoader
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

def create_dataloader(data_dir, batch_size=8, return_dataset=False):
    print(f"Loading data from {data_dir}...")
    dataset = ObjaverseDataset(data_dir)
    if return_dataset:
        return dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example usage for dataloader:
# dataloader = create_dataloader("/path/to/processed_models", batch_size=8)
