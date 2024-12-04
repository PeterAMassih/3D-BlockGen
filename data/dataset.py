from torch.utils.data import Dataset
from transformers import CLIPTokenizer
from pathlib import Path
import torch
import json
from tqdm import tqdm
from..configs.voxel_config import VoxelConfig 

class VoxelTextDataset(Dataset):
    """Dataset for text-conditioned voxel generation"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32") # Now static to all instances

    def __init__(self, voxel_dir, annotation_file, config: VoxelConfig):
        self.voxel_dir = Path(voxel_dir)
        self.config = config
        
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Get all files recursively
        self.files = []
        for pt_file in tqdm(self.voxel_dir.rglob("*.pt"), desc="Finding voxel files"):
            if "_aug" not in pt_file.name:
                self.files.append(pt_file)
                for i in range(1, 4):
                    aug_file = pt_file.with_name(f"{pt_file.stem}_aug{i}.pt")
                    if aug_file.exists():
                        self.files.append(aug_file)
        
        print(f"\nFound {len(self.files)} files (including augmentations)")
        
    def _process_voxel_data(self, voxel_data):
        """Process voxel data according to configuration"""
        if self.config.use_rgb:
            if voxel_data.shape[0] == 4:
                # Already in RGBA format
                alpha = (voxel_data[3:4] > 0).float()
                rgb = voxel_data[:3]
                return torch.cat([alpha, rgb], dim=0)
            else:
                # Convert occupancy to RGBA
                alpha = (voxel_data > 0).float()  # [1, H, W, D]
                rgb = self.config.default_color.view(3, 1, 1, 1).to(voxel_data.device)
                rgb = rgb.repeat(1, *voxel_data.shape[1:])  # [3, H, W, D]
                return torch.cat([alpha, rgb], dim=0)  # [4, H, W, D]
        else:
            if voxel_data.shape[0] > 1:
                voxel_data = voxel_data[-1:] # take the last one which is occupancy TODO IF RGBAO change

            return (voxel_data > 0).float()
    
    def _create_simple_prompt(self, model_id):
        if model_id in self.annotations:
            name = self.annotations[model_id].get('name', 'an object')
            return f"{name}" # remove a 3d model of
        return ""
    
    def _create_detailed_prompt(self, model_id):
        if model_id in self.annotations:
            name = self.annotations[model_id].get('name', 'an object')
            categories = [cat['name'] for cat in self.annotations[model_id].get('categories', [])]
            tags = [tag['name'] for tag in self.annotations[model_id].get('tags', [])]
            
            prompt_parts = [f"{name}"] # REMOVE a 3d model of
            if categories:
                prompt_parts.append(f"in category {', '.join(categories)}")
            if tags:
                prompt_parts.append(f"with traits: {', '.join(tags)}")
            
            return ' '.join(prompt_parts)
        return ""
    
    def _get_random_prompt(self, model_id):
        rand = torch.rand(1).item()
        if rand < 0.10:
            #print("")
            return ""
        elif rand < 0.55:
            #print(self._create_detailed_prompt(model_id))
            return self._create_detailed_prompt(model_id)
        else:
            #print(self._create_simple_prompt(model_id))
            return self._create_simple_prompt(model_id)
    
    def __len__(self):
        return len(self.files)
    
    @staticmethod
    def collate_fn(batch):
        voxels = torch.stack([item['voxels'] for item in batch])
        prompts = [item['text'] for item in batch]
        
        
        tokens = VoxelTextDataset.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
            # clean_up_tokenization_spaces=True  # Not available in current version Sometimes fires a warning
        )
        
        return {
            'voxels': voxels,
            **tokens
        }
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        model_id = file_path.stem.split('_aug')[0]
        
        voxel_data = torch.load(file_path)
        voxel_data = self._process_voxel_data(voxel_data)
        
        return {
            'voxels': voxel_data,
            'text': self._get_random_prompt(model_id)
        }