from torch.utils.data import Dataset
from transformers import CLIPTokenizer
from pathlib import Path
import torch
import json
from tqdm import tqdm
from ..configs.voxel_config import VoxelConfig


class VoxelTextDataset(Dataset):
    """Dataset for text-conditioned voxel generation"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def __init__(self, voxel_dir, annotation_file, config: VoxelConfig, use_label_mapping: bool = False):
        """
        Args:
            voxel_dir: Directory containing voxel files
            annotation_file: Path to annotation file (either standard Objaverse or label mapping)
            config: VoxelConfig instance
            use_label_mapping: If True, uses simple label mapping format instead of full annotations
        """
        self.voxel_dir = Path(voxel_dir)
        self.config = config
        self.use_label_mapping = use_label_mapping

        with open(annotation_file, 'r') as f:
            if use_label_mapping:
                # Load simple label mapping (filename -> label)
                self.label_mapping = json.load(f)
                self.annotations = None  # Not used in this mode
            else:
                # Load standard Objaverse annotations
                self.annotations = json.load(f)
                self.label_mapping = None  # Not used in this mode

        self.files = []
        seen_paths = set()  # To track duplicates

        if self.use_label_mapping:
            # For finetuning dataset: Only look for original files, no augmentations
            for pt_file in tqdm(self.voxel_dir.rglob("*.pt"), desc="Finding voxel files"):
                # Extract original GLB filename from PT filename
                glb_filename = f"{pt_file.stem}.glb"
                if glb_filename in self.label_mapping:
                    pt_path_str = str(pt_file)
                    if pt_path_str not in seen_paths:
                        self.files.append(pt_file)
                        seen_paths.add(pt_path_str)
        else:
            # For standard dataset: Include augmentations
            for pt_file in tqdm(self.voxel_dir.rglob("*.pt"), desc="Finding voxel files"):
                if "_aug" not in pt_file.name:
                    # Verify path is valid
                    if pt_file.exists():
                        pt_path_str = str(pt_file)
                        if pt_path_str not in seen_paths:
                            self.files.append(pt_file)
                            seen_paths.add(pt_path_str)

                    # Check augmentations
                    for i in range(1, 4):
                        aug_file = pt_file.parent / f"{pt_file.stem}_aug{i}.pt"
                        if aug_file.exists():
                            aug_path_str = str(aug_file)
                            if aug_path_str not in seen_paths:
                                self.files.append(aug_file)
                                seen_paths.add(aug_path_str)

        print(f"\nFound {len(self.files)} unique files" +
              ("" if self.use_label_mapping else " (including augmentations)"))

        # Debug: Print a few paths
        print("\nSample paths:")
        for path in self.files[:5]:
            print(f"  {path}")

    def _get_label_for_model(self, model_id: str) -> str:
        """Get the label for a model based on the annotation format"""
        if self.use_label_mapping:
            # For label mapping, just return the label directly
            glb_filename = f"{model_id}.glb"
            return self.label_mapping.get(glb_filename, "")
        else:
            # For standard annotations, use existing logic
            if model_id in self.annotations:
                return self.annotations[model_id].get('name', 'an object')
            return ""

    def _create_simple_prompt(self, model_id):
        """Create a simple prompt based on the model ID"""
        return self._get_label_for_model(model_id)

    def _create_detailed_prompt(self, model_id):
        """Create a detailed prompt based on the model ID"""
        if self.use_label_mapping:
            # For label mapping, we only have the simple label
            return self._get_label_for_model(model_id)
        else:
            if model_id in self.annotations:
                name = self.annotations[model_id].get('name', 'an object')
                categories = [cat['name'] for cat in self.annotations[model_id].get('categories', [])]
                tags = [tag['name'] for tag in self.annotations[model_id].get('tags', [])]

                prompt_parts = [f"{name}"]
                if categories:
                    prompt_parts.append(f"in category {', '.join(categories)}")
                if tags:
                    prompt_parts.append(f"with traits: {', '.join(tags)}")

                return ' '.join(prompt_parts)
            return ""

    def _get_random_prompt(self, model_id):
        """Get a random prompt for the model"""
        # For finetuning dataset, always return the simple label
        if self.use_label_mapping:
            return self._get_label_for_model(model_id)

        rand = torch.rand(1).item()
        if rand < 0.10:
            return ""
        elif rand < 0.55:
            return self._create_detailed_prompt(model_id)
        else:
            return self._create_simple_prompt(model_id)

    def _process_voxel_data(self, voxel_data):
        """Process voxel data according to configuration"""
        if self.config.mode == 'two_stage':
            if self.config.stage == 'shape':
                # Shape stage: Handle binary occupancy
                # Input: [1, H, W, D] or potentially other shapes
                if voxel_data.shape[0] > 1:
                    voxel_data = voxel_data[-1:]  # Take the last channel (occupancy)
                # Output: [1, H, W, D] (binary occupancy)
                # print(voxel_data.shape)
                return (voxel_data > 0).float()
            elif self.config.stage == 'color':
                # Color stage: Handle RGBA format
                # Input: [4, H, W, D] or potentially [1, H, W, D]
                if voxel_data.shape[0] == 4:
                    # Already in RGBA format
                    # Output: [4, H, W, D] (RGBA with binary alpha)
                    alpha = (voxel_data[3:4] > 0).float()
                    rgb = voxel_data[:3]
                    return torch.cat([rgb, alpha], dim=0)
                elif voxel_data.shape[0] == 1:
                    # Convert occupancy to RGBA
                    # Output: [4, H, W, D] (RGBA)
                    alpha = (voxel_data > 0).float()  # [1, H, W, D]
                    rgb = self.config.default_color.view(3, 1, 1, 1).to(voxel_data.device)
                    rgb = rgb.repeat(1, *voxel_data.shape[1:])  # [3, H, W, D]
                    return torch.cat([rgb, alpha], dim=0)
                else:
                    raise ValueError("Unexpected voxel data shape for color stage")
        elif self.config.mode == 'combined':
            # Combined mode: Handle RGBA or convert occupancy to RGBA
            # Input: [4, H, W, D] or [1, H, W, D]
            if voxel_data.shape[0] == 4:
                # Already in RGBA format
                # Output: [4, H, W, D] (RGBA with binary alpha)
                alpha = (voxel_data[3:4] > 0).float()
                rgb = voxel_data[:3]
                return torch.cat([rgb, alpha], dim=0)
            elif voxel_data.shape[0] == 1:
                # Convert occupancy to RGBA
                # Output: [4, H, W, D] (RGBA)
                alpha = (voxel_data > 0).float()  # [1, H, W, D]
                rgb = self.config.default_color.view(3, 1, 1, 1).to(voxel_data.device)
                rgb = rgb.repeat(1, *voxel_data.shape[1:])  # [3, H, W, D]
                return torch.cat([rgb, alpha], dim=0)
            else:
                raise ValueError("Unexpected voxel data shape for combined mode")
        elif self.config.mode == 'shape':
            # Shape-only mode: Handle binary occupancy
            # Input: [1, H, W, D] or potentially other shapes
            if voxel_data.shape[0] > 1:
                voxel_data = voxel_data[-1:]  # Take the last channel (occupancy)
            # Output: [1, H, W, D] (binary occupancy)
            return (voxel_data > 0).float()
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

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

        try:
            voxel_data = torch.load(file_path)
        except Exception as e:
            print(f"Error loading file: {file_path}")
            print(f"Error: {str(e)}")
            raise

        voxel_data = self._process_voxel_data(voxel_data)

        return {
            'voxels': voxel_data,
            'text': self._get_random_prompt(model_id)
        }
