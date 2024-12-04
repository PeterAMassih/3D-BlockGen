# scripts/process_dataset.py
from pathlib import Path
import json
import traceback
from blockgen.data.processing.data_voxelization import VoxelizerWithAugmentation
import tqdm
import gc
import torch

def process_dataset(input_dir: str, output_dir: str, resolution: int = 32, visualize_all: bool = False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    voxelizer = VoxelizerWithAugmentation(resolution=resolution)
    
    # Load or create processing log
    log_file = output_dir / "processing_log.json"
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_data = json.load(f)
            processed_set = set(log_data["processed"])
            failed_set = set(log_data["failed"])
    else:
        log_data = {"processed": [], "failed": []}
        processed_set = set()
        failed_set = set()
    
    glb_files = list(input_dir.rglob("*.glb"))
    files_to_process = [f for f in glb_files if str(f) not in processed_set]
    
    print(f"\nTotal files found: {len(glb_files)}")
    print(f"Already processed: {len(processed_set)}")
    print(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        print("All files have been processed!")
        return
        
    all_results = []
    
    for glb_file in tqdm(files_to_process, desc="Processing files"):
        try:
            results = voxelizer.process_mesh(str(glb_file))
            
            if results:
                # Keep only first 5 results for visualization
                if len(all_results) < 5:
                    all_results.append(results[0])
                
                # Save with directory structure
                relative_path = glb_file.relative_to(input_dir)
                save_dir = output_dir / relative_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                
                for i, tensor in enumerate(results):
                    suffix = "" if i == 0 else f"_aug{i}"
                    save_path = save_dir / f"{relative_path.stem}{suffix}.pt"
                    torch.save(tensor, save_path)
                
                # Update tracking
                file_str = str(glb_file)
                processed_set.add(file_str)
                log_data["processed"].append(file_str)
                
                # Save log periodically
                if len(processed_set) % 100 == 0:
                    with open(log_file, 'w') as f:
                        json.dump(log_data, f, indent=2)
            
            # Clear memory
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {glb_file}: {str(e)}")
            file_str = str(glb_file)
            failed_set.add(file_str)
            log_data["failed"].append(file_str)
    
    # Save final log
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print("\nProcessing Complete!")
    print(f"Successfully processed: {len(processed_set)}")
    print(f"Failed: {len(failed_set)}")
    
    if visualize_all and all_results:
        voxelizer.visualize_results(all_results, "Dataset Visualization")

if __name__ == "__main__":
    input_dir = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data"
    output_dir = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data_voxelized"
    try:
        process_dataset(input_dir, output_dir, resolution=32, visualize_all=False)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        traceback.print_exc()