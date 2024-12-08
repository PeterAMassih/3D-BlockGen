from pathlib import Path
import json
import traceback
from blockgen.data.processing.data_voxelization import VoxelizerWithAugmentation
from tqdm import tqdm
import torch
import gc

def load_processing_log(log_file):
    """
    Load the processing log from a JSON file.

    Args:
        log_file (Path): Path to the log file.

    Returns:
        processed_set (set): Set of processed file paths.
        failed_set (set): Set of permanently failed file paths.
    """
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                processed_set = set(log_data.get("processed", []))
                failed_set = set(log_data.get("permanent_failures", []))
                print(f"Loaded processing log: {len(processed_set)} processed, {len(failed_set)} failed.")
                return processed_set, failed_set
        except Exception as e:
            print(f"Failed to load log file {log_file}: {e}")
            return set(), set()
    print("No existing processing log found. Starting fresh.")
    return set(), set()

def save_processing_log(log_file, processed_list, failed_list):
    """
    Save the processing log to a JSON file.

    Args:
        log_file (Path): Path to the log file.
        processed_list (list): List of processed file paths.
        failed_list (list): List of permanently failed file paths.
    """
    try:
        with open(log_file, 'w') as f:
            json.dump({
                "processed": processed_list,
                "permanent_failures": failed_list
            }, f, indent=2)
        print(f"Processing log saved: {len(processed_list)} processed, {len(failed_list)} failed.")
    except Exception as e:
        print(f"Failed to save processing log {log_file}: {e}")

def process_dataset(input_dir: str, output_dir: str, resolution: int = 32, visualize_all: bool = False):
    """
    Process 3D datasets by voxelizing .glb files with optional augmentation and visualization.

    Args:
        input_dir (str): Path to the input directory containing .glb files.
        output_dir (str): Path to the output directory to save voxelized data.
        resolution (int, optional): Voxel resolution. Defaults to 32.
        visualize_all (bool, optional): Whether to visualize processed results. Defaults to False.
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "processing_log.json"
    processed_set, failed_set = load_processing_log(log_file)
    
    # Gather all .glb files
    all_glb_files = set(str(f) for f in input_dir.rglob("*.glb"))
    files_to_process = sorted(all_glb_files - processed_set - failed_set)
    
    print(f"\nTotal .glb files found: {len(all_glb_files)}")
    print(f"Already processed: {len(processed_set)}")
    print(f"Previously failed: {len(failed_set)}")
    print(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        print("All files have been processed! Exiting.")
        return
    
    voxelizer = VoxelizerWithAugmentation(resolution=resolution)
    all_results = []
    processed_list = list(processed_set)
    failed_list = list(failed_set)
    
    # Configuration for intervals
    gc_interval = 50      # Perform GC every 500 files
    save_interval = 40    # Save log every 100 files
    
    # Precompute CUDA availability
    cuda_available = torch.cuda.is_available()
    
    try:
        with tqdm(total=len(files_to_process), desc="Processing files", dynamic_ncols=True) as pbar:
            for idx, file_path_str in enumerate(files_to_process, 1):
                file_path = Path(file_path_str)
                try:
                    results = voxelizer.process_mesh(str(file_path))
                    
                    if results:
                        # Collect results for visualization (up to first 5)
                        if len(all_results) < 5:
                            all_results.append(results[0])
                        
                        # Define save directory maintaining the input's structure
                        relative_path = file_path.relative_to(input_dir)
                        save_dir = output_dir / relative_path.parent
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save each tensor result
                        for i, tensor in enumerate(results):
                            suffix = "" if i == 0 else f"_aug{i}"
                            save_path = save_dir / f"{relative_path.stem}{suffix}.pt"
                            torch.save(tensor, save_path)
                        
                        # Update processed tracking
                        processed_list.append(file_path_str)
                    
                    else:
                        # No results implies failure
                        failed_list.append(file_path_str)
                        print(f"Error processing {file_path_str}: No results returned.")
                
                except Exception as e:
                    # Log exception immediately
                    failed_list.append(file_path_str)
                    print(f"Error processing {file_path_str}: {str(e)}")
                
                # Update progress bar after every file
                total_processed = len(processed_list) + len(failed_list)
                success_rate = (len(processed_list) / max(total_processed, 1)) * 100
                pbar.set_postfix({
                    'processed': f"{len(processed_list)}",
                    'failed': f"{len(failed_list)}",
                    'success_rate': f"{success_rate:.1f}%"
                })
                pbar.update(1)
                
                # Garbage collection and CUDA cache clearing
                if idx % gc_interval == 0:
                    gc.collect()
                    if cuda_available:
                        torch.cuda.empty_cache()
                
                # Save processing log periodically
                if idx % save_interval == 0:
                    save_processing_log(log_file, processed_list, failed_list)
        
        # Save any remaining progress in the log
        save_processing_log(log_file, processed_list, failed_list)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        # Save current progress before exiting
        save_processing_log(log_file, processed_list, failed_list)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        # Save current progress before exiting
        save_processing_log(log_file, processed_list, failed_list)
    
    finally:
        # Final summary
        print("\nProcessing Complete!")
        print(f"Successfully processed: {len(processed_list)}")
        print(f"Failed: {len(failed_list)}")
        
        # Optional visualization
        if visualize_all and all_results:
            try:
                voxelizer.visualize_results(all_results, "Dataset Visualization")
                print("Visualization completed.")
            except Exception as e:
                print(f"Visualization failed: {e}")

if __name__ == "__main__":
    INPUT_DIR = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data"
    OUTPUT_DIR = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data_voxelized"
    
    try:
        process_dataset(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            resolution=32,
            visualize_all=False  # Set to True if visualization is desired
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"Fatal error during processing: {str(e)}")
        traceback.print_exc()
