from pathlib import Path
import json
import traceback
from blockgen.data.processing.data_voxelization import VoxelizerWithAugmentation
from tqdm import tqdm
import torch
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processing_log(log_file):
    """Load or initialize the processing log with error handling."""
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                return set(log_data.get("processed", [])), set(log_data.get("permanent_failures", []))
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading log file: {e}. Starting fresh.")
    return set(), set()

def process_dataset(input_dir: str, output_dir: str, resolution: int = 32):
    """Process the dataset sequentially with optimized memory usage."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed files tracking
    log_file = output_dir / "processing_log.json"
    processed_files, failed_files = load_processing_log(log_file)
    
    # Find all GLB files efficiently
    all_files = {str(f) for f in Path(input_dir).rglob("*.glb")}
    files_to_process = all_files - processed_files - failed_files
    
    logger.info(f"Total files found: {len(all_files)}")
    logger.info(f"Already processed: {len(processed_files)}")
    logger.info(f"Permanent failures: {len(failed_files)}")
    logger.info(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        logger.info("All files have been processed!")
        return
    
    # Create single voxelizer instance
    voxelizer = VoxelizerWithAugmentation(resolution=resolution)
    
    # Process files sequentially
    processed_count = 0
    failed_count = 0
    
    try:
        for file_path in tqdm(sorted(files_to_process), desc="Processing files"):
            results = None
            try:
                # Process mesh
                results = voxelizer.process_mesh(file_path)
                
                if not results:
                    failed_files.add(file_path)
                    failed_count += 1
                    logger.error(f"Failed to process {file_path}: No results generated")
                    continue
                
                # Save tensors
                relative_path = Path(file_path).relative_to(input_dir)
                save_dir = output_dir / relative_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                
                for i, tensor in enumerate(results):
                    suffix = "" if i == 0 else f"_aug{i}"
                    save_path = save_dir / f"{relative_path.stem}{suffix}.pt"
                    torch.save(tensor, save_path)
                
                processed_files.add(file_path)
                processed_count += 1
                
            except Exception as e:
                failed_files.add(file_path)
                failed_count += 1
                logger.error(f"Failed to process {file_path}: {str(e)}")
            
            finally:
                # Clean up memory
                if results is not None:
                    del results
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save progress periodically
            if (processed_count + failed_count) % 20 == 0:
                current_log = {
                    "processed": list(processed_files),
                    "permanent_failures": list(failed_files)
                }
                with open(log_file, 'w') as f:
                    json.dump(current_log, f, indent=2)
    
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    
    finally:
        # Save final log
        final_log = {
            "processed": list(processed_files),
            "permanent_failures": list(failed_files)
        }
        with open(log_file, 'w') as f:
            json.dump(final_log, f, indent=2)
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Failed: {failed_count}")

if __name__ == "__main__":
    input_dir = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data"
    output_dir = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data_voxelized"
    
    try:
        process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            resolution=32
        )
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}")
        traceback.print_exc()