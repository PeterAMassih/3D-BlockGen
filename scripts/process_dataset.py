from pathlib import Path
import json
import traceback
from blockgen.data.processing.data_voxelization import VoxelizerWithAugmentation
from tqdm import tqdm
import torch
import gc
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_file(args):
    """Process a single GLB file with optimized memory management."""
    glb_file, input_dir, output_dir, resolution = args
    results = None
    
    try:
        # Create voxelizer instance for this process
        voxelizer = VoxelizerWithAugmentation(resolution=resolution)
        
        # Process mesh
        results = voxelizer.process_mesh(str(glb_file))
        
        if not results:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return str(Path(glb_file).resolve()), True, None
        
        # Save with directory structure
        relative_path = Path(glb_file).relative_to(input_dir)
        save_dir = output_dir / relative_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug: Print tensor information
        # logger.info(f"Processing {len(results)} tensors for {glb_file}")
        
        # Save all tensors
        for i, tensor in enumerate(results):
            suffix = "" if i == 0 else f"_aug{i}"
            save_path = save_dir / f"{relative_path.stem}{suffix}.pt"
            
            # logger.info(f"Saving tensor {i} to {save_path}, shape: {tensor.shape}")
            torch.save(tensor, save_path)
            # logger.info(f"Successfully saved tensor {i} to {save_path}")
        
        # Clean up after successful processing
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return str(glb_file), True, None
        
    except Exception as e:
        # Clean up on error
        if results is not None:
            del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return str(Path(glb_file).resolve()), False, str(e)

def load_processing_log(log_file):
    """Load or initialize the processing log with error handling."""
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                # Ensure all required keys exist
                log_data.setdefault("processed", [])
                log_data.setdefault("permanent_failures", [])
                return log_data, set(log_data["processed"]), set(log_data["permanent_failures"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading log file: {e}. Starting fresh.")
    
    return {"processed": [], "permanent_failures": []}, set(), set()

def process_dataset(input_dir: str, output_dir: str, resolution: int = 32, num_processes: int = None):
    """Process the dataset with optimized multiprocessing."""
    input_dir = Path(input_dir).resolve()  # Make absolute
    output_dir = Path(output_dir).resolve()  # Make absolute
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create processing log
    log_file = output_dir / "processing_log.json"
    log_data, processed_set, permanent_failures_set = load_processing_log(log_file)
    
    # Collect files to process
    glb_files = list(input_dir.rglob("*.glb"))
    files_to_process = [f.resolve() for f in glb_files if str(f.resolve()) not in processed_set and str(f.resolve()) not in permanent_failures_set]
    
    logger.info(f"Total files found: {len(glb_files)}")
    logger.info(f"Already processed: {len(processed_set)}")
    logger.info(f"Permanent failures: {len(permanent_failures_set)}")
    logger.info(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        logger.info("All files have been processed!")
        return
    
    # Determine optimal number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), 12)  # Limit to reasonable number
    
    logger.info(f"Using {num_processes} processes")
    
    # Prepare arguments for multiprocessing
    process_args = [(f, input_dir, output_dir, resolution) for f in files_to_process]
    
    # Process files in parallel with progress tracking
    processed_count = 0
    failed_count = 0
    
    try:
        with Pool(processes=num_processes) as pool:
            with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
                for file_path, success, error in pool.imap_unordered(process_single_file, process_args):
                    if success:
                        processed_set.add(file_path)
                        log_data["processed"].append(file_path)
                        processed_count += 1
                    else:
                        permanent_failures_set.add(file_path)
                        log_data["permanent_failures"].append(file_path)
                        failed_count += 1
                        logger.error(f"Failed to process {file_path}: {error}")
                    
                    # Save log periodically (every 20 files)
                    if (processed_count + failed_count) % 20 == 0:
                        with open(log_file, 'w') as f:
                            json.dump(log_data, f, indent=2)
                    
                    pbar.update(1)
    
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    finally:
        # Save final log
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Failed: {failed_count}")

if __name__ == "__main__":
    input_dir = "objaverse_data"
    output_dir = "objaverse_data_voxelized"
    
    try:
        process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            resolution=32,
            num_processes=12  # Adjust based on your system
        )
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}")
        traceback.print_exc()