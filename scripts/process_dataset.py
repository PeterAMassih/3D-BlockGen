from pathlib import Path
import json
import traceback
from blockgen.data.processing.data_voxelization import VoxelizerWithAugmentation
from tqdm import tqdm
import torch
from multiprocessing import Pool, Value, Lock
from functools import partial
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_file(glb_file, input_dir, output_dir, resolution):
    """Process a single glb file and return its status."""
    voxelizer = VoxelizerWithAugmentation(resolution=resolution)
    
    try:
        results = voxelizer.process_mesh(str(glb_file))
        
        if not results:
            return str(glb_file), False, "No results generated"
            
        # Save with directory structure
        relative_path = glb_file.relative_to(input_dir)
        save_dir = output_dir / relative_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tensors
        for i, tensor in enumerate(results):
            suffix = "" if i == 0 else f"_aug{i}"
            save_path = save_dir / f"{relative_path.stem}{suffix}.pt"
            torch.save(tensor, save_path)
        
        return str(glb_file), True, None
        
    except Exception as e:
        return str(glb_file), False, str(e)

def process_dataset(input_dir: str, output_dir: str, resolution: int = 32, num_workers: int = 8):
    """Process the entire dataset with parallel processing and proper error handling."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create processing log
    log_file = output_dir / "processing_log.json"
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_data = json.load(f)
            processed_set = set(log_data.get("processed", []))
            permanent_failures = set(log_data.get("permanent_failures", []))
    else:
        log_data = {"processed": [], "permanent_failures": []}
        processed_set = set()
        permanent_failures = set()
    
    # Get files to process
    glb_files = list(input_dir.rglob("*.glb"))
    files_to_process = [
        f for f in glb_files 
        if str(f) not in processed_set and str(f) not in permanent_failures
    ]
    
    logger.info(f"Total files found: {len(glb_files)}")
    logger.info(f"Already processed: {len(processed_set)}")
    logger.info(f"Permanent failures: {len(permanent_failures)}")
    logger.info(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        logger.info("All files have been processed!")
        return
    
    # Initialize counters
    processed_count = Value('i', 0)
    failed_count = Value('i', 0)
    log_lock = Lock()
    
    def save_log():
        """Save processing log with lock."""
        with log_lock:
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
    
    def update_progress(result):
        """Update progress and logs atomically."""
        file_path, success, error = result
        nonlocal log_data
        
        with log_lock:
            if success:
                with processed_count.get_lock():
                    processed_count.value += 1
                log_data["processed"].append(file_path)
            else:
                with failed_count.get_lock():
                    failed_count.value += 1
                if "Scene' object" in str(error) or "vertices" in str(error):
                    log_data["permanent_failures"].append(file_path)
                logger.error(f"Failed to process {file_path}: {error}")
            
            # Save log periodically
            if (processed_count.value + failed_count.value) % 100 == 0:
                save_log()
    
    # Process files in parallel
    # Process files in parallel
    try:
        with Pool(num_workers) as pool:
            # Create TQDM instance that will work with logging
            pbar = tqdm(total=len(files_to_process), 
                       desc="Processing files",
                       dynamic_ncols=True,
                       position=0,
                       leave=True)
            
            # Create the process function
            process_func = partial(process_single_file,
                               input_dir=input_dir,
                               output_dir=output_dir,
                               resolution=resolution)
            
            def update_with_pbar(result):
                update_progress(result)
                pbar.update(1)
                pbar.refresh()  # Force refresh of the progress bar
            
            for _ in pool.imap_unordered(process_func, files_to_process):
                update_with_pbar(_)
            
            pbar.close()
            
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    finally:
        save_log()
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Successfully processed: {processed_count.value}")
        logger.info(f"Failed: {failed_count.value}")

if __name__ == "__main__":
    input_dir = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data"
    output_dir = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data_voxelized"
    
    try:
        process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            resolution=32,
            num_workers=12
        )
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}")
        traceback.print_exc()