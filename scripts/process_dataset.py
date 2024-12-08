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
                # Ensure all required keys exist
                log_data.setdefault("processed", [])
                log_data.setdefault("permanent_failures", [])
                return log_data, set(log_data["processed"]), set(log_data["permanent_failures"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading log file: {e}. Starting fresh.")
    
    return {"processed": [], "permanent_failures": []}, set(), set()

def process_dataset(input_dir: str, output_dir: str, resolution: int = 32):
    """Process the dataset sequentially."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create processing log
    log_file = output_dir / "processing_log.json"
    log_data, processed_set, permanent_failures_set = load_processing_log(log_file)
    
    # Collect files to process
    glb_files = list(input_dir.rglob("*.glb"))
    files_to_process = [f for f in glb_files if str(f) not in processed_set and str(f) not in permanent_failures_set]
    
    logger.info(f"Total files found: {len(glb_files)}")
    logger.info(f"Already processed: {len(processed_set)}")
    logger.info(f"Permanent failures: {len(permanent_failures_set)}")
    logger.info(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        logger.info("All files have been processed!")
        return
    
    # Create single voxelizer instance
    voxelizer = VoxelizerWithAugmentation(resolution=resolution)
    
    # Process files sequentially with progress tracking
    processed_count = 0
    failed_count = 0
    
    try:
        for glb_file in tqdm(files_to_process, desc="Processing files"):
            results = None
            try:
                # Process mesh
                results = voxelizer.process_mesh(str(glb_file))
                
                if not results:
                    permanent_failures_set.add(str(glb_file))
                    log_data["permanent_failures"].append(str(glb_file))
                    failed_count += 1
                    logger.error(f"Failed to process {glb_file}: No results generated")
                    continue
                
                # Save with directory structure
                relative_path = Path(glb_file).relative_to(input_dir)
                save_dir = output_dir / relative_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save all tensors
                for i, tensor in enumerate(results):
                    suffix = "" if i == 0 else f"_aug{i}"
                    save_path = save_dir / f"{relative_path.stem}{suffix}.pt"
                    torch.save(tensor, save_path)
                
                # Mark as processed
                processed_set.add(str(glb_file))
                log_data["processed"].append(str(glb_file))
                processed_count += 1
                
            except Exception as e:
                permanent_failures_set.add(str(glb_file))
                log_data["permanent_failures"].append(str(glb_file))
                failed_count += 1
                logger.error(f"Failed to process {glb_file}: {str(e)}")
            
            finally:
                # Clean up memory after each file
                if results is not None:
                    del results
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save log periodically (every 20 files)
            if (processed_count + failed_count) % 20 == 0:
                with open(log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)
                    
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