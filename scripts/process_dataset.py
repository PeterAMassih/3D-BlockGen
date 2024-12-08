from pathlib import Path
import json
import traceback
from blockgen.data.processing.data_voxelization import VoxelizerWithAugmentation
from tqdm import tqdm
import torch
import gc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processing_log(log_file):
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                return log_data.get("processed", []), log_data.get("permanent_failures", [])
        except:
            return [], []
    return [], []

def process_dataset(input_dir: str, output_dir: str, resolution: int = 32):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "processing_log.json"
    processed_list, failed_list = load_processing_log(log_file)
    
    processed_set = set(processed_list)
    failed_set = set(failed_list)
    all_files = {str(f) for f in Path(input_dir).rglob("*.glb")}
    files_to_process = all_files - processed_set - failed_set
    
    logger.info(f"Total files found: {len(all_files)}")
    logger.info(f"Already processed: {len(processed_set)}")
    logger.info(f"Permanent failures: {len(failed_set)}")
    logger.info(f"Files to process: {len(files_to_process)}")
    
    if not files_to_process:
        logger.info("All files have been processed!")
        return
    
    voxelizer = VoxelizerWithAugmentation(resolution=resolution)
    processed_count = failed_count = 0
    gc_interval = 100
    save_interval = 20
    
    try:
        with tqdm(files_to_process, desc="Processing files", dynamic_ncols=True) as pbar:
            for file_path in pbar:
                try:
                    results = voxelizer.process_mesh(file_path)
                    
                    if not results:
                        failed_list.append(file_path)
                        failed_count += 1
                        continue
                    
                    relative_path = Path(file_path).relative_to(input_dir)
                    save_dir = output_dir / relative_path.parent
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    for i, tensor in enumerate(results):
                        suffix = "" if i == 0 else f"_aug{i}"
                        torch.save(tensor, save_dir / f"{relative_path.stem}{suffix}.pt")
                    
                    processed_list.append(file_path)
                    processed_count += 1
                    
                except Exception as e:
                    failed_list.append(file_path)
                    failed_count += 1
                    logger.error(f"Failed to process {file_path}: {str(e)}")
                
                total_processed = processed_count + failed_count
                pbar.set_postfix({
                    'processed': f"{processed_count}/{total_processed}",
                    'success_rate': f"{(processed_count/max(total_processed, 1))*100:.1f}%"
                })
                
                if total_processed % gc_interval == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if total_processed % save_interval == 0:
                    with open(log_file, 'w') as f:
                        json.dump({
                            "processed": processed_list,
                            "permanent_failures": failed_list
                        }, f)
    
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    
    finally:
        with open(log_file, 'w') as f:
            json.dump({
                "processed": processed_list,
                "permanent_failures": failed_list
            }, f)
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Failed: {failed_count}")

if __name__ == "__main__":
    try:
        process_dataset(
            input_dir="/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data",
            output_dir="/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/objaverse_data_voxelized",
            resolution=32
        )
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        traceback.print_exc()