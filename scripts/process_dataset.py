# blockgen/scripts/process_dataset.py
# This script processes a dataset of 3D models in GLB format by voxelizing them and saving the results.

from pathlib import Path  # Import Path before using it
import json
import traceback
import torch
import gc
import logging
from tqdm import tqdm
import warnings
import signal
from contextlib import contextmanager

# Silence warnings
warnings.filterwarnings("ignore")
import torchvision

torchvision.disable_beta_transforms_warning()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import and create global voxelizer
from blockgen.data.processing.data_voxelization import VoxelizerWithAugmentation

VOXELIZER = None


# Timeout context manager
class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    def _handle_timeout(signum, frame):
        raise TimeoutException("Timeout exceeded during processing")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def process_single_file(glb_file: Path, input_dir: Path, output_dir: Path, resolution: int) -> tuple[bool, str]:
    """Process a single GLB file with detailed logging and memory management."""
    global VOXELIZER
    results = None

    try:
        logger.debug(f"Starting to process {glb_file}")

        # Initialize voxelizer if needed
        if VOXELIZER is None:
            logger.debug("Creating voxelizer instance...")
            VOXELIZER = VoxelizerWithAugmentation(resolution=resolution)

        # Process mesh with timeout
        with timeout(60):  # Timeout after 60 seconds
            results = VOXELIZER.process_mesh(str(glb_file))

        if not results:
            return False, "No results generated"

        # Save results
        logger.debug("Saving results...")
        relative_path = glb_file.relative_to(input_dir)
        save_dir = output_dir / relative_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, tensor in enumerate(results):
            suffix = "" if i == 0 else f"_aug{i}"
            save_path = save_dir / f"{relative_path.stem}{suffix}.pt"
            torch.save(tensor, save_path)
            del tensor  # Free memory

        return True, None

    except TimeoutException as e:
        logger.error(f"Timeout processing file: {glb_file}")
        return False, str(e)
    except Exception as e:
        logger.error(f"Error in process_single_file: {traceback.format_exc()}")
        return False, str(e)
    finally:
        if results is not None:
            del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_processing_log(log_file: Path) -> tuple[dict, set, set]:
    """Load or initialize the processing log with error handling."""
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                log_data.setdefault("processed", [])
                log_data.setdefault("permanent_failures", [])
                return log_data, set(log_data["processed"]), set(log_data["permanent_failures"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading log file: {e}. Starting fresh.")
    return {"processed": [], "permanent_failures": []}, set(), set()


def save_log(log_file: Path, log_data: dict):
    """Save log with error handling."""
    try:
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving log: {e}")


def process_dataset(input_dir: str, output_dir: str, resolution: int = 32):
    """Process dataset with improved memory management and logging."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "processing_log.json"

    try:
        # Load log
        logger.info("Loading processing log...")
        log_data, processed_set, permanent_failures_set = load_processing_log(log_file)

        # Stream and filter files
        logger.info("Finding files to process...")
        glb_files = []
        total_files = 0
        skipped_large = 0

        for glb_file in tqdm(input_dir.rglob("*.glb"), desc="Scanning files"):
            total_files += 1
            file_str = str(glb_file)

            if file_str not in processed_set and file_str not in permanent_failures_set:
                try:
                    if glb_file.stat().st_size > 50 * 1024 * 1024:
                        logger.debug(f"Skipping large file: {file_str}")
                        permanent_failures_set.add(file_str)
                        log_data["permanent_failures"].append(file_str)
                        skipped_large += 1
                    else:
                        glb_files.append(glb_file)
                except Exception as e:
                    logger.error(f"Error accessing file {file_str}: {e}")
                    continue

        logger.info(f"Total files found: {total_files}")
        logger.info(f"Already processed: {len(processed_set)}")
        logger.info(f"Previously failed: {len(permanent_failures_set)}")
        logger.info(f"Skipped large files: {skipped_large}")
        logger.info(f"Files to process: {len(glb_files)}")

        # Save initial state
        save_log(log_file, log_data)

        # Process files
        processed_count = failed_count = 0

        for i, glb_file in enumerate(tqdm(glb_files, desc="Processing files")):
            file_str = str(glb_file)
            logger.info(f"Processing {i + 1}/{len(glb_files)}: {file_str}")

            try:
                success, error = process_single_file(glb_file, input_dir, output_dir, resolution)

                if success:
                    processed_set.add(file_str)
                    log_data["processed"].append(file_str)
                    processed_count += 1
                else:
                    permanent_failures_set.add(file_str)
                    log_data["permanent_failures"].append(file_str)
                    failed_count += 1
                    logger.error(f"Failed to process {file_str}: {error}")

                # Save log periodically
                if (i + 1) % 2 == 0:
                    save_log(log_file, log_data)

            except Exception as e:
                logger.error(f"Error processing {file_str}: {e}")
                permanent_failures_set.add(file_str)
                log_data["permanent_failures"].append(file_str)
                failed_count += 1

            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        save_log(log_file, log_data)
        logger.info("\nProcessing Summary:")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Failed: {failed_count}")


if __name__ == "__main__":
    # Change this to objaverse_data or objaverse_finetune
    input_dir = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/evaluation_set"
    output_dir = "/scratch/students/2024-fall-sp-pabdel/3D-BlockGen/evaluation_set_voxelized"

    try:
        process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            resolution=32
        )
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}")
        traceback.print_exc()
