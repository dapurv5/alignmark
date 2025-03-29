import glob
import logging
import multiprocessing
import os

import fire
import numpy as np

from model_based_scorer import ModelBasedScorerRegistry
from utils import get_device_to_use

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set start method to spawn
multiprocessing.set_start_method("spawn", force=True)


def run_model_based_scorer(
    input_path: str,
    output_path: str,
    scorer_model: str = "llm-blender/PairRM",
    num_processes: int = 1,
    num_gpus_per_process: int = 0,
    gpu_start_id: int = 0,
    debug_mode: bool = False,
    text_field: str = "prompt",
    file_name_suffix: str = "_rewards",
    score_field_name: str = "reward_score",
):

    def process_single_file(
        input_path: str,
        output_path: str,
        text_field: str,
        num_processes: int,
        num_gpus_per_process: int,
        gpu_start_id: int,
    ):
        """Process a single input file, either directly or by splitting into parts for parallel processing"""
        output_dir = os.path.dirname(output_path)

        if num_processes == 1 and num_gpus_per_process == 0:
            scorer = ModelBasedScorerRegistry.get(scorer_model)(
                text_field=text_field,
                file_name_suffix=file_name_suffix,
                score_field_name=score_field_name,
                device="cpu",
            )
            scorer.compute_scores(input_path, output_path)
            return

        # Split file and process in parallel if using GPUs
        if num_processes == 1 and num_gpus_per_process > 0:
            process_file_in_parallel(
                input_path,
                output_dir,
                num_processes,
                num_gpus_per_process,
                gpu_start_id,
                scorer_model,
            )

    def process_file_in_parallel(
        input_path: str,
        output_dir: str,
        text_field: str,
        num_processes: int,
        num_gpus_per_process: int,
        gpu_start_id: int,
        scorer_model: str,
    ):
        """Split input file into parts and process them in parallel"""
        temp_dir = os.path.join(output_dir, "file_parts")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Split input file into parts
            with open(input_path, "r") as f:
                lines = f.readlines()
            lines_per_process = np.array_split(lines, num_processes)

            # Write temporary split files
            temp_files = []
            for i, lines_chunk in enumerate(lines_per_process):
                temp_file = os.path.join(temp_dir, f"temp_{i}.jsonl")
                with open(temp_file, "w") as f:
                    f.writelines(lines_chunk)
                temp_files.append(temp_file)

            # Process temporary files in parallel
            compute_parallel(
                temp_files,
                text_field,
                num_processes,
                num_gpus_per_process,
                scorer_model,
                output_dir,
                gpu_start_id,
                debug_mode,
                file_name_suffix,
                score_field_name,
            )

        finally:
            # Clean up temporary files
            for f in glob.glob(os.path.join(temp_dir, "*.jsonl")):
                os.remove(f)
            os.rmdir(temp_dir)

    def process_directory(
        input_path: str,
        output_path: str,
        text_field: str,
        num_processes: int,
        num_gpus_per_process: int,
        gpu_start_id: int,
        debug_mode: bool,
    ):
        """Process all JSONL files in a directory in parallel"""
        # If output_path does not exist, create it
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        filelist = glob.glob(os.path.join(input_path, "*.jsonl"))
        filelist = [f for f in filelist if not f.endswith(file_name_suffix + ".jsonl")]

        # Adjust number of processes if needed
        num_processes = min(num_processes, len(filelist))

        compute_parallel(
            filelist,
            text_field,
            num_processes,
            num_gpus_per_process,
            scorer_model,
            output_path,
            gpu_start_id,
            debug_mode,
            file_name_suffix,
            score_field_name,
        )

    # Main processing logic
    if os.path.isfile(input_path) and os.path.isfile(output_path):
        process_single_file(
            input_path,
            output_path,
            text_field,
            num_processes,
            num_gpus_per_process,
            gpu_start_id,
        )
    else:
        process_directory(
            input_path,
            output_path,
            text_field,
            num_processes,
            num_gpus_per_process,
            gpu_start_id,
            debug_mode,
        )


def process_func(
    files,
    text_field,
    gpu_ids_to_use: list[int],
    scorer_model: str,
    output_dir: str,
    device_to_use: str,
    file_name_suffix: str,
    score_field_name: str,
):
    try:
        process_id = os.getpid()
        print(f"Process {process_id} starting with GPU IDs {gpu_ids_to_use}")

        # Set CUDA environment variables before any CUDA operations
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(gpu_id) for gpu_id in gpu_ids_to_use]
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print(f"Process {process_id}: Loading model {scorer_model}")
        scorer = ModelBasedScorerRegistry.get(scorer_model)(
            text_field=text_field,
            file_name_suffix=file_name_suffix,
            score_field_name=score_field_name,
            device=device_to_use,
            gpu_ids=gpu_ids_to_use,
        )
        for input_file in files:
            logger.info(f"Process {process_id}: Processing {input_file}")
            input_filename = os.path.basename(input_file)
            output_filename = (
                os.path.splitext(input_filename)[0]
                + file_name_suffix
                + os.path.splitext(input_filename)[1]
            )
            output_filename = os.path.join(output_dir, output_filename)
            if not os.path.exists(output_filename) or (
                os.path.exists(output_filename)
                and sum(1 for _ in open(input_file))
                != sum(1 for _ in open(output_filename))
            ):
                scorer.compute_scores(input_file, output_filename)
                logger.info(f"Process {process_id}: Completed {input_file}")
    except Exception as e:
        logger.error(
            f"Error in process {process_id} with GPU IDs {gpu_ids_to_use}: {str(e)}"
        )
        raise e


def compute_parallel(
    filelist,
    text_field,
    num_processes,
    num_gpus_per_process,
    scorer_model,
    output_dir,
    gpu_start_id: int = 0,
    debug_mode: bool = False,
    file_name_suffix: str = "_rewards",
    score_field_name: str = "reward_score",
):
    files_to_process = np.array_split(filelist, num_processes)
    device_to_use = get_device_to_use(num_gpus_per_process, num_processes)

    if debug_mode:
        print("Running in debug mode (single process)...")
        for process_idx in range(num_processes):
            files_to_process_for_this_process = files_to_process[process_idx]
            gpu_ids_for_this_process = [
                gpu_start_id + process_idx * num_gpus_per_process + gpu_id
                for gpu_id in range(num_gpus_per_process)
            ]
            process_func(
                files_to_process_for_this_process,
                text_field,
                gpu_ids_for_this_process,
                scorer_model,
                output_dir,
                device_to_use,
                file_name_suffix,
                score_field_name,
            )
    else:
        print(f"Launching {num_processes} processes...")
        processes = []
        for process_idx in range(num_processes):
            files_to_process_for_this_process = files_to_process[process_idx]
            gpu_ids_for_this_process = [
                gpu_start_id + process_idx * num_gpus_per_process + gpu_id
                for gpu_id in range(num_gpus_per_process)
            ]

            p = multiprocessing.Process(
                target=process_func,
                args=(
                    files_to_process_for_this_process,
                    text_field,
                    gpu_ids_for_this_process,
                    scorer_model,
                    output_dir,
                    device_to_use,
                    file_name_suffix,
                    score_field_name,
                ),
            )
            p.daemon = False  # Ensure process isn't daemonic
            print(
                f"Starting process {process_idx} for GPU(s) {gpu_ids_for_this_process}"
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for idx, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Process {idx} failed with exit code {p.exitcode}")
            print(f"Process {idx} completed successfully")


if __name__ == "__main__":
    fire.Fire(run_model_based_scorer)
