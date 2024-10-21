# scripts/train_optuna.py

import argparse
import logging
import os
import sys
from multiprocessing import Process, Queue

# Add the project root to sys.path to allow imports from other packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pytorch_lightning import seed_everything

from optim.multi_gpu_optuna import run_multi_gpu_optuna
from utils.config import Args
from utils.helpers import remove_directory_contents
from utils.logging import log_listener, worker_logging_configurer


def initialize_args() -> Args:
    """
    Initialize and return the Args object by loading configuration from a YAML file.

    Returns:
        Args: Configuration object populated from YAML.
    """
    config_path = "./configs/bbu_config.yaml"
    return Args.from_yaml(config_path)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Optuna Hyperparameter Optimization"
    )
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--n_procs_per_gpu", type=int, default=1, help="Number of processes per GPU"
    )
    cmd_args = parser.parse_args()

    # Initialize arguments
    args = initialize_args()

    # Set a global seed for reproducibility
    seed_everything(args.seed, workers=True)

    # Create a multiprocessing Queue for logging
    log_queue = Queue()

    # Set up global logger
    global_log_dir = os.path.join(args.ltn_log_dir, args.study_name)
    if args.delete_existing_study_folder and os.path.exists(global_log_dir):
        print(f"Attempting to remove existing study folder: {global_log_dir}")
        if remove_directory_contents(global_log_dir):
            print(f"Successfully removed contents of {global_log_dir}")
        else:
            print(
                f"Failed to remove contents of {global_log_dir}. Proceeding with existing directory."
            )

    os.makedirs(global_log_dir, exist_ok=True)
    global_log_file = os.path.join(global_log_dir, f"{args.study_name}.log")

    # Start the logging listener process
    listener = Process(target=log_listener, args=(log_queue, global_log_file))
    listener.start()

    # Configure worker processes to use the queue
    worker_logging_configurer(log_queue)

    try:
        # Run multi-GPU Optuna hyperparameter optimization
        run_multi_gpu_optuna(args, cmd_args.n_gpus, cmd_args.n_procs_per_gpu, log_queue)
    except Exception as e:
        logging.error(f"An error occurred during optimization: {str(e)}")
    finally:
        # Signal the logging process to finish and wait for it
        log_queue.put(None)
        listener.join()
        sys.exit(0)


if __name__ == "__main__":
    main()
