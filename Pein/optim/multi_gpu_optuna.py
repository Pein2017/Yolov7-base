import logging

# Add import for logging listener
import logging.handlers
import os
import time
from dataclasses import dataclass
from multiprocessing import Lock, Manager, Process, Queue
from typing import Any, Dict

import optuna
from optuna.exceptions import DuplicatedStudyError

from optim.logging import OptunaExperimentLogger
from optim.objective import objective
from utils import Args
from utils.logging import worker_logging_configurer


@dataclass
class WorkerConfig:
    gpu_id: int
    proc_id: int
    n_trials: int
    args: Args
    shared_best_metrics: Dict[str, float]
    lock: Any


def run_worker(config: WorkerConfig, log_queue: Queue):
    """
    Run Optuna optimization on a single GPU.

    Args:
        config (WorkerConfig): Configuration for the worker.
        log_queue (Queue): Queue for logging.
    """
    # Configure logging for worker
    worker_logging_configurer(log_queue)

    # Get the global logger
    global_logger = logging.getLogger("global_logger")

    # Assign GPU
    config.args.device_id = config.gpu_id

    try:
        study = optuna.load_study(
            study_name=config.args.study_name,
            storage=config.args.mysql_url,
        )
    except Exception as e:
        global_logger.error(f"Failed to load study: {e}")
        return

    # Create an OptunaExperimentLogger instance with shared best metrics and lock
    experiment_logger = OptunaExperimentLogger(
        config.proc_id, config.shared_best_metrics, config.lock, config.args.study_name
    )

    # Run the optimization with the enhanced logger
    study.optimize(
        lambda trial: objective(trial, config.args, config.args.study_name),
        n_trials=config.n_trials,
        callbacks=[experiment_logger],
    )


def configure_listener(log_queue: Queue, global_log_dir: str, study_name: str):
    """
    Configure the logging listener to handle log records from all worker processes.
    """
    log_file = os.path.join(global_log_dir, f"{study_name}.log")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    listener = logging.handlers.QueueListener(log_queue, handler)
    listener.start()
    return listener


def run_multi_gpu_optuna(
    args: Args, n_gpus: int, n_procs_per_gpu: int, log_queue: Queue
):
    """
    Run Optuna optimization across multiple GPUs and processes.

    Args:
        args (Args): Configuration object.
        n_gpus (int): Number of GPUs to use.
        n_procs_per_gpu (int): Number of processes to run per GPU.
        log_queue (Queue): Queue for logging.
    """
    global_logger = logging.getLogger("global_logger")

    # Utilize the global_log_dir from args
    global_log_dir = os.path.join(args.ltn_log_dir, args.study_name)
    os.makedirs(global_log_dir, exist_ok=True)  # Single directory creation

    # Configure and start the logging listener with global_log_dir
    listener = configure_listener(log_queue, global_log_dir, args.study_name)

    # Initialize Manager for shared best metrics and lock
    manager = Manager()
    shared_best_metrics = manager.dict(
        {
            "val": float("inf") if args.direction == "minimize" else -float("inf"),
            "val_epoch": -1,
            "val_cheat": float("inf")
            if args.direction == "minimize"
            else -float("inf"),
            "val_cheat_epoch": -1,
        }
    )
    lock = Lock()

    # Check and delete existing study at the beginning if required
    if args.delete_existing_study:
        try:
            optuna.delete_study(study_name=args.study_name, storage=args.mysql_url)
            global_logger.info(f"Existing study '{args.study_name}' deleted.")
        except KeyError:
            # Study doesn't exist, so no need to delete
            global_logger.warning(
                f"Study '{args.study_name}' does not exist. Skipping deletion."
            )

    # Create the study
    try:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.mysql_url,
            load_if_exists=not args.delete_existing_study,
            direction=args.direction,  # Use the direction from args
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
        )
        global_logger.info(f"Study '{args.study_name}' created or loaded successfully.")
    except DuplicatedStudyError:
        global_logger.info(
            f"Study '{args.study_name}' already exists and will be used."
        )
        study = optuna.load_study(study_name=args.study_name, storage=args.mysql_url)
    except Exception as e:
        global_logger.error(f"Error in study setup: {str(e)}")
        return  # Exit if study setup fails

    # Set selection_criterion as a study attribute
    study.set_user_attr("selection_criterion", args.selection_criterion)

    global_logger.info(
        f"Starting optimization with {args.n_trials} trials across {n_gpus} GPUs and {n_procs_per_gpu} processes per GPU."
    )

    start_time = time.time()  # Record start time

    # Determine trials per process
    total_procs = n_gpus * n_procs_per_gpu
    trials_per_proc = args.n_trials // total_procs
    remaining_trials = args.n_trials % total_procs

    processes = []
    proc_id = 0
    for gpu in range(n_gpus):
        for _ in range(n_procs_per_gpu):
            if remaining_trials > 0:
                n_trials = trials_per_proc + 1
                remaining_trials -= 1
            else:
                n_trials = trials_per_proc

            worker_config = WorkerConfig(
                gpu_id=gpu,
                proc_id=proc_id,
                n_trials=n_trials,
                args=args,
                shared_best_metrics=shared_best_metrics,
                lock=lock,
            )

            p = Process(
                target=run_worker,
                args=(worker_config, log_queue),
            )
            p.start()
            processes.append(p)
            proc_id += 1

    # Wait for all processes to finish
    for p in processes:
        p.join()

    end_time = time.time()  # Record end time
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, _ = divmod(remainder, 60)

    global_logger.info("All workers have completed.")
    global_logger.info(
        f"Total time for all trials: {int(hours)} hours: {int(minutes)} minutes"
    )

    # Stop the logging listener
    listener.stop()


def main():
    # Initialize Args instance with appropriate parameters
    args = Args(
        study_name="your_study_name",
        mysql_url="mysql://username:password@host:port/dbname",
        n_trials=6,
        delete_existing_study=False,
        selection_criterion="avg_error",
        device_id=None,  # Will be set by worker
        ltn_log_dir="ltn_logs",
        pos_dir="path_to_positive_data",
        neg_dir="path_to_negative_data",
        classes_file="path_to_classes_file",
        batch_size=32,
        balance_ratio=1.0,
        resample_method="none",
        seed=42,
        num_classes=10,
        # Add other necessary attributes
    )

    n_gpus = 2
    n_procs_per_gpu = 2

    # Set up logging queue if using a logging handler that requires it
    log_queue = Queue()

    # Run multi-GPU Optuna optimization
    run_multi_gpu_optuna(args, n_gpus, n_procs_per_gpu, log_queue)


if __name__ == "__main__":
    main()
