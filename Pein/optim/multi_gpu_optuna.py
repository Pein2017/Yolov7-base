import logging
from multiprocessing import Process, Queue

import optuna
from optuna.exceptions import DuplicatedStudyError

from optim.logging import OptunaExperimentLogger
from optim.objective import objective
from utils import Args
from utils.logging import worker_logging_configurer  # Updated import


def run_worker(gpu_id: int, n_trials: int, args: Args, log_queue: Queue):
    """
    Run Optuna optimization on a single GPU.

    Args:
        gpu_id (int): ID of the GPU to use.
        n_trials (int): Number of trials to run on this worker.
        args (Args): Configuration object.
        log_queue (Queue): Queue for logging.
    """
    # Configure logging for worker
    worker_logging_configurer(log_queue)

    # Assign GPU
    args.device_id = gpu_id

    experiment_logger = logging.getLogger("train_optuna")
    experiment_logger.info(f"Worker started on GPU {gpu_id}")

    # Load the study (it's already created or loaded in run_multi_gpu_optuna)
    try:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=args.mysql_url,
        )
    except Exception as e:
        experiment_logger.error(f"Failed to load study: {e}")
        return

    # Run the optimization with the enhanced logger
    study.optimize(
        lambda trial: objective(trial, args, args.study_name),
        n_trials=n_trials,
        callbacks=[
            OptunaExperimentLogger(
                min_improvement=0.001, start_logging_epoch=args.log_after_epoch
            )
        ],
    )

    experiment_logger.info(f"Worker on GPU {gpu_id} completed {n_trials} trials")


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
    experiment_logger = logging.getLogger("train_optuna")

    # Check and delete existing study at the beginning if required
    try:
        if args.delete_existing_study:
            optuna.delete_study(study_name=args.study_name, storage=args.mysql_url)
            experiment_logger.info(f"Existing study '{args.study_name}' deleted.")

        # Create the study
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.mysql_url,
            load_if_exists=not args.delete_existing_study,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
        )
        experiment_logger.info(
            f"Study '{args.study_name}' created or loaded successfully."
        )
    except DuplicatedStudyError:
        experiment_logger.info(
            f"Study '{args.study_name}' already exists and will be used."
        )
    except Exception as e:
        experiment_logger.error(f"Error in study setup: {str(e)}")
        return  # Exit if study setup fails

    # Determine trials per process
    total_procs = n_gpus * n_procs_per_gpu
    trials_per_proc = args.n_trials // total_procs
    remaining_trials = args.n_trials % total_procs

    processes = []
    for gpu in range(n_gpus):
        for proc in range(n_procs_per_gpu):
            # Distribute remaining trials
            n_trials = trials_per_proc + (1 if remaining_trials > 0 else 0)
            remaining_trials -= 1 if remaining_trials > 0 else 0

            # Start worker process
            p = Process(target=run_worker, args=(gpu, n_trials, args, log_queue))
            p.start()
            processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    experiment_logger.info("All workers have completed.")

    # Summarize results
    try:
        study = optuna.load_study(study_name=args.study_name, storage=args.mysql_url)
        best_trial = study.best_trial
        experiment_logger.info(f"Best trial: {best_trial.number}")
        experiment_logger.info(f"Best value: {best_trial.value}")
        experiment_logger.info(f"Best params: {best_trial.params}")
    except Exception as e:
        experiment_logger.error(f"Error loading study results: {str(e)}")
