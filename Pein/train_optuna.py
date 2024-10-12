import argparse
import logging
import os
import time
from multiprocessing import Process, Queue
from typing import Any, Dict

import optuna
import pytorch_lightning as pl
from optuna.exceptions import DuplicatedStudyError
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from pytorch_lightning import seed_everything

from dataset.data_factory import get_dataloader
from model.classifier_2 import DetectionClassificationModel
from trainer.bbu_ltn import DetectLtnModule
from trainer.utils import (
    Args,
    log_new_best,
    setup_global_logger,
    setup_trainer,
    worker_logging_configurer,
)

# ================================================================
# Argument Initialization
# ================================================================


def initialize_args() -> Args:
    """
    Initialize and return the Args object with fixed parameters.

    Returns:
        Args: Configuration object containing fixed parameters.
    """
    return Args(
        # Data-related parameters
        pos_dir="/data/training_code/yolov7/runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3/pos/labels",
        neg_dir="/data/training_code/yolov7/runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3/neg/labels",
        classes_file="/data/dataset/bbu_training_data/bbu_and_shield/classes.txt",
        balance_ratio=1.0,
        resample_method="both",  # Fixed to 'both'. 'downsample' for debugging
        # Training parameters
        num_epochs=40,
        valid_epoch=20,
        log_after_epoch=15,
        batch_size=4096,
        lr=1e-3,  # Initial learning rate; overridden by Optuna
        optimizer="Adam",  # Fixed to Adam
        gradient_clip_val=0.1,
        scheduler_type="onecycle",
        # Model architecture parameters
        num_layers=1,  # Fixed to 1
        attn_heads=2,
        dropout=0.1,
        embedding_dim=16,
        fc_hidden_dims=[32, 16],
        # Evaluation and optimization parameters
        selection_criterion="avg_error",  # For threshold optimization
        early_stop_metric="val/loss",  # For early stopping
        patience=10,
        mode="min",
        # Hardware and environment parameters
        device_id=0,
        seed=17,
        # Logging and storage parameters
        ltn_log_dir="./ltn_logs",
        mysql_url="mysql://root:@localhost:3306/optuna_db",
        # Optuna-specific parameters
        delete_existing_study=True,  # Control study deletion
        n_trials=500,  # Number of Optuna trials
        study_name="24-10-10_with_tpe",  # Name of the Optuna study
        sampler="tpe",  # Options: "tpe", "random", "cmaes", "nsgaii", "qmc"
        pruner="median",  # Options: "median", "nop", "hyperband", "threshold"
    )


# ================================================================
# Hyperparameter Search Space Definition
# ================================================================


def define_search_space(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """
    Define the hyperparameter search space for Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial object for hyperparameter suggestions.

    Returns:
        Dict[str, Any]: Dictionary of suggested hyperparameters.
    """
    search_space = {
        "optimizer": "Adam",  # Fixed to Adam as per user's requirement
        "lr": trial.suggest_float("lr", 5e-4, 2e-2, log=True),
        "embedding_dim": trial.suggest_int(
            "embedding_dim", 16, high=128, step=16
        ),  # Flexible range
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2, 4, 8]),
        "attn_heads": trial.suggest_int("attn_heads", 1, 64),  # Flexible range
        "dropout": 0.1,  # Fixed dropout as per user's setting
        "fc_hidden_dims_2": trial.suggest_categorical(
            "fc_hidden_dims_2", [16, 32, 64, 128]
        ),
    }

    # Set fc_hidden_dims_1 to be twice fc_hidden_dims_2
    search_space["fc_hidden_dims_1"] = search_space["fc_hidden_dims_2"] * 2

    # Combine fc_hidden_dims into a list
    search_space["fc_hidden_dims"] = [
        search_space["fc_hidden_dims_1"],
        search_space["fc_hidden_dims_2"],
    ]

    return search_space


# ================================================================
# Experimental Setting Generation
# ================================================================


def generate_exp_setting(args: Args, search_space: Dict[str, Any]) -> str:
    """
    Generate a shorthand string representing the hyperparameter settings.

    Args:
        args (Args): Configuration object.
        search_space (Dict[str, Any]): Dictionary of hyperparameters and their suggested values.

    Returns:
        str: Shorthand string.
    """
    exp_settings = []
    # List of hyperparameters to include in the exp_setting with their shorthand names
    keys = [
        ("lr", "lr"),
        ("hidden_dim", "hid_dim"),
        ("attn_heads", "attn_h"),
        ("dropout", "drop"),
        ("fc_hidden_dims_1", "fc_hid1"),
        ("fc_hidden_dims_2", "fc_hid2"),
        ("embedding_dim", "emb_dim"),
    ]
    for long_name, short_name in keys:
        value = search_space.get(long_name, getattr(args, long_name, None))
        if isinstance(value, float):
            if long_name == "lr":
                value = f"{value:.4f}"
            elif long_name == "dropout":
                value = f"{value:.2f}"
        exp_settings.append(f"{short_name}_{value}")
    return "-".join(exp_settings)


# ================================================================
# Model Creation
# ================================================================


def create_model(args: Args, trial: optuna.trial.Trial) -> DetectionClassificationModel:
    """
    Create and return the DetectionClassificationModel with hyperparameters suggested by Optuna.

    Args:
        args (Args): Configuration object.
        trial (optuna.trial.Trial): Optuna trial object for hyperparameter suggestions.

    Returns:
        DetectionClassificationModel: Configured model instance.
    """
    search_space = define_search_space(trial)

    num_classes = args.train_loader.dataset.num_classes
    num_labels = args.train_loader.dataset.num_labels
    PAD_LABEL = args.train_loader.dataset.PAD_LABEL
    SEP_LABEL = args.train_loader.dataset.SEP_LABEL

    embedding_dim = search_space["embedding_dim"]
    suggested_attn_heads = search_space["attn_heads"]

    # Adjust attn_heads to ensure it divides embedding_dim
    def find_largest_divisor(n, max_val):
        # Start from max_val and find the largest divisor <= max_val
        for i in range(max_val, 0, -1):
            if n % i == 0:
                return i
        return 1  # Fallback to 1 if no divisor found

    attn_heads = find_largest_divisor(embedding_dim, suggested_attn_heads)

    model = DetectionClassificationModel(
        num_classes=num_classes,
        num_labels=num_labels,
        PAD_LABEL=PAD_LABEL,
        SEP_LABEL=SEP_LABEL,
        embedding_dim=embedding_dim,
        hidden_dim=search_space["hidden_dim"],
        attn_heads=attn_heads,  # Use the adjusted attn_heads
        fc_hidden_dims=search_space["fc_hidden_dims"],
        num_layers=search_space["num_layers"],
        dropout=search_space["dropout"],
    )

    # Log the adjustment if it was made
    if attn_heads != suggested_attn_heads:
        logging.getLogger("train_optuna").info(
            f"Adjusted attn_heads from {suggested_attn_heads} to {attn_heads} "
            f"to be compatible with embedding_dim {embedding_dim}"
        )

    return model


# ================================================================
# Module Setup
# ================================================================


def setup_module(args: Args, model: DetectionClassificationModel) -> DetectLtnModule:
    """
    Initialize the DetectLtnModule with the given args and model.

    Args:
        args (Args): Configuration object.
        model (DetectionClassificationModel): The model instance.

    Returns:
        DetectLtnModule: Configured LightningModule.
    """
    return DetectLtnModule(args, model)


# ================================================================
# Trainer Setup
# ================================================================


def setup_trainer_instance(
    args: Args, trial_number: int, study_name: str, exp_setting: str
) -> pl.Trainer:
    """
    Set up and return the PyTorch Lightning Trainer instance.

    Args:
        args (Args): Configuration object.
        trial_number (int): Current trial number for logging.
        study_name (str): Name of the Optuna study.
        exp_setting (str): String representing the experimental settings.

    Returns:
        pl.Trainer: Configured Trainer instance.
    """
    # Incorporate study_name and exp_setting into the log directory
    log_dir = os.path.join(args.ltn_log_dir, study_name, exp_setting)

    # Initialize trainer using the setup_trainer function from utils
    return setup_trainer(
        log_dir=log_dir,
        max_epochs=args.num_epochs,
        device_id=args.device_id,
        early_stop_metric=args.early_stop_metric,
        gradient_clip_val=args.gradient_clip_val,
        patience=args.patience,
        mode=args.mode,
        additional_callbacks=None,
    )


# ================================================================
# Optuna Experiment Logger
# ================================================================


class OptunaExperimentLogger:
    """
    Callback class to log when new best val or val_cheat metrics are found after a specified epoch.
    """

    def __init__(self, min_improvement: float = 0.001, start_logging_epoch: int = 10):
        """
        Initialize the logger with a minimum improvement threshold and logging start epoch.

        Args:
            min_improvement (float, optional): Minimum required improvement to log. Defaults to 0.001.
            start_logging_epoch (int, optional): Epoch number after which logging is enabled. Defaults to 10.
        """
        self.best_val = float("inf")
        self.best_val_cheat = float("inf")
        self.min_improvement = min_improvement
        self.start_logging_epoch = start_logging_epoch

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return  # Skip trials that didn't complete successfully

        best_epoch = trial.user_attrs.get("best_epoch")
        if best_epoch is None:
            logging.getLogger("train_optuna").warning(
                f"Trial {trial.number} has no 'best_epoch' attribute."
            )
            return

        selection_criterion = trial.user_attrs.get(
            "selection_criterion"
        ) or study.best_trial.params.get("selection_criterion", "avg_error")
        val_metric = trial.user_attrs.get(selection_criterion)
        val_cheat_metric = trial.user_attrs.get(f"val_cheat_{selection_criterion}")

        # Ensure logging only after the specified log_after_epoch
        if best_epoch < self.start_logging_epoch:
            return  # Skip logging before start_logging_epoch

        # Log val metric if improved
        if val_metric is not None and val_metric < self.best_val:
            self.best_val = val_metric
            fp_over_tp_fp_val = trial.user_attrs["fp_over_tp_fp"]
            fn_over_fn_tn_val = trial.user_attrs["fn_over_fn_tn"]
            log_new_best(
                logger_name="train_optuna",
                metric_type="val",
                metric_value=val_metric,
                epoch=best_epoch,
                fp_over=fp_over_tp_fp_val,
                fn_over=fn_over_fn_tn_val,
                selection_criterion=selection_criterion,
            )

        # Log val_cheat metric if improved
        if val_cheat_metric is not None and val_cheat_metric < self.best_val_cheat:
            self.best_val_cheat = val_cheat_metric
            fp_over_tp_fp_val_cheat = trial.user_attrs["val_cheat_fp_over_tp_fp"]
            fn_over_fn_tn_val_cheat = trial.user_attrs["val_cheat_fn_over_fn_tn"]
            log_new_best(
                logger_name="train_optuna",
                metric_type="val_cheat",
                metric_value=val_cheat_metric,
                epoch=best_epoch,
                fp_over=fp_over_tp_fp_val_cheat,
                fn_over=fn_over_fn_tn_val_cheat,
                selection_criterion=selection_criterion,
            )


# ================================================================
# Objective Function
# ================================================================


def objective(trial: optuna.trial.Trial, args: Args, study_name: str) -> float:
    """
    Objective function for Optuna to optimize.

    Args:
        trial (optuna.trial.Trial): Optuna trial object for hyperparameter suggestions.
        args (Args): Configuration object.
        study_name (str): Name of the Optuna study.

    Returns:
        float: Metric value to optimize.
    """
    start_time = time.time()

    # Suggest hyperparameters
    search_space = define_search_space(trial)

    # Load data with suggested hyperparameters
    dataloaders = get_dataloader(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        classes_file=args.classes_file,
        split=[0.8, 0.2],
        batch_size=args.batch_size,
        balance_ratio=args.balance_ratio,
        resample_method=args.resample_method,
        seed=args.seed,
    )
    args.train_loader = dataloaders["train"]
    args.val_loader = dataloaders["val"]

    # Update args with suggested hyperparameters
    args.lr = search_space["lr"]
    args.optimizer = search_space[
        "optimizer"
    ]  # Ensuring optimizer is up-to-date if not fixed
    args.num_layers = search_space["num_layers"]

    # Set selection_criterion from search space or existing args
    selection_criterion = args.selection_criterion or search_space.get(
        "selection_criterion", "avg_error"
    )

    # Create model with suggested hyperparameters
    model = create_model(args, trial)

    # Initialize DetectLtnModule
    detect_ltn_module = setup_module(args, model)

    # Generate experimental setting string
    exp_setting = generate_exp_setting(args, search_space)

    # Set up trainer
    trainer = setup_trainer_instance(args, trial.number, study_name, exp_setting)

    # Log trial start
    current_trial = trial.number + 1
    total_trials = args.n_trials
    logging.getLogger("train_optuna").info(
        f"Starting Trial {current_trial}/{total_trials}"
    )

    best_metric = float("inf")
    best_epoch = None
    logger = logging.getLogger("train_optuna")

    for epoch in range(args.num_epochs):
        trainer.fit(detect_ltn_module, args.train_loader, args.val_loader)

        # Get the current metrics
        current_val_metric = detect_ltn_module.best_metrics["val"].get(
            selection_criterion
        )
        current_val_fp_over = detect_ltn_module.best_metrics["val"].get("fp_over_tp_fp")
        current_val_fn_over = detect_ltn_module.best_metrics["val"].get("fn_over_fn_tn")
        current_val_cheat_metric = detect_ltn_module.best_metrics["val_cheat"].get(
            selection_criterion
        )
        current_val_cheat_fp_over = detect_ltn_module.best_metrics["val_cheat"].get(
            "fp_over_tp_fp"
        )
        current_val_cheat_fn_over = detect_ltn_module.best_metrics["val_cheat"].get(
            "fn_over_fn_tn"
        )

        # Only consider metrics after valid_epoch
        if epoch + 1 >= args.valid_epoch:
            if current_val_metric is not None and current_val_cheat_metric is not None:
                trial.report(current_val_metric, epoch)

                # Handle pruning based on the intermediate value
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at epoch {epoch + 1}")
                    raise optuna.TrialPruned()

                # Update best_metric and best_epoch
                if current_val_metric < best_metric:
                    best_metric = current_val_metric
                    best_epoch = epoch + 1

                    # Set user attributes
                    trial.set_user_attr("best_epoch", best_epoch)
                    trial.set_user_attr("selection_criterion", selection_criterion)
                    trial.set_user_attr(selection_criterion, best_metric)
                    trial.set_user_attr("fp_over_tp_fp", current_val_fp_over)
                    trial.set_user_attr("fn_over_fn_tn", current_val_fn_over)
                    trial.set_user_attr(
                        f"val_cheat_{selection_criterion}", current_val_cheat_metric
                    )
                    trial.set_user_attr(
                        "val_cheat_fp_over_tp_fp", current_val_cheat_fp_over
                    )
                    trial.set_user_attr(
                        "val_cheat_fn_over_fn_tn", current_val_cheat_fn_over
                    )

                # Log info after log_after_epoch
                if epoch + 1 >= args.log_after_epoch:
                    logger.info(
                        f"Epoch {epoch + 1}: "
                        f"val_{selection_criterion} = {current_val_metric:.4f}, "
                        f"val_fp_over = {current_val_fp_over:.4f}, "
                        f"val_fn_over = {current_val_fn_over:.4f}, "
                        f"val_cheat_{selection_criterion} = {current_val_cheat_metric:.4f}, "
                        f"val_cheat_fp_over = {current_val_cheat_fp_over:.4f}, "
                        f"val_cheat_fn_over = {current_val_cheat_fn_over:.4f}"
                    )
            else:
                logger.warning(f"No valid metric found at epoch {epoch + 1}")

    # Calculate duration
    duration = time.time() - start_time

    # Log trial completion
    if best_epoch is not None:
        logger.info(
            f"Completed Trial {trial.number + 1}/{args.n_trials} in {duration:.2f} seconds "
            f"with {selection_criterion}: {best_metric:.4f} at epoch {best_epoch}"
        )
    else:
        logger.info(
            f"Completed Trial {trial.number + 1}/{args.n_trials} in {duration:.2f} seconds "
            f"but did not reach valid epoch {args.valid_epoch}"
        )

    return best_metric if best_metric != float("inf") else float("inf")


# ================================================================
# Worker Function
# ================================================================


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


# ================================================================
# Multi-GPU Optuna Optimization Runner
# ================================================================


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


# ================================================================
# Main Execution
# ================================================================

if __name__ == "__main__":
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

    # Set up logging queue and listener
    log_queue = Queue()
    listener = setup_global_logger(log_queue)

    try:
        # Run multi-GPU Optuna hyperparameter optimization
        run_multi_gpu_optuna(args, cmd_args.n_gpus, cmd_args.n_procs_per_gpu, log_queue)
    finally:
        listener.stop()

    listener = setup_global_logger(log_queue)

    try:
        # Run multi-GPU Optuna hyperparameter optimization
        run_multi_gpu_optuna(args, cmd_args.n_gpus, cmd_args.n_procs_per_gpu, log_queue)
    finally:
        listener.stop()
