# scripts/train_optuna.py

import argparse

from pytorch_lightning import seed_everything

from optim.multi_gpu_optuna import run_multi_gpu_optuna
from utils.config import Args
from utils.logging import setup_global_logger


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

    # Set up global logger
    setup_global_logger(
        name="train_optuna",
        log_file=args.ltn_log_dir + "/train_optuna.log",
        level="info",
    )

    try:
        # Run multi-GPU Optuna hyperparameter optimization
        run_multi_gpu_optuna(args, cmd_args.n_gpus, cmd_args.n_procs_per_gpu)
    finally:
        # If you have any cleanup, handle it here
        pass


if __name__ == "__main__":
    main()
