import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from typing import Any, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def setup_trainer(
    log_dir: str,
    max_epochs: int,
    device_id: int,
    early_stop_metric: str,
    gradient_clip_val: float = 0.1,
    patience: int = 10,
    mode: str = "min",
    additional_callbacks: Optional[List] = None,
) -> pl.Trainer:
    """Set up and return a PyTorch Lightning Trainer with all necessary components."""
    ltn_logger = TensorBoardLogger(save_dir=log_dir, name=None, version=None)

    # Set up callbacks
    callbacks = []

    # EarlyStopping callback
    early_stop_callback = EarlyStopping(
        monitor=early_stop_metric,
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode=mode,
    )
    callbacks.append(early_stop_callback)

    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=False,
        monitor=early_stop_metric,
        mode=mode,
    )
    callbacks.append(checkpoint_callback)

    # Add any additional callbacks
    if additional_callbacks:
        callbacks.extend(additional_callbacks)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=[device_id],
        logger=ltn_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=0,
    )

    return trainer


def setup_global_logger(log_queue: Queue) -> QueueListener:
    """Set up the global logger to handle logs from all processes."""
    global_logger = logging.getLogger()
    global_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # File Handler (overwrite mode)
    file_handler = logging.FileHandler(
        "train_optuna.log",
        mode="w",  # Overwrite log file each run
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # Listener will handle logs from the queue and dispatch to handlers
    listener = QueueListener(log_queue, console_handler, file_handler)
    listener.start()
    return listener


def worker_logging_configurer(log_queue: Queue):
    """Configure logging for worker processes to send logs to the queue."""
    queue_handler = QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove all handlers associated with the logger to prevent duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.addHandler(queue_handler)


def log_new_best(
    logger_name: str,
    metric_type: str,
    metric_value: float,
    epoch: int,
    fp_over: Any,
    fn_over: Any,
    selection_criterion: str,
):
    """
    Helper function to log new best metrics with additional information.

    Args:
        logger_name (str): Name of the logger.
        metric_type (str): Type of the metric (e.g., 'val', 'val_cheat').
        metric_value (float): Value of the metric.
        epoch (int): Epoch number where the metric was achieved.
        fp_over (Any): Value of fp_over_tp_fp.
        fn_over (Any): Value of fn_over_fn_tn.
        selection_criterion (str): The criterion used for selection.
    """
    logger = logging.getLogger(logger_name)
    logger.info(
        f"New best {metric_type}/{selection_criterion}: {metric_value:.4f} at epoch {epoch}, "
        f"fp_over_tp_fp: {fp_over:.4f}, fn_over_fn_tn: {fn_over:.4f}"
    )
