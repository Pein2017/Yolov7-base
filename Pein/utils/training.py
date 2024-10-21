# Remove or relocate functions if they've been moved to `trainer/utils.py`

# Example: If `setup_trainer_instance` has been moved, remove it from here
# def setup_trainer_instance(...):
#     pass

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils import Args


def setup_trainer_instance(args: Args, study_name: str, exp_setting: str):
    """Set up and return a PyTorch Lightning Trainer instance."""
    log_dir = os.path.join(args.ltn_log_dir, study_name, "trial_logs", exp_setting)
    os.makedirs(log_dir, exist_ok=True)

    ltn_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=None,
        version="",
    )

    early_stop_callback = EarlyStopping(
        monitor=args.early_stop_metric,
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode=args.mode,
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        devices=[args.device_id],
        logger=ltn_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[early_stop_callback],
        enable_checkpointing=args.enable_checkpointing,
    )

    return trainer
