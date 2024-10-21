import logging
import os
import time

from optuna.trial import Trial

from data.data_factory import get_dataloader
from models.classifier_2 import DetectionClassificationModel
from optim.search_space import define_search_space
from trainer.bbu_ltn import DetectLtnModule
from utils import Args
from utils.helpers import generate_exp_setting
from utils.logging import setup_logger
from utils.training import setup_trainer_instance


def objective(trial: Trial, args: Args, study_name: str) -> float:
    """
    Objective function for Optuna to optimize.

    Args:
        trial (optuna.trial.Trial): Optuna trial object for hyperparameter suggestions.
        args (Args): Configuration object.
        study_name (str): Name of the Optuna study.

    Returns:
        float: Metric value to optimize.
    """
    # Get the global logger
    global_logger = logging.getLogger("global_logger")

    start_time = time.time()  # Record start time

    # Suggest hyperparameters
    search_space = define_search_space(trial, args.direction)

    # Generate experimental setting string
    exp_setting = generate_exp_setting(args, search_space)

    # Set up logger for this trial
    trial_log_dir = os.path.join(
        args.ltn_log_dir, study_name, "trial_logs", exp_setting
    )
    os.makedirs(trial_log_dir, exist_ok=True)
    trial_log_file = os.path.join(trial_log_dir, f"trial_{trial.number}.log")
    trial_logger = setup_logger(
        log_file=trial_log_file,
        name=f"trial_{trial.number}",
        level="INFO",
        log_to_console=False,
        overwrite=True,
    )

    # Disable propagation to prevent trial logs from appearing in the global log
    trial_logger.propagate = False

    # Log trial start with 1-based indexing and current/total format using trial_logger
    trial_logger.info(f"Starting Trial {trial.number + 1}/{args.n_trials}")

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
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Update args with suggested hyperparameters
    args.lr = search_space["lr"]
    args.optimizer = search_space["optimizer"]
    args.num_layers = search_space["num_layers"]

    if args.selection_criterion is None:
        raise ValueError("selection_criterion is not set")

    selection_criterion = args.selection_criterion

    # Set selection_criterion as a trial user attribute
    trial.set_user_attr("selection_criterion", selection_criterion)

    # Adjust attn_heads to ensure it divides embedding_dim
    embedding_dim = search_space["embedding_dim"]
    suggested_attn_heads = search_space["attn_heads"]

    def find_largest_divisor(n, max_val):
        for i in range(max_val, 0, -1):
            if n % i == 0:
                return i
        return 1  # Fallback to 1 if no divisor found

    attn_heads = find_largest_divisor(embedding_dim, suggested_attn_heads)

    # Update search_space with adjusted attn_heads
    search_space["attn_heads"] = attn_heads

    # Create model with suggested hyperparameters
    model = DetectionClassificationModel(
        num_classes=args.num_classes,
        embedding_dim=embedding_dim,
        hidden_dim=search_space["hidden_dim"],
        attn_heads=attn_heads,
        fc_hidden_dims=search_space["fc_hidden_dims"],
        num_layers=search_space["num_layers"],
        dropout=search_space["dropout"],
    )

    # Initialize DetectLtnModule without passing trial
    detect_ltn_module = DetectLtnModule(
        args,
        model,
        trial_logger=trial_logger,
    )

    # Set up trainer
    trainer = setup_trainer_instance(args, study_name, exp_setting)

    # Fit the model (trainer.fit handles multiple epochs)
    trainer.fit(detect_ltn_module, train_loader, val_loader)

    # Retrieve the best metrics and records after training
    best_metrics = detect_ltn_module.get_best_metrics()
    best_val_records = detect_ltn_module.best_val_records
    best_val_cheat_records = detect_ltn_module.best_val_cheat_records

    # Record best_selection_criterion and best_epoch for each relevant phase
    # {{ edit_1 }}
    # Setting attributes only for 'val' and 'val_cheat' phases
    if best_val_records:
        # Assuming the latest best_val_record is the best one
        best_val_epoch = max(best_val_records.keys())
        best_val_record = best_val_records[best_val_epoch]
        trial.set_user_attr("best_val_epoch", best_val_epoch)
        trial.set_user_attr("val_value", best_val_record["val_cheat_select_criterion"])
        trial.set_user_attr("train_value", best_val_record["train_select_criterion"])
    else:
        trial.set_user_attr("best_val_epoch", None)
        trial.set_user_attr("val_value", None)
        trial.set_user_attr("train_value", None)

    if best_val_cheat_records:
        # Assuming the latest best_val_cheat_record is the best one
        best_val_cheat_epoch = max(best_val_cheat_records.keys())
        best_val_cheat_record = best_val_cheat_records[best_val_cheat_epoch]
        trial.set_user_attr("best_val_cheat_epoch", best_val_cheat_epoch)
        trial.set_user_attr(
            "val_cheat_value", best_val_cheat_record["val_select_criterion"]
        )
        trial.set_user_attr(
            "train_value_cheat", best_val_cheat_record["train_select_criterion"]
        )
    else:
        trial.set_user_attr("best_val_cheat_epoch", None)
        trial.set_user_attr("val_cheat_value", None)
        trial.set_user_attr("train_value_cheat", None)
    # {{ end_edit_1 }}

    # Determine if the trial is valid based on best_epoch and valid_epoch
    if args.valid_epoch is not None:
        if_valid = (
            best_metrics["val"]["epoch"] > args.valid_epoch
            or best_metrics["val_cheat"]["epoch"] > args.valid_epoch
        )
    else:
        if_valid = True  # Treat as valid if valid_epoch is not set

    trial.set_user_attr("if_valid", str(if_valid))

    # Ensure 'val_cheat' metrics are considered in validity
    for phase in ["train", "val", "val_cheat"]:
        if phase not in best_metrics:
            global_logger.error(f"Unexpected phase '{phase}' encountered.")
            raise ValueError(f"Unexpected phase '{phase}' encountered.")

    end_time = time.time()  # Record end time
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)
    # Log trial completion with current/total and formatted duration using trial_logger
    trial_logger.info(
        f"Complete trial {trial.number + 1}/{args.n_trials} in {int(minutes)}m:{int(seconds)}s"
    )

    return best_metrics["val"][args.selection_criterion] if if_valid else int(100)
