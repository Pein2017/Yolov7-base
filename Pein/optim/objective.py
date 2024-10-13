import logging
import time

import optuna
from optuna.trial import Trial

from data.data_factory import get_dataloader
from models.classifier_2 import DetectionClassificationModel
from optim.search_space import define_search_space
from trainer.bbu_ltn import DetectLtnModule
from utils import Args
from utils.helpers import generate_exp_setting
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
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

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
    model = DetectionClassificationModel(
        num_classes=args.num_classes,
        num_labels=args.num_labels,
        PAD_LABEL=args.PAD_LABEL,
        SEP_LABEL=args.SEP_LABEL,
        embedding_dim=search_space["embedding_dim"],
        hidden_dim=search_space["hidden_dim"],
        attn_heads=search_space["attn_heads"],
        fc_hidden_dims=search_space["fc_hidden_dims"],
        num_layers=search_space["num_layers"],
        dropout=search_space["dropout"],
    )

    # Initialize DetectLtnModule
    detect_ltn_module = DetectLtnModule(args, model)

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
        trainer.fit(detect_ltn_module, train_loader, val_loader)

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
