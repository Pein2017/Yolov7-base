import logging

from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


def log_new_best(
    logger_name: str,
    metric_type: str,
    metric_value: float,
    epoch: int,
    fp_over: float,
    fn_over: float,
    selection_criterion: str,
):
    """
    Log when a new best metric is found.

    Args:
        logger_name (str): Name of the logger.
        metric_type (str): Type of metric ('val' or 'val_cheat').
        metric_value (float): Value of the metric.
        epoch (int): Epoch number.
        fp_over (float): False positive rate.
        fn_over (float): False negative rate.
        selection_criterion (str): The criterion used for selection.
    """
    logger = logging.getLogger(logger_name)
    logger.info(
        f"New best {metric_type} {selection_criterion} = {metric_value:.4f} at epoch {epoch}. "
        f"FP Over: {fp_over:.4f}, FN Over: {fn_over:.4f}"
    )


class OptunaExperimentLogger:
    """
    Callback class to log when new best val or val_cheat metrics are found after a specified epoch.
    """

    def __init__(self, min_improvement: float = 0.001, start_logging_epoch: int = 10):
        self.best_val = float("inf")
        self.best_val_cheat = float("inf")
        self.min_improvement = min_improvement
        self.start_logging_epoch = start_logging_epoch

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        logger = logging.getLogger("train_optuna")
        if trial.state != TrialState.COMPLETE:
            return  # Skip incomplete trials

        best_epoch = trial.user_attrs.get("best_epoch")
        if best_epoch is None:
            logger.warning(f"Trial {trial.number} has no 'best_epoch' attribute.")
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
            fp_over_tp_fp_val = trial.user_attrs.get("fp_over_tp_fp", 0.0)
            fn_over_fn_tn_val = trial.user_attrs.get("fn_over_fn_tn", 0.0)
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
            fp_over_tp_fp_val_cheat = trial.user_attrs.get(
                "val_cheat_fp_over_tp_fp", 0.0
            )
            fn_over_fn_tn_val_cheat = trial.user_attrs.get(
                "val_cheat_fn_over_fn_tn", 0.0
            )
            log_new_best(
                logger_name="train_optuna",
                metric_type="val_cheat",
                metric_value=val_cheat_metric,
                epoch=best_epoch,
                fp_over=fp_over_tp_fp_val_cheat,
                fn_over=fn_over_fn_tn_val_cheat,
                selection_criterion=selection_criterion,
            )
