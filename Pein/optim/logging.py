import logging  # Ensure logging is imported
from typing import Any, Dict

from optuna.study import Study, StudyDirection
from optuna.trial import FrozenTrial, TrialState


class OptunaExperimentLogger:
    """
    Callback class to log when new best metrics are found across all trials.
    """

    def __init__(
        self,
        process_id: int,
        shared_best_metrics: Dict[str, float],
        lock: Any,
        study_name: str,
    ):
        # Initialize with shared best metrics and lock
        self.shared_best_metrics = shared_best_metrics
        self.lock = lock
        self.process_id = process_id
        self.study_name = study_name  # Add study_name
        self.logger = logging.getLogger("global_logger")  # Use the standard logger

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE or self.process_id != 0:
            return

        current_val_metric = trial.user_attrs.get("best_val_value")
        current_val_cheat_metric = trial.user_attrs.get("best_val_cheat_value")

        if current_val_metric is None or current_val_cheat_metric is None:
            self.logger.error(
                f"Trial {trial.number}: Required metrics not found in trial user attributes."
            )
            return

        if_valid = trial.user_attrs.get("if_valid") == "True"

        if not if_valid:
            return  # Skip invalid trials

        log_message_lines = []

        with self.lock:
            # Check and update best_val_metric
            if self._is_better(
                current_val_metric, self.shared_best_metrics["val"], study.direction
            ):
                self.shared_best_metrics["val"] = current_val_metric
                self.shared_best_metrics["val_epoch"] = trial.user_attrs.get(
                    "best_val_epoch", "N/A"
                )
                log_message_lines.append(
                    f"val: Metric={current_val_metric:.4f}, Epoch={self.shared_best_metrics['val_epoch']}"
                )

            # Check and update best_val_cheat_metric
            if self._is_better(
                current_val_cheat_metric,
                self.shared_best_metrics["val_cheat"],
                study.direction,
            ):
                self.shared_best_metrics["val_cheat"] = current_val_cheat_metric
                self.shared_best_metrics["val_cheat_epoch"] = trial.user_attrs.get(
                    "best_val_cheat_epoch", "N/A"
                )
                log_message_lines.append(
                    f"val_cheat: Metric={current_val_cheat_metric:.4f}, Epoch={self.shared_best_metrics['val_cheat_epoch']}"
                )

        if log_message_lines:
            duration = trial.datetime_complete - trial.datetime_start
            duration_str = f"{int(duration.total_seconds() // 60)}m:{int(duration.total_seconds() % 60)}s"
            message = (
                f"Trial {trial.number}: New Best Metrics:\n"
                + "\n".join(log_message_lines)
                + f"\nDuration: {duration_str}"
            )
            self.logger.info(message)

    def _get_selection_criterion(self, trial: FrozenTrial, study: Study) -> str:
        selection_criterion = trial.user_attrs.get("selection_criterion")
        if selection_criterion is None:
            selection_criterion = study.user_attrs.get("selection_criterion")
        if selection_criterion is None:
            raise ValueError(
                "selection_criterion not found in trial user_attrs or study user_attrs"
            )
        return selection_criterion

    def _is_better(
        self, current: float, best: float, direction: StudyDirection
    ) -> bool:
        if direction == StudyDirection.MINIMIZE:
            return current < best
        elif direction == StudyDirection.MAXIMIZE:
            return current > best
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def get_best_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the best metrics tracked by the logger.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing best metrics and their epochs.
        """
        return {
            "val": {
                "best_score": self.shared_best_metrics["val"],
                "best_epoch": self.shared_best_metrics["val_epoch"],
            },
            "val_cheat": {
                "best_score": self.shared_best_metrics["val_cheat"],
                "best_epoch": self.shared_best_metrics["val_cheat_epoch"],
            },
        }
