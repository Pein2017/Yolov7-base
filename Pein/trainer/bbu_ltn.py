import logging
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from trainer.base_ltn import LtnBaseModule, Phase
from utils import Args


class DetectLtnModule(LtnBaseModule):
    def __init__(
        self,
        args: Args,
        model: nn.Module,
        trial_logger: logging.Logger,
    ):
        super(DetectLtnModule, self).__init__(args, model)

        self.trial_logger = trial_logger

        self.metric_names = getattr(
            args,
            "metric_names",
            [
                "accuracy",
                "fp_over_tp_fp",
                "fn_over_fn_tn",
                "avg_error",
                "f1_score",
            ],
        )

        # Initialize metrics tracking with metrics and epoch
        self.current_metrics: Dict[str, Dict[str, float]] = {
            phase.name.lower(): {metric: 0.0 for metric in self.metric_names}
            for phase in Phase
            if phase in [Phase.train, Phase.val, Phase.val_cheat]
        }

        self.log_after_epoch = getattr(args, "log_after_epoch", None)

        # Hyperparameters
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.scheduler_type = args.scheduler_type
        self.selection_criterion = args.selection_criterion
        self.thresholds_num = getattr(args, "thresholds_num", 200)

        # Configure loss function
        self.loss_fn = self.configure_loss_function(args)

        # Initialize best_train_threshold with a default value
        self.best_train_threshold = 0.5

        self.direction = (
            "maximize"
            if self.selection_criterion in {"accuracy", "f1_score"}
            else "minimize"
        )

        self.best_metrics: Dict[str, Dict[str, Any]] = {
            phase.name.lower(): {
                **{
                    metric: float("inf")
                    if self.direction == "minimize"
                    else float("-inf")
                    for metric in self.metric_names
                },
                "epoch": 0,
            }
            for phase in Phase
            if phase in [Phase.train, Phase.val, Phase.val_cheat]
        }

        # Initialize dictionaries to store best val and val_cheat records
        self.best_val_records: Dict[int, Dict[str, float]] = {}
        self.best_val_cheat_records: Dict[int, Dict[str, float]] = {}

    def custom_epoch_end_logic(
        self,
        phase: Phase,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Handle custom end-of-epoch logic for different phases.

        Args:
            phase (Phase): The current phase (Phase.train, Phase.val, or Phase.test).
            outputs (torch.Tensor): Model outputs for the entire epoch.
            labels (torch.Tensor): True labels for the entire epoch.
        """
        sigmoid_outputs = torch.sigmoid(outputs)
        threshold_list = self._generate_threshold_list(sigmoid_outputs)

        if phase == Phase.train:
            # Perform threshold search based on selection criterion
            train_metrics = self._search_optimal_threshold(
                sigmoid_outputs,
                labels,
                threshold_list,
                self.selection_criterion,
            )
            self.best_train_threshold = train_metrics["threshold"]

            # Record and log train metrics
            self._record_metrics(phase, train_metrics)

        elif phase == Phase.val:
            # Compute validation metrics using the best training threshold
            preds = (sigmoid_outputs >= self.best_train_threshold).float()
            val_metrics = self._compute_metrics_for_threshold(labels, preds)
            self._record_metrics(phase, val_metrics)

            # Perform threshold search based on selection criterion
            val_cheat_metrics = self._search_optimal_threshold(
                sigmoid_outputs,
                labels,
                threshold_list,
                self.selection_criterion,
            )
            best_val_cheat_threshold = val_cheat_metrics["threshold"]

            # Compute val_cheat metrics using the best threshold
            preds_cheat = (sigmoid_outputs >= best_val_cheat_threshold).float()
            val_cheat_computed_metrics = self._compute_metrics_for_threshold(
                labels, preds_cheat
            )
            val_cheat_computed_metrics["threshold"] = best_val_cheat_threshold

            # Record and log val_cheat metrics
            self._record_metrics(Phase.val_cheat, val_cheat_computed_metrics)

            self.trial_logger.debug(
                f"[Phase: val_cheat] Metrics: {val_cheat_computed_metrics}"
            )
            self.trial_logger.debug(
                f"Computed val_cheat metrics: {val_cheat_computed_metrics}"
            )

        elif phase == Phase.test:
            # Handle test phase metrics
            preds = torch.sigmoid(outputs)
            test_metrics = self._compute_metrics_for_threshold(labels, preds)
            self._record_metrics(phase, test_metrics)

        else:
            raise ValueError(f"Unexpected phase: {phase}")

    def _record_metrics(self, phase: Phase, metrics: Dict[str, float]) -> None:
        """
        Record and log metrics dynamically based on the improvement of the selection criterion.

        Args:
            phase (Phase): Current phase (train, val, val_cheat).
            metrics (Dict[str, float]): Calculated metrics.
        """
        phase_key = phase.name.lower()

        # Get the current value of the selection criterion
        selection_value = metrics.get(self.selection_criterion, None)
        if selection_value is None:
            raise KeyError(
                f"Selection criterion '{self.selection_criterion}' not found in metrics."
            )

        # Determine the optimization direction based on predefined direction
        optimization_direction = self.direction

        # Get the current best value for the selection criterion
        current_best = self.best_metrics[phase_key].get(
            self.selection_criterion,
            float("inf") if optimization_direction == "minimize" else float("-inf"),
        )

        # Check if the current epoch has a better selection criterion value
        improved = False
        if optimization_direction == "minimize" and selection_value < current_best:
            improved = True
        elif optimization_direction == "maximize" and selection_value > current_best:
            improved = True

        # Always update best_metrics with the new metrics
        for metric_name, value in metrics.items():
            if metric_name == "threshold":
                continue  # Skip threshold as it's handled separately
            rounded_value = round(value, 5)
            self.log(
                f"{phase_key}/{metric_name}",
                rounded_value,
                on_epoch=True,
                logger=True,
            )
            self.current_metrics[phase_key][metric_name] = rounded_value

            # Update best_metrics regardless of improvement
            self.best_metrics[phase_key][metric_name] = rounded_value

        self.best_metrics[phase_key]["epoch"] = self.current_epoch

        # Conditionally log improvements based on log_after_epoch and if_improved
        if (
            self.log_after_epoch is None or self.current_epoch > self.log_after_epoch
        ) and improved:
            # Consolidate logging into a single log entry with a distinct prefix
            improved_metrics = {
                metric: round(value, 5)
                for metric, value in metrics.items()
                if metric != "threshold"
            }
            self.trial_logger.info(
                f"[NEW BEST in {phase_key}] Metrics: {improved_metrics}, Epoch: {self.current_epoch}"
            )

            # Update best_metrics only on improvement
            for metric_name, value in metrics.items():
                if metric_name == "threshold":
                    continue
                rounded_value = round(value, 5)
                self.log(
                    f"{phase_key}/{metric_name}",
                    rounded_value,
                    on_epoch=True,
                    logger=True,
                )
                self.current_metrics[phase_key][metric_name] = rounded_value

                self.best_metrics[phase_key][metric_name] = rounded_value

            self.best_metrics[phase_key]["epoch"] = self.current_epoch

        # Record additional metrics when a new best val is achieved
        if phase == Phase.val and improved:
            best_val_epoch = self.current_epoch
            self.best_val_records[best_val_epoch] = {
                "train_select_criterion": self.best_metrics["train"][
                    self.selection_criterion
                ],
                "val_cheat_select_criterion": self.best_metrics["val_cheat"][
                    self.selection_criterion
                ],
            }
            self.trial_logger.info(
                f"[Best Val] Epoch: {best_val_epoch}, Train Select Criterion: {self.best_val_records[best_val_epoch]['train_select_criterion']}, Val Cheat Select Criterion: {self.best_val_records[best_val_epoch]['val_cheat_select_criterion']}"
            )

        # Record additional metrics when a new best val_cheat is achieved
        if phase == Phase.val_cheat and improved:
            best_val_cheat_epoch = self.current_epoch
            self.best_val_cheat_records[best_val_cheat_epoch] = {
                "train_select_criterion": self.best_metrics["train"][
                    self.selection_criterion
                ],
                "val_select_criterion": self.best_metrics["val"][
                    self.selection_criterion
                ],
            }
            self.trial_logger.info(
                f"[Best Val Cheat] Epoch: {best_val_cheat_epoch}, Train Select Criterion: {self.best_val_cheat_records[best_val_cheat_epoch]['train_select_criterion']}, Val Select Criterion: {self.best_val_cheat_records[best_val_cheat_epoch]['val_select_criterion']}"
            )

        self.trial_logger.debug(
            f"Recording metrics for phase '{phase_key}': {self.current_metrics[phase_key]}"
        )

        if phase == Phase.val_cheat:
            # Custom logic for val_cheat if necessary
            pass

    def _search_optimal_threshold(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        threshold_list: List[float],
        selection_criterion: str,
    ) -> Dict[str, float]:
        best_metrics = {}
        direction = self.direction
        if direction == "minimize":
            best_criterion_value = float("inf")
        elif direction == "maximize":
            best_criterion_value = float("-inf")
        else:
            raise ValueError(
                f"Invalid optimization direction for {selection_criterion}: {direction}"
            )

        for threshold in threshold_list:
            preds = (outputs >= threshold).float()
            metrics = self._compute_metrics_for_threshold(labels, preds)

            # Remove the phase prefix from selection_criterion if it exists
            criterion_key = selection_criterion.split("/")[-1]

            if criterion_key not in metrics:
                raise KeyError(
                    f"Selection criterion '{criterion_key}' not found in computed metrics. Available metrics: {list(metrics.keys())}"
                )

            criterion_value = metrics[criterion_key]

            if (direction == "minimize" and criterion_value < best_criterion_value) or (
                direction == "maximize" and criterion_value > best_criterion_value
            ):
                best_criterion_value = criterion_value
                best_metrics = metrics.copy()
                best_metrics["threshold"] = threshold.item()

        # Ensure selection_criterion exists
        if selection_criterion not in best_metrics:
            raise KeyError(
                f"Selection criterion '{selection_criterion}' not found in best_metrics."
            )

        return best_metrics

    def _compute_metrics_for_threshold(
        self, labels: torch.Tensor, preds: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for a given threshold.
        """
        metrics = {}

        # Convert tensors to boolean for comparison
        tp = torch.sum((preds == 1) & (labels == 1)).item()
        tn = torch.sum((preds == 0) & (labels == 0)).item()
        fp = torch.sum((preds == 1) & (labels == 0)).item()
        fn = torch.sum((preds == 0) & (labels == 1)).item()
        support = labels.size(0)

        # Compute metrics
        metrics["accuracy"] = (tp + tn) / support if support > 0 else 0

        # False Positive over all predicted positive: FP / (TP + FP)
        total_predicted_positive = tp + fp
        metrics["fp_over_tp_fp"] = (
            fp / total_predicted_positive if total_predicted_positive > 0 else 0
        )

        # False Negative over all predicted negative: FN / (FN + TN)
        total_predicted_negative = fn + tn
        metrics["fn_over_fn_tn"] = (
            fn / total_predicted_negative if total_predicted_negative > 0 else 0
        )

        # Arithmetic mean of FP rate and FN rate
        fp_rate = metrics["fp_over_tp_fp"]
        fn_rate = metrics["fn_over_fn_tn"]
        metrics["avg_error"] = (fp_rate + fn_rate) / 2  # Arithmetic mean

        # F1 Score
        metrics["f1_score"] = (
            (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        )

        return metrics

    def _generate_threshold_list(self, sigmoid_outputs: torch.Tensor) -> torch.Tensor:
        min_val = sigmoid_outputs.min().item()
        max_val = sigmoid_outputs.max().item()
        return torch.linspace(min_val, max_val, self.thresholds_num)

    def get_best_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Return the best metrics across all phases in the desired structure.
        Example:
        {
            'train': {'accuracy': 0.85, 'loss': 0.3, ..., 'epoch': 8},
            'val': {'accuracy': 0.84, 'loss': 0.35, ..., 'epoch': 6},
            'val_cheat': {'accuracy': 0.80, 'loss': 0.40, ..., 'epoch': 7}
        }
        """
        return self.best_metrics

    """
    fix methods
    """

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        if self.scheduler_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs * len(self.train_dataloader()),
            )
        else:
            return optimizer  # No scheduler

        return [optimizer], [scheduler]

    def configure_loss_function(self, args: Args) -> nn.Module:
        """
        Configure the loss function for the model.

        Args:
            args (Args): Configuration object.

        Returns:
            nn.Module: Configured loss function.
        """
        # Use Binary Cross Entropy with Logits
        pos_weight = getattr(args, "pos_weight", None)
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            return nn.BCEWithLogitsLoss()


class ExplicitModel(nn.Module):
    def __init__(self, input_dim1, input_dim2, input_dim3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim1 + input_dim2 + input_dim3, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x1, x2, x3):
        combined_input = torch.cat((x1, x2, x3), dim=1)
        x = self.linear1(combined_input)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def setup_trainer(
    log_dir: str,
    max_epochs: int,
    device_id: int,
    early_stop_metric: str,
    gradient_clip_val: float = 0.1,
    patience: int = 10,
    mode: str = "min",
) -> pl.Trainer:
    """Set up and return a PyTorch Lightning Trainer."""
    ltn_logger = TensorBoardLogger(save_dir=log_dir, name=None, version=None)

    # Set up early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=early_stop_metric,
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode=mode,
    )

    # Set up callbacks
    callbacks = [early_stop_callback]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=[device_id],
        logger=ltn_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
    )

    return trainer


def main():
    # Example configuration
    args = Args()
    args.num_epochs = 15
    args.lr = 0.001
    args.scheduler_type = "onecycle"
    args.selection_criterion = "avg_error"  # For threshold optimization
    args.early_stop_metric = "val/loss"  # For early stopping
    bs = 1
    input_dim1, input_dim2, input_dim3 = 2, 2, 2

    # Create the explicit model
    model = ExplicitModel(input_dim1, input_dim2, input_dim3)

    # Initialize the trainer
    # For single experiment, do not pass trial and trial_logger
    bbu_ltn_module = DetectLtnModule(args, model)

    # Create a simple random dataset for demonstration
    num_samples = 100
    dataset = []
    for _ in range(num_samples):
        x1 = torch.randn(input_dim1)
        x2 = torch.randn(input_dim2)
        x3 = torch.randn(input_dim3)
        labels = torch.randint(0, 2, (1,)).float()
        dataset.append((x1, x2, x3, labels))

    # Define a collate function for the DataLoader
    def collate_fn(batch):
        return (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            torch.stack([item[2] for item in batch]),
            torch.stack([item[3] for item in batch]),
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=bs,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=bs,
        collate_fn=collate_fn,
    )

    trainer = setup_trainer(
        log_dir="./test.log",
        max_epochs=10,
        device_id=0,
        early_stop_metric=args.early_stop_metric,  # 'val/loss'
    )

    # Run training
    trainer.fit(bbu_ltn_module, train_loader, val_loader)


if __name__ == "__main__":
    main()
