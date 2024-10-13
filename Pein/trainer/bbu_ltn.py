# Instead, get the global logger
import logging
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from trainer.base_ltn import LtnBaseModule, Phase
from utils import Args  # Updated import

logger = logging.getLogger("train_optuna")


class DetectLtnModule(LtnBaseModule):
    def __init__(self, args: Args, model: nn.Module):
        super(DetectLtnModule, self).__init__(args, model)

        # Initialize user_attrs
        self.user_attrs = {}

        # Hyperparameters
        self.lr = args.lr
        self.weight_decay = getattr(args, "weight_decay", 0.0)
        self.num_epochs = args.num_epochs
        self.scheduler_type = args.scheduler_type
        self.selection_criterion = args.selection_criterion  # e.g., "avg_error"
        self.thresholds_num = getattr(args, "thresholds_num", 200)
        self.metric_names = getattr(
            args,
            "metric_names",
            [
                "loss",
                "accuracy",
                "fp_over_tp_fp",
                "fn_over_fn_tn",
                "avg_error",
                "f1_score",
            ],
        )

        # Initialize thresholds with default values
        self.threshold = 0.5  # Default threshold
        self.best_train_threshold = 0.5  # Default best train threshold
        self.best_val_threshold = 0.5  # Default best val threshold

        # Initialize best metrics tracking
        self.best_metrics = {
            "train": {metric: float("inf") for metric in self.metric_names},
            "val": {metric: float("inf") for metric in self.metric_names},
            "val_cheat": {metric: float("inf") for metric in self.metric_names},
        }

        # Initialize best_epoch to track when the best metric was achieved
        self.best_epoch = None

        # Configure loss function
        self.loss_fn = self.configure_loss_function(args)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
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

    def configure_loss_function(self, args: Args):
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

    def custom_epoch_end_logic(
        self, phase: Phase, outputs: torch.Tensor, labels: torch.Tensor
    ) -> None:
        sigmoid_outputs = torch.sigmoid(outputs)
        threshold_list = self.generate_threshold_list(sigmoid_outputs)

        if phase == Phase.train:
            # Perform threshold search based on selection criterion
            train_metrics = self.evaluate(
                sigmoid_outputs,
                labels,
                threshold_list,
                self.selection_criterion,
            )
            self.best_train_threshold = train_metrics["threshold"]

            # Log all train metrics with best threshold found in train
            for metric_name, value in train_metrics.items():
                if metric_name != "threshold":
                    rounded_value = round(value, 4)
                    self.log(
                        f"train/{metric_name}",
                        rounded_value,
                        on_epoch=True,
                        logger=True,
                    )
                    logger.debug(f"Logged train/{metric_name}: {rounded_value:.4f}")

            # Update best metrics if current selection criterion is better
            if (
                train_metrics[self.selection_criterion]
                < self.best_metrics["train"][self.selection_criterion]
            ):
                self.best_metrics["train"][self.selection_criterion] = round(
                    train_metrics[self.selection_criterion], 4
                )

        elif phase == Phase.val:
            # Compute validation metrics using the best training threshold
            val_metrics = self.compute_metrics(
                labels, (sigmoid_outputs >= self.best_train_threshold).float()
            )
            for metric_name, value in val_metrics.items():
                rounded_value = round(value, 4)
                self.log(
                    f"val/{metric_name}", rounded_value, on_epoch=True, logger=True
                )
                if rounded_value < self.best_metrics["val"][metric_name]:
                    self.best_metrics["val"][metric_name] = rounded_value

            # Perform threshold search based on selection criterion for val_cheat
            val_cheat_metrics = self.evaluate(
                sigmoid_outputs,
                labels,
                threshold_list,
                self.selection_criterion,
            )
            for metric_name, value in val_cheat_metrics.items():
                if metric_name != "threshold":
                    rounded_value = round(value, 4)
                    self.log(
                        f"val_cheat/{metric_name}",
                        rounded_value,
                        on_epoch=True,
                        logger=True,
                    )
                    if rounded_value < self.best_metrics["val_cheat"][metric_name]:
                        self.best_metrics["val_cheat"][metric_name] = rounded_value

            # Update best_epoch if selection_criterion improved
            if self.best_metrics["val"][self.selection_criterion] < float("inf"):
                self.best_epoch = self.current_epoch + 1

        else:
            raise ValueError(f"Unexpected phase: {phase}")

    def evaluate(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        threshold_list: List,
        selection_criterion: str,
    ) -> Dict[str, float]:
        best_metrics = {}
        best_criterion_value = float("inf")  # Assuming lower is better

        for threshold in threshold_list:
            preds = (outputs >= threshold).float()
            metrics = self.compute_metrics(labels, preds)

            # Remove the phase prefix from selection_criterion if it exists
            criterion_key = selection_criterion.split("/")[-1]

            if criterion_key not in metrics:
                raise KeyError(
                    f"Selection criterion '{criterion_key}' not found in computed metrics. Available metrics: {list(metrics.keys())}"
                )

            criterion_value = metrics[criterion_key]

            if criterion_value < best_criterion_value:
                best_criterion_value = criterion_value
                best_metrics = metrics.copy()
                best_metrics["threshold"] = threshold.item()

        # Ensure that a threshold is always returned
        if "threshold" not in best_metrics:
            best_metrics["threshold"] = (
                0.5  # Default threshold if no better one is found
            )

        # Ensure selection_criterion exists
        if selection_criterion not in best_metrics:
            raise KeyError(
                f"Selection criterion '{selection_criterion}' not found in best_metrics."
            )

        return best_metrics

    def compute_metrics(
        self, labels: torch.Tensor, preds: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
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

        # Harmonic mean of FP rate and FN rate
        fp_rate = metrics["fp_over_tp_fp"]
        fn_rate = metrics["fn_over_fn_tn"]
        metrics["avg_error"] = (
            0.5 * (fp_rate + fn_rate) if (fp_rate + fn_rate) > 0 else 0
        )

        # F1 Score
        metrics["f1_score"] = (
            (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        )

        return metrics

    def generate_threshold_list(self, sigmoid_outputs: torch.Tensor) -> torch.Tensor:
        min_val = sigmoid_outputs.min().item()
        max_val = sigmoid_outputs.max().item()
        return torch.linspace(min_val, max_val, self.thresholds_num)

    def get_best_metric(self, metric_name: str) -> float:
        """Get the weighted average of the best val and val_cheat metric values."""
        val_weight = 0.6
        val_cheat_weight = 0.4
        val_metric = self.best_metrics.get(f"val/{metric_name}")
        val_cheat_metric = self.best_metrics.get(f"val_cheat/{metric_name}")

        # Check if both metrics exist
        if val_metric is None or val_cheat_metric is None:
            missing = []
            if val_metric is None:
                missing.append(f"val/{metric_name}")
            if val_cheat_metric is None:
                missing.append(f"val_cheat/{metric_name}")
            missing_metrics = ", ".join(missing)
            raise KeyError(
                f"Missing metrics: {missing_metrics}. "
                f"Ensure that both 'val/{metric_name}' and 'val_cheat/{metric_name}' are being logged."
            )

        weighted_average = val_weight * val_metric + val_cheat_weight * val_cheat_metric
        return round(weighted_average, 4)


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
