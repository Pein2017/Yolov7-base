import argparse
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from data.data_factory import get_dataloader
from models.classifier_2 import DetectionClassificationModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Set up the logger
from utils.setup import (
    setup_logger,  # Ensure this import matches your project structure
)

logger = setup_logger(
    log_file="trainer.log", level="debug", name="trainer_logger", log_to_console=True
)


class BbuDetectModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        lr: float,
        scheduler_type: str = "onecycle",
        selection_criterion: str = "avg_error",
        pos_weight: Optional[float] = None,
        thresholds: Optional[np.ndarray] = None,
        metric_names: Optional[List[str]] = None,
    ):
        super(BbuDetectModule, self).__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.pos_weight = pos_weight
        self.selection_criterion = selection_criterion
        self.threshold = None  # will be set during training
        self.thresholds = thresholds
        self.metric_names = (
            metric_names
            if metric_names is not None
            else [
                "val_loss",
                "accuracy",
                "fp_over_tp_fp",
                "fn_over_fn_tn",
                "avg_error",
                "f1_score",
            ]
        )

        # Loss function
        self.criterion = self._get_loss_function()

        self.train_outputs = []
        self.train_labels = []
        self.val_outputs = []
        self.val_labels = []

        # Initialize best metrics tracking
        self.best_metrics = {
            "epoch": 0,
            "threshold": None,
            self.selection_criterion: None,
        }

        ## py_lightning >=2.0
        self.train_step_outputs = []
        self.validation_step_outputs = []

    def _get_loss_function(self):
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor(self.pos_weight, dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        return criterion

    def forward(
        self,
        object_features,
        figure_features,
        group_features,
        object_lengths,
        figure_lengths,
    ):
        return self.model(
            object_features,
            figure_features,
            group_features,
            object_lengths,
            figure_lengths,
        ).squeeze()

    def training_step(self, batch, batch_idx):
        data, labels, object_lengths, figure_lengths = batch
        object_features, figure_features, group_features = data

        # Forward pass
        outputs = self(
            object_features,
            figure_features,
            group_features,
            object_lengths,
            figure_lengths,
        )

        loss = self.criterion(outputs, labels.float())

        # Manually collect outputs, labels, and loss
        self.train_step_outputs.append(
            {
                "loss": loss.detach(),
                "outputs": outputs.detach(),
                "labels": labels.detach(),
            }
        )

        return {"loss": loss}

    def on_train_epoch_end(self):
        # Ensure train_step_outputs is not empty
        if not self.train_step_outputs:
            logger.warning("No training outputs collected.")
            return

        # Compute average training loss
        avg_loss = torch.stack([x["loss"] for x in self.train_step_outputs]).mean()
        self.log("Loss/Train", avg_loss, on_epoch=True, logger=True, prog_bar=True)

        # Concatenate all outputs and labels
        train_outputs = (
            torch.cat([x["outputs"] for x in self.train_step_outputs]).cpu().numpy()
        )
        train_labels = (
            torch.cat([x["labels"] for x in self.train_step_outputs]).cpu().numpy()
        )

        # Evaluate to find the best threshold on training set
        train_metrics = self.evaluate(train_outputs, train_labels, prefix="Train")

        # Update threshold based on training metrics
        self.threshold = train_metrics["threshold"]

        # Log train metrics
        self.log_metrics(train_metrics, prefix="Train")

        # Reset `self.train_step_outputs` for the next epoch
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        data, labels, object_lengths, figure_lengths = batch
        object_features, figure_features, group_features = data

        # Forward pass
        outputs = self(
            object_features,
            figure_features,
            group_features,
            object_lengths,
            figure_lengths,
        )

        loss = self.criterion(outputs, labels.float())

        # Manually collect outputs, labels, and loss
        self.validation_step_outputs.append(
            {
                "val_loss": loss.detach(),
                "outputs": outputs.detach(),
                "labels": labels.detach(),
            }
        )

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Ensure validation_step_outputs is not empty
        if not self.validation_step_outputs:
            logger.warning("No validation outputs collected.")
            return

        # Compute average validation loss
        avg_val_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        self.log(
            "Loss/Validation", avg_val_loss, on_epoch=True, logger=True, prog_bar=True
        )

        # Concatenate all outputs and labels
        val_outputs = (
            torch.cat([x["outputs"] for x in self.validation_step_outputs])
            .cpu()
            .numpy()
        )
        val_labels = (
            torch.cat([x["labels"] for x in self.validation_step_outputs]).cpu().numpy()
        )

        # Check if threshold is set
        if self.threshold is None:
            logger.warning(
                "Threshold is not set. Skipping validation metrics computation for this epoch."
            )
            return

        # Compute validation metrics using the threshold from training set
        sigmoid_outputs = torch.sigmoid(torch.tensor(val_outputs)).numpy()
        preds = (sigmoid_outputs >= self.threshold).astype(float)

        val_metrics = self.compute_metrics(val_labels, preds)
        val_metrics["threshold"] = self.threshold

        # Log validation metrics
        self.log_metrics(val_metrics, prefix="Val")

        # Evaluate validation set to find the best threshold on validation set
        val_treat_metrics = self.evaluate(val_outputs, val_labels, prefix="Val_Treat")

        # Log val_treat metrics
        self.log_metrics(val_treat_metrics, prefix="Val_Treat")

        # Update best metrics if current validation metrics are better
        criterion_value = val_treat_metrics[self.selection_criterion]
        best_criterion_value = self.best_metrics.get(self.selection_criterion, None)

        if best_criterion_value is None or (
            (
                self.selection_criterion in ["accuracy", "f1_score"]
                and criterion_value > best_criterion_value
            )
            or (
                self.selection_criterion
                in ["fp_over_tp_fp", "fn_over_fn_tn", "avg_error"]
                and criterion_value < best_criterion_value
            )
        ):
            self.best_metrics = val_treat_metrics.copy()
            self.best_metrics["epoch"] = self.current_epoch + 1

        # Reset validation outputs for the next epoch
        self.validation_step_outputs.clear()

    def evaluate(self, outputs, labels, prefix=""):
        sigmoid_outputs = torch.sigmoid(torch.tensor(outputs)).numpy()
        best_metrics = {}
        best_criterion_value = None

        # Dynamically set thresholds based on min and max of sigmoid outputs
        if self.thresholds is not None:
            thresholds = self.thresholds
        else:
            min_output = np.min(sigmoid_outputs)
            max_output = np.max(sigmoid_outputs)
            # Handle the case when min and max are equal
            if min_output == max_output:
                thresholds = [min_output]
            else:
                thresholds = np.linspace(min_output, max_output, 100)

        # Evaluate metrics for each threshold
        for thresh in thresholds:
            preds = (sigmoid_outputs >= thresh).astype(float)
            metrics = self.compute_metrics(labels, preds)
            metrics["threshold"] = thresh

            criterion_value = metrics.get(self.selection_criterion, None)
            if criterion_value is None:
                continue  # Skip if criterion not found

            is_better = False
            if best_criterion_value is None:
                is_better = True
            else:
                if self.selection_criterion in [
                    "accuracy",
                    "f1_score",
                ]:
                    is_better = criterion_value > best_criterion_value
                elif self.selection_criterion in [
                    "fp_over_tp_fp",
                    "fn_over_fn_tn",
                    "avg_error",
                ]:
                    is_better = criterion_value < best_criterion_value
                else:
                    raise ValueError(
                        f"Unknown selection criterion: {self.selection_criterion}"
                    )

            if is_better:
                best_metrics = metrics.copy()
                best_criterion_value = criterion_value

        if not best_metrics:
            logger.warning(f"No valid metrics found for {prefix}")
            return {}

        return best_metrics

    def log_metrics(self, metrics, prefix=""):
        # Log to console
        message = f"{prefix} Metrics at Epoch {self.current_epoch + 1}: "
        metric_strs = [
            f"{k}: {v:.4f}" for k, v in metrics.items() if k in self.metric_names
        ]
        message += ", ".join(metric_strs)
        logger.info(message)

        # Log to TensorBoard
        metrics_to_log = {
            f"{prefix}/{k}": v for k, v in metrics.items() if k in self.metric_names
        }
        self.log_dict(metrics_to_log, on_epoch=True, logger=True)

    def compute_metrics(self, labels, preds):
        """
        Compute evaluation metrics.
        """
        metrics = {}
        labels = np.array(labels)
        preds = np.array(preds)

        # Compute confusion matrix components
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        support = len(labels)

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if self.scheduler_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
            )
        elif self.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs * len(self.train_dataloader()),
            )
        else:
            return optimizer  # No scheduler

        return [optimizer], [scheduler]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a detection classification model"
    )

    # Add command-line arguments
    parser.add_argument(
        "--pos_dir", type=str, required=True, help="Path to positive samples directory"
    )
    parser.add_argument(
        "--neg_dir", type=str, required=True, help="Path to negative samples directory"
    )
    parser.add_argument(
        "--classes_file", type=str, required=True, help="Path to classes file"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # Model-specific arguments
    parser.add_argument(
        "--embedding_dim", type=int, default=16, help="Embedding dimension"
    )
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument(
        "--attn_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers in the model"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Additional model structure arguments
    parser.add_argument(
        "--fc_hidden_dims",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Fully connected layer dimensions",
    )

    # Training-specific arguments
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="onecycle",
        help="Type of learning rate scheduler",
    )

    parser.add_argument(
        "--ltn_log_dir",
        type=str,
        default="./ltn_log",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--seed", type=int, default=17, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resample_method",
        type=str,
        default="both",
        help="Method to resample the data",
    )

    # Training criterion and device
    parser.add_argument(
        "--criterion", type=str, default="f1_score", help="Selection criterion"
    )
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")

    # Parse arguments
    args, unknown = parser.parse_known_args()  # Capture unknown arguments
    if unknown:
        print(f"Unknown arguments passed: {unknown}")

    return args


def setup_trainer(log_dir, max_epochs, device_id, early_stop_metric):
    # Initialize TensorBoard Logger
    ltn_logger = TensorBoardLogger(save_dir=log_dir, name=None, version=None)

    # Add the EarlyStopping callback
    early_stop_callback = EarlyStopping(
        monitor=early_stop_metric,
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode="min",
    )

    # Set up PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=[device_id],
        logger=ltn_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
    )

    return trainer


def build_model(args, dataset):
    """Build the model and initialize the DetectionClassificationModel."""
    model = DetectionClassificationModel(
        num_classes=dataset.num_classes,
        num_labels=dataset.num_labels,
        PAD_LABEL=dataset.PAD_LABEL,
        SEP_LABEL=dataset.SEP_LABEL,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        attn_heads=args.attn_heads,
        fc_hidden_dims=args.fc_hidden_dims,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    return model


def main():
    args = parse_args()

    # Create exp_settings string
    exp_settings = (
        f"resamp_{args.resample_method}-"
        f"lr_{args.lr}-"
        f"hid_dim_{args.hidden_dim}-"
        f"embed_dim_{args.embedding_dim}-"
        f"layers_{args.num_layers}-"
        f"attn_heads_{args.attn_heads}-"
        f"dropout_{args.dropout}"
    )

    logger.info(f"Exp settings are:\n{exp_settings}")

    seed = args.seed
    if seed == -1:
        seed = np.random.randint(0, 1000)
    pl.seed_everything(seed, workers=True)

    # Get the dataloaders
    dataloaders = get_dataloader(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        classes_file=args.classes_file,
        split=[0.8, 0.2],  # 80% train, 20% validation
        batch_size=args.batch_size,
        balance_ratio=1.0,
        resample_method=args.resample_method,
        seed=seed,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Get dataset size and calculate the number of batches
    train_dataset_size = len(train_loader.dataset)

    # Calculate the total number of batches in one epoch
    num_batches_per_epoch = len(train_loader)

    # Dynamically adjust the number of epochs based on dataset size and batch size
    total_dataset_iterations = args.num_epochs
    adjusted_epochs = int(
        total_dataset_iterations
        * train_dataset_size
        / (num_batches_per_epoch * args.batch_size)
    )

    logger.info(f"Dataset size: {train_dataset_size}, Batch size: {args.batch_size}")
    logger.info(
        f"Adjusted number of epochs: {adjusted_epochs} (Original: {args.num_epochs})"
    )

    # Get PAD_LABEL, SEP_LABEL, num_classes, and num_labels from the dataset
    # dataset = train_loader.dataset
    # PAD_LABEL = dataset.PAD_LABEL
    # SEP_LABEL = dataset.SEP_LABEL
    # num_classes = dataset.num_classes
    # num_labels = dataset.num_labels

    model = build_model(args, train_loader.dataset)

    # Initialize the LightningModule
    detect_ltn_model = BbuDetectModule(
        model=model,
        num_epochs=adjusted_epochs,  # Use adjusted epochs
        lr=args.lr,
        scheduler_type=args.scheduler_type,
        selection_criterion=args.criterion,
        metric_names=[
            "val_loss",
            "accuracy",
            "fp_over_tp_fp",
            "fn_over_fn_tn",
            "avg_error",
            "f1_score",
        ],
    )

    # Use exp_settings only in save_dir to avoid duplication
    log_dir = f"{args.ltn_log_dir}/{exp_settings}"

    # Set up the trainer by calling the setup_trainer function
    trainer = setup_trainer(
        log_dir=log_dir,
        max_epochs=adjusted_epochs,
        device_id=args.device_id,
        early_stop_metric="Loss/Validation",  # This metric is used for early stopping
    )

    # Train the model
    trainer.fit(
        detect_ltn_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    print("Training completed!")


# Main training code
if __name__ == "__main__":
    main()
