import argparse
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.data_factory import get_dataloader
from model.classifier_1 import DetectionClassificationModel
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Set up the logger
from utils.setup import (
    setup_logger,  # Ensure this import matches your project structure
)

logger = setup_logger(
    log_file="trainer.log", level="debug", name="trainer_logger", log_to_console=True
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        lr: float = 0.001,
        threshold: float = 0.5,
        device: Optional[str] = None,
        pos_weight: Optional[float] = None,
        criterion: str = "f1_score",
        tb_log_dir: str = "./tb_logs",
        output_dir: str = "./plots",
        thresholds: Optional[np.ndarray] = None,
        metric_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # DataLoaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Criterion and optimizer
        self.criterion = self._get_loss_function(pos_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Threshold for prediction
        self.threshold = threshold
        self.selection_criterion = criterion

        # Best metrics tracking
        self.best_metrics = {
            "epoch": 0,
            "threshold": None,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "fp_ratio": float("inf"),
            "fn_ratio": float("inf"),
            "fp_over_tp_fp": float("inf"),
            "fn_over_fn_tn": float("inf"),
            "val_loss": float("inf"),
            "confusion_matrix": None,
        }

        # Directories and thresholds
        self.log_dir = tb_log_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.thresholds = thresholds or np.arange(0.1, 1.0, 0.01)

        # Metrics to track
        self.metric_names = metric_names or [
            "val_loss",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "fp_ratio",
            "fn_ratio",
            "fp_over_tp_fp",
            "fn_over_fn_tn",
        ]

    def _run_epoch(self, dataloader, training=False):
        if training:
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0.0
        all_labels = []
        all_outputs = []

        for data, labels, lengths in dataloader:
            data = data.to(self.device)
            labels = labels.float().to(self.device)
            lengths = lengths.to(self.device)

            if training:
                # Zero the gradients
                self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data, lengths)
            outputs = outputs.squeeze()

            # Compute loss
            loss = self.criterion(outputs, labels)

            if training:
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()

            # Collect outputs and labels
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())

        # Average loss over the epoch
        epoch_loss /= len(dataloader)

        return epoch_loss, all_outputs, all_labels

    def train(self, num_epochs: int):
        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=self.log_dir)

        for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
            # Run training epoch
            train_loss, all_outputs, all_labels = self._run_epoch(
                self.train_loader, training=True
            )

            # Log training loss
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

            # Log training loss to TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch + 1)

            # Validate the model
            best_val_metrics = self.validate(current_epoch=epoch)

            # Log validation metrics to TensorBoard
            self.log_to_tensorboard(
                writer, best_val_metrics, epoch, prefix="Validation"
            )

            # Update best_metrics if current validation metrics are better
            criterion_value = best_val_metrics[self.selection_criterion]
            best_criterion_value = self.best_metrics.get(self.selection_criterion, None)

            if best_criterion_value is None:
                is_better = True
            else:
                if self.selection_criterion in [
                    "f1_score",
                    "accuracy",
                    "precision",
                    "recall",
                ]:
                    is_better = criterion_value > best_criterion_value
                elif self.selection_criterion in [
                    "fp_ratio",
                    "fn_ratio",
                    "fp_over_tp_fp",
                    "fn_over_fn_tn",
                ]:
                    is_better = criterion_value < best_criterion_value
                else:
                    raise ValueError(
                        f"Unknown selection criterion: {self.selection_criterion}"
                    )

            if is_better:
                self.best_metrics = best_val_metrics.copy()
                self.best_metrics["epoch"] = epoch + 1  # Store the epoch number

                # # Optionally save the model checkpoint
                # self.save_model(path=f"best_model_epoch_{epoch+1}.pth")
                # logger.info(f"Best model updated at epoch {epoch+1}")

        # Close the TensorBoard writer
        writer.close()

        # After training, log the best metrics
        self.log_metrics(self.best_metrics, prefix="Best Validation")

    def validate(self, current_epoch=None):
        # Run validation epoch once to get outputs and labels
        val_loss, all_outputs, all_labels = self._run_epoch(
            self.val_loader, training=False
        )

        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)

        sigmoid_outputs = torch.sigmoid(torch.tensor(all_outputs)).numpy()

        best_metrics = {
            "threshold": None,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "fp_ratio": float("inf"),
            "fn_ratio": float("inf"),
            "fp_over_tp_fp": float("inf"),
            "fn_over_fn_tn": float("inf"),
            "val_loss": val_loss,
            "confusion_matrix": None,
        }

        for thresh in self.thresholds:
            # Apply threshold to outputs
            preds = (sigmoid_outputs > thresh).astype(float)

            # Compute metrics
            metrics = self.compute_metrics(all_labels, preds)
            metrics["threshold"] = thresh
            metrics["val_loss"] = val_loss
            criterion_value = metrics.get(self.selection_criterion, None)

            if criterion_value is None:
                raise ValueError(
                    f"Unknown selection criterion: {self.selection_criterion}"
                )

            # Determine if this is the best metric
            is_better = False
            if self.selection_criterion in [
                "f1_score",
                "accuracy",
                "precision",
                "recall",
            ]:
                is_better = criterion_value > best_metrics.get("criterion_value", 0)
            elif self.selection_criterion in [
                "fp_ratio",
                "fn_ratio",
                "fp_over_tp_fp",
                "fn_over_fn_tn",
            ]:
                is_better = criterion_value < best_metrics.get(
                    "criterion_value", float("inf")
                )
            else:
                raise ValueError(
                    f"Unknown selection criterion: {self.selection_criterion}"
                )

            if is_better:
                best_metrics = metrics.copy()
                best_metrics["criterion_value"] = criterion_value

        # Log the best metrics
        self.log_metrics(best_metrics, epoch=current_epoch, prefix="Validation")

        # Update the model threshold to the best threshold found
        self.threshold = best_metrics["threshold"]

        return best_metrics

    def evaluate(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_loader
        if dataloader is None:
            raise ValueError("No dataloader provided for evaluation.")

        # Run evaluation epoch to get outputs and labels
        val_loss, all_outputs, all_labels = self._run_epoch(dataloader, training=False)

        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)

        # Apply the best threshold
        sigmoid_outputs = torch.sigmoid(torch.tensor(all_outputs)).numpy()
        preds = (sigmoid_outputs > self.threshold).astype(float)

        # Compute metrics
        metrics = self.compute_metrics(all_labels, preds)
        metrics["threshold"] = self.threshold
        metrics["val_loss"] = val_loss

        # Log metrics
        self.log_metrics(metrics, prefix="Evaluation")

    def save_model(self, path="model.pth"):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path="model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")

    def _get_loss_function(self, pos_weight: Optional[float]):
        """Initialize the loss function with optional positive class weighting."""
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(
                self.device
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        return criterion

    def _compute_class_weights(self):
        """Compute class weights based on the training data."""
        labels = []
        for _, batch_labels, _ in self.train_loader:
            labels.extend(batch_labels.numpy())
        num_positives = np.sum(labels)
        num_negatives = len(labels) - num_positives
        total_samples = num_positives + num_negatives

        weight_for_0 = total_samples / num_negatives if num_negatives > 0 else 1.0
        weight_for_1 = total_samples / num_positives if num_positives > 0 else 1.0

        return weight_for_0, weight_for_1

    def compute_metrics(self, labels, preds):
        """Compute evaluation metrics."""
        total = len(labels)
        correct = (preds == labels).sum()
        accuracy = correct / total if total > 0 else 0

        # Use labels=[1, 0] to treat 1 as positive and 0 as negative
        confusion_mat = confusion_matrix(labels, preds, labels=[1, 0])

        # Extract TP, FN, FP, TN
        tp, fn, fp, tn = confusion_mat.ravel()

        # Calculate additional metrics
        fp_ratio = fp / total if total > 0 else 0
        fn_ratio = fn / total if total > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Additional metrics
        fp_over_tp_fp = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        fn_over_fn_tn = fn / (fn + tn) if (fn + tn) > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "fp_ratio": fp_ratio,
            "fn_ratio": fn_ratio,
            "fp_over_tp_fp": fp_over_tp_fp,
            "fn_over_fn_tn": fn_over_fn_tn,
            "confusion_matrix": confusion_mat,
        }

        return metrics

    def log_metrics(self, metrics, epoch=None, prefix="Validation"):
        """Log metrics to the logger."""
        if epoch is not None:
            epoch_info = f" at Epoch {epoch + 1}"
        else:
            epoch_info = ""

        logger.info(
            f"{prefix} Results{epoch_info}: "
            f"Threshold: {metrics.get('threshold', 'N/A'):.4f}, "
            f"Val Loss: {metrics.get('val_loss', 'N/A'):.4f}, "
            f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}, "
            f"F1-Score: {metrics.get('f1_score', 'N/A'):.4f}, "
            f"Precision: {metrics.get('precision', 'N/A'):.4f}, "
            f"Recall: {metrics.get('recall', 'N/A'):.4f}, "
            f"FP Ratio: {metrics.get('fp_ratio', 'N/A'):.4f}, "
            f"FN Ratio: {metrics.get('fn_ratio', 'N/A'):.4f}, "
            f"FP/(TP+FP): {metrics.get('fp_over_tp_fp', 'N/A'):.4f}, "
            f"FN/(FN+TN): {metrics.get('fn_over_fn_tn', 'N/A'):.4f}\n"
            f"Confusion Matrix:\n{metrics.get('confusion_matrix', 'N/A')}"
        )

    def log_to_tensorboard(self, writer, metrics, epoch, prefix="Validation"):
        """Log metrics to TensorBoard."""
        for key in self.metric_names:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                writer.add_scalar(f"{prefix}/{key}", value, epoch + 1)


# Add argument parser
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
    parser.add_argument(
        "--embedding_dim", type=int, default=16, help="Embedding dimension"
    )
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument(
        "--fc_hidden_dim",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Fully connected layer dimensions",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers in the model"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--class_counts_dim", type=int, default=8, help="Dimension for class counts"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./ckpt/class_on_detect.pth",
        help="Path to save the model checkpoint",
    )

    parser.add_argument(
        "--criterion", type=str, default="f1_score", help="Selection criterion"
    )
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")

    parser.add_argument(
        "--tb_log_dir", type=str, default="./tb_logs", help="TensorBoard log directory"
    )

    args, unknown = parser.parse_known_args()  # Capture unknown arguments
    if unknown:
        print(f"Unknown arguments passed: {unknown}")

    return args


# Main training code
if __name__ == "__main__":
    args = parse_args()

    # Directories and files
    pos_dir = args.pos_dir
    neg_dir = args.neg_dir
    classes_file = args.classes_file
    batch_size = args.batch_size
    lr = args.lr
    criterion = args.criterion
    device_id = args.device_id
    tb_log_dir = args.tb_log_dir

    # Device configuration
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Get the dataloaders
    dataloaders = get_dataloader(
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        classes_file=classes_file,
        split=[0.8, 0.2],
        batch_size=batch_size,
        balance_ratio=1,
        resample_method="both",
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Get PAD_LABEL and num_labels from the dataset
    dataset = train_loader.dataset
    PAD_LABEL = dataset.PAD_LABEL
    num_classes = dataset.num_classes
    num_labels = dataset.num_labels
    CLASS_COUNTS_LABEL = dataset.CLASS_COUNTS_LABEL

    # Model
    model = DetectionClassificationModel(
        num_classes=num_classes,
        num_labels=num_labels,
        CLASS_COUNTS_LABEL=CLASS_COUNTS_LABEL,
        PAD_LABEL=PAD_LABEL,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        fc_hidden_dim=args.fc_hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        class_counts_dim=args.class_counts_dim,
    ).to(device)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        device=device,
        criterion=criterion,
        tb_log_dir=tb_log_dir,
    )

    # Train the model
    trainer.train(num_epochs=args.num_epochs)

    # Save the trained model
    if not os.path.exists(os.path.dirname(args.checkpoint_path)):
        os.makedirs(os.path.dirname(args.checkpoint_path))
