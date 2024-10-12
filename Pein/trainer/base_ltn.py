from enum import Enum, auto
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torchmetrics

from .utils import Args


class Phase(Enum):
    train = auto()
    val = auto()
    test = auto()

    @staticmethod
    def from_string(phase_str: str):
        phase_str = phase_str.lower()
        if phase_str == "train":
            return Phase.train
        elif phase_str in ["val", "validation"]:
            return Phase.val
        elif phase_str == "test":
            return Phase.test
        else:
            raise ValueError(
                f"Invalid phase: {phase_str}. Choose from 'train', 'val', 'test'."
            )


class LtnBaseModule(pl.LightningModule):
    def __init__(self, args: Args, model: torch.nn.Module = None):
        """
        Base module for training. Subclasses must implement specific methods.

        Args:
            args (Args): Configuration arguments.
            model (torch.nn.Module, optional): The model to train. If None, should be created by `create_model`.
        """
        super(LtnBaseModule, self).__init__()
        self.args = args
        self.model = self.create_model(args) if model is None else model
        self.criterion = self.configure_loss_function(args)

        # Initialize metrics
        self.loss_metrics = {phase: torchmetrics.MeanMetric() for phase in Phase}
        self.epoch_results = {phase: {"outputs": [], "labels": []} for phase in Phase}

    def on_fit_start(self):
        # Move metrics to the correct device when training starts
        device = self.device
        self.loss_metrics = {
            phase: metric.to(device) for phase, metric in self.loss_metrics.items()
        }

    def forward(self, *inputs) -> torch.Tensor:
        """General forward pass through the model."""
        outputs = self.model(*inputs)
        if not isinstance(outputs, torch.Tensor):
            raise TypeError("Model output must be a torch.Tensor.")
        # Ensure outputs always have at least 2 dimensions
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(-1)
        return outputs

    def step(self, batch: tuple, phase: Phase) -> torch.Tensor:
        """
        Perform a forward pass and compute loss.

        Args:
            batch (tuple): Batch of data.
            phase (Phase): One of Phase.train, Phase.validation, or Phase.test.

        Returns:
            torch.Tensor: Computed loss.
        """
        *inputs, labels = batch
        inputs = [input.to(self.device) for input in inputs]
        labels = labels.to(self.device)

        outputs = self(*inputs)

        # Ensure labels have the same shape as outputs
        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)

        # Ensure outputs and labels have the same number of dimensions
        if outputs.ndim != labels.ndim:
            raise ValueError(
                f"Dimension mismatch: outputs.shape={outputs.shape}, labels.shape={labels.shape}"
            )

        loss = self.criterion(outputs, labels)

        # Update metrics
        self.loss_metrics[phase].update(loss.detach())
        self.epoch_results[phase]["outputs"].append(outputs.detach())
        self.epoch_results[phase]["labels"].append(labels.detach())

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step executed during training."""
        return self.step(batch, Phase.train)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step executed during validation."""
        self.step(batch, Phase.val)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step executed during testing."""
        self.step(batch, Phase.test)

    def on_epoch_end(self, phase: Phase) -> None:
        """
        Handle end of epoch for a given phase.

        Args:
            phase (Phase): One of Phase.train, Phase.val, or Phase.test.
        """
        avg_loss = self.loss_metrics[phase].compute()
        # Remove the following line to prevent logging 'train_loss' and 'val_loss'
        # self.log(f"{phase.name}_loss", avg_loss, prog_bar=True, logger=True)
        self.log(f"{phase.name}/loss", avg_loss, prog_bar=True, logger=True)
        self.loss_metrics[phase].reset()

        outputs = torch.cat(self.epoch_results[phase]["outputs"])
        labels = torch.cat(self.epoch_results[phase]["labels"])

        self.custom_epoch_end_logic(phase, outputs, labels)
        self.epoch_results[phase]["outputs"].clear()
        self.epoch_results[phase]["labels"].clear()

    # Epoch end handlers for different phases
    def on_train_epoch_end(self) -> None:
        """Handle end of training epoch."""
        self.on_epoch_end(Phase.train)

    def on_validation_epoch_end(self) -> None:
        """Handle end of validation epoch."""
        self.on_epoch_end(Phase.val)

    def on_test_epoch_end(self) -> None:
        """Handle end of test epoch."""
        self.on_epoch_end(Phase.test)

    # Abstract methods to be implemented by subclasses
    def create_model(self, args: Args) -> torch.nn.Module:
        raise NotImplementedError("Subclasses must implement this method")

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.

        Must be implemented by subclasses.

        Returns:
            Optimizer or list of optimizers.
        """
        raise NotImplementedError(
            "Subclasses must implement `configure_optimizers` method."
        )

    def configure_loss_function(self, args: Args):
        """
        Configure the loss function.
        Args:
            args (Args): Configuration arguments.
        Returns:
            torch.nn.Module: Loss function.
        """
        raise NotImplementedError(
            "Subclasses must implement `configure_loss_function` method."
        )

    def custom_epoch_end_logic(
        self, phase: Phase, outputs: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """
        Hook for custom end-of-epoch logic. Must be implemented by subclasses.

        Args:
            phase (Phase): The current phase (Phase.train, Phase.val, or Phase.test).
            outputs (torch.Tensor): Model outputs for the entire epoch.
            labels (torch.Tensor): True labels for the entire epoch.
        """
        raise NotImplementedError(
            "Subclasses must implement `custom_epoch_end_logic` method."
        )


def main():
    pass


if __name__ == "__main__":
    main()
