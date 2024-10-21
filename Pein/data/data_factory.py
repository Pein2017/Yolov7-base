# Instead, get the global logger
import logging  # noqa
import os
import random
import sys
from collections import OrderedDict
from functools import partial

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to the system path
sys.path.append(parent_dir)
from utils.logging import setup_logger  # noqa

# logger = logging.getLogger("train_optuna")  # for training
logger = setup_logger(
    log_file="data_debug.log",
    name="data",
    level="DEBUG",
    log_to_console=True,
    overwrite=True,
)  # for debug and test


class GroupedDetectionDataset(Dataset):
    """
    A PyTorch Dataset class for handling grouped detection data.
    Each sample consists of a group of figures (images) associated with an experiment ID.
    The dataset processes YOLO detection outputs (text files) and constructs sequences
    suitable for input into a model that accepts object-level, figure-level, and group-level features.
    """

    def __init__(
        self,
        pos_dir,
        neg_dir,
        classes_file,
        csv_file,  # Add csv_file parameter
        transform=None,
        balance_ratio=None,
        resample_method="downsample",
        indices=None,
    ):
        # Initialization code (unchanged)
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.transform = transform

        # Ensure class_names is properly set
        self.class_names = self._load_classes(classes_file)
        self.num_classes = len(self.class_names)

        # Initialize sample containers
        self.samples_dict = {}
        self.positive_samples = []
        self.negative_samples = []

        # Process positive and negative directories
        self._load_samples_from_directory(self.pos_dir, label=1)
        self._load_samples_from_directory(self.neg_dir, label=0)

        # Combine all samples into a list
        self.samples = self.positive_samples + self.negative_samples

        # Subset samples if indices are provided
        if indices is not None:
            self._subset_samples(indices)

        # Resample to achieve desired balance ratio
        if balance_ratio is not None:
            self._resample_samples(balance_ratio, resample_method)

        # Find the maximum number of figures in a group
        self.max_figures_in_group = max(len(sample["files"]) for sample in self.samples)

        # Log the dataset statistics
        self._log_statistics()

        # Define special labels
        self.CLASS_COUNTS_LABEL = self.num_classes  # 5
        self.SEP_LABEL = self.num_classes + 1  # 6
        self.PAD_LABEL = self.num_classes + 2  # 7
        self.num_labels = self.num_classes + 3  # 8 labels (0 to 7)

        # Define per-object features and their computation functions
        self.object_feature_functions = OrderedDict(
            [
                (
                    "coarse_label",
                    lambda obj_data: 0 if int(obj_data[0]) in [0, 1, 3] else 1,
                ),  # 0 for 'bbu', 1 for 'shield'
                ("class_label", lambda obj_data: int(obj_data[0])),
                ("x", lambda obj_data: obj_data[1]),
                ("y", lambda obj_data: obj_data[2]),
                ("w", lambda obj_data: obj_data[3]),
                ("h", lambda obj_data: obj_data[4]),
                ("confidence", lambda obj_data: obj_data[5]),
                ("area", lambda obj_data: obj_data[3] * obj_data[4]),
                # Add more features here as needed
            ]
        )

        # Total number of features per object entry
        self.num_object_features = len(self.object_feature_functions)

        self.feature_vector_length = (
            self.num_object_features  # Per-object features length
            + (self.num_classes + 1)  # Figure-wise features length
            + (self.num_classes + 1)  # Group-wise features length
        )

        # Load CSV data for figure features
        self.figure_features = pd.read_csv(csv_file)
        self.figure_features["id"] = (
            self.figure_features["id"].astype(str).str.strip().str.lower()
        )

        logger.debug(f"Total rows loaded from CSV: {len(self.figure_features)}")

        # Check for duplicate IDs
        duplicate_ids = self.figure_features[self.figure_features["id"].duplicated()][
            "id"
        ].unique()
        if len(duplicate_ids) > 0:
            logger.warning(f"Duplicate IDs found in CSV: {duplicate_ids.tolist()}")

        # Check for IDs with unusual formats
        unusual_ids = [
            idx
            for idx in self.figure_features["id"]
            if not idx.replace("-", "").isdigit()
        ]
        if unusual_ids:
            logger.warning(f"IDs with unusual format found: {unusual_ids[:10]}...")

    def _load_classes(self, classes_file):
        """Load class names from a file."""
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def _load_samples_from_directory(self, directory, label):
        """Process files in a directory and group them by unique experiment ID."""
        files = os.listdir(directory)
        for file_name in files:
            # Assume file names start with experiment ID separated by '-'
            experiment_id = file_name.split("-")[0]
            file_path = os.path.join(directory, file_name)

            if experiment_id not in self.samples_dict:
                sample = {"files": [], "label": label}
                self.samples_dict[experiment_id] = sample
                if label == 1:
                    self.positive_samples.append(sample)
                else:
                    self.negative_samples.append(sample)

            self.samples_dict[experiment_id]["files"].append(file_path)

    def _subset_samples(self, indices):
        """Subset the samples based on provided indices."""
        self.samples = [self.samples[i] for i in indices]
        # Update positive_samples and negative_samples
        self.positive_samples = [s for s in self.samples if s["label"] == 1]
        self.negative_samples = [s for s in self.samples if s["label"] == 0]

    def _resample_samples(self, desired_ratio, resample_method):
        """Resample samples to achieve the desired balance ratio."""
        num_positive = len(self.positive_samples)
        num_negative = len(self.negative_samples)

        if num_positive == 0 or num_negative == 0:
            logger.error("Cannot resample because one of the classes has zero samples.")
            return

        # Define majority and minority groups
        if num_positive > num_negative:
            majority_samples = self.positive_samples
            minority_samples = self.negative_samples
            majority_label = "positive"
            minority_label = "negative"
        else:
            majority_samples = self.negative_samples
            minority_samples = self.positive_samples
            majority_label = "negative"
            minority_label = "positive"

        majority_count = len(majority_samples)
        minority_count = len(minority_samples)

        # Resampling logic based on method
        if resample_method == "upsample":
            logger.debug(
                f"Upsampling the {minority_label} samples to match the {majority_label}."
            )
            minority_samples = self._adjust_samples(
                minority_samples, majority_count, "upsample"
            )
        elif resample_method == "downsample":
            logger.debug(
                f"Downsampling the {majority_label} samples to match the {minority_label}."
            )
            majority_samples = self._adjust_samples(
                majority_samples, minority_count, "downsample"
            )
        elif resample_method == "both":
            # Compute desired counts based on the ratio
            total_samples = num_positive + num_negative
            desired_num_positive = int(
                (desired_ratio * total_samples) / (1 + desired_ratio)
            )
            desired_num_negative = total_samples - desired_num_positive

            # Ensure counts are at least 1
            desired_num_positive = max(desired_num_positive, 1)
            desired_num_negative = max(desired_num_negative, 1)

            logger.debug(
                f"Resampling both classes based on desired ratio: {desired_ratio}."
            )
            self.positive_samples = self._adjust_samples(
                self.positive_samples, desired_num_positive, "both"
            )
            self.negative_samples = self._adjust_samples(
                self.negative_samples, desired_num_negative, "both"
            )
        else:
            logger.warning(
                f"Resample method '{resample_method}' not recognized. No resampling performed."
            )
            return

        # Combine the resampled positive and negative samples
        if resample_method in ["upsample", "downsample"]:
            if majority_label == "positive":
                self.positive_samples = majority_samples
                self.negative_samples = minority_samples
            else:
                self.positive_samples = minority_samples
                self.negative_samples = majority_samples

        self.samples = self.positive_samples + self.negative_samples
        random.shuffle(self.samples)

    def _adjust_samples(self, samples, desired_count, resample_method):
        """Adjust samples to match the desired count based on the resampling method."""
        current_count = len(samples)

        if resample_method == "downsample":
            if current_count > desired_count:
                samples = random.sample(samples, desired_count)
        elif resample_method == "upsample":
            if current_count < desired_count:
                factor = desired_count // current_count
                remainder = desired_count % current_count
                samples = samples * factor + random.choices(samples, k=remainder)
        elif resample_method == "both":
            if current_count > desired_count:
                samples = random.sample(samples, desired_count)
            elif current_count < desired_count:
                factor = desired_count // current_count
                remainder = desired_count % current_count
                samples = samples * factor + random.choices(samples, k=remainder)
        else:
            logger.warning(
                f"Resample method '{resample_method}' not recognized. No resampling performed."
            )

        return samples

    def _log_statistics(self):
        """Log statistics about the dataset."""
        num_positive = len(self.positive_samples)
        num_negative = len(self.negative_samples)
        total_samples = num_positive + num_negative
        logger.debug(f"Number of positive samples: {num_positive}")
        logger.debug(f"Number of negative samples: {num_negative}")
        logger.debug(f"Max number of figures in a group: {self.max_figures_in_group}")
        logger.debug(
            f"Label distribution: {num_positive / total_samples:.2%} positive, {num_negative / total_samples:.2%} negative."
        )

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        files = sample["files"]
        label = sample["label"]

        # Lists to hold features
        object_features = []
        figure_features = []

        # Initialize group-level variables
        total_coarse_counts = [0.0, 0.0]  # [total_bbu_count, total_shield_count]
        total_class_counts = [0.0] * self.num_classes
        total_num_figures = len(files)

        logger.debug(f"Processing sample {idx} with {total_num_figures} figures")

        for i, file_path in enumerate(files):
            # Process each figure and get its entries along with integrated figure feature
            obj_entries, fig_feature = self._process_figure(file_path)

            # Append object entries
            object_features.extend(obj_entries)

            # Append figure feature
            figure_features.append(fig_feature)

            # Append SEP_LABEL between figure entries (except after the last figure)
            if i < len(files) - 1:
                sep_entry = self._get_sep_figure_entry()
                figure_features.append(sep_entry)

            # Validate fig_feature length
            expected_length = (
                2 + self.num_classes + 2
            )  # coarse_count + class_count + width + height
            if len(fig_feature) != expected_length:
                raise ValueError(
                    f"fig_feature length ({len(fig_feature)}) does not match expected length ({expected_length})"
                )

            # Accumulate total coarse counts and class counts
            total_coarse_counts = [
                x + y for x, y in zip(total_coarse_counts, fig_feature[:2])
            ]
            total_class_counts = [
                x + y
                for x, y in zip(
                    total_class_counts, fig_feature[2 : 2 + self.num_classes]
                )
            ]

            logger.debug(f"After processing figure {i+1}:")
            logger.debug(f"  fig_feature: {fig_feature}")
            logger.debug(f"  total_coarse_counts: {total_coarse_counts}")
            logger.debug(f"  total_class_counts: {total_class_counts}")

        # Prepare group_feature
        group_feature = (
            total_coarse_counts + total_class_counts + [float(total_num_figures)]
        )

        # Assert the correct length of group_feature
        assert len(group_feature) == 2 + self.num_classes + 1, (
            f"group_feature length ({len(group_feature)}) does not match "
            f"expected length ({2 + self.num_classes + 1})"
        )

        logger.debug(f"Final group_feature: {group_feature}")

        # Calculate lengths
        object_length = len(object_features)
        figure_length = len(figure_features)

        return [
            object_features,
            figure_features,
            group_feature,
            object_length,
            figure_length,
            label,
        ]

    def _process_figure(self, file_path):
        obj_entries = []

        # Initialize class_counts and coarse_counts for this figure
        fig_count = [0.0] * self.num_classes
        coarse_count = [0.0, 0.0]  # [bbu_count, shield_count]

        # Extract the unique_id from the file_path
        unique_id = os.path.splitext(os.path.basename(file_path))[0].strip().lower()

        logger.debug(f"Processing unique_id: '{unique_id}'")

        # Retrieve width and height from the CSV using the unique_id
        matching_row = self.figure_features[self.figure_features["id"] == unique_id]
        if matching_row.empty:
            logger.error(f"ID '{unique_id}' not found in the CSV file.")
            raise KeyError(f"Image ID '{unique_id}' not found in fig_size_path CSV.")

        fig_size = matching_row[["width", "height"]].astype(float).values[0].tolist()

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            object_data = list(map(float, line.strip().split()))

            # Compute per-object features (including coarse_label and class_label)
            features = [
                func(object_data) for func in self.object_feature_functions.values()
            ]

            class_label = int(object_data[0])
            coarse_label = 0 if class_label in [0, 1, 3] else 1
            confidence = object_data[5]

            # Update weighted class counts and coarse counts
            if 0 <= class_label < self.num_classes:
                fig_count[class_label] += confidence
                coarse_count[coarse_label] += confidence

            obj_entries.append(features)

        assert (
            len(coarse_count) == 2
        ), f"Expected 2 coarse counts, got {len(coarse_count)}"
        assert (
            len(fig_count) == self.num_classes
        ), f"Expected {self.num_classes} class counts, got {len(fig_count)}"
        assert len(fig_size) == 2, f"Expected 2 size values, got {len(fig_size)}"

        # Flatten the fig_feature
        fig_feature = coarse_count + fig_count + fig_size

        expected_length = (
            2 + self.num_classes + 2
        )  # coarse_count + class_count + width + height
        if len(fig_feature) != expected_length:
            raise ValueError(
                f"fig_feature length ({len(fig_feature)}) does not match expected length ({expected_length})"
            )

        logger.debug("_process_figure output:")
        logger.debug(f"coarse_count: {coarse_count}")
        logger.debug(f"fig_count: {fig_count}")
        logger.debug(f"fig_size: {fig_size}")
        logger.debug(f"fig_feature: {fig_feature}")

        return obj_entries, fig_feature

    def _get_sep_object_entry(self):
        # Create a separator entry for object_feature
        # The length should match the length of obj_entry
        sep_entry = [self.SEP_LABEL] + [0.0] * (self.num_object_features - 1)
        return sep_entry

    def _get_sep_figure_entry(self):
        # Create a separator entry for figure_feature
        return [self.SEP_LABEL] + [0.0] * (
            2 + self.num_classes + 2 - 1
        )  # coarse_counts + class_counts + fig_size - 1 for SEP_LABEL

    def _get_separator_entry(self):
        # For per-object features, use default values
        features = [0.0] * self.num_object_features

        # class_label
        class_label = self.SEP_LABEL

        # Figure-wise features: zeros
        figure_features = [0.0] * (
            2 + self.num_classes + 2
        )  # coarse_counts + class_counts + fig_size

        # Group-wise features: zeros
        group_features = [0.0] * (
            2 + self.num_classes + 1
        )  # coarse_counts + class_counts + total_figures

        entry = [class_label] + features + figure_features + group_features
        return entry


def custom_collate_fn(batch, padding_label):
    """
    Custom collate function to pad sequences and organize batch data.

    Args:
        batch (list): List of tuples (object_features, figure_features, group_features, object_length, figure_length, label).
        padding_label (int): The label to use for padding.

    Returns:
        tuple: (object_features_padded, figure_features_padded, group_features_tensor, object_lengths, figure_lengths, labels_tensor)
    """
    # Unpack each component from the batch
    batch_data = list(zip(*batch))

    (
        object_features,
        figure_features,
        group_features,
        object_lengths,
        figure_lengths,
        labels,
    ) = batch_data

    # Convert lengths to integer tensors
    object_lengths = torch.tensor(object_lengths, dtype=torch.long)
    figure_lengths = torch.tensor(figure_lengths, dtype=torch.long)

    # Debug log
    logger.debug("In custom_collate_fn:")
    logger.debug(f"object_features[0] length: {len(object_features[0])}")
    logger.debug(f"figure_features[0] length: {len(figure_features[0])}")
    logger.debug(f"group_features[0] length: {len(group_features[0])}")

    # Pad object_features
    max_object_length = max(len(of) for of in object_features)
    object_feature_dim = len(object_features[0][0])
    pad_entry_obj = [padding_label] + [0.0] * (object_feature_dim - 1)

    object_features_padded = []
    for of in object_features:
        pad_length = max_object_length - len(of)
        if pad_length > 0:
            padding = [pad_entry_obj] * pad_length
            padded_of = of + padding
        else:
            padded_of = of
        object_features_padded.append(padded_of)
    object_features_padded = torch.tensor(object_features_padded, dtype=torch.float32)

    # Pad figure_features
    max_figure_length = max(len(ff) for ff in figure_features)
    fig_feature_dim = len(figure_features[0][0]) if figure_features[0] else 0
    pad_entry_fig = [padding_label] + [0.0] * (fig_feature_dim - 1)

    figure_features_padded = []
    for ff in figure_features:
        pad_length = max_figure_length - len(ff)
        if pad_length > 0:
            padding = [pad_entry_fig] * pad_length
            padded_ff = ff + padding
        else:
            padded_ff = ff

        # Ensure all entries have the same length
        if len(padded_ff) > 0 and len(padded_ff[0]) != fig_feature_dim:
            logger.error(
                f"Mismatch in figure feature dimension: expected {fig_feature_dim}, got {len(padded_ff[0])}"
            )
            padded_ff = [entry[:fig_feature_dim] for entry in padded_ff]

        figure_features_padded.append(padded_ff)

    # Convert to tensor after padding
    figure_features_padded = torch.tensor(figure_features_padded, dtype=torch.float32)

    # Stack group_features
    group_features_tensor = torch.tensor(group_features, dtype=torch.float32)

    logger.debug("In custom_collate_fn:")
    logger.debug(f"group_features: {group_features}")
    logger.debug(f"group_features_tensor shape: {group_features_tensor.shape}")

    # Convert labels to tensors
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Debug logging for figure features
    logger.debug("Figure features before padding:")
    for i, ff in enumerate(figure_features):
        logger.debug(f"Sample {i}: shape = {len(ff)}, {len(ff[0]) if ff else 0}")

    return (
        object_features_padded,
        figure_features_padded,
        group_features_tensor,
        object_lengths,
        figure_lengths,
        labels_tensor,
    )


def get_dataloader(
    pos_dir,
    neg_dir,
    classes_file,
    csv_file,  # Added parameter for fig_size_path
    split=[0.8, 0.2],
    batch_size=32,
    balance_ratio: int = None,
    resample_method: str = "downsample",
    seed: int = 17,
):
    """Create DataLoaders for training, validation, and testing."""
    assert len(split) in [
        2,
        3,
    ], "Split should be either [train, val] or [train, val, test]."

    # Create the full dataset with fig_size_path
    full_dataset = GroupedDetectionDataset(
        pos_dir,
        neg_dir,
        classes_file,
        csv_file,  # Pass csv_file
    )

    PAD_LABEL = full_dataset.PAD_LABEL  # Assuming PAD_LABEL is num_classes + 2

    # Get the total number of samples
    dataset_size = len(full_dataset)

    # Generate indices for splitting
    train_val_indices = list(range(dataset_size))
    random.seed(seed)
    random.shuffle(train_val_indices)

    split1 = int(split[0] * dataset_size)
    train_indices = train_val_indices[:split1]
    val_indices = train_val_indices[split1:]

    # Create the training dataset with resampling
    train_dataset = GroupedDetectionDataset(
        pos_dir,
        neg_dir,
        classes_file,
        csv_file,  # Pass csv_file
        balance_ratio=balance_ratio,
        resample_method=resample_method,
        indices=train_indices,
    )

    # Create the validation dataset without resampling
    val_dataset = GroupedDetectionDataset(
        pos_dir,
        neg_dir,
        classes_file,
        csv_file,  # Pass csv_file
        balance_ratio=balance_ratio,
        resample_method=resample_method,
        indices=val_indices,
    )

    # Create the test dataset without resampling
    test_dataset = None

    # Create the collate function with PAD_LABEL
    collate_fn_with_pad = partial(
        custom_collate_fn,
        padding_label=PAD_LABEL,
    )

    # Create DataLoaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_with_pad,
            drop_last=False,
            num_workers=0,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_pad,
            drop_last=False,
        ),
    }

    if test_dataset:
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_pad,
        )

    return dataloaders


def print_class_counts(feature, class_names, is_figure=False, is_group=False):
    """Helper function to print class counts with class names."""
    if isinstance(feature, torch.Tensor):
        feature = feature.tolist()

    if is_group:
        # For group features, we expect 2 + num_classes + 1 elements
        expected_length = len(class_names) + 3
        if len(feature) != expected_length:
            raise ValueError(
                f"group_feature length ({len(feature)}) does not match "
                f"expected length ({expected_length})"
            )
        coarse_count = feature[:2]
        class_count = feature[2:-1]  # All but the last element are class counts
        total_num_figures = feature[-1]  # Last element is total number of figures
    elif is_figure:
        # For figure features, we expect 2 + num_classes + 2 elements (including figure size)
        expected_length = len(class_names) + 4
        if len(feature) != expected_length:
            raise ValueError(
                f"figure_feature length ({len(feature)}) does not match "
                f"expected length ({expected_length})"
            )
        coarse_count = feature[:2]
        class_count = feature[2:-2]  # All but the last two elements are class counts
        fig_size = feature[-2:]  # Last two elements are figure size
    else:
        raise ValueError("Must specify either is_figure=True or is_group=True")

    logger.debug("Coarse-grained Counts:")
    logger.debug(f"  BBU: {coarse_count[0]:.4f}")
    logger.debug(f"  Shield: {coarse_count[1]:.4f}")

    logger.debug("Class Counts:")
    for cls_idx, count in enumerate(class_count):
        cls_name = class_names[cls_idx]
        logger.debug(f"  {cls_name}: {count:.4f}")

    if is_group:
        logger.debug(f"Total Number of Figures: {total_num_figures:.0f}")
    elif is_figure:
        logger.debug(f"Figure Size: {fig_size[0]:.0f} x {fig_size[1]:.0f}")


def main():
    # Directories and files
    pos_dir = "/data/training_code/yolov7/runs/detect/v7-p5-lr_1e-3/pos/labels"
    neg_dir = "/data/training_code/yolov7/runs/detect/v7-p5-lr_1e-3/neg/labels"
    classes_file = "/data/dataset/bbu_training_data/bbu_and_shield/classes.txt"
    fig_size_path = "/data/training_code/yolov7/Pein/fig_size.csv"

    # Desired balance ratio and resampling method
    balance_ratio = 1.0  # Desired positive to negative ratio (1:1)
    resample_method = "no"  # Can be 'downsample', 'upsample', or 'both'

    # Create DataLoaders using the get_dataloader function
    dataloaders = get_dataloader(
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        classes_file=classes_file,
        csv_file=fig_size_path,
        split=[0.8, 0.2],
        batch_size=2,  # For clearer logging
        balance_ratio=balance_ratio,
        resample_method=resample_method,
    )

    # Access the train_loader
    train_loader = dataloaders["train"]

    # Get special labels and class names from the dataset
    PAD_LABEL = train_loader.dataset.PAD_LABEL
    SEP_LABEL = train_loader.dataset.SEP_LABEL
    class_names = train_loader.dataset.class_names

    # Iterate over the train_loader
    for batch_idx, batch in enumerate(train_loader):
        logger.debug(f"\n{'='*50}")
        logger.debug(f"Batch {batch_idx + 1}")
        logger.debug(f"{'='*50}")

        # Unpack the batch
        (
            object_features_padded,
            figure_features_padded,
            group_features_tensor,
            object_lengths,
            figure_lengths,
            labels_tensor,
        ) = batch

        logger.debug(f"Batch Labels: {labels_tensor.numpy()}")

        # Iterate over the samples in the batch
        for sample_idx in range(object_features_padded.size(0)):
            logger.debug(f"\n{'-'*50}")
            logger.debug(f"Sample {sample_idx + 1}")
            logger.debug(f"{'-'*50}")

            # Get lengths
            obj_length = object_lengths[sample_idx].item()
            fig_length = figure_lengths[sample_idx].item()

            # Get object features and figure features for this sample
            object_sequence = object_features_padded[sample_idx][:obj_length]
            figure_sequence = figure_features_padded[sample_idx][:fig_length]
            group_feature = group_features_tensor[sample_idx]

            # Initialize variables to keep track of the current figure
            current_figure_idx = 1
            current_figure_objects = []

            # 1. Object-level and Figure-level data
            logger.debug("\n1. Object-level and Figure-level Data:")
            logger.debug("=" * 40)

            for obj_idx, obj_feature in enumerate(object_sequence):
                coarse_label = int(obj_feature[0].item())
                class_label = int(obj_feature[1].item())

                if class_label == SEP_LABEL:
                    # Log the objects for the current figure
                    logger.debug(f"\nFigure {current_figure_idx} Objects:")
                    for i, obj in enumerate(current_figure_objects, 1):
                        logger.debug(f"  Object {i}:")
                        for fname, fvalue in obj:
                            logger.debug(f"    {fname}: {fvalue:.4f}")

                    # Log the figure-level data
                    logger.debug(f"\nFigure {current_figure_idx} Summary:")
                    try:
                        print_class_counts(
                            figure_sequence[current_figure_idx - 1].tolist(),
                            class_names,
                            is_figure=True,
                        )
                    except ValueError as e:
                        logger.error(f"Error in Figure {current_figure_idx}: {str(e)}")

                    logger.debug("-" * 40)

                    # Reset for the next figure
                    current_figure_idx += 1
                    current_figure_objects = []

                elif class_label != PAD_LABEL:
                    # Add object to the current figure's list
                    obj_data = []
                    feature_names = list(
                        train_loader.dataset.object_feature_functions.keys()
                    )
                    for fname, fvalue in zip(feature_names, obj_feature):
                        obj_data.append((fname, fvalue.item()))
                    current_figure_objects.append(obj_data)

            # Log the last figure if it wasn't followed by a separator
            if current_figure_objects:
                logger.debug(f"\nFigure {current_figure_idx} Objects:")
                for i, obj in enumerate(current_figure_objects, 1):
                    logger.debug(f"  Object {i}:")
                    for fname, fvalue in obj:
                        logger.debug(f"    {fname}: {fvalue:.4f}")

                logger.debug(f"\nFigure {current_figure_idx} Summary:")
                try:
                    print_class_counts(
                        figure_sequence[current_figure_idx - 1].tolist(),
                        class_names,
                        is_figure=True,
                    )
                except ValueError as e:
                    logger.error(f"Error in Figure {current_figure_idx}: {str(e)}")

            # 2. Group-level data
            logger.debug("\n2. Group-level Data:")
            logger.debug("=" * 40)
            try:
                print_class_counts(group_feature.numpy(), class_names, is_group=True)
            except ValueError as e:
                logger.error(f"Error in Group-level data: {str(e)}")
                logger.error(f"group_feature content: {group_feature.numpy()}")

        # Break after the first batch for testing
        break


if __name__ == "__main__":
    main()
