# Instead, get the global logger
import logging
import os
import random
import sys
from collections import OrderedDict
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to the system path
sys.path.append(parent_dir)


from utils.setup import setup_logger  # noqa

logger = logging.getLogger("train_optuna")


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
        transform=None,
        balance_ratio=None,
        resample_method="downsample",
        indices=None,
    ):
        # Initialization code (unchanged)
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self.transform = transform

        # Load class names and set number of classes
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
        object_feature = []
        figure_feature = []

        # Initialize group-level variables
        total_class_counts = [0.0] * self.num_classes
        total_num_figures = len(files)

        for i, file_path in enumerate(files):
            # Process each figure and get its entries
            obj_entries, fig_class_counts = self._process_figure(file_path)

            # Append object entries
            object_feature.extend(obj_entries)

            # Append SEP_LABEL between figures (except after the last figure)
            if i < len(files) - 1:
                object_feature.append(self._get_sep_object_entry())

            # Append figure entry
            figure_feature.append(fig_class_counts)

            # Append SEP_LABEL between figure entries (except after the last figure)
            if i < len(files) - 1:
                figure_feature.append(self._get_sep_figure_entry())

            # Accumulate total class counts
            total_class_counts = [
                x + y for x, y in zip(total_class_counts, fig_class_counts)
            ]

        # Prepare group_feature
        group_feature = total_class_counts + [float(total_num_figures)]

        # Convert features to tensors
        object_feature = torch.tensor(object_feature, dtype=torch.float32)
        figure_feature = torch.tensor(figure_feature, dtype=torch.float32)
        group_feature = torch.tensor(group_feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Calculate lengths
        object_length = object_feature.size(0)
        figure_length = figure_feature.size(0)

        return (
            object_feature,
            figure_feature,
            group_feature,
            object_length,
            figure_length,
            label,
        )

    def _process_figure(self, file_path):
        obj_entries = []

        # Initialize class_counts for this figure
        class_counts = [0.0] * self.num_classes

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            object_data = list(map(float, line.strip().split()))
            class_label = int(object_data[0])

            # Compute per-object features (including class_label)
            features = [
                func(object_data) for func in self.object_feature_functions.values()
            ]

            # Update weighted class counts
            if 0 <= class_label < self.num_classes:
                confidence = object_data[5]
                class_counts[class_label] += confidence

            obj_entries.append(features)

        return obj_entries, class_counts

    def _get_sep_object_entry(self):
        # Create a separator entry for object_feature
        # The length should match the length of obj_entry
        sep_entry = [self.SEP_LABEL] + [0.0] * (self.num_object_features - 1)
        return sep_entry

    def _get_sep_figure_entry(self):
        # Create a separator entry for figure_feature
        # The length should match the number of classes
        sep_entry = [self.SEP_LABEL] + [0.0] * (self.num_classes - 1)
        return sep_entry

    def _get_separator_entry(self):
        # For per-object features, use default values
        features = [0.0] * self.num_object_features

        # class_label
        class_label = self.SEP_LABEL

        # Figure-wise features: zeros
        figure_features = [0.0] * (self.num_classes + 1)

        # Group-wise features: zeros
        group_features = [0.0] * (self.num_classes + 1)

        entry = [class_label] + features + figure_features + group_features
        return entry

    def _get_default_value(self, feature_name, entry_type="object"):
        """Provide default values for features based on entry type."""
        if feature_name == "class_label":
            if entry_type == "figure":
                return self.CLASS_COUNTS_LABEL
            elif entry_type == "separator":
                return self.SEP_LABEL
            elif entry_type == "group":
                return self.TOTAL_CLASS_COUNTS_LABEL
            elif entry_type == "padding":
                return self.PAD_LABEL
            else:
                return 0.0  # For object entries
        else:
            return 0.0  # Default for other features


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
    (
        object_features,
        figure_features,
        group_features,
        object_lengths,
        figure_lengths,
        labels,
    ) = zip(*batch)

    # Convert lengths to integer tensors
    object_lengths = torch.tensor(object_lengths, dtype=torch.long)
    figure_lengths = torch.tensor(figure_lengths, dtype=torch.long)

    # Pad object_features
    max_object_length = object_lengths.max().item()
    object_feature_dim = object_features[0].size(1)
    pad_entry_obj = [padding_label] + [0.0] * (object_feature_dim - 1)
    pad_entry_obj = torch.tensor(pad_entry_obj, dtype=torch.float32)

    object_features_padded = []
    for of in object_features:
        pad_length = max_object_length - of.size(0)
        if pad_length > 0:
            # Ensure pad_length is an integer
            padding = pad_entry_obj.unsqueeze(0).repeat(pad_length, 1)
            padded_of = torch.cat([of, padding], dim=0)
        else:
            padded_of = of
        object_features_padded.append(padded_of)
    object_features_padded = torch.stack(object_features_padded)

    # Pad figure_features
    max_figure_length = figure_lengths.max().item()
    figure_feature_dim = figure_features[0].size(1)
    pad_entry_fig = [padding_label] + [0.0] * (figure_feature_dim - 1)
    pad_entry_fig = torch.tensor(pad_entry_fig, dtype=torch.float32)

    figure_features_padded = []
    for ff in figure_features:
        pad_length = max_figure_length - ff.size(0)
        if pad_length > 0:
            # Ensure pad_length is an integer
            padding = pad_entry_fig.unsqueeze(0).repeat(pad_length, 1)
            padded_ff = torch.cat([ff, padding], dim=0)
        else:
            padded_ff = ff
        figure_features_padded.append(padded_ff)
    figure_features_padded = torch.stack(figure_features_padded)

    # Stack group_features
    group_features_tensor = torch.stack(group_features)

    # Convert labels to tensors
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

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

    # Create the full dataset without resampling
    full_dataset = GroupedDetectionDataset(pos_dir, neg_dir, classes_file)

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
        balance_ratio=balance_ratio,
        resample_method=resample_method,
        indices=train_indices,
    )

    # Create the validation dataset without resampling
    val_dataset = GroupedDetectionDataset(
        pos_dir,
        neg_dir,
        classes_file,
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


def print_class_counts(class_counts, class_names, is_group=False):
    """Helper function to print class counts with class names."""
    if is_group:
        for cls_idx, count in enumerate(
            class_counts[:-1]
        ):  # Exclude the last element (num_figures)
            cls_name = class_names[cls_idx]
            logger.info(f"  {cls_name}: {count}")
        logger.info(
            f"  Total Number of Figures: {class_counts[-1]}"
        )  # Last element is num_figures
    else:
        for cls_idx, count in enumerate(class_counts):
            cls_name = class_names[cls_idx]
            logger.info(f"  {cls_name}: {count}")


def main():
    # Directories and files
    pos_dir = "/data/training_code/yolov7/runs/detect/v7-p5-lr_1e-3/pos/labels"
    neg_dir = "/data/training_code/yolov7/runs/detect/v7-p5-lr_1e-3/neg/labels"
    classes_file = "/data/dataset/bbu_training_data/bbu_and_shield/classes.txt"

    # Desired balance ratio and resampling method
    balance_ratio = 1.0  # Desired positive to negative ratio (1:1)
    resample_method = "no"  # Can be 'downsample', 'upsample', or 'both'

    # Create DataLoaders using the get_dataloader function
    dataloaders = get_dataloader(
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        classes_file=classes_file,
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
    for batch_idx, (*inputs, labels) in enumerate(train_loader):
        (
            object_features_padded,
            figure_features_padded,
            group_features_tensor,
            object_lengths,
            figure_lengths,
        ) = inputs
        labels_tensor = labels

        logger.info(f"\n===== Batch {batch_idx + 1} =====")

        # Log batch labels
        logger.info(f"Batch Labels: {labels_tensor.numpy()}")

        # Iterate over the batch
        for sample_idx in range(object_features_padded.size(0)):
            logger.info(f"\n--- Sample {sample_idx + 1} ---")

            # Get lengths
            obj_length = object_lengths[sample_idx].item()
            fig_length = figure_lengths[sample_idx].item()

            # Get object features and figure features for this sample
            object_sequence = object_features_padded[sample_idx][:obj_length]
            figure_sequence = figure_features_padded[sample_idx][:fig_length]
            group_feature = group_features_tensor[sample_idx]

            # Initialize variables for tracking figures
            figure_idx = 1
            obj_seq_idx = 0
            fig_seq_idx = 0

            logger.info("Object Features:")
            while obj_seq_idx < obj_length:
                data = object_sequence[obj_seq_idx].numpy()
                class_label = int(data[0])
                other_features = data[1:]

                if class_label == PAD_LABEL:
                    obj_seq_idx += 1
                    continue  # Skip padding entries

                elif class_label == SEP_LABEL:
                    logger.info(f"----- End of Figure {figure_idx} -----\n")
                    figure_idx += 1
                    obj_seq_idx += 1
                    continue

                else:
                    logger.info(f"----- Figure {figure_idx} -----")
                    # Object entry
                    logger.info(f"Object at index {obj_seq_idx + 1}:")
                    logger.info(f"  Class Label: {class_label}")
                    # Extract and log feature values
                    feature_names = list(
                        train_loader.dataset.object_feature_functions.keys()
                    )[1:]  # Exclude class_label
                    for fname, fvalue in zip(feature_names, other_features):
                        logger.info(f"  {fname}: {fvalue}")
                    obj_seq_idx += 1

            # Reset figure index for figure features
            figure_idx = 1
            fig_seq_idx = 0

            logger.info("\nFigure Features:")
            while fig_seq_idx < fig_length:
                data = figure_sequence[fig_seq_idx].numpy()
                class_counts = data

                if (class_counts[0] == SEP_LABEL) and all(class_counts[1:] == 0):
                    logger.info(f"----- End of Figure {figure_idx} -----\n")
                    figure_idx += 1
                    fig_seq_idx += 1
                    continue

                else:
                    logger.info(f"Class Counts for Figure {figure_idx}:")
                    print_class_counts(class_counts, class_names)
                    fig_seq_idx += 1

            # Log group features
            logger.info("\nGroup Features:")
            logger.info("Total Class Counts across all Figures:")
            print_class_counts(group_feature.numpy(), class_names, is_group=True)

        # Break after the first batch for testing
        break


if __name__ == "__main__":
    main()
