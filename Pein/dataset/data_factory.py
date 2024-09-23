import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to the system path
sys.path.append(parent_dir)


from utils.setup import setup_logger  # noqa

# Set up the logger
logger = setup_logger(
    log_file="data_factory.log", level="debug", name="data_factory_logger"
)


class GroupedDetectionDataset(Dataset):
    """
    A PyTorch Dataset class for handling grouped detection data.
    Each sample consists of a group of figures (images) associated with an experiment ID.
    The dataset processes YOLO detection outputs (text files) and constructs sequences
    suitable for input into an RNN (e.g., GRU) model for classification tasks.
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
        """
        Initialize the dataset.

        Args:
            pos_dir (str): Directory containing positive sample files.
            neg_dir (str): Directory containing negative sample files.
            classes_file (str): Path to the classes file.
            transform (callable, optional): Optional transform to be applied on a sample.
            balance_ratio (float, optional): Desired ratio of positive to negative samples.
            resample_method (str, optional): Method to resample data ('downsample', 'upsample', 'both').
            indices (list, optional): List of indices to subset the dataset.
        """
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
        self.TOTAL_CLASS_COUNTS_LABEL = self.num_classes + 2  # 7
        self.PAD_LABEL = self.num_classes + 3  # 8
        self.num_labels = self.num_classes + 4  # 9 labels (0 to 8)

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
        total_samples = num_positive + num_negative

        if num_positive == 0 or num_negative == 0:
            logger.error("Cannot resample because one of the classes has zero samples.")
            return

        # Compute desired counts
        desired_num_positive = int(
            (desired_ratio * total_samples) / (1 + desired_ratio)
        )
        desired_num_negative = total_samples - desired_num_positive

        # Ensure desired counts are at least 1
        desired_num_positive = max(desired_num_positive, 1)
        desired_num_negative = max(desired_num_negative, 1)

        # Resampling logic
        self.positive_samples = self._adjust_samples(
            self.positive_samples, desired_num_positive, resample_method
        )
        self.negative_samples = self._adjust_samples(
            self.negative_samples, desired_num_negative, resample_method
        )

        # Combine the positive and negative samples
        self.samples = self.positive_samples + self.negative_samples

        # Shuffle the combined samples
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
        logger.info(f"Number of positive samples: {num_positive}")
        logger.info(f"Number of negative samples: {num_negative}")
        logger.info(f"Max number of figures in a group: {self.max_figures_in_group}")
        logger.info(
            f"Label distribution: {num_positive / total_samples:.2%} positive, {num_negative / total_samples:.2%} negative."
        )

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        files = sample["files"]
        label = sample["label"]

        sequence = []

        # Initialize total_class_counts and total_num_figures
        total_class_counts = [0.0] * self.num_classes
        total_num_figures = len(files)

        for i, file_path in enumerate(files):
            # Process each figure and get its entries, class counts, and num_objects
            figure_entries, figure_class_counts, num_objects_per_figure = (
                self._process_figure(file_path)
            )
            sequence.extend(figure_entries)

            # Accumulate total class counts
            total_class_counts = [
                x + y for x, y in zip(total_class_counts, figure_class_counts)
            ]

            # Add separator entry if not the last figure
            if i < len(files) - 1:
                sequence.append(self._get_separator_entry())

        # Group-wise entry: total class counts and total number of figures
        group_entry = self._get_group_entry(total_class_counts, total_num_figures)
        sequence.append(group_entry)

        # Convert sequence to tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        return sequence_tensor, label

    def _process_figure(self, file_path):
        figure_entries = []

        # Initialize class_counts and num_objects for this figure
        class_counts = [0.0] * self.num_classes
        num_objects_per_figure = 0

        with open(file_path, "r") as f:
            lines = f.readlines()

        # Collect object data and compute weighted class counts
        for line in lines:
            object_data = list(map(float, line.strip().split()))
            class_label = int(object_data[0])

            # Compute per-object features (exclude class_label)
            features = [
                func(object_data)
                for func in list(self.object_feature_functions.values())[1:]
            ]

            # Update weighted class counts
            if 0 <= class_label < self.num_classes:
                confidence = object_data[5]
                class_counts[class_label] += confidence

            num_objects_per_figure += 1

            # Prepare the full feature vector
            # [class_label, features, zeros for figure-wise features, zeros for group-wise features]
            entry = [class_label] + features
            # Add zeros for figure-wise features and group-wise features
            entry += [0.0] * (self.num_classes + 1 + self.num_classes + 1)
            figure_entries.append(entry)

        # Figure-wise entry
        figure_entry = self._get_figure_entry(class_counts, num_objects_per_figure)
        figure_entries.append(figure_entry)

        return figure_entries, class_counts, num_objects_per_figure

    def _get_class_counts_entry(self, class_counts):
        """Create an entry for weighted class counts of a figure."""
        # Use default values for per-object features
        features = [
            self._get_default_value(feature_name, entry_type="class_counts")
            for feature_name in self.object_feature_functions.keys()
        ]
        # Append class counts
        features += class_counts
        return features

    def _get_total_class_counts_entry(self, total_class_counts):
        """Create an entry for total class counts across all figures in a sample."""
        # Use default values for per-object features
        features = [
            self._get_default_value(feature_name, entry_type="total_class_counts")
            for feature_name in self.object_feature_functions.keys()
        ]
        # Append total class counts
        features += total_class_counts
        return features

    def _get_figure_entry(self, class_counts, num_objects_per_figure):
        # For per-object features, use default values
        features = [0.0] * self.num_object_features  # Since we've excluded class_label

        # class_label
        class_label = self.CLASS_COUNTS_LABEL

        # Figure-wise features: class_counts + [num_objects_per_figure]
        figure_features = class_counts + [float(num_objects_per_figure)]

        # Group-wise features: zeros
        group_features = [0.0] * (self.num_classes + 1)

        entry = [class_label] + features + figure_features + group_features
        return entry

    def _get_group_entry(self, total_class_counts, total_num_figures):
        # Features should be consistent in length
        features = [0.0] * self.num_object_features
        class_label = self.TOTAL_CLASS_COUNTS_LABEL

        # Ensure figure-wise features length is correct
        figure_features = [0.0] * (self.num_classes + 1)

        # Ensure group-wise features length is consistent
        group_features = total_class_counts + [float(total_num_figures)]

        # Make sure total entry length matches self.feature_vector_length
        entry = [class_label] + features + figure_features + group_features
        assert (
            len(entry) == self.feature_vector_length
        ), f"Entry length mismatch: expected {self.feature_vector_length}, got {len(entry)}"
        return entry

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


def custom_collate_fn(batch, padding_label, feature_vector_length):
    sequences, labels = zip(*batch)

    # Find the maximum sequence length in the batch
    max_seq_length = max(seq.shape[0] for seq in sequences)
    feature_dim = sequences[0].shape[1]  # Should be consistent across sequences

    # Define pad_entry with padding_label
    pad_entry = [padding_label] + [0.0] * (
        feature_vector_length - 1
    )  # Length: feature_dim

    # Pad sequences
    padded_sequences = []
    sequence_lengths = []
    for seq in sequences:
        seq_length = seq.shape[0]
        sequence_lengths.append(seq_length)
        pad_length = max_seq_length - seq_length

        if pad_length > 0:
            pad_tensor = torch.tensor([pad_entry] * pad_length, dtype=torch.float32)
            padded_seq = torch.cat([seq, pad_tensor], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    # Stack sequences into a tensor
    sequences_tensor = torch.stack(padded_sequences)

    # Stack labels into a tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Convert sequence_lengths to tensor
    sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long)

    return sequences_tensor, labels_tensor, sequence_lengths


def get_dataloader(
    pos_dir,
    neg_dir,
    classes_file,
    split=[0.8, 0.2],
    batch_size=32,
    balance_ratio=None,
    resample_method="downsample",
    seed=None,
):
    """Create DataLoaders for training, validation, and testing."""
    assert len(split) in [
        2,
        3,
    ], "Split should be either [train, val] or [train, val, test]."

    if seed is None:
        seed = np.random.randint(0, 1000)

    # Create the full dataset without resampling
    full_dataset = GroupedDetectionDataset(pos_dir, neg_dir, classes_file)

    PAD_LABEL = full_dataset.num_classes + 2  # Assuming PAD_LABEL is num_classes + 2
    feature_vector_length = full_dataset.feature_vector_length

    # Get the total number of samples
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    # Extract labels for stratification
    labels = [full_dataset.samples[i]["label"] for i in indices]

    # Split the dataset into train, val, and optionally test
    if len(split) == 2:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=split[1],
            train_size=split[0],
            random_state=seed,
            stratify=labels,
        )
        test_indices = []
    else:
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=(split[1] + split[2]),
            train_size=split[0],
            random_state=seed,
            stratify=labels,
        )
        temp_labels = [labels[i] for i in temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(split[2] / (split[1] + split[2])),
            random_state=seed,
            stratify=temp_labels,
        )

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
    test_dataset = (
        GroupedDetectionDataset(pos_dir, neg_dir, classes_file, indices=test_indices)
        if test_indices
        else None
    )

    # Create the collate function with PAD_LABEL
    from functools import partial

    collate_fn_with_pad = partial(
        custom_collate_fn,
        padding_label=PAD_LABEL,
        feature_vector_length=feature_vector_length,
    )

    # Create DataLoaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_with_pad,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_pad,
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


def print_class_counts(class_counts, class_names):
    """Helper function to print class counts with class names."""
    for cls_idx, count in enumerate(
        class_counts[:-1]
    ):  # Exclude the last element (num_objects or num_figures)
        cls_name = class_names[cls_idx]
        logger.info(f"  {cls_name}: {count}")
    logger.info(
        f"  Number: {class_counts[-1]}"
    )  # Last element is num_objects or num_figures


if __name__ == "__main__":
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
        batch_size=1,  # For clearer logging
        balance_ratio=balance_ratio,
        resample_method=resample_method,
    )

    # Access the train_loader
    train_loader = dataloaders["train"]

    # Get special labels and class names from the dataset
    PAD_LABEL = train_loader.dataset.PAD_LABEL
    SEP_LABEL = train_loader.dataset.SEP_LABEL
    CLASS_COUNTS_LABEL = train_loader.dataset.CLASS_COUNTS_LABEL
    TOTAL_CLASS_COUNTS_LABEL = CLASS_COUNTS_LABEL + 1
    class_names = train_loader.dataset.class_names
    num_classes = train_loader.dataset.num_classes
    num_features = len(train_loader.dataset.object_feature_functions)

    # Iterate over the train_loader
    for batch_idx, (sequences_tensor, labels, sequence_lengths) in enumerate(
        train_loader
    ):
        logger.info(f"\n===== Batch {batch_idx + 1} =====")

        # Log batch labels
        logger.info(f"Batch Labels: {labels.numpy()}")

        # Iterate over the batch
        for sample_idx in range(sequences_tensor.size(0)):
            logger.info(f"\n--- Sample {sample_idx + 1} ---")

            sequence = sequences_tensor[sample_idx]
            seq_length = sequence_lengths[sample_idx].item()
            data_dim = sequence.shape[1]

            # Initialize variables for tracking figures
            figure_idx = 1
            new_figure = True

            # Iterate over the sequence
            for seq_idx in range(seq_length):
                data = sequence[seq_idx].numpy()

                class_label = data[0]
                # Extract features
                entry_features = data[1:7]  # bbox(4), confidence, area
                figure_features = data[
                    7:13
                ]  # class_counts_per_figure + num_objects_per_figure
                group_features = data[
                    13:
                ]  # total_class_counts_per_group + total_num_figures_per_group

                if class_label == PAD_LABEL:
                    continue  # Skip padding entries

                elif class_label == SEP_LABEL:
                    logger.info(f"----- End of Figure {figure_idx} -----")
                    figure_idx += 1
                    new_figure = True
                    continue

                elif class_label == CLASS_COUNTS_LABEL:
                    logger.info(f"Class Counts for Figure {figure_idx}:")
                    print_class_counts(figure_features, class_names)
                    continue

                elif class_label == TOTAL_CLASS_COUNTS_LABEL:
                    logger.info("----- Total Class Counts across all Figures -----")
                    print_class_counts(group_features, class_names)
                    continue

                else:
                    if new_figure:
                        logger.info(f"----- Start of Figure {figure_idx} -----")
                        new_figure = False
                    # Object entry
                    logger.info(f"Object at index {seq_idx + 1}:")
                    logger.info(f"  Class Label: {class_label}")
                    # Extract and log feature values
                    feature_names = list(
                        train_loader.dataset.object_feature_functions.keys()
                    )[1:]  # Exclude class_label
                    for fname, fvalue in zip(feature_names, entry_features):
                        logger.info(f"  {fname}: {fvalue}")
            break  # Remove or adjust this break to process more batches
