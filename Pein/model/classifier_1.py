import os
import sys

import torch
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append the parent directory to the system path
sys.path.append(parent_dir)


from utils.setup import setup_logger  # noqa

# Set up the logger
logger = setup_logger(log_file="model.log", level="info", name="model_logger")


class DetectionClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes,
        num_labels,  # Total number of labels including special labels
        CLASS_COUNTS_LABEL,
        PAD_LABEL,
        embedding_dim=4,
        hidden_dim=16,
        fc_hidden_dim=[64, 32],
        num_layers=1,
        dropout=0.1,
        class_counts_dim=8,  # Dimension to project class_counts
    ):
        super(DetectionClassificationModel, self).__init__()

        self.PAD_LABEL = PAD_LABEL
        self.CLASS_COUNTS_LABEL = CLASS_COUNTS_LABEL

        # Embedding layer for class labels
        self.class_embedding = nn.Embedding(
            num_embeddings=num_labels,
            embedding_dim=embedding_dim,
            padding_idx=PAD_LABEL,
        )

        # Linear layer to project class_counts
        self.class_counts_projection = nn.Linear(num_classes, class_counts_dim)

        # Input dimension to the GRU
        input_dim = (
            embedding_dim + 4 + 1 + class_counts_dim
        )  # embedding + bbox(4) + confidence(1) + class_counts_dim

        # LayerNorm for normalization
        self.layer_norm = nn.LayerNorm(input_dim)

        # GRU (bidirectional)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dynamically define fully connected layers based on fc_hidden_dim
        fc_layers = []
        last_hidden_dim = hidden_dim * 2  # GRU is bidirectional

        for fc_dim in fc_hidden_dim:
            fc_layers.append(nn.Linear(last_hidden_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            last_hidden_dim = fc_dim

        # Final output layer
        fc_layers.append(nn.Linear(last_hidden_dim, 1))

        self.fc_layers = nn.Sequential(*fc_layers)

        # Apply weight initialization
        self.apply(self._init_weights)

    def forward(self, X, lengths):
        # X: [batch_size, seq_length, data_dim]
        batch_size, seq_length, data_dim = X.size()
        num_classes = data_dim - 6  # data_dim = 6 + num_classes

        # Split input
        class_labels = X[:, :, 0].long()  # [batch_size, seq_length]
        bbox = X[:, :, 1:5]  # [batch_size, seq_length, 4]
        confidence = X[:, :, 5:6]  # [batch_size, seq_length, 1]
        class_counts = X[:, :, 6:]  # [batch_size, seq_length, num_classes]

        # Embedding for class labels
        class_label_embeddings = self.class_embedding(
            class_labels
        )  # [batch_size, seq_length, embedding_dim]

        # Project class_counts
        class_counts_proj = self.class_counts_projection(
            class_counts
        )  # [batch_size, seq_length, class_counts_dim]

        # Create masks
        pad_mask = (class_labels != self.PAD_LABEL).unsqueeze(
            -1
        )  # Shape: [batch_size, seq_length, 1]
        class_counts_mask = (class_labels == self.CLASS_COUNTS_LABEL).unsqueeze(
            -1
        )  # Shape: [batch_size, seq_length, 1]

        # For entries that are not class_counts entries, set class_counts_proj to zeros
        class_counts_proj = class_counts_proj * class_counts_mask.float()

        # For entries that are padding, set everything to zeros
        pad_mask_float = pad_mask.float()
        class_label_embeddings = class_label_embeddings * pad_mask_float
        bbox = bbox * pad_mask_float
        confidence = confidence * pad_mask_float
        class_counts_proj = class_counts_proj * pad_mask_float

        # Concatenate inputs
        X_input = torch.cat(
            [class_label_embeddings, bbox, confidence, class_counts_proj], dim=-1
        )

        # Apply LayerNorm
        X_input = self.layer_norm(X_input)

        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            X_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through GRU
        packed_output, _ = self.gru(packed_input)

        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Use output from the last valid time step for each sequence
        idx = (
            (lengths - 1)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(batch_size, 1, self.gru.hidden_size * 2)
        )
        last_output = output.gather(1, idx).squeeze(1)  # [batch_size, hidden_dim * 2]

        # Pass through fully connected layers
        logits = self.fc_layers(last_output).squeeze(1)  # [batch_size]

        # Return final output (binary classification)
        return torch.sigmoid(logits)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)


if __name__ == "__main__":
    # Example parameters
    batch_size = 2
    num_classes = 5  # Number of classes excluding special labels
    CLASS_COUNTS_LABEL = num_classes
    SEP_LABEL = num_classes + 1
    PAD_LABEL = num_classes + 2
    num_labels = num_classes + 3
    class_counts_dim = 8  # Dimension to project class_counts

    # Example input
    seq_lengths = [4, 6]  # Variable sequence lengths

    # Maximum sequence length
    max_seq_length = max(seq_lengths)

    # Data dimension
    data_dim = 6 + num_classes  # 6 + num_classes

    # Create random data for two batches
    sequences = []

    for seq_len in seq_lengths:
        sequence = []

        # Random object entries
        num_object_entries = seq_len - 1  # Leave space for class_counts entry
        for _ in range(num_object_entries):
            class_label = torch.randint(0, num_classes, (1,)).item()
            bbox = torch.rand(4).tolist()
            confidence = torch.rand(1).item()
            class_counts = [0.0] * num_classes  # Zeros for object entries
            entry = [class_label] + bbox + [confidence] + class_counts
            sequence.append(entry)

        # Class counts entry
        class_counts = torch.rand(num_classes).tolist()
        entry = [CLASS_COUNTS_LABEL] + [0.0] * 5 + class_counts
        sequence.append(entry)

        # Pad sequence if necessary
        pad_length = max_seq_length - len(sequence)
        for _ in range(pad_length):
            entry = [PAD_LABEL] + [0.0] * (data_dim - 1)
            sequence.append(entry)

        sequences.append(sequence)

    # Convert sequences to tensor
    X = torch.tensor(
        sequences, dtype=torch.float32
    )  # Shape: [batch_size, max_seq_length, data_dim]
    lengths = torch.tensor(seq_lengths, dtype=torch.long)  # Sequence lengths
    labels = torch.randint(0, 2, (batch_size,)).float()  # Random labels (0 or 1)

    # Initialize model
    model = DetectionClassificationModel(
        num_classes=num_classes,
        num_labels=num_labels,
        CLASS_COUNTS_LABEL=CLASS_COUNTS_LABEL,
        PAD_LABEL=PAD_LABEL,
        embedding_dim=4,
        hidden_dim=16,
        fc_hidden_dim=[64, 32],
        num_layers=1,
        dropout=0.1,
        class_counts_dim=8,
    )

    # Forward pass
    output = model(X, lengths)

    # Print outputs
    print("Input Sequences (X):")
    print(X)
    print("\nSequence Lengths:")
    print(lengths)
    print("\nLabels:")
    print(labels)
    print("\nModel Output:")
    print(output)
