import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes,
        num_labels,  # Total number of labels including special labels
        CLASS_COUNTS_LABEL,
        PAD_LABEL,
        embedding_dim=8,
        hidden_dim=64,
        num_attention_heads=8,
        num_transformer_layers=4,
        fc_hidden_dim=[128, 64],
        dropout=0.1,
        max_seq_length=512,  # Maximum sequence length for positional encoding
    ):
        super(DetectionClassificationModel, self).__init__()

        self.PAD_LABEL = PAD_LABEL
        self.CLASS_COUNTS_LABEL = CLASS_COUNTS_LABEL
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Embedding layer for class labels
        self.class_embedding = nn.Embedding(
            num_embeddings=num_labels,
            embedding_dim=embedding_dim,
            padding_idx=PAD_LABEL,
        )

        # Linear layer to project class_counts
        self.class_counts_projection = nn.Linear(num_classes, embedding_dim)

        # Input dimension after concatenation
        # embedding + bbox(4) + confidence(1) + projected class_counts(embedding_dim)
        self.input_dim = (
            embedding_dim + 4 + 1 + embedding_dim
        )  # Update if extra features are added

        # Positional encoding
        self.positional_encoding = self._generate_positional_encoding(
            max_seq_length, self.input_dim
        )

        # LayerNorm
        self.layer_norm = nn.LayerNorm(self.input_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim,
            activation="silu",
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            norm=nn.LayerNorm(self.input_dim),
        )

        # Fully connected layers
        fc_layers = []
        last_hidden_dim = self.input_dim  # Output dimension of TransformerEncoder

        for fc_dim in fc_hidden_dim:
            fc_layers.append(nn.Linear(last_hidden_dim, fc_dim))
            fc_layers.append(nn.SiLU())
            fc_layers.append(nn.Dropout(dropout))
            last_hidden_dim = fc_dim

        # Final output layer
        fc_layers.append(nn.Linear(last_hidden_dim, 1))

        self.fc_layers = nn.Sequential(*fc_layers)

        # Initialize weights
        self._init_weights()

    def forward(self, X, lengths):
        # X: [batch_size, seq_length, data_dim]
        batch_size, seq_length, data_dim = X.size()
        num_classes = data_dim - (6 + self.num_classes_extra_features)
        num_classes = self.num_classes  # Ensure consistency

        # Split input
        class_labels = X[:, :, 0].long()  # [batch_size, seq_length]
        bbox = X[:, :, 1:5]  # [batch_size, seq_length, 4]
        confidence = X[:, :, 5:6]  # [batch_size, seq_length, 1]
        class_counts = X[
            :, :, 6 : 6 + num_classes
        ]  # [batch_size, seq_length, num_classes]
        extra_features = X[:, :, 6 + num_classes :]  # Additional features if any

        # Embedding for class labels
        class_label_embeddings = self.class_embedding(
            class_labels
        )  # [batch_size, seq_length, embedding_dim]

        # Project class_counts
        class_counts_proj = self.class_counts_projection(
            class_counts
        )  # [batch_size, seq_length, embedding_dim]

        # Create masks
        pad_mask = class_labels != self.PAD_LABEL  # Shape: [batch_size, seq_length]

        # For entries that are padding, set everything to zeros
        pad_mask_float = pad_mask.unsqueeze(-1).float()

        class_label_embeddings = class_label_embeddings * pad_mask_float
        bbox = bbox * pad_mask_float
        confidence = confidence * pad_mask_float
        class_counts_proj = class_counts_proj * pad_mask_float
        if extra_features.size(-1) > 0:
            extra_features = extra_features * pad_mask_float

        # Concatenate inputs
        X_input = torch.cat(
            [
                class_label_embeddings,
                bbox,
                confidence,
                class_counts_proj,
                extra_features,
            ],
            dim=-1,
        )  # [batch_size, seq_length, input_dim]

        # Apply positional encoding (truncate if sequence is longer than max_seq_length)
        pos_enc = self.positional_encoding[:, :seq_length, :].to(X_input.device)
        X_input = X_input + pos_enc

        # Apply LayerNorm
        X_input = self.layer_norm(X_input)

        # Create attention mask (True where padding)
        attn_mask = ~pad_mask  # [batch_size, seq_length]

        # Pass through Transformer encoder
        transformer_output = self.transformer_encoder(
            X_input, src_key_padding_mask=attn_mask
        )  # [batch_size, seq_length, input_dim]

        # Apply mask to output
        transformer_output = transformer_output * pad_mask_float

        # Compute mean over valid time steps
        sum_output = transformer_output.sum(dim=1)  # [batch_size, input_dim]
        lengths = lengths.unsqueeze(-1)  # [batch_size, 1]
        avg_output = sum_output / lengths  # [batch_size, input_dim]

        # Pass through fully connected layers
        logits = self.fc_layers(avg_output).squeeze(1)  # [batch_size]

        # Return final output (logits for binary classification)
        return logits

    def _init_weights(self):
        # Initialize weights
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _generate_positional_encoding(self, max_seq_len, d_model):
        """
        Generates positional encoding for the Transformer.
        """
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe


if __name__ == "__main__":
    # Example parameters
    batch_size = 2
    num_classes = 5  # Number of classes excluding special labels
    CLASS_COUNTS_LABEL = num_classes
    TOTAL_CLASS_COUNTS_LABEL = num_classes + 1
    SEPARATOR_LABEL = num_classes + 2
    PADDING_LABEL = num_classes + 3
    num_labels = num_classes + 4
    class_counts_dim = 8  # Dimension to project class_counts

    # Example input
    seq_lengths = [10, 15]  # Variable sequence lengths

    # Maximum sequence length
    max_seq_length = max(seq_lengths)

    # Data dimension
    num_object_features = 7  # class_label + bbox(4) + confidence(1) + area(1)
    data_dim = num_object_features + num_classes  # Features + class_counts

    # Create random data for two batches
    sequences = []

    for seq_len in seq_lengths:
        sequence = []

        # Random object entries
        num_object_entries = (
            seq_len - 2
        )  # Leave space for class_counts and total_class_counts entries
        for _ in range(num_object_entries):
            class_label = torch.randint(0, num_classes, (1,)).item()
            bbox = torch.rand(4).tolist()
            confidence = torch.rand(1).item()
            area = torch.rand(1).item()
            class_counts = [0.0] * num_classes  # Zeros for object entries
            entry = [class_label] + bbox + [confidence, area] + class_counts
            sequence.append(entry)

        # Class counts entry
        class_counts = torch.rand(num_classes).tolist()
        entry = [CLASS_COUNTS_LABEL] + [0.0] * (num_object_features - 1) + class_counts
        sequence.append(entry)

        # Total class counts entry
        total_class_counts = torch.rand(num_classes).tolist()
        entry = (
            [TOTAL_CLASS_COUNTS_LABEL]
            + [0.0] * (num_object_features - 1)
            + total_class_counts
        )
        sequence.append(entry)

        # Pad sequence if necessary
        pad_length = max_seq_length - len(sequence)
        for _ in range(pad_length):
            entry = [PADDING_LABEL] + [0.0] * (data_dim - 1)
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
        PAD_LABEL=PADDING_LABEL,
        embedding_dim=8,
        hidden_dim=16,
        num_attention_heads=4,
        num_transformer_layers=2,
        fc_hidden_dim=[32, 16],
        dropout=0.1,
        max_seq_length=max_seq_length,
    )

    # Forward pass
    logits = model(X, lengths)

    # Print outputs
    print("Input Sequences (X):")
    print(X)
    print("\nSequence Lengths:")
    print(lengths)
    print("\nLabels:")
    print(labels)
    print("\nModel Output (Logits):")
    print(logits)
    print("\nPredicted Probabilities:")
    print(torch.sigmoid(logits))
