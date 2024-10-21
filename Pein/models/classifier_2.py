import os
import sys

import torch
import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data.data_factory import get_dataloader  # noqa

# Append the parent directory to the system path
from utils.logging import setup_logger  # noqa

logger = setup_logger(
    log_file="model_debug.log",
    name="model",
    level="DEBUG",
    log_to_console=True,
    overwrite=True,
)  # for debug and test


class DetectionClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes,
        num_labels=None,
        PAD_LABEL=None,
        SEP_LABEL=None,
        embedding_dim=16,
        hidden_dim=64,
        attn_heads=4,
        fc_hidden_dims=[16, 8],
        num_layers=2,
        dropout=0.1,
        use_fig_size=False,
    ):
        super(DetectionClassificationModel, self).__init__()

        # Set default values if not provided
        #! Note: Ensure class labels are zero-indexed (0 to num_classes-1)
        # * SEP_LABEL and PAD_LABEL are assigned values outside this range
        self.num_classes = num_classes
        self.SEP_LABEL = SEP_LABEL if SEP_LABEL is not None else num_classes
        self.PAD_LABEL = PAD_LABEL if PAD_LABEL is not None else num_classes + 1
        self.num_labels = num_labels if num_labels is not None else num_classes + 2

        self.embedding_dim = embedding_dim
        self.num_numerical_features = 6  # x, y, w, h, conf, area
        self.fc_hidden_dims = fc_hidden_dims

        # Embedding layer for class labels
        self.class_embedding = nn.Embedding(
            num_embeddings=self.num_labels,
            embedding_dim=embedding_dim,
            padding_idx=self.PAD_LABEL,
        )

        # Linear layer to project numerical features
        self.numerical_feature_proj = nn.Linear(
            self.num_numerical_features, embedding_dim
        )

        # Positional encoding for objects within figures
        self.obj_positional_encoding = PositionalEncoding(embedding_dim * 2)

        # LayerNorm for object inputs
        self.obj_layer_norm = nn.LayerNorm(embedding_dim * 2)

        # Object-level Transformer
        encoder_layer_obj = nn.TransformerEncoderLayer(
            d_model=embedding_dim * 2,
            nhead=attn_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.object_encoder = nn.TransformerEncoder(
            encoder_layer_obj,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim * 2),
        )

        # Projection layer to reduce dimension back to embedding_dim
        self.obj_output_proj = nn.Linear(embedding_dim * 2, embedding_dim)

        # LayerNorm for numerical features
        self.other_feature_norm = nn.LayerNorm(self.num_numerical_features)

        # Figure-level processing remains the same
        # Positional encoding for figures
        self.fig_positional_encoding = PositionalEncoding(embedding_dim)

        # LayerNorm for figure inputs
        self.fig_layer_norm = nn.LayerNorm(embedding_dim)

        # Figure-level Transformer
        encoder_layer_fig = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attn_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.figure_encoder = nn.TransformerEncoder(
            encoder_layer_fig,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

        # Linear layer to project class counts
        self.class_counts_projection = nn.Linear(num_classes, embedding_dim)

        # Fully connected layers
        fc_input_dim = embedding_dim + embedding_dim + num_classes + 1
        fc_layers = []
        last_hidden_dim = fc_input_dim

        for fc_dim in self.fc_hidden_dims:
            fc_layers.append(nn.Linear(last_hidden_dim, fc_dim))
            fc_layers.append(nn.GELU())
            fc_layers.append(nn.Dropout(dropout))
            last_hidden_dim = fc_dim

        fc_layers.append(nn.Linear(last_hidden_dim, 1))
        self.fc_layers = nn.Sequential(*fc_layers)

        # Initialize weights
        self._init_weights()

        self.use_fig_size = use_fig_size  # Store the flag

        # Adjust figure feature dimension based on use_fig_size
        fig_feature_dim = num_classes + 2 if use_fig_size else num_classes

        # Linear layer to project figure features
        self.figure_feature_proj = nn.Linear(fig_feature_dim, embedding_dim)

        logger.debug(
            f"Initializing DetectionClassificationModel with num_classes={num_classes}, use_fig_size={use_fig_size}"
        )

    def forward(
        self,
        object_features,
        figure_features,
        group_features,
        object_lengths,
        figure_lengths,
    ):
        logger.debug(f"Forward pass: batch_size={object_features.size(0)}")
        logger.debug(f"Object features shape: {object_features.shape}")
        logger.debug(f"Figure features shape: {figure_features.shape}")
        logger.debug(f"Group features shape: {group_features.shape}")

        batch_size = object_features.size(0)

        # Process object features
        object_outputs = []
        for i in range(batch_size):
            obj_feat = object_features[
                i, : object_lengths[i]
            ]  # [obj_seq_length, obj_feature_dim]
            obj_embedded = self._process_object_sequence(obj_feat)
            object_outputs.append(obj_embedded)
        object_outputs = torch.stack(object_outputs)  # [batch_size, embedding_dim]

        # Process figure features
        figure_outputs = []
        for i in range(batch_size):
            fig_feat = figure_features[
                i, : figure_lengths[i]
            ]  # [fig_seq_length, fig_feature_dim]
            fig_embedded = self._process_figure_sequence(fig_feat)
            figure_outputs.append(fig_embedded)
        figure_outputs = torch.stack(figure_outputs)  # [batch_size, embedding_dim]

        # Concatenate object, figure, and group features
        group_features = group_features  # [batch_size, group_feature_dim]
        combined_features = torch.cat(
            [object_outputs, figure_outputs, group_features], dim=-1
        )

        # Pass through fully connected layers
        logits = self.fc_layers(combined_features).squeeze(1)  # [batch_size]

        logger.debug(f"Model output (logits) shape: {logits.shape}")

        return logits

    def _process_object_sequence(self, obj_seq):
        logger.debug(f"Processing object sequence of shape: {obj_seq.shape}")
        class_labels = obj_seq[:, 0].long()
        other_features = obj_seq[:, 1:]  # [seq_length, num_numerical_features]

        # Create mask
        pad_mask = class_labels != self.PAD_LABEL
        sep_mask = class_labels != self.SEP_LABEL
        mask = pad_mask & sep_mask  # Valid object entries

        if mask.sum() == 0:
            # If all entries are PAD or SEP, return zeros
            return torch.zeros(self.embedding_dim).to(obj_seq.device)

        # Embed class labels
        class_embeds = self.class_embedding(class_labels)  # [seq_length, embedding_dim]

        # Normalize numerical features
        other_features = self.other_feature_norm(other_features)

        # Project numerical features
        other_features_proj = self.numerical_feature_proj(
            other_features
        )  # [seq_length, embedding_dim]

        # Combine embeddings
        obj_input = torch.cat(
            [class_embeds, other_features_proj], dim=-1
        )  # [seq_length, embedding_dim * 2]

        # Apply positional encoding
        obj_input = self.obj_positional_encoding(obj_input.unsqueeze(0)).squeeze(0)

        # Apply LayerNorm
        obj_input = self.obj_layer_norm(obj_input)

        # Create attention mask
        attn_mask = ~mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length]

        # Pass through object encoder
        obj_output = self.object_encoder(
            obj_input.unsqueeze(0), src_key_padding_mask=attn_mask.squeeze(0)
        )  # [1, seq_length, embedding_dim * 2]

        # Pool over valid entries
        obj_output = obj_output[0][mask].mean(dim=0)  # [embedding_dim * 2]

        # Project back to embedding_dim
        obj_output = self.obj_output_proj(obj_output)  # [embedding_dim]

        logger.debug(f"Object output shape: {obj_output.shape}")
        return obj_output

    def _process_figure_sequence(self, fig_seq):
        logger.debug(f"Processing figure sequence of shape: {fig_seq.shape}")
        # Assume that SEP_LABEL is used in fig_seq[:, 0] for separators
        class_labels = fig_seq[:, 0].long()
        pad_mask = class_labels != self.PAD_LABEL
        sep_mask = class_labels != self.SEP_LABEL
        mask = pad_mask & sep_mask  # Valid figure entries

        if mask.sum() == 0:
            # If all entries are PAD or SEP, return zeros
            return torch.zeros(self.embedding_dim).to(fig_seq.device)

        # Extract class counts and figure size based on use_fig_size flag
        if self.use_fig_size:
            class_counts = fig_seq[mask, : self.num_classes]
            fig_sizes = fig_seq[mask, self.num_classes : self.num_classes + 2]
        else:
            class_counts = fig_seq[mask, : self.num_classes]

        logger.debug(f"Class counts shape: {class_counts.shape}")
        logger.debug(f"Class counts content: {class_counts}")

        # Ensure class_counts is 2D
        if class_counts.dim() == 1:
            class_counts = class_counts.unsqueeze(0)

        # Project class counts to embeddings
        fig_embeds = self.class_counts_projection(
            class_counts
        )  # [num_figures, embedding_dim]
        logger.debug(f"Figure embeddings shape after projection: {fig_embeds.shape}")

        # If using figure size, concatenate it with the embeddings
        if self.use_fig_size:
            logger.debug(f"Figure sizes shape: {fig_sizes.shape}")
            fig_embeds = torch.cat([fig_embeds, fig_sizes], dim=-1)
            fig_embeds = self.figure_feature_proj(fig_embeds)

        # Add positional encoding
        fig_input = self.fig_positional_encoding(fig_embeds.unsqueeze(0)).squeeze(
            0
        )  # [num_figures, embedding_dim]

        # Apply LayerNorm
        fig_input = self.fig_layer_norm(fig_input)

        # Pass through figure encoder
        fig_output = self.figure_encoder(
            fig_input.unsqueeze(0), src_key_padding_mask=~mask.unsqueeze(0)
        )  # [1, num_figures, embedding_dim]

        # Pool over valid entries
        fig_output = fig_output[0][mask].mean(dim=0)  # [embedding_dim]

        logger.debug(f"Final figure output shape: {fig_output.shape}")
        return fig_output

    def _init_weights(self):
        # Initialize weights
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _create_padding_mask(self, lengths, max_len):
        batch_size = lengths.size(0)
        mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create constant 'pe' matrix with values dependent on position and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return x


def main():
    # Directories and files
    pos_dir = "/data/training_code/yolov7/runs/detect/v7-p5-lr_1e-3/pos/labels"
    neg_dir = "/data/training_code/yolov7/runs/detect/v7-p5-lr_1e-3/neg/labels"
    classes_file = "/data/dataset/bbu_training_data/bbu_and_shield/classes.txt"
    fig_size_path = "/data/training_code/yolov7/Pein/fig_size.csv"

    # Model parameters
    num_classes = 5  # Adjust this based on your actual number of classes
    embedding_dim = 4
    hidden_dim = 6
    num_attention_heads = 2
    num_layers = 1
    dropout = 0.1
    batch_size = 2
    use_fig_size = False

    # Get dataloaders
    dataloaders = get_dataloader(
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        classes_file=classes_file,
        csv_file=fig_size_path,
        split=[0.8, 0.2],
        batch_size=batch_size,
        balance_ratio=1.0,
        resample_method="downsample",
    )

    # Access the train_loader
    train_loader = dataloaders["train"]

    # Get special labels from the dataset
    PAD_LABEL = train_loader.dataset.PAD_LABEL
    SEP_LABEL = train_loader.dataset.SEP_LABEL
    num_labels = train_loader.dataset.num_labels

    # Initialize model
    model = DetectionClassificationModel(
        num_classes=num_classes,
        SEP_LABEL=SEP_LABEL,
        PAD_LABEL=PAD_LABEL,
        num_labels=num_labels,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        attn_heads=num_attention_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_fig_size=use_fig_size,
    )

    # Set the model to evaluation mode
    model.eval()

    logger.debug(f"Loaded dataloaders with batch_size={batch_size}")
    logger.debug(
        f"PAD_LABEL: {PAD_LABEL}, SEP_LABEL: {SEP_LABEL}, num_labels: {num_labels}"
    )

    # After model initialization
    logger.debug(f"Model initialized with use_fig_size={use_fig_size}")

    # Before processing a batch
    logger.debug("Processing a single batch")

    for batch in train_loader:
        (
            object_features_padded,
            figure_features_padded,
            group_features_tensor,
            object_lengths,
            figure_lengths,
            labels_tensor,
        ) = batch

        logger.debug("Batch shapes:")
        logger.debug(f"  object_features_padded: {object_features_padded.shape}")
        logger.debug(f"  figure_features_padded: {figure_features_padded.shape}")
        logger.debug(f"  group_features_tensor: {group_features_tensor.shape}")
        logger.debug(f"  object_lengths: {object_lengths.shape}")
        logger.debug(f"  figure_lengths: {figure_lengths.shape}")
        logger.debug(f"  labels_tensor: {labels_tensor.shape}")

        # Forward pass
        with torch.no_grad():
            logits = model(
                object_features_padded,
                figure_features_padded,
                group_features_tensor,
                object_lengths,
                figure_lengths,
            )

        logger.debug(f"Model output (logits) shape: {logits.shape}")

        # Print outputs
        print("Model Output (Logits):")
        print(logits)
        print("\nPredicted Probabilities:")
        print(torch.sigmoid(logits))
        print("\nTrue Labels:")
        print(labels_tensor)

        # Process only one batch for demonstration
        break


if __name__ == "__main__":
    main()
