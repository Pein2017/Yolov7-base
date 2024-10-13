import torch
import torch.nn as nn


class DetectionClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes,
        num_labels,  # Total number of labels including special labels
        PAD_LABEL,
        SEP_LABEL,
        embedding_dim=16,
        hidden_dim=64,
        attn_heads=4,
        fc_hidden_dims=[32, 16],
        num_layers=2,
        dropout=0.1,
    ):
        super(DetectionClassificationModel, self).__init__()

        self.PAD_LABEL = PAD_LABEL
        self.SEP_LABEL = SEP_LABEL
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.num_numerical_features = 6  # x, y, w, h, conf, area
        self.fc_hidden_dims = fc_hidden_dims

        # Embedding layer for class labels
        self.class_embedding = nn.Embedding(
            num_embeddings=num_labels,
            embedding_dim=embedding_dim,
            padding_idx=PAD_LABEL,
        )

        # Linear layer to project numerical features
        self.numerical_feature_proj = nn.Linear(
            self.num_numerical_features, embedding_dim
        )

        # Positional encoding for objects within figures
        self.obj_positional_encoding = PositionalEncoding(embedding_dim * 2)

        # LayerNorm for object inputs
        self.obj_layer_norm = nn.LayerNorm(embedding_dim * 2)

        # Transformer encoder layers for object features within figures
        encoder_layer_obj = nn.TransformerEncoderLayer(
            d_model=embedding_dim * 2,
            nhead=attn_heads,
            dim_feedforward=hidden_dim,
            activation="gelu",
            dropout=dropout,
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

        # Transformer encoder layers for figure features
        encoder_layer_fig = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attn_heads,
            dim_feedforward=hidden_dim,
            activation="gelu",
            dropout=dropout,
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
        fc_layers = []
        last_hidden_dim = (
            embedding_dim + embedding_dim + num_classes + 1
        )  # Object + Figure + Group feature

        # Adjusted for the example
        for fc_dim in self.fc_hidden_dims:
            fc_layers.append(nn.Linear(last_hidden_dim, fc_dim))
            fc_layers.append(nn.GELU())
            fc_layers.append(nn.Dropout(dropout))
            last_hidden_dim = fc_dim

        # Final output layer
        fc_layers.append(nn.Linear(last_hidden_dim, 1))
        self.fc_layers = nn.Sequential(*fc_layers)

        # Initialize weights
        self._init_weights()

    def forward(
        self,
        object_features,
        figure_features,
        group_features,
        object_lengths,
        figure_lengths,
    ):
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

        return logits

    def _process_object_sequence(self, obj_seq):
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
        attn_mask = ~mask.unsqueeze(0)  # [1, seq_length]

        # Pass through object encoder
        obj_output = self.object_encoder(
            obj_input.unsqueeze(0), src_key_padding_mask=attn_mask
        )  # [1, seq_length, embedding_dim * 2]

        # Pool over valid entries
        obj_output = obj_output[0][mask].mean(dim=0)  # [embedding_dim * 2]

        # Project back to embedding_dim
        obj_output = self.obj_output_proj(obj_output)  # [embedding_dim]

        return obj_output

    def _process_figure_sequence(self, fig_seq):
        class_counts = fig_seq  # [fig_seq_length, fig_feature_dim]

        # Assume that SEP_LABEL is used in class_counts[:, 0] for separators
        class_labels = class_counts[:, 0].long()
        pad_mask = class_labels != self.PAD_LABEL
        sep_mask = class_labels != self.SEP_LABEL
        mask = pad_mask & sep_mask  # Valid figure entries

        if mask.sum() == 0:
            # If all entries are PAD or SEP, return zeros
            return torch.zeros(self.embedding_dim).to(fig_seq.device)

        # Project class counts to embeddings
        fig_embeds = self.class_counts_projection(
            class_counts[mask]
        )  # [num_figures, embedding_dim]

        # Add positional encoding
        fig_input = self.fig_positional_encoding(fig_embeds.unsqueeze(0)).squeeze(
            0
        )  # [num_figures, embedding_dim]

        # Apply LayerNorm
        fig_input = self.fig_layer_norm(fig_input)

        # No need for attention mask since we've filtered invalid entries
        # Pass through figure encoder
        fig_output = self.figure_encoder(
            fig_input.unsqueeze(0)
        )  # [1, num_figures, embedding_dim]

        # Pool over valid entries
        fig_output = fig_output[0].mean(dim=0)  # [embedding_dim]

        return fig_output

    def _init_weights(self):
        # Initialize weights
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


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
    # Example parameters
    batch_size = 2
    num_classes = 5  # Number of classes excluding special labels
    SEP_LABEL = num_classes + 1  # 6
    PAD_LABEL = num_classes + 2  # 7
    num_labels = num_classes + 3  # 8 labels (0 to 7)

    embedding_dim = 16
    hidden_dim = 64
    num_attention_heads = 4
    num_layers = 2
    dropout = 0.1

    # Create random data for two batches
    batch_data = []
    labels = []
    batch_size = 2

    for _ in range(batch_size):
        num_figures = torch.randint(
            1, 4, (1,)
        ).item()  # Random number of figures between 1 and 3
        object_feature = []
        figure_feature = []
        total_class_counts = torch.zeros(num_classes)
        total_num_figures = num_figures

        for fig_idx in range(num_figures):
            # Random number of objects in this figure
            num_objects = torch.randint(1, 5, (1,)).item()  # Between 1 and 4 objects
            class_counts = torch.zeros(num_classes)

            for obj_idx in range(num_objects):
                class_label = torch.randint(0, num_classes, (1,)).item()
                x = torch.rand(1).item()
                y = torch.rand(1).item()
                w = torch.rand(1).item()
                h = torch.rand(1).item()
                conf = torch.rand(1).item()
                area = w * h
                other_features = [x, y, w, h, conf, area]
                obj_entry = [class_label] + other_features
                object_feature.append(obj_entry)

                # Update class counts
                class_counts[class_label] += conf

            # Append SEP_LABEL to object_feature if not last figure
            if fig_idx < num_figures - 1:
                object_feature.append([SEP_LABEL] + [0.0] * 6)

            # Figure feature (class counts)
            figure_feature.append(class_counts.tolist())

            # Append SEP_LABEL to figure_feature if not last figure
            if fig_idx < num_figures - 1:
                figure_feature.append([SEP_LABEL] + [0.0] * (num_classes - 1))

            # Update total class counts
            total_class_counts += class_counts

        # Group feature
        group_feature = total_class_counts.tolist() + [float(total_num_figures)]

        # Convert to tensors
        object_feature = torch.tensor(object_feature, dtype=torch.float32)
        figure_feature = torch.tensor(figure_feature, dtype=torch.float32)
        group_feature = torch.tensor(group_feature, dtype=torch.float32)

        # Append to batch_data
        batch_data.append((object_feature, figure_feature, group_feature))

        # Random label
        label = torch.randint(0, 2, (1,)).item()
        labels.append(label)

    # Now pad the sequences to create batch tensors
    object_features, figure_features, group_features = zip(*[d for d in batch_data])
    labels = torch.tensor(labels, dtype=torch.long)

    # Pad object_features
    object_lengths = torch.tensor(
        [of.size(0) for of in object_features], dtype=torch.long
    )
    max_object_length = object_lengths.max()
    object_feature_dim = object_features[0].size(1)

    pad_entry_obj = [PAD_LABEL] + [0.0] * (object_feature_dim - 1)
    pad_entry_obj = torch.tensor(pad_entry_obj, dtype=torch.float32)

    object_features_padded = []
    for of in object_features:
        length = of.size(0)
        pad_length = max_object_length - length
        if pad_length > 0:
            padding = pad_entry_obj.unsqueeze(0).repeat(pad_length, 1)
            padded_of = torch.cat([of, padding], dim=0)
        else:
            padded_of = of
        object_features_padded.append(padded_of)
    object_features_padded = torch.stack(object_features_padded)

    # Pad figure_features
    figure_lengths = torch.tensor(
        [ff.size(0) for ff in figure_features], dtype=torch.long
    )
    max_figure_length = figure_lengths.max()
    figure_feature_dim = figure_features[0].size(1)

    pad_entry_fig = [PAD_LABEL] + [0.0] * (figure_feature_dim - 1)
    pad_entry_fig = torch.tensor(pad_entry_fig, dtype=torch.float32)

    figure_features_padded = []
    for ff in figure_features:
        length = ff.size(0)
        pad_length = max_figure_length - length
        if pad_length > 0:
            padding = pad_entry_fig.unsqueeze(0).repeat(pad_length, 1)
            padded_ff = torch.cat([ff, padding], dim=0)
        else:
            padded_ff = ff
        figure_features_padded.append(padded_ff)
    figure_features_padded = torch.stack(figure_features_padded)

    group_features_tensor = torch.stack(group_features)

    # Initialize model
    model = DetectionClassificationModel(
        num_classes=num_classes,
        num_labels=num_labels,
        PAD_LABEL=PAD_LABEL,
        SEP_LABEL=SEP_LABEL,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        attn_heads=num_attention_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Forward pass
    logits = model(
        object_features_padded,
        figure_features_padded,
        group_features_tensor,
        object_lengths,
        figure_lengths,
    )

    # Print outputs
    print("Model Output (Logits):")
    print(logits)
    print("\nPredicted Probabilities:")
    print(torch.sigmoid(logits))


if __name__ == "__main__":
    main()
