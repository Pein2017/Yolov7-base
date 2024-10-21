# BBU Task: Group-based Image Analysis and Prediction

## Task Overview
The task involves analyzing groups of images sharing the same unique ID prefix and predicting a boolean output (True or False) for each group. These images represent different views of the same location, including both close-up (局部特写) and panoramic (全景图) shots. The goal is to detect the existence of specific objects (BBUs and shields) and verify if certain rules are satisfied, such as:

1. Each BBU is properly closed/assigned to a wind shield.
2. The installation of BBUs maintains a minimum distance (not too close to each other).

The model aims to learn implicit rules that determine whether a location passes or fails these criteria. By aggregating information from multiple views, the model can overcome limitations of single-view analysis, especially when crucial objects might be missed in local close-up images.

## Dataset Structure
1. **Object-level features**: 
   - Derived from YOLO detection outputs (*.txt files)
   - Each object has the following features:
     - **粗颗粒度 (coarse-grained) label**: Integer representing 'bbu' (0) or 'shield' (1)
       - 0 ('bbu') for class labels [0, 1, 3]
       - 1 ('shield') for class labels [2, 4]
     - **class_label**: Integer representing the original object class (0-4)
     - **x**: X-coordinate of the object center
     - **y**: Y-coordinate of the object center
     - **w**: Width of the object
     - **h**: Height of the object
     - **confidence**: Detection confidence score
     - **area**: Calculated as w * h
   - These features are stored in an ordered dictionary `object_feature_functions`

2. **Figure-level features**:
   - Aggregated from object-level features
   - Includes weighted count of objects for each **coarse label** (bbu, shield):
     - Weighted by the confidence score of each detected object
     - Results in a vector of length 2
   - Includes weighted count of objects for each **class**:
     - Weighted by the confidence score of each detected object
     - Results in a vector of length equal to the number of classes
   - Includes figure size (width, height) from a separate CSV file
   - **Total feature vector**: `[coarse_counts (2), class_counts (num_classes), width, height]`

3. **Group-level features**:
   - Aggregated from figure-level features
   - Contains:
     - Total weighted count of objects for each **coarse label** across all images in the group
     - Total weighted count of objects for each **class** across all images in the group
     - Number of total images in this group
   - **Total feature vector**: `[total_coarse_counts (2), total_class_counts (num_classes), total_num_figures]`

## Data Processing Pipeline
1. **Data Loading**:
   - Load positive and negative samples from separate directories
   - Read classes from a file
   - Load figure sizes from a CSV file

2. **Feature Extraction**:
   - Extract object-level features from YOLO output files
   - Aggregate object features to create figure-level features:
     - Sum confidence scores for each class
     - Append figure size from CSV
   - Aggregate figure features to create group-level features:
     - Sum class counts across all figures in the group
     - Count total number of figures

3. **Sequence Creation**:
   - Create sequences of object features for each image
   - Add separator tokens (SEP_LABEL) between images in a group
   - Pad sequences to a uniform length with padding tokens (PAD_LABEL)

4. **Batch Processing**:
   - Implement custom collate function for batching
   - Handle variable-length sequences with padding

5. **Data Balancing**:
   - Support resampling to balance positive and negative samples

## Model Input Preparation

The data is prepared for a transformer-based model with three main components:

1. **Sequences of object features**: `[batch_size, max_objects, num_object_features]`
   - `batch_size`: Number of groups in the batch
   - `max_objects`: Maximum number of objects across all figures in the batch
   - `num_object_features`: Number of features for each object (8 in this case)
     - coarse_label (0 or 1)
     - class_label (0 to 4)
     - x, y, w, h (coordinates and dimensions)
     - confidence
     - area

2. **Sequences of figure features**: `[batch_size, max_figures, num_classes + 4]`
   - `batch_size`: Number of groups in the batch
   - `max_figures`: Maximum number of figures in any group in the batch
   - `num_classes + 4`: Total number of figure-level features
     - First 2 elements: Coarse-grained counts [bbu_count, shield_count]
     - Next `num_classes` elements: Class-specific counts
     - Last 2 elements: Figure dimensions [width, height]

3. **Group-level features**: `[batch_size, num_classes + 3]`
   - `batch_size`: Number of groups in the batch
   - `num_classes + 3`: Total number of group-level features
     - First 2 elements: Total coarse-grained counts [total_bbu_count, total_shield_count]
     - Next `num_classes` elements: Total class-specific counts
     - Last element: Total number of figures in the group

### Special Tokens

- **SEP_LABEL** (Separator): Used to mark boundaries between figures within a group.
  - Value: `num_classes + 1`
  - Example: If there are 5 classes, SEP_LABEL = 6

- **PAD_LABEL** (Padding): Used to ensure uniform sequence lengths within a batch.
  - Value: `num_classes + 2`
  - Example: If there are 5 classes, PAD_LABEL = 7

These special tokens are crucial for maintaining the structure of the input data and allowing the model to distinguish between actual data, figure boundaries, and padding.

### Example: Data Processing for a Group with Two Figures

#### Class Definitions
1. huawei_bbu (0)
2. alx_bbu (1)
3. huawei_shield (2)
4. zhongxing_bbu (3)
5. zhongxing_shield (4)

#### Input Data

##### 1. Object-level Features
Figure 1:
```
[[0, 0, 0.1, 0.2, 0.3, 0.4, 0.9, 0.12],  # ['bbu', 'huawei_bbu', ...]
 [0, 1, 0.5, 0.6, 0.1, 0.1, 0.8, 0.01],  # ['bbu', 'alx_bbu', ...]
 [0, 3, 0.7, 0.3, 0.2, 0.2, 0.7, 0.04]]  # ['bbu', 'zhongxing_bbu', ...]
```
Figure 2:
```
[[1, 2, 0.3, 0.4, 0.2, 0.2, 0.7, 0.04],    # ['shield', 'huawei_shield', ...]
 [1, 4, 0.6, 0.5, 0.15, 0.15, 0.6, 0.0225]]  # ['shield', 'zhongxing_shield', ...]
```

##### 2. Figure-level Features
```
Figure 1: [1.7, 0.0, 0.9, 0.8, 0.0, 0.7, 0.0, 100, 200]  # [coarse_counts, class_counts, width, height]
Figure 2: [0.0, 1.3, 0.0, 0.0, 0.7, 0.0, 0.6, 150, 250]
```

##### 3. Group-level Feature
```
[1.7, 1.3, 0.9, 0.8, 0.7, 0.7, 0.6, 2]  # [total_coarse_counts, total_class_counts, total_figures]
```

#### Processing Steps

##### Step 1: Create Object Sequence
```
[[0, 0, 0.1, 0.2, 0.3, 0.4, 0.9, 0.12],
 [0, 1, 0.5, 0.6, 0.1, 0.1, 0.8, 0.01],
 [0, 3, 0.7, 0.3, 0.2, 0.2, 0.7, 0.04],
 SEP,
 [1, 2, 0.3, 0.4, 0.2, 0.2, 0.7, 0.04],
 [1, 4, 0.6, 0.5, 0.15, 0.15, 0.6, 0.0225],
 PAD]
```

##### Step 2: Create Figure Sequence
```
[[1.7, 0.0, 0.9, 0.8, 0.0, 0.7, 0.0, 100, 200],
 SEP,
 [0.0, 1.3, 0.0, 0.0, 0.7, 0.0, 0.6, 150, 250]]
```

##### Step 3: Final Output for this Group
- **Object features**: (as in Step 1)
- **Figure features**: (as in Step 2)
- **Group feature**: `[1.7, 1.3, 0.9, 0.8, 0.7, 0.7, 0.6, 2]`

#### Batch Processing
In the `custom_collate_fn`:
- **Object sequences** are padded to match the longest sequence in the batch.
- **Figure sequences** are padded to match the longest group in the batch.

> **Key Point**: This structure allows the transformer model to distinguish between different hierarchical levels (objects, figures, groups) while maintaining a consistent input shape across different samples.

## Implementation Details
- `GroupedDetectionDataset.__getitem__()` method creates the sequences with SEP tokens.
- `custom_collate_fn()` function handles the padding of sequences to create uniform-length batches.
- **Special labels**:
  - `CLASS_COUNTS_LABEL = num_classes`
  - `SEP_LABEL = num_classes + 1`
  - `PAD_LABEL = num_classes + 2`

> **Note**: In this example, `SEP_LABEL = 6` and `PAD_LABEL = 7`.

## Model Architecture: Transformer-based Approach

To effectively process the hierarchical nature of our data (objects within figures within groups), we propose a transformer-based model architecture that leverages the available features at each level.

### Key Components

1. **Object-level Transformer**
   - **Input**: Sequence of object features for each figure
   - **Purpose**: Capture relationships between objects within a single image
   - **Output**: Aggregated representation of objects in an image

2. **Figure-level Transformer**
   - **Input**: Sequence of figure features (including aggregated object representations)
   - **Purpose**: Capture relationships between different figures within a group
   - **Output**: Aggregated representation of all figures in a group

3. **Group-level Processing**
   - **Input**: Aggregated figure representation and group-level features
   - **Purpose**: Final classification based on all available information
   - **Output**: Binary classification (Pass/Fail) for the group

### Advanced Aggregation Techniques

To better aggregate the hierarchical features and capture the complex relationships between objects across different figures, we propose the following approaches:

1. **Hierarchical Multi-Head Attention (MHA)**
   - **Description**: 
     - Apply an MHA layer to the `object_feature_sequence` within each figure to capture intra-figure relationships.
     - Apply another MHA layer to the `figure_feature_sequence` to capture inter-figure relationships across the entire group.
     - Combine the outputs using concatenation or an MLP to form a comprehensive representation.

2. **Cross-Attention with Figure Constraints**
   - **Description**:
     - Apply MHA to the `object_feature_sequence` within each figure to capture object-level relationships.
     - Use cross-attention between object features and their corresponding figure-wise features to integrate detailed object information with figure-level context.
     - **Constraint**: Ensure that object features of each figure are only exposed to the corresponding figure-wise feature, preventing cross-talk between different figures.

3. **Enhanced Feature Aggregation**
   - **Description**:
     - Utilize advanced pooling methods, such as attention-based pooling, to aggregate transformer outputs instead of using simple mean or max pooling.
     - **Goal**: Focus on the most salient features within the transformer outputs, improving the informativeness of the aggregated representations.

4. **Rule-Guided Attention**
   - **Description**:
     - Incorporate prior knowledge about important rules (e.g., BBU-shield pairing, minimum distance) into the attention mechanism.
     - Use these rules to guide the attention weights, focusing the model on relevant object pairs or spatial relationships that are critical for classification.

### Implementation Considerations

- **Hierarchical Structure**:
  - Ensure that the transformer layers are organized hierarchically, first processing object-level features within each figure and then figure-level features across the group.
  - Maintain clear boundaries between different levels to preserve the hierarchical relationships within the data.

- **Attention Mechanism Configuration**:
  - Carefully configure the number of attention heads and layers to balance model complexity and performance.
  - Utilize residual connections and layer normalization to stabilize training and improve gradient flow.

- **Attention Masks**:
  - Implement attention masks to handle padding and to enforce figure constraints in cross-attention mechanisms.
  - Ensure that the model does not attend to padded elements or across unrelated figures.

- **Feature Dimensionality**:
  - Align the dimensionality of object-level and figure-level features to ensure seamless integration during aggregation.
  - Use projection layers where necessary to match feature dimensions before applying attention mechanisms.

- **Incorporating Rules**:
  - Translate business rules (e.g., BBU-shield pairing) into mechanisms that influence the attention weights or connections within the model.
  - This can involve initializing attention patterns, adding bias terms, or designing custom attention layers that account for these rules.

- **Training Strategy**:
  - Utilize appropriate loss functions, such as binary cross-entropy for the Pass/Fail classification.
  - Employ regularization techniques like dropout to prevent overfitting.
  - Consider using transfer learning by initializing transformer layers with pre-trained weights if applicable.

### Multi-head Attention

- **Essential Component**: 
  - **Functionality**: Allows the model to focus on different parts of the input sequence simultaneously, capturing diverse aspects of the data.
  - **Implementation**: Integral to transformer encoders at both object-level and figure-level.

- **Benefits**:
  - **Parallel Attention**: Multiple attention heads can attend to different positions and representations, enhancing the model's ability to capture intricate patterns.
  - **Feature Diversity**: Encourages the learning of diverse representations, improving the model's overall expressiveness.

### Suggested Transformer and Cross-Attention Applications

Given the nature of the task and the available data, here are specific suggestions on how to apply the Transformer architecture and cross-attention mechanisms:

1. **Object-level Transformer**:
   - **Input Representation**:
     - Each object within a figure is represented by its feature vector: `[coarse_label, class_label, x, y, w, h, confidence, area]`.
   - **Processing**:
     - Apply an embedding layer to categorical features (`coarse_label` and `class_label`).
     - Project numerical features (`x`, `y`, `w`, `h`, `confidence`, `area`) into the same embedding space.
     - Concatenate embeddings and projected numerical features.
     - Add positional encoding to preserve the order of objects.
     - Pass the sequence through a Transformer Encoder to capture intra-object relationships within the figure.
     - Aggregate the transformer outputs (e.g., via mean pooling) to obtain a fixed-size representation for the figure.

2. **Figure-level Transformer**:
   - **Input Representation**:
     - Each figure is represented by its aggregated object representation and figure-level features: `[bbu_count, shield_count, class0_count, class1_count, class2_count, class3_count, class4_count, width, height]`.
   - **Processing**:
     - Project figure-level features into the same embedding space as object-level representations.
     - Concatenate the aggregated object representation with figure-level features.
     - Add positional encoding to preserve the order of figures within the group.
     - Pass the sequence through a second Transformer Encoder to capture inter-figure relationships across the group.
     - Aggregate the transformer outputs to obtain a fixed-size representation for the entire group.

3. **Cross-Attention Mechanism**:
   - **Objective**:
     - Enable interaction between object-level and figure-level representations within the same figure.
   - **Implementation**:
     - For each figure, use a cross-attention layer where the figure-level representation attends to the object-level representations.
     - Incorporate masking to ensure that attention is confined within the boundaries of each figure, preventing leakage between figures.

4. **Final Classification**:
   - **Input Representation**:
     - Combine the aggregated group-level representation with any additional group-level features.
   - **Processing**:
     - Pass the combined representation through fully connected layers with activation functions and dropout.
     - Apply a sigmoid activation to obtain probability scores for the Pass/Fail classification.
   - **Output**:
     - Binary classification indicating whether the location passes or fails based on the presence and arrangement of BBUs and shields.

### Aggregating 'Full' and 'Detailed' Pictures

To effectively aggregate information from both full (panoramic) and detailed (close-up) pictures, consider the following strategies:

1. **Dual-pathway Architecture**:
   - **Description**:
     - Implement separate pathways within the model for processing full and detailed pictures.
     - Each pathway consists of its own object-level and figure-level transformers.
   - **Processing**:
     - Full pictures provide global context, capturing overall arrangements and spatial relationships.
     - Detailed pictures provide granular information about individual objects and their precise relationships.
   - **Aggregation**:
     - Combine the outputs of both pathways (e.g., by concatenation or attention-based fusion) to form a comprehensive representation for the group.

2. **Attention-based Fusion**:
   - **Description**:
     - After processing full and detailed pictures through their respective transformers, use an attention mechanism to fuse their representations.
   - **Processing**:
     - The model learns to weight the importance of global and local features dynamically based on the input data.

3. **Positional Encoding Adjustment**:
   - **Description**:
     - Modify positional encoding to differentiate between full and detailed pictures.
     - For example, prepend a special token indicating the type of view (full or detailed) to each figure's feature sequence.

### Final Recommendations

Given the constraints of your available data, here are the recommended steps to implement the Transformer-based model with effective hierarchical aggregation:

1. **Model Design**:
   - **Hierarchical Transformers**:
     - Design the model with clear hierarchical transformer layers, first processing object-level features within each figure and then figure-level features across the group.
   - **Cross-Attention with Constraints**:
     - Implement cross-attention layers that allow figure-level transformers to attend to their corresponding object-level representations without cross-interacting between different figures.
   - **Dual-pathway for Full and Detailed Pictures**:
     - If your data explicitly differentiates between full and detailed pictures, consider implementing dual pathways to process them separately before aggregating their representations.

2. **Attention Mechanism Configuration**:
   - **Separate Attention Heads**:
     - Use separate attention heads for capturing intra-figure and inter-group relationships to maintain clarity in feature interactions.
   - **Masked Attention**:
     - Apply attention masks to enforce that attention computation is confined within the hierarchical boundaries (e.g., objects within the same figure).

3. **Aggregation Strategies**:
   - **Mean Pooling with Attention-based Weights**:
     - Instead of simple mean pooling, use attention-based pooling to allow the model to focus on more important objects and figures when aggregating representations.
   - **Learned Aggregation Layers**:
     - Introduce MLP layers that learn to combine object-level and figure-level representations effectively.

4. **Incorporating Business Rules**:
   - **Rule-Guided Features**:
     - Encode business rules as additional features or priors within the model to guide the attention mechanisms toward critical relationships.
   - **Constraint-based Attention**:
     - Design custom attention layers that prioritize or enforce specific relationships based on predefined rules (e.g., each BBU must be paired with a shield).

### Summary

This transformer-based architecture leverages the hierarchical structure of the BBU classification task, enabling the model to effectively process and aggregate information from objects to figures to groups. By incorporating advanced attention mechanisms and hierarchical processing, the model can capture complex relationships and improve classification performance. The proposed aggregation techniques focus on utilizing the available data structure to its fullest potential, ensuring that both 'full' and 'detailed' pictures contribute effectively to the final classification decision.

## Next Steps
1. **Implement the Transformer-based Model**:
   - Develop the model architecture as outlined in the Detailed Architecture section.
   - Ensure proper integration of embedding layers, positional encoding, transformer encoders, and aggregation mechanisms.
