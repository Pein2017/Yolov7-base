#!/bin/bash

# Define directories and class file
POS_DIR="/data/training_code/yolov7/runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3/pos/labels"
NEG_DIR="/data/training_code/yolov7/runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3/neg/labels"
CLASSES_FILE="/data/dataset/bbu_training_data/bbu_and_shield/classes.txt"
PYTHON_SCRIPT="/data/training_code/yolov7/Pein/trainer.py"

# Define the testing parameters
BATCH_SIZE=128
CRITERION='f1_score'
NUM_EPOCHS=100
LEARNING_RATE=0.01
EMBEDDING_DIM=8
HIDDEN_DIM=8
NUM_LAYERS=1
DROPOUT=0.1
DEVICE_ID=0
TB_LOG_DIR="./tb_logs/test_run"

# Run the Python script with the specified arguments
/opt/conda/envs/yolov7/bin/python $PYTHON_SCRIPT \
    --pos_dir $POS_DIR \
    --neg_dir $NEG_DIR \
    --classes_file $CLASSES_FILE \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --embedding_dim $EMBEDDING_DIM \
    --hidden_dim $HIDDEN_DIM \
    --fc_hidden_dim 128 64 \
    --num_layers $NUM_LAYERS \
    --dropout $DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --class_counts_dim 8 \
    --criterion $CRITERION \
    --device_id $DEVICE_ID \
    --tb_log_dir $TB_LOG_DIR \
    --checkpoint_path "./ckpt/model_test.pth"
