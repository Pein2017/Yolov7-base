#!/bin/bash

# Define directories and class file
POS_DIR="/data/training_code/yolov7/runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3/pos/labels"
NEG_DIR="/data/training_code/yolov7/runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3/neg/labels"
CLASSES_FILE="/data/dataset/bbu_training_data/bbu_and_shield/classes.txt"
PYTHON_SCRIPT="/data/training_code/yolov7/Pein/trainer.py"

# Define the testing parameters
BATCH_SIZE=256
CRITERION='f1_score'
NUM_EPOCHS=30
LEARNING_RATE=0.002
EMBEDDING_DIM=32
HIDDEN_DIM=32
NUM_LAYERS=2
DROPOUT=0.1
RESAMPLE_METHOD='both'
DEVICE_ID=2

LTN_LOG_DIR="./ltn_logs/test_run"
TRAIN_LOG_DIR='./train_logs/test_run'
LOG_FILE="$TRAIN_LOG_DIR/output.log"

# scheduler_type
SCHEDULER_TYPE='onecycle'

mkdir -p $TRAIN_LOG_DIR

# Run the Python script with the specified arguments
/opt/conda/envs/yolov7/bin/python $PYTHON_SCRIPT \
    --pos_dir $POS_DIR \
    --neg_dir $NEG_DIR \
    --classes_file $CLASSES_FILE \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --scheduler_type $SCHEDULER_TYPE \
    --embedding_dim $EMBEDDING_DIM \
    --hidden_dim $HIDDEN_DIM \
    --fc_hidden_dims 128 64 \
    --num_layers $NUM_LAYERS \
    --dropout $DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --criterion $CRITERION \
    --device_id $DEVICE_ID \
    --ltn_log_dir $LTN_LOG_DIR \
    --resample_method $RESAMPLE_METHOD \
    >"$LOG_FILE" 2>&1
