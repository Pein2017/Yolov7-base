#!/bin/bash

# Define variables
NPROC_PER_NODE=6
MASTER_PORT=9570
WORKERS=8
DEVICES="0,1,2,3,4,5"
BATCH_SIZE=192
EPOCHS=1000
DATA_CONFIG="data/bbu.yaml"
WEIGHTS="./weights/yolov7-e6e_training.pt"
# WEIGHTS='/data/training_code/yolov7/runs/train/v7-e6e/p5/bs192-ep400-lr_1e-3/weights/best_385.pt'
MODEL="v7-e6e"             # Model checkpoint
CFG_TYPE="ori"             # Can be either 'ori' or 'added'
CFG="${MODEL}-${CFG_TYPE}" # Full config name based on $MODEL and $CFG_TYPE
HYP_VERSION="p6"           # Hyperparameter version
HYP="data/hyp.scratch.$HYP_VERSION.yaml"
IMG_SIZE="640 640"

EXPERIMENT_NAME="bs${BATCH_SIZE}-ep${EPOCHS}-lr_5e-4"

# Set the folder structure
OUTPUT_FOLDER="runs/train/$MODEL/$HYP_VERSION"

# Set the script and the torchrun path as variables
TRAIN_SCRIPT="train_aux.py"
TORCHRUN_PATH="/opt/conda/envs/yolov7/bin/torchrun"

# Run the training script with the variables
$TORCHRUN_PATH --nproc_per_node $NPROC_PER_NODE --master_port $MASTER_PORT \
    $TRAIN_SCRIPT --workers $WORKERS --device $DEVICES --batch-size $BATCH_SIZE --epochs $EPOCHS \
    --data $DATA_CONFIG --img $IMG_SIZE \
    --cfg cfg/training/$CFG.yaml --weights $WEIGHTS \
    --name $EXPERIMENT_NAME --hyp $HYP --sync-bn \
    --project "$OUTPUT_FOLDER" --exist-ok
