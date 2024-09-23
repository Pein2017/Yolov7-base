#!/bin/bash

# Define variables
NPROC_PER_NODE=6
MASTER_PORT=9566
WORKERS=8
DEVICES="0,1,2,3,4,5"
BATCH_SIZE=192
EPOCHS=1000
DATA_CONFIG="data/bbu.yaml"
WEIGHTS="./weights/yolov7.pt"
MODEL="v7"     # Model checkpoint
CFG_TYPE="ori" # Fix to be ori, default using kmeans
CFG="${MODEL}-${CFG_TYPE}"
HYP_VERSION="p5" # Hyperparameter version
HYP="data/hyp.scratch.$HYP_VERSION.yaml"
IMG_SIZE="640 640"

EXPERIMENT_NAME="bs${BATCH_SIZE}-ep${EPOCHS}-lr_1e-3"

# Set the folder structure
OUTPUT_FOLDER="runs/train/$MODEL/$HYP_VERSION"

# Set the script and the torchrun path as variables
TRAIN_SCRIPT="train.py"
TORCHRUN_PATH="/opt/conda/envs/yolov7/bin/torchrun"

# Run the training script with the variables
$TORCHRUN_PATH --nproc_per_node $NPROC_PER_NODE --master_port $MASTER_PORT \
    $TRAIN_SCRIPT --workers $WORKERS --device $DEVICES --batch-size $BATCH_SIZE --epochs $EPOCHS \
    --data $DATA_CONFIG --img $IMG_SIZE \
    --cfg cfg/training/$CFG.yaml --weights $WEIGHTS \
    --name $EXPERIMENT_NAME --hyp $HYP --sync-bn \
    --project "$OUTPUT_FOLDER" --exist-ok
