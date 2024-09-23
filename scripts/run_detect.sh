#!/bin/bash

# Set variables
WEIGHTS="/data/training_code/yolov7/runs/train/v7-e6e/p6/bs96-ep1000-lr_1e-3/weights/best.pt"
NAME="pos"
SOURCE="/data/dataset/BBU_wind_shield/$NAME"
IMG_SIZE=640
DEVICE=0
SAVE_TXT="--save-txt"
EXIST_OK="--exist-ok"
SAVE_CONF="--save-conf"
PROJECT="runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3"

# Run YOLOv7 detect.py script with the specified options
/opt/conda/envs/yolov7/bin/python detect.py \
    --weights "$WEIGHTS" \
    --source "$SOURCE" \
    --img-size "$IMG_SIZE" \
    --device "$DEVICE" \
    $EXIST_OK \
    $SAVE_TXT \
    $SAVE_CONF \
    --name "$NAME" \
    --project "$PROJECT"
