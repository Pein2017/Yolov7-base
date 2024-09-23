# Define variables
WORKERS=8
DEVICES="5" # Use only the first GPU (device 0)
BATCH_SIZE=32
EPOCHS=300
DATA_CONFIG="data/bbu.yaml"
CFG="cfg/training/yolov7-pein.yaml"
WEIGHTS="./weights/yolov7.pt"
HYP="data/hyp.scratch.custom.yaml"
IMG_SIZE="640 640"

EXPERIMENT_NAME="bbu-bs192-ep300-v7_pt-custom-img640"

# Run the training script with the variables
python train.py --workers $WORKERS --device $DEVICES --batch-size $BATCH_SIZE --epochs $EPOCHS \
    --data $DATA_CONFIG --img $IMG_SIZE \
    --cfg $CFG --weights $WEIGHTS \
    --name $EXPERIMENT_NAME --hyp $HYP
