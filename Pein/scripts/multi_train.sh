#!/bin/bash

# Define directories and class file
POS_DIR="/data/training_code/yolov7/runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3/pos/labels"
NEG_DIR="/data/training_code/yolov7/runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3/neg/labels"
CLASSES_FILE="/data/dataset/bbu_training_data/bbu_and_shield/classes.txt"
PYTHON_SCRIPT="/data/training_code/yolov7/Pein/trainer.py"

BATCH_SIZE=128
CRITERION='f1_score'
NUM_EPOCHS=100

# Define numerical lists for learning rate, embedding_dim, hidden_dim, and num_layers
lr_list=(0.02 0.01 0.08 0.005 0.001) # Narrow down around 0.001 and 0.0005
embedding_dim_list=(32 64)           # Focus on 32 and 64
hidden_dim_list=(64 128)             # Focus on 64 and 128
num_layers_list=(1 2)                # Focus on 1 and 2 layers

mkdir -p ./train_logs
mkdir -p ./ckpt

# List of available GPUs
gpu_list=(0 1 2 3 4 5)
num_gpus=${#gpu_list[@]}

# Control how many processes each GPU can handle
num_processes_per_gpu=20

# Total number of processes to run in parallel across all GPUs
total_processes=$((num_gpus * num_processes_per_gpu))

# Prepare an array to store all commands
commands=()

# Initialize GPU index
gpu_index=0

# Generate the list of commands with assigned GPUs
for lr in "${lr_list[@]}"; do
    for embedding_dim in "${embedding_dim_list[@]}"; do
        for hidden_dim in "${hidden_dim_list[@]}"; do
            for num_layers in "${num_layers_list[@]}"; do

                # Assign a GPU in round-robin fashion
                gpu_id=${gpu_list[$gpu_index]}
                gpu_index=$(((gpu_index + 1) % num_gpus))

                # Generate a log file name based on the hyperparameters
                log_file="./train_logs/lr_${lr}_emb_${embedding_dim}_hid_${hidden_dim}_layers_${num_layers}.log"

                # Generate a unique TensorBoard log directory based on the hyperparameters
                tb_log_dir="./tb_logs/lr_${lr}_emb_${embedding_dim}_hid_${hidden_dim}_layers_${num_layers}"

                # Create the command with the assigned GPU
                cmd="CUDA_VISIBLE_DEVICES=$gpu_id /opt/conda/envs/yolov7/bin/python $PYTHON_SCRIPT \
                    --device_id 0 \
                    --criterion $CRITERION \
                    --pos_dir $POS_DIR \
                    --neg_dir $NEG_DIR \
                    --classes_file $CLASSES_FILE \
                    --batch_size $BATCH_SIZE \
                    --lr $lr \
                    --embedding_dim $embedding_dim \
                    --hidden_dim $hidden_dim \
                    --fc_hidden_dim 128 64 \
                    --num_layers $num_layers \
                    --dropout 0.1 \
                    --num_epochs $NUM_EPOCHS \
                    --class_counts_dim 8 \
                    --tb_log_dir \"$tb_log_dir\" \
                    --checkpoint_path \"./ckpt/model_lr_${lr}_emb_${embedding_dim}_hid_${hidden_dim}_layers_${num_layers}.pth\" \
                    > \"$log_file\" 2>&1"

                # Add the command to the list
                commands+=("$cmd")

                echo "Prepared command for GPU $gpu_id: lr=$lr, embedding_dim=$embedding_dim, hidden_dim=$hidden_dim, num_layers=$num_layers, tb_log_dir=$tb_log_dir"

            done
        done
    done
done

# Export the necessary environment variable for GNU Parallel
export CUDA_VISIBLE_DEVICES

# Run the commands in parallel using GNU Parallel, allowing multiple processes per GPU
printf "%s\n" "${commands[@]}" | parallel -j $total_processes

echo "All training jobs completed."
