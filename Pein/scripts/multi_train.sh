#!/bin/bash

# Function to map GPU device IDs to their UUIDs
map_device_id_to_uuid() {
    declare -A gpu_map
    while IFS=, read -r device_id gpu_uuid; do
        gpu_map[$device_id]=$gpu_uuid
    done < <(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader,nounits)
    echo ${gpu_map[@]}
    echo ${!gpu_map[@]}
}

# Function to count the number of running processes on a specific GPU by UUID
get_gpu_process_count_by_uuid() {
    gpu_uuid=$1
    # Query nvidia-smi and count the processes associated with the given GPU UUID
    count=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$gpu_uuid")
    echo $count
}

# Record the start time
start_time=$(date +%s)

# Define directories and class file
POS_DIR="/data/training_code/yolov7/runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3/pos/labels"
NEG_DIR="/data/training_code/yolov7/runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3/neg/labels"
CLASSES_FILE="/data/dataset/bbu_training_data/bbu_and_shield/classes.txt"
PYTHON_SCRIPT="/data/training_code/yolov7/Pein/trainer.py"

# Fixed parameters
BATCH_SIZE=256
CRITERION='avg_error'
NUM_EPOCHS=50
DROPOUT=0.1
SCHEDULER_TYPE='onecycle'
SEED=17

# Parameter lists for multiple training runs
LEARNING_RATES=(0.005 0.002)
EMBEDDING_DIMS=(16 32 64)
HIDDEN_DIMS=(64 128 256)
NUM_LAYERS=(2 4 8)
ATTENTION_HEADS=(4 8 16)
RESAMPLE_METHODS=('both' 'upsample' 'downsample')

# Create directories for logs
LOG_DIR_FOLDER="prod_run/9-26"
LOG_DIR_BASE="./train_logs/$LOG_DIR_FOLDER"
LTN_LOG_DIR_BASE="./ltn_logs/$LOG_DIR_FOLDER"
mkdir -p $LOG_DIR_BASE
mkdir -p $LTN_LOG_DIR_BASE

# GPU management
gpu_list=(0 1 2 3 4 5) # Available GPUs
num_gpus=${#gpu_list[@]}
max_process_per_gpu=10 # Max processes each GPU can handle

# Step 1: Map GPU Device IDs to UUIDs
declare -A gpu_map
while IFS=, read -r device_id gpu_uuid; do
    gpu_map[$device_id]=$gpu_uuid
done < <(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader,nounits)

# Array to store all commands
commands=()

# Initialize GPU index for round-robin GPU assignment
gpu_index=0

# Loop through each combination of hyperparameters and prepare commands
for lr in "${LEARNING_RATES[@]}"; do
    for embedding_dim in "${EMBEDDING_DIMS[@]}"; do
        for hidden_dim in "${HIDDEN_DIMS[@]}"; do
            for num_layers in "${NUM_LAYERS[@]}"; do
                for resample_method in "${RESAMPLE_METHODS[@]}"; do
                    for attn_head in "${ATTENTION_HEADS[@]}"; do

                        # Assign a GPU in round-robin fashion
                        gpu_id=${gpu_list[$gpu_index]}
                        gpu_index=$(((gpu_index + 1) % num_gpus))

                        # Generate a log file name based on the hyperparameters
                        log_file="$LOG_DIR_BASE/resamp_${resample_method}-lr_${lr}-emb_${embedding_dim}-hid_${hidden_dim}-layers_${num_layers}-attn_h_${attn_head}.log"

                        # Create the command with the assigned GPU
                        cmd="CUDA_VISIBLE_DEVICES=$gpu_id /opt/conda/envs/yolov7/bin/python $PYTHON_SCRIPT \
                            --pos_dir $POS_DIR \
                            --neg_dir $NEG_DIR \
                            --classes_file $CLASSES_FILE \
                            --batch_size $BATCH_SIZE \
                            --lr $lr \
                            --embedding_dim $embedding_dim \
                            --hidden_dim $hidden_dim \
                            --fc_hidden_dim 128 64 \
                            --num_layers $num_layers \
                            --dropout $DROPOUT \
                            --num_epochs $NUM_EPOCHS \
                            --criterion $CRITERION \
                            --ltn_log_dir \"$LTN_LOG_DIR_BASE\" \
                            --resample_method $resample_method \
                            --seed $SEED \
                            --attn_heads $attn_head \
                            > \"$log_file\" 2>&1"

                        # Add the command to the list
                        commands+=("$cmd")

                        echo "Prepared command for GPU $gpu_id: lr=$lr, embedding_dim=$embedding_dim, hidden_dim=$hidden_dim, num_layers=$num_layers"
                    done
                done
            done
        done
    done
done

# Total number of commands
num_commands=${#commands[@]}

# Track how many processes are running on each GPU (by UUID)
gpu_process_count=()
for gpu_id in "${gpu_list[@]}"; do
    gpu_uuid=${gpu_map[$gpu_id]}
    count=$(get_gpu_process_count_by_uuid $gpu_uuid)
    gpu_process_count+=($count)
done

# Create a flag to track if a GPU has already reached the max process limit
gpu_maxed_out=()

# Initialize maxed out flags for each GPU to false
for ((j = 0; j < num_gpus; j++)); do
    gpu_maxed_out[$j]=0
done

# Adjust the assignment loop to allow more concurrency
for ((i = 0; i < num_commands; i++)); do
    while true; do
        # Find the GPU with the least number of processes assigned
        min_processes=${gpu_process_count[0]}
        gpu_id=${gpu_list[0]}
        for ((j = 1; j < num_gpus; j++)); do
            if [ ${gpu_process_count[$j]} -lt $min_processes ]; then
                min_processes=${gpu_process_count[$j]}
                gpu_id=${gpu_list[$j]}
            fi
        done

        # Check if the selected GPU can handle more processes
        if [ ${gpu_process_count[$gpu_id]} -lt $max_process_per_gpu ]; then
            # Assign the command to this GPU
            cmd="CUDA_VISIBLE_DEVICES=$gpu_id ${commands[$i]}"
            echo "Assigning command to GPU $gpu_id: ${commands[$i]}"
            eval $cmd & # Run the command in the background

            # Increment the process count for this GPU
            gpu_process_count[$gpu_id]=$((gpu_process_count[$gpu_id] + 1))
            break
        fi
        sleep 1 # Wait for a bit before checking again
    done
done

# Wait for all background processes to finish
wait

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
total_time=$((end_time - start_time))

# Output the total time taken and the number of scripts executed
echo "Training completed!"
echo "Total time taken: $((total_time / 60)) minutes and $((total_time % 60)) seconds."
echo "Number of experiments executed: $num_commands"
