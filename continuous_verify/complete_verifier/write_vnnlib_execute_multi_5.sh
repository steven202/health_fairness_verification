#!/bin/bash

rm -rf write_vnnlib_execute_dir

# Set the directory containing VNNLIB config directories
write_vnnlib_config_dir="write_vnnlib_config_dir"
script_path="abcrown.py"

# Allow the user to break the script with Ctrl-C
trap "exit" INT

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate alpha-beta-crown

# List of available GPU IDs
gpus=(2 3)  # Modify this list based on the available GPU IDs

# Function to run the verification process
run_verification() {
    local fold=$1
    local output_dir=$2
    local gpu_id=$3

    echo "Executing $fold on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python "$script_path" --config "$fold" > "$output_dir/$(basename ${fold%_config.yaml}).txt"
}

# Initialize GPU counter
gpu_counter=0
num_gpus=${#gpus[@]}
process_counter=0

# Loop through each model directory in the specified path
for model_dir in "$write_vnnlib_config_dir"/*/; do

    model_name=$(basename "$model_dir")

    # Loop through each combination directory within the model directory
    for combination_dir in "$model_dir"/*/; do
        combination_name=$(basename "$combination_dir")

        # Loop through each config file within the combination directory
        for fold in "$combination_dir"/*.yaml; do
            output_dir="./write_vnnlib_execute_dir/$model_name/$combination_name"
            mkdir -p "$output_dir"

            # Run the verification in the background, using GPUs in a round-robin fashion
            gpu_id=${gpus[gpu_counter]}
            run_verification "$fold" "$output_dir" $gpu_id &

            # Update the GPU counter for round-robin assignment
            gpu_counter=$(( (gpu_counter + 1) % num_gpus ))
            process_counter=$((process_counter + 1))

            # Check if we have started 64 processes
            if [[ $process_counter -ge 64 ]]; then
                # Wait for all background processes to finish
                wait
                process_counter=0  # Reset process counter
            fi
        done
    done
done

# Wait for any remaining background processes to finish
wait
