#!/bin/bash

# Set the directory containing VNNLIB config directories
write_vnnlib_config_dir="write_vnnlib_config_dir"

# Allow the user to break the script with Ctrl-C
trap "exit" INT

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate alpha-beta-crown

# Loop through each model directory in the specified path
for model_dir in "$write_vnnlib_config_dir"/*/; do
    model_name=$(basename "$model_dir")

    # Loop through each combination directory within the model directory
    for combination_dir in "$model_dir"*; do
        combination_name=$(basename "$combination_dir")

        # Loop through each config file within the combination directory
        for fold in "$combination_dir"/*.yaml; do
            fold_name=$(basename "$fold")
            fold_name=${fold_name%_config.yaml}

            # Execute the VNNLIB file and redirect output to a log file
            echo "Executing $fold"
            output_dir="./write_vnnlib_execute_dir/$model_name/$combination_name"
            mkdir -p "$output_dir"
            CUDA_VISIBLE_DEVICES=3 python abcrown.py --config "$fold" > "$output_dir/$fold_name.txt"
        done
    done
done
