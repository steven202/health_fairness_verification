#!/bin/bash

# Constants
TOTAL_FILES=$((30*63))  # Total number of files expected

# Function to calculate the elapsed time
function elapsed_time {
    local start_time=$1
    local end_time=$2
    local elapsed=$((end_time - start_time))
    echo $elapsed
}

# Function to estimate remaining time
function estimate_remaining_time {
    local previous_progress=$1
    local current_progress=$2
    local elapsed_since_last_update=$3
    
    local progress_diff=$(echo "$current_progress - $previous_progress" | bc -l)
    local remaining_progress=$(echo "100 - $current_progress" | bc -l)
    local remaining_time=$(echo "$elapsed_since_last_update * $remaining_progress / $progress_diff" | bc -l)
    
    echo ${remaining_time%.*}
}

# Function to convert seconds to days, hours, minutes, and seconds
function format_time {
    local seconds=$1
    local days=$((seconds / 86400))
    local hours=$(( (seconds % 86400) / 3600))
    local minutes=$(( (seconds % 3600) / 60))
    local seconds=$((seconds % 60))
    printf "%d days, %d hours, %d minutes, %d seconds" $days $hours $minutes $seconds
}

# Start time
start_time=$(date +%s)
previous_progress=0
previous_time=$start_time
estimated_remaining=0

# Monitor the progress
while true; do
    # Get the number of .txt files
    num_files=$(ls write_vnnlib_execute_dir/*/*/*.txt 2>/dev/null | wc -l)
    
    # Calculate current progress in percentage
    current_progress=$(echo "scale=2; $num_files / $TOTAL_FILES * 100" | bc -l)
    
    # Get the current time
    current_time=$(date +%s)
    
    # Calculate the elapsed time since the start
    elapsed_total=$(elapsed_time $start_time $current_time)
    
    # Calculate the elapsed time since the last progress update
    if (( $(echo "$current_progress != $previous_progress" | bc -l) )); then
        elapsed_since_last_update=$(elapsed_time $previous_time $current_time)
        estimated_remaining=$(estimate_remaining_time $previous_progress $current_progress $elapsed_since_last_update)
        previous_progress=$current_progress
        previous_time=$current_time
    fi
    
    # Format the remaining time
    formatted_remaining=$(format_time $estimated_remaining)
    
    # Display the progress and remaining time, overwriting the previous line
    printf "\rProgress: %.2f%%, Elapsed Time: %s, Estimated Remaining Time: %s" "$current_progress" "$(format_time $elapsed_total)" "$formatted_remaining"
    
    # Wait for 1 second before checking again
    sleep 1
done
