#!/bin/bash
# Allow the user to break the script with Ctrl-C
trap "exit" INT

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate alpha-beta-crown

# Set the log folder name as a variable
log_folder="log_07042024"
new_dir="output_year_eps_07042024"

# Create the log folder and new directory if they don't exist
mkdir -p $log_folder
mkdir -p $new_dir
for year_eps in -1 3 6 10 15 1 5 7 9 11 13 2 4 8 12 14
do
    # clean any previous results
    rm -rf write_vnnlib_config_dir
    rm -rf write_vnnlib_csv_dir
    rm -rf write_vnnlib_execute_dir
    rm -rf write_vnnlib_save_dir
    rm -rf write_vnnlib_verified_result_dir

    python write_vnnlib_save_1.py --year-eps $year_eps
    python write_vnnlib_csv_2.py
    python write_vnnlib_config_3.py
    python write_vnnlib_execute_4.py
    bash write_vnnlib_execute_multi_5.sh
    python write_vnnlib_extract_6.py
    
    # Write it to a log file for each year_eps
    log_file="$log_folder/analysis_$year_eps.txt"
    echo "year_eps: $year_eps" > $log_file
    python write_vnnlib_analysis_7.py >> $log_file
    
    # Create a new folder for the current year_eps
    new_folder="$new_dir/output_$year_eps"
    mkdir -p $new_folder
    
    # Make a copy of the log file in the new folder
    cp $log_file $new_folder
    
    # Move the specified directories to the new folder
    mv write_vnnlib_config_dir $new_folder
    mv write_vnnlib_csv_dir $new_folder
    mv write_vnnlib_execute_dir $new_folder
    mv write_vnnlib_save_dir $new_folder
    mv write_vnnlib_verified_result_dir $new_folder
done
