#!/bin/bash

trap 'echo "Stopping the script"; exit 1' INT

# List of Python files used directly or indirectly in the script
used_python_files=("dataset2.py" "discrete_verify.py" "models.py" "model_verify.py" "test.py" "utils.py" "dataset_numpy_0104_fixed.py" "draft_model_combine_share_vertical.py" "model_verify_analyze.py" "pd_to_csv_combine.py" "train_model.py" "discrete_verify_analyze_spec.py" "k_fold_data_rent.py" "model_verify_analyze_spec.py" "pd_to_csv.py" "train.py" "extract_all_pnr_fpr.py")

# List of Bash files used in the script
used_bash_files=("run_all.sh" "fin5.sh" "fin6.sh" "fin7.sh" "fin8.sh" "fin9.sh" "clean.sh")

# Find all Python files in the current directory
all_python_files=($(ls *.py 2>/dev/null))

# Find all Bash files in the current directory
all_bash_files=($(ls *.sh 2>/dev/null))

# Identify unused Python files
unused_python_files=()
for file in "${all_python_files[@]}"; do
  if [[ ! " ${used_python_files[@]} " =~ " ${file} " ]]; then
    unused_python_files+=("$file")
  fi
done

# Identify unused Bash files
unused_bash_files=()
for file in "${all_bash_files[@]}"; do
  if [[ ! " ${used_bash_files[@]} " =~ " ${file} " ]]; then
    unused_bash_files+=("$file")
  fi
done

# Print and remove unused Python files
echo "Unused Python files: ${unused_python_files[@]}"
for file in "${unused_python_files[@]}"; do
  echo "Removing $file"
  rm "$file"
done

# Print and remove unused Bash files
echo "Unused Bash files: ${unused_bash_files[@]}"
for file in "${unused_bash_files[@]}"; do
  echo "Removing $file"
  rm "$file"
done

# Remove all Python compiled files in directory and subdirectories
echo "Removing all Python compiled files (.pyc and .pyo) in directory and subdirectories"
find . -type f -name "*.pyc" -exec rm -f {} +
find . -type f -name "*.pyo" -exec rm -f {} +

# Remove all __pycache__ directories in directory and subdirectories
echo "Removing all __pycache__ directories in directory and subdirectories"
find . -type d -name "__pycache__" -exec rm -rf {} +

rm -rf cbig/Nguyen2020/output/*
rm -f log_split.txt new_df_split_0.csv new_df_split_1.csv unique_rids_split_0.csv unique_rids_split_1.csv
rm -rf log_kfold k_fold_rent
rm -rf log_train ckpt2
rm -rf log7
rm -rf log7_analyze
rm -rf year_*.pdf
rm -rf tables_analyze_spec_*
rm -f frame_indices.pkl processed_data.csv saved_pd_indices.pkl sampled_data_split_0.csv sampled_data_split_1.csv
rm -rf split_plots
rm -rf tmp
rm -f ckpt.pth
rm -rf run7 log7 run7_analyze log7_analyze run7_analyze_spec log7_analyze_spec log_analyze_specs
rm -f  unique_rids_split_0.csv unique_rids_split_1.csv new_df_split_0.csv new_df_split_1.csv 
rm -f whole_df.csv
rm -rf log_fnr_fpr plots_for_paper
echo "Cleanup complete."
