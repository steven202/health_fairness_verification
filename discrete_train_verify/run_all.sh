#!/bin/bash

trap 'echo "Stopping the script"; exit 1' INT
eval "$(conda shell.bash hook)"
conda activate alpha-beta-crown

for repeat in no yes
do
    for selected_year in 3.0 5.0
    do  
        # first, split the data into two cases: 0 for NC and MCI, 1 for MCI and AD, using dataset_numpy_0104_fixed.py, log saved to log_split.txt, data saved to frame_indices.pkl, processed_data.csv, saved_pd_indices.pkl, sampled_data_split_0.csv, sampled_data_split_1.csv, split_plots saved to split_plots directory.
        rm -f frame_indices.pkl processed_data.csv saved_pd_indices.pkl sampled_data_split_0.csv sampled_data_split_1.csv
        rm -rf split_plots
        rm -f log_split.txt new_df_split_0.csv new_df_split_1.csv unique_rids_split_0.csv unique_rids_split_1.csv
        python dataset_numpy_0104_fixed.py --data-dir "" --selected-year "$selected_year" --plots split_plots > log_split.txt
        
        # make sure the file frame_indices.pkl, processed_data.csv, saved_pd_indices.pkl are created:
        if [ ! -f frame_indices.pkl ] || [ ! -f processed_data.csv ] || [ ! -f saved_pd_indices.pkl ]; then
            echo "Error: file frame_indices.pkl, processed_data.csv, saved_pd_indices.pkl not found"
            exit 1
        fi
        
        # saved_pd_indices.pkl will contain 4 indices: 0 split with unchanged, 0 split with changed, 1 split with unchanged, 1 split with changed
        # kfold the data
        rm -rf log_kfold k_fold_rent
        mkdir log_kfold k_fold_rent
        python k_fold_data_rent.py --dataset-split 0 
        python k_fold_data_rent.py --dataset-split 1
        
        # training the model, with cross-validation, log saved to log_train, model saved to ckpt2, using train_model.py
        rm -rf log_train ckpt2
        mkdir log_train ckpt2 
        # train the split 0
        CUDA_VISIBLE_DEVICES=2 python train_model.py --dataset-split 0 --model logistic --discrete  --train --train-epoch 200 >> log_train/train_log_model_logistic_data_0_discrete.txt
        CUDA_VISIBLE_DEVICES=2 python train_model.py --dataset-split 0 --model mlp3 --discrete  --train --train-epoch 200 >> log_train/train_log_model_mlp3_data_0_discrete.txt
        CUDA_VISIBLE_DEVICES=2 python train_model.py --dataset-split 0 --model mlp6 --discrete  --train  --train-epoch 200 >> log_train/train_log_model_mlp6_data_0_discrete.txt

        # train the split 1
        CUDA_VISIBLE_DEVICES=3 python train_model.py --dataset-split 1 --model logistic --discrete  --train --train-epoch 200 >> log_train/train_log_model_logistic_data_1_discrete.txt
        CUDA_VISIBLE_DEVICES=3 python train_model.py --dataset-split 1 --model mlp3 --discrete  --train --train-epoch 200 >> log_train/train_log_model_mlp3_data_1_discrete.txt
        CUDA_VISIBLE_DEVICES=3 python train_model.py --dataset-split 1 --model mlp6 --discrete  --train  --train-epoch 200 >> log_train/train_log_model_mlp6_data_1_discrete.txt
        
        # verify the model, using model_verify.py, log saved to log7, data saved to run7 directory
        rm -rf log7
        mkdir log7
        rm -rf run7
        mkdir run7
        bash fin5.sh
        bash fin6.sh
        
        # analyze the model, log saved to log7_analyze, using model_verify_analyze.py, data saved to run7_analyze directory
        # this analysis is only for the samples that are correctly predicted by the model, and the result is not included in the paper.
        rm -rf log7_analyze
        mkdir log7_analyze
        rm -rf run7_analyze
        mkdir run7_analyze
        bash fin7.sh
        bash fin8.sh
        
        # draw plots, using draft_model_combine_share_vertical.py, pd_to_csv.py, pd_to_csv_combine.py, log files are pdf files, log saved to log7_analyze_spec directory, data saved to run7_analyze_spec directory, table saved to log_analyze_specs
        # this analysis is only for the samples that are correctly predicted by the model, and we create tables to better understand the results. The result is not included in the paper.
        rm -rf log7_analyze_spec run7_analyze_spec log_analyze_specs
        mkdir log7_analyze_spec run7_analyze_spec log_analyze_specs
        rm -f year_*.pdf
        rm -rf tables_analyze_spec_*
        bash fin9.sh
        # extract all the pnr and fpr from run7, and save the results to log_fnr_fpr directory.
        rm -rf log_fnr_fpr
        mkdir log_fnr_fpr
        python extract_all_pnr_fpr.py
        # move all the log files into the logs/$selected_year directory
        LOG_DIR="logs/year_${selected_year}_repeat_${repeat}"
        mkdir -p "$LOG_DIR"
        mv log_kfold log_train log7 log7_analyze log_split.txt log7_analyze_spec "$LOG_DIR"

        # move all the generated files into the generated_files/$selected_year directory
        GEN_FILES_DIR="generated_files/year_${selected_year}_repeat_${repeat}"
        mkdir -p "$GEN_FILES_DIR"
        mv frame_indices.pkl processed_data.csv saved_pd_indices.pkl k_fold_rent ckpt2 split_plots sampled_data_split_0.csv sampled_data_split_1.csv run7 run7_analyze run7_analyze_spec log_analyze_specs unique_rids_split_0.csv unique_rids_split_1.csv new_df_split_0.csv new_df_split_1.csv log_fnr_fpr "$GEN_FILES_DIR"
        
        mv year_*.pdf "$GEN_FILES_DIR"
        mv tables_analyze_spec_* "$GEN_FILES_DIR"

        rm -rf tmp
        rm -f ckpt.pth
    done
done
