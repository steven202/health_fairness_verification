#! /bin/bash
trap 'echo "Stopping the script"; exit 1' INT
eval "$(conda shell.bash hook)"
conda activate alpha-beta-crown
today=`date +%Y%m%d_%H%M%S`
for model in logistic mlp3 mlp6; do
    for split in 0 1; do
        # CUDA_VISIBLE_DEVICES=$cuda python train_model.py --dataset-split $split --model $model --discrete  --train --train-epoch 2 >> "log_train/train_log_model_"$model"_data_"$split"_discrete.txt"
        # CUDA_VISIBLE_DEVICES=$cuda python model_verify.py --dataset-split $split --model $model --discrete  --train >> ./log7/'model_'$model'_split_'$split'_log_disc.txt'
        CUDA_VISIBLE_DEVICES=3 python model_verify_analyze_spec.py --dataset-split $split --model $model --discrete  --train >> ./log7_analyze_spec/'model_'$model'_split_'$split'_log_disc.txt'
    done
done
rm -rf *.pdf
python draft_model_combine_share_vertical.py
python pd_to_csv.py
python pd_to_csv_combine.py