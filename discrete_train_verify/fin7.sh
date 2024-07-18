#! /bin/bash
trap 'echo "Stopping the script"; exit 1' INT
eval "$(conda shell.bash hook)"
conda activate alpha-beta-crown
today=`date +%Y%m%d_%H%M%S`
# model=mlp6

for model in logistic mlp3 mlp6; do
    for split in 0; do
        CUDA_VISIBLE_DEVICES=3 python model_verify_analyze.py --dataset-split $split --model $model --discrete  --train >> ./log7_analyze/'model_'$model'_split_'$split'_log_disc.txt'
    done
done