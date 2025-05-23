#! /bin/bash
trap 'echo "Stopping the script"; exit 1' INT
eval "$(conda shell.bash hook)"
conda activate alpha-beta-crown
today=`date +%Y%m%d_%H%M%S`
# model=mlp6

for model in logistic mlp3 mlp6; do
    for split in 1; do
        CUDA_VISIBLE_DEVICES=2 python model_verify.py --dataset-split $split --model $model --discrete  --train >> ./log7/'model_'$model'_split_'$split'_log_disc.txt'
    done
done