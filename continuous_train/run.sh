#!/bin/bash
trap 'echo "Stopping the script"; exit 1' INT
eval "$(conda shell.bash hook)"
conda activate alpha-beta-crown

mkdir -p checkpoints_03062024
# data split for cross-validation
python main.py --split-data --save-path checkpoints_03062024 > checkpoints_03062024/fold_split_log.txt
for phase in "train" "test"; do
    if [ "$phase" == "train" ]; then
        train_flag="--train"
    else
        train_flag=""
    fi
    echo $phase
    # "logistic"
    CUDA_VISIBLE_DEVICES=2 python main.py $train_flag -bs 64 --model logistic --save-path checkpoints_03062024 --train-epoch 50 > checkpoints_03062024/${phase}_logistic_log.txt
    # "mlp3": "/home/cw3344@drexel.edu/fairness_verification_continuous_training/log_03062024/model_mlp3_T1700_b1_0.9_b2_0.98_eta_1e-4_lr_0.001_epoch_1700_hidden_64.txt"
    CUDA_VISIBLE_DEVICES=2 python main.py $train_flag -bs 64 --model mlp3 --save-path checkpoints_03062024 --beta1 0.9 --beta2 0.98 --eta-min 1e-4 --learning-rate 0.001 --train-epoch 1700 --hidden-dim 64 > checkpoints_03062024/${phase}_mlp3_log.txt
    # "mlp6": "/home/cw3344@drexel.edu/fairness_verification_continuous_training/log_03062024/model_mlp6_T1700_b1_0.95_b2_0.999_eta_0.0_lr_0.001_epoch_1700_hidden_64.txt" 
    CUDA_VISIBLE_DEVICES=3 python main.py $train_flag -bs 64 --model mlp6 --save-path checkpoints_03062024 --beta1 0.95 --beta2 0.999 --eta-min 0.0 --learning-rate 0.001 --train-epoch 1700 --hidden-dim 64 > checkpoints_03062024/${phase}_mlp6_log.txt

    cat checkpoints_03062024/${phase}_logistic_log.txt | tail -n 1
    cat checkpoints_03062024/${phase}_mlp3_log.txt | tail -n 1
    cat checkpoints_03062024/${phase}_mlp6_log.txt | tail -n 1
done
