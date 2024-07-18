# training:
CUDA_VISIBLE_DEVICES=2 python main.py -bs 64 --model logistic --train > checkpoints/train_logistic_log.txt
CUDA_VISIBLE_DEVICES=2 python main.py -bs 64 --model mlp3 --init --hidden-dim 256 --learning-rate 0.001 --beta1 0.95 --beta2 0.98 --T-max 100 --eta-min 1e-5 --train-epoch 1700 --train --comb > checkpoints/train_mlp3_log.txt
CUDA_VISIBLE_DEVICES=3 python main.py -bs 64 --model mlp6 --init --hidden-dim 256 --learning-rate 0.001 --beta1 0.95 --beta2 0.98 --T-max 100 --eta-min 1e-5 --train-epoch 1700 --train --comb > checkpoints/train_mlp6_log.txt
# summarize the training results
echo "logistic" > checkpoints/summary.txt
cat checkpoints/train_logistic_log.txt | grep "Final" >> checkpoints/summary.txt
cat checkpoints/train_logistic_log.txt | grep "Average" >> checkpoints/summary.txt
echo "" >> checkpoints/summary.txt
echo "mlp3" >> checkpoints/summary.txt
cat checkpoints/train_mlp3_log.txt | grep "Final" >> checkpoints/summary.txt
cat checkpoints/train_mlp3_log.txt | grep "Average" >> checkpoints/summary.txt
echo "" >> checkpoints/summary.txt
echo "mlp6" >> checkpoints/summary.txt
cat checkpoints/train_mlp6_log.txt | grep "Final" >> checkpoints/summary.txt
cat checkpoints/train_mlp6_log.txt | grep "Average" >> checkpoints/summary.txt
echo "Done training"

# testing:
CUDA_VISIBLE_DEVICES=2 python main.py -bs 64 --model logistic > checkpoints/test_logistic_log.txt
CUDA_VISIBLE_DEVICES=2 python main.py -bs 64 --model mlp3 > checkpoints/test_mlp3_log.txt
CUDA_VISIBLE_DEVICES=3 python main.py -bs 64 --model mlp6 > checkpoints/test_mlp6_log.txt