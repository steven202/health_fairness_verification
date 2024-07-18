import argparse
import os

from write_vnnlib_src.utils import seed_func


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vnnlib-config-dir", default="write_vnnlib_config_dir")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--vnnlib-csv-dir", default="write_vnnlib_csv_dir")
    parser.add_argument("--model-ckpt-dir", default="/home/cw3344@drexel.edu/fairness_verification_continuous_training/checkpoints_03062024")
    parser.add_argument("--vnnlib-execute-dir", default="/")
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

def main(args):
    seed_func(args.seed)
    for model, onnx_dir in zip(["mlp3", "mlp6", "logistic"], ["mlp3/split_{0}.onnx", "mlp6/split_{0}.onnx", "logistic/torch_split_{0}.onnx"]):
        for combination in sorted(os.listdir(args.vnnlib_csv_dir)):
            for fold in range(args.folds):
                execute_file_path = os.path.join(args.vnnlib_config_dir, model, combination, f"fold_{fold}_config.yaml")

if __name__ == "__main__":
    args = get_arg_parser()
    main(args)