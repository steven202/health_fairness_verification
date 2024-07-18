import argparse
import os

from write_vnnlib_src.utils import seed_func


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vnnlib-config-dir", default="write_vnnlib_config_dir")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--vnnlib-csv-dir", default="write_vnnlib_csv_dir")
    parser.add_argument("--model-ckpt-dir", default="/home/cw3344@drexel.edu/fairness_verification_continuous_training/checkpoints_03062024")
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

def main(args):
    seed_func(args.seed)
    for model, onnx_dir in zip(["mlp3", "mlp6", "logistic"], ["mlp3/split_{0}.onnx", "mlp6/split_{0}.onnx", "logistic/torch_split_{0}.onnx"]):
        for combination in sorted(os.listdir(args.vnnlib_csv_dir)):
            for fold in range(args.folds):
                csv_path = os.path.join(args.vnnlib_csv_dir, combination, f"fold_{fold}_instances.csv")
                assert os.path.exists(csv_path), f"{csv_path} does not exist"
                onnx_path = os.path.join(args.model_ckpt_dir, onnx_dir.format(fold))
                assert os.path.exists(onnx_path), f"{onnx_path} does not exist"
                save_file_path = os.path.join(args.vnnlib_config_dir, model, combination, f"fold_{fold}_config.yaml")
                os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                write_a_yaml_config(save_file_path, csv_path, model, onnx_path, "[-1, 28]", 8192, 60, True)
def write_a_yaml_config(save_file_path, csv_path, model, onnx_path, input_shape, batch_size=8192, timeout=60, enable=True):
    with open(save_file_path, "w") as f:
        f.write("general:\n")
        f.write("  root_path: ./\n")
        f.write(f"  csv_name: {csv_path}\n")
        f.write("model:\n")
        f.write(f"  onnx_path: {onnx_path}\n")
        f.write(f"  input_shape: {input_shape}\n")
        f.write("solver:\n")
        f.write(f"  batch_size: {batch_size}\n")
        f.write("bab:\n")
        f.write(f"  timeout: {timeout}\n")
        f.write("  branching:\n")
        f.write("    method: sb\n")
        f.write("    input_split:\n")
        f.write(f"      enable: {enable}\n")

if __name__ == "__main__":
    args = get_arg_parser()
    main(args)