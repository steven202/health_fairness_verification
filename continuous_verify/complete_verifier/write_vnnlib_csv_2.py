
import argparse
import os

from tqdm import tqdm


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vnnlib-save-dir", default="write_vnnlib_save_dir")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--vnnlib-csv-dir", default="write_vnnlib_csv_dir")
    args, _ = parser.parse_known_args()
    return args


def main(args):
    # find all directories in the save path
    vnnlib_dirs = os.listdir(args.vnnlib_save_dir)
    # make sure only directories are selected
    vnnlib_dirs = sorted([vnnlib_dir for vnnlib_dir in vnnlib_dirs if os.path.isdir(os.path.join(args.vnnlib_save_dir, vnnlib_dir))])
    instances = dict()
        
    # for each directory in the save path, there are args.folds number of directories
    for vnnlib_dir in tqdm(vnnlib_dirs):
        for fold in range(args.folds):
            vnnlib_dir_fold = os.path.join(args.vnnlib_save_dir, vnnlib_dir, f"fold_{fold}")
            vnnlib_instance_dir = os.path.join(args.vnnlib_csv_dir, vnnlib_dir)
            os.makedirs(vnnlib_instance_dir, exist_ok=True)
            # find all path in the vnnlib directory
            vnnlib_paths = sorted([tmp for tmp in os.listdir(vnnlib_dir_fold) if tmp.endswith(".vnnlib")], key=lambda x: int(x.split("_")[-1].split(".vnnlib")[0]))
            # print the vnnlib paths
            with open(os.path.join(vnnlib_instance_dir, f"fold_{fold}_instances.csv"), "w") as f:
                # print(os.path.join(vnnlib_dir_fold, vnnlib_path))
                for vnnlib_path in vnnlib_paths:
                    f.write(f"{os.path.join(vnnlib_dir_fold, vnnlib_path)}\n")

if __name__ == "__main__":
    args = get_arg_parser()
    main(args)