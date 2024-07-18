#!/usr/bin/env python
import argparse
from datetime import datetime
import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from itertools import combinations
from write_vnnlib_src.dataset import intialize_generator_mlp, iterate_through_generator
from write_vnnlib_src.utils import seed_func
from write_vnnlib_src.vnnlib_utils import save_vnnlib
pd.options.mode.chained_assignment = None  # default='warn'

# https://www.geeksforgeeks.org/itertools-combinations-module-python-print-possible-combinations/
def rSubset(arr, r):
    return list(combinations(arr, r))
def flatten_combinations(l):
    """flatten all sublist in a list"""
    return [item for sublist in l for item in sublist]
def get_arg_parser():
    """get arguments"""
    parser = argparse.ArgumentParser(description="Continuous verification model training")
    parser.add_argument("--save-dir", "-s", type=str, default=f"write_vnnlib_save_dir", help="save path")
    parser.add_argument(
        "--dataset-split",
        "-ds",
        type=int,
        default=-1,
        choices=[0, 1, -1],
        help="0 for NL=0, MCI=1, 1 for MCI=0, Dementia=1, -1 means no dataset split",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spreadsheet", default="./fairness_data/TADPOLE_D1_D2.csv")
    parser.add_argument("--features", default="./fairness_data/features")
    parser.add_argument("--indices-dir", default="/home/cw3344@drexel.edu/fairness_verification_continuous_training/checkpoints_03062024")
    parser.add_argument("--fairness-data-dir", default="/home/cw3344@drexel.edu/fairness_verification_continuous_training/fairness_data/")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--year-eps", type=int, default=-1)
    args, _ = parser.parse_known_args()
    return args

def main(args):
    seed_func(args.seed)
    args.train = True
    today = datetime.now().strftime("%m%d%Y_%H%M%S")
    num_classes = 3 if args.dataset_split == -1 else 2
    print(f"Dataset Split: {args.dataset_split}, Num of Classes: {num_classes}, Seed: {args.seed}")
    # should be loaded from the dataset
    patients, labels, groups = torch.load(os.path.join(args.fairness_data_dir, "fairness.pt")).values()
    ######################### set input range ####################################
    column_bounds = []
    attr_names = "AGE	PTGENDER	PTEDUCAT	PTETHCAT	PTRACCAT	PTMARRY".split()
    for column in range(0, 6):
        column_bounds.append(
            (
                column,
                min(patients[:, column]).item(),
                max(patients[:, column]).item(),
                attr_names[column],
            )
        )
    with open(os.path.join(args.indices_dir, "data_split_indices.json"), "r") as f:
        data_splits = json.load(f)
    ######################## verify with different input attributes #################
    for i in tqdm(flatten_combinations([rSubset(list(range(0, 6)), tmp) for tmp in range(1, 7)])):
        # if not AGE_PTGENDER_PTEDUCAT_PTETHCAT_PTRACCAT_PTMARRY
        # metrics = {}
        # metrics["clean"] = {"FPR": [], "FNR": [], "ACC": []}
        # metrics["pgd"] = {"FPR": [], "FNR": [], "ACC": []}
        # metrics["verify"] = {"FPR": [], "FNR": [], "ACC": []}
        # print(f"==== START ====\nFor {len(i)} attribute", [column_bounds[i[tmp]][3] for tmp in range(len(i))], ":\n")
        multi_column_bound = [column_bounds[tmp] for tmp in i]
        multi_name = "_".join([bound[3] for bound in multi_column_bound])
        assert "_".join([column_bounds[i[tmp]][3] for tmp in range(len(i))]) == multi_name
        # if multi_name not in ["AGE_PTGENDER_PTEDUCAT_PTETHCAT_PTRACCAT_PTMARRY", "AGE", "PTEDUCAT"]:
            # continue
        seen_test_idx = set()
        for n_split in range(args.folds):
            train_idx = data_splits[str(n_split)]["train"]
            val_idx = data_splits[str(n_split)]["val"]
            test_idx = data_splits[str(n_split)]["test"]
            # check no overlap between train, val, test
            assert len(set(train_idx).intersection(set(val_idx))) == 0
            assert len(set(train_idx).intersection(set(test_idx))) == 0
            assert len(set(val_idx).intersection(set(test_idx))) == 0
            # continue
            ################ prepare test image and labels ###############################
            save_vnnlib_path = os.path.join(args.save_dir, "_".join([column_bounds[i[tmp]][3] for tmp in range(len(i))]), f"fold_{n_split}")
            os.makedirs(save_vnnlib_path, exist_ok=True) 
            # for j in range(patients[test_idx].shape[0]):
            #     save_vnnlib(multi_column_bound, patients[test_idx][j], labels[test_idx][j], os.path.join(save_vnnlib_path, f"test_{j}.vnnlib"), num_classes)
            # use j in test_idx to get the test image and labels
            for j in test_idx:
                save_vnnlib(multi_column_bound, patients[j], labels[j], os.path.join(save_vnnlib_path, f"test_{j}.vnnlib"), num_classes, args.year_eps)
            # make sure test_idx have not seen previously
            assert len(seen_test_idx.intersection(set(test_idx))) == 0
            # insert test_idx to seen_test_idx
            seen_test_idx = seen_test_idx.union(set(test_idx))
if __name__ == "__main__":
    args = get_arg_parser()
    main(args)
