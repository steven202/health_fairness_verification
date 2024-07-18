#!/usr/bin/env python
import argparse
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from cbig.Nguyen2020.misc2 import REVERSE_CONVERTERS
from dataset2 import intialize_generator_mlp, iterate_through_generator
from utils import set_seed


pd.options.mode.chained_assignment = None  # default='warn'


def get_arg_parser():
    """get arguments"""
    parser = argparse.ArgumentParser(description="fanirness verification with mlp models")
    parser.add_argument("--batch-size", "-bs", type=int, default=256, help="batch size")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="logistic",
        choices=["mlp4", "mlp6", "logistic", "rnn1", "rnn2"],
        help="model type",
    )
    parser.add_argument("--save-path", "-s", type=str, default="ckpt.pth", help="save path")
    parser.add_argument(
        "--discrete",
        "-d",
        action="store_true",
        help="use discrete verification instead of continuous",
    )
    parser.add_argument(
        "--dataset-split",
        "-ds",
        type=int,
        default=1,
        choices=[0, 1, -1],
        help="0 for NL=0, MCI=1, 1 for MCI=0, Dementia=1, -1 means no dataset split",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spreadsheet", default="./fairness_data/TADPOLE_D1_D2.csv")
    parser.add_argument("--features", default="./fairness_data/features")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--eps", default=0.3)
    parser.add_argument("--train-epoch", default=1000)
    parser.add_argument("--train", action="store_true")
    args, _ = parser.parse_known_args()
    args.model = f"{args.model}_ckpt.pth" if args.model == "ckpt.pth" else args.model

    return args


trans = {
    "AGE": "age",
    "PTGENDER": "gender",
    "PTEDUCAT": "education",
    "PTETHCAT": "ethnicity",
    "PTRACCAT": "race",
    "PTMARRY": "marriage",
}
ATTRS = ["AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY"]


def main(args):
    set_seed(args.seed)
    args.discrete = True
    records = {}
    today = datetime.now().strftime("%m%d%Y_%H%M%S")
    mode = "disc" if args.discrete else "cont"
    save_path = f"run5/{today}_model_{args.model}_split_{args.dataset_split}_log_{mode}"
    print(f"Model: {args.model}")
    print(f"Verification Mode: {'Discrete' if args.discrete else 'Continuous'}")
    print(f"classes: {'NL=0, MCI=1' if args.dataset_split==0 else 'MCI=0, Dementia=1'}")
    num_classes = 3 if args.dataset_split == -1 else 2

    # patients, labels, groups, generator, splits_num = intialize_generator_mlp(args)
    ######################### set input range ####################################

    # reintilize generator
    patients, labels, groups, generator, splits_num, new_df = intialize_generator_mlp(args)

    for n_split in tqdm(range(splits_num)):
        train_idx, val_idx, test_idx = iterate_through_generator(patients, labels, groups, generator)
        assert not (test_idx > new_df.shape[0]).any()
        ### borrow at least one from other folds
        for cdx in [0, 1]:
            for attri, value in zip(
                [
                    "PTGENDER",
                    "PTGENDER",
                    "PTETHCAT",
                    "PTETHCAT",
                    "PTRACCAT",
                    "PTRACCAT",
                    "PTRACCAT",
                    "PTMARRY",
                    "PTMARRY",
                    "PTMARRY",
                    "PTMARRY",
                ],
                [0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 3],
            ):
                train_ = new_df.loc[train_idx].loc[
                    (new_df.loc[train_idx]["CDX"] == cdx) & (new_df.loc[train_idx][attri] == value)
                ]
                val_ = new_df.loc[val_idx].loc[
                    (new_df.loc[val_idx]["CDX"] == cdx) & (new_df.loc[val_idx][attri] == value)
                ]
                test_ = new_df.loc[test_idx].loc[
                    (new_df.loc[test_idx]["CDX"] == cdx) & (new_df.loc[test_idx][attri] == value)
                ]
                if train_.empty:
                    if not val_.empty:
                        idx = val_.index[0]
                        indices = np.where(val_ == idx)
                        val_idx = np.delete(val_idx, indices)
                        train_idx = np.append(train_idx, idx)
                    elif not test_.empty:
                        idx = test_.index[0]
                        indices = np.where(test_ == idx)
                        test_idx = np.delete(test_idx, indices)
                        train_idx = np.append(train_idx, idx)
                    else:
                        raise Exception(f"not found attribute {attri} with value {value} in CDX {cdx}")

        dataframes = [
            new_df.loc[train_idx],
            new_df.loc[val_idx],
            new_df.loc[test_idx],
        ]

        names = ["train", "validation", "test"]
        with open(f"log_kfold/{splits_num}_fold_data_{args.dataset_split}_num_{n_split}.txt", "w") as f:
            for df, name in zip(dataframes, names):
                f.write(f"beginning: {name} dataset: ++++++\n")
                f.write(f"total visits: {df.shape[0]}\n")
                df_rids = df.sort_values(by=["RID", "VISCODE"], ascending=False).drop_duplicates(subset=["RID"])
                f.write(f"total rids: {df_rids.shape[0]}\n")
                unchanged = df_rids.loc[(np.abs(df_rids["CDX"]) <= 1e-14)]
                changed = df_rids.loc[(np.abs(df_rids["CDX"] - 1.0) <= 1e-14)]
                f.write(f"unchanged rids: {unchanged.shape[0]}\n")
                f.write(f"changed rids: {changed.shape[0]}\n")
                f.write("\n")
                for rids_, name_ in zip([df_rids, unchanged, changed], ["all rids", "unchanged rids", "changed rids"]):
                    f.write(f"starting: for {name_} <<<:\n")
                    f.write("\n")
                    for attr in ATTRS:
                        f.write(f"for attribute {trans[attr]}:\n")
                        f.write(f"{rids_[attr].apply(REVERSE_CONVERTERS[attr]).value_counts()}\n")
                        f.write("\n")
                    f.write(f"ending: {name_} >>>\n")
                    f.write("\n")
                f.write(f"finishing {name} dataset -----\n")
                f.write("\n")
            f.write("end of file\n")

        ckpt = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "patients": patients,
            "labels": labels,
        }

        torch.save(
            ckpt,
            f"./k_fold_rent/data_{args.dataset_split}_fold_{n_split}.pth",
        )


if __name__ == "__main__":
    args = get_arg_parser()
    main(args)
