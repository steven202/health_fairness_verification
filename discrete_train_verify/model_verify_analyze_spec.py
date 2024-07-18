#!/usr/bin/env python
# Written by Minh Nguyen and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
import argparse
from datetime import datetime
import os
import pickle
import numpy as np
import pandas as pd
import torch
import itertools

from cbig.Nguyen2020.evaluation import MAUC
import sklearn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from dataset2 import intialize_generator_mlp
from discrete_verify_analyze_spec import discrete_method
from models import LogisticRegressionModel, mlp_3_layer, mlp_4_layer, mlp_6_layer
from test import test_model_mlp

from utils import calculate_metrics, flatten_combinations
from sklearn.linear_model import LogisticRegression
from utils import rSubset
from utils import mcnemar

pd.options.mode.chained_assignment = None  # default='warn'


def find_attr(attributes, keywords):
    for key in keywords:
        for attr in attributes:
            if key in attr[3]:
                # print("found", attr[3])
                return True

    return False


def get_arg_parser():
    """get arguments"""
    parser = argparse.ArgumentParser(description="fanirness verification with mlp models")
    parser.add_argument("--batch-size", "-bs", type=int, default=256, help="batch size")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="logistic",
        choices=["logistic", "mlp3", "mlp4", "mlp6", "rnn1", "rnn2"],
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
        default=0,
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


def main(args):
    args.discrete = True
    np.random.seed(args.seed)
    records = {}
    today = datetime.now().strftime("%m%d%Y_%H%M%S")
    mode = "disc" if args.discrete else "cont"
    save_path = (
        f"run7_analyze_spec/model_{args.model}_split_{args.dataset_split}_log_{mode}"
    )
    if not os.path.exists("run7_analyze_spec"):
        os.makedirs("run7_analyze_spec")
    print(f"Model: {args.model}")
    print(f"Verification Mode: {'Discrete' if args.discrete else 'Continuous'}")
    # NL=0, MCI=1, Dementia=2
    print(f"classes: {'NL=0, MCI=1' if args.dataset_split==0 else 'MCI=0, Dementia=1'}")
    num_classes = 3 if args.dataset_split == -1 else 2

    patients, labels, groups, generator, splits_num, new_df = intialize_generator_mlp(args)
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
    # print(column_bounds)
    ######################## verify with different input attributes #################
    all_attr_analyze = {}
    df_attrs_lst = []
    for i in tqdm(flatten_combinations([rSubset(list(range(0, 6)), tmp) for tmp in range(1, 7)])):
        multi_column_bound = [column_bounds[tmp] for tmp in i]
        if not len(multi_column_bound)==1:
            continue
        if find_attr(multi_column_bound, ["PTEDUCAT", "AGE"]):
            continue
        attr_name = multi_column_bound[0][3]
        all_attr_analyze[attr_name] = {}
        all_attr_analyze[attr_name]["all_rids"] = []
        all_attr_analyze[attr_name]["unchange_rids"] = []
        all_attr_analyze[attr_name]["change_rids"] = []
        all_rids = []
        unchange_rids =[]
        change_rids = []
        metrics = {}
        metrics["clean"] = {"FPR": [], "FNR": [], "ACC": []}
        metrics["pgd"] = {"FPR": [], "FNR": [], "ACC": []}
        metrics["verify"] = {"FPR": [], "FNR": [], "ACC": [], "MN": []}

        print(
            f"==== START ====\nFor {len(i)} attribute",
            [column_bounds[i[tmp]][3] for tmp in range(len(i))],
            ":\n",
        )

        # reintilize generator
        # patients, labels, groups, generator, splits_num = intialize_generator_mlp(args)

        ################ prepare test image and labels ###############################

        for n_split in range(splits_num):
            ckpt = torch.load(
                f"./ckpt2/model_{args.model}_data_{args.dataset_split}_fold_{n_split}.pth",
            )

            # image = ckpt["test_x"]
            # true_label = ckpt["test_y"]
            k_fold_data = torch.load(
                f"./k_fold_rent/data_{args.dataset_split}_fold_{n_split}.pth",
            )
            train_idx, val_idx, test_idx, patients, labels = (
                k_fold_data["train_idx"],
                k_fold_data["val_idx"],
                k_fold_data["test_idx"],
                k_fold_data["patients"],
                k_fold_data["labels"],
            )
            mean, std = torch.mean(patients[train_idx], dim=0), torch.std(patients[train_idx], dim=0)
            patients = (patients - mean) / (std + 1e-6)
            image = patients[test_idx]
            true_label = labels[test_idx]

            if torch.cuda.is_available():
                image = image.cuda()
                true_label = true_label.cuda()
            # print("Running on", image.device)
            ############ prepare model ###############
            if args.model == "mlp3":
                model = mlp_3_layer(num_classes=num_classes)
                criterion = torch.nn.CrossEntropyLoss()
            elif args.model == "mlp4":
                model = mlp_4_layer(num_classes=num_classes)
                criterion = torch.nn.CrossEntropyLoss()
            elif args.model == "mlp6":
                model = mlp_6_layer(num_classes=num_classes)
                criterion = torch.nn.CrossEntropyLoss()
            elif args.model == "logistic":
                model = LogisticRegressionModel(1)
                criterion = torch.nn.BCELoss()
            ############## model test #########################
            if torch.cuda.is_available():
                model = model.cuda()

            # model.load_state_dict(torch.load(args.save_path))
            model.load_state_dict(ckpt["model"])
            model.eval()
            if False:  # args.model == "logistic":
                clf = LogisticRegression(random_state=0, max_iter=1000).fit(
                    patients[train_idx].cpu().detach().numpy(), labels[train_idx].cpu().detach().numpy()
                )
                predicted = clf.predict(patients[test_idx].cpu().detach().numpy())
            else:
                outputs, clean_predicted, acc = test_model_mlp(model, image, true_label)
            # acc = ckpt["acc"]
            # outputs = ckpt["outputs"]
            # predicted = ckpt["predicted"]
            # print(f"Accuracy of the network on test images: {100 * acc:.4f} %")
            append_metrics("clean", metrics, true_label, clean_predicted)
            # assert len(metrics["clean"]["ACC"]) != 0
            # FPR, FNR, ACC = calculate_metrics(predicted, true_label)
            # assert not np.isnan(ACC)

            # data = [(lbl, logit) for lbl, logit in zip(true_label, outputs)]
            # print("MAUC:", MAUC(data, num_classes))

            ################## select continuous or discrete verification algorithm, only discrete is implemented ############################
            if args.discrete:
                predicted = torch.ones(true_label.shape).to(true_label.device)
            else:
                raise NotImplementedError
            append_metrics("pgd", metrics, true_label, predicted)
            # print("pgd acc:", 100.0 * pgd_acc, "%\n")
            mask = true_label == clean_predicted
            if not args.discrete:
                raise NotImplementedError
            else:
                outputs, predicted, discrete_acc,all_rid,unchange_rid,change_rid = discrete_method(
                    image[mask], true_label[mask], model, num_classes, multi_column_bound, None, mean, std
                )
                all_rids.append(all_rid)
                unchange_rids.append(unchange_rid)
                change_rids.append(change_rid)
                # print("discrete acc:", 100.0 * discrete_acc, "%\n")
            append_metrics("verify", metrics, true_label[mask], predicted)

            metrics["verify"]["MN"].append(mcnemar(true_label[mask], clean_predicted[mask], predicted))
            # FPR, FNR, ACC = calculate_metrics(predicted, true_label)
            # assert not np.isnan(ACC)
        for key, value in metrics.items():
            FPR_avg = np.mean(value["FPR"]) if len(value["FPR"]) != 0 else 0.0
            FNR_avg = np.mean(value["FNR"]) if len(value["FNR"]) != 0 else 0.0
            ACC_avg = np.mean(value["ACC"]) if len(value["ACC"]) != 0 else 0.0
            FPR_std = np.std(value["FPR"]) if len(value["FPR"]) != 0 else 0.0
            FNR_std = np.std(value["FNR"]) if len(value["FNR"]) != 0 else 0.0
            ACC_std = np.std(value["ACC"]) if len(value["ACC"]) != 0 else 0.0
            assert len(value["ACC"]) != 0
            print(f"{key} avg: FPR: { FPR_avg * 100:.3f} ,FNR: {FNR_avg * 100:.3f} ,Accuracy: {ACC_avg * 100:.3f}")
            print(f"{key} std: FPR: { FPR_std * 100:.3f} ,FNR: {FNR_std * 100:.3f} ,Accuracy: {ACC_std * 100:.3f}")
        records[tuple(multi_column_bound)] = metrics
        # attr_columns = list(itertools.product(*[["White","Black","Asian"],["White","Black","Asian"]]))
        # attr_columns = [tmp[0]+"_"+tmp[1] for tmp in attr_columns if tmp[0]!=tmp[1]]
        attr_columns = sorted(list(set(itertools.chain.from_iterable([list(tmp.keys()) for tmp in all_rids]))))
        if len(attr_columns)==0:
            continue
        df_attr = pd.DataFrame(index=np.arange(30), columns=attr_columns)
        for i, result in enumerate(all_rids):
            for k,v in result.items():
                df_attr.iloc[i*3][attr_columns[0]]=0
                df_attr.iloc[i*3+1][attr_columns[0]]=0
                df_attr.iloc[i*3+2][attr_columns[0]]=0
                if k in attr_columns:
                    df_attr.iloc[i*3][k]=all_rids[i][k]
                    df_attr.iloc[i*3+1][k]=unchange_rids[i][k]
                    df_attr.iloc[i*3+2][k]=change_rids[i][k]
        df_attrs_lst.append(df_attr)
        all_attr_analyze[attr_name]["all_rids"] = all_rids
        all_attr_analyze[attr_name]["unchange_rids"] = unchange_rids
        all_attr_analyze[attr_name]["change_rids"] = change_rids
    all_table  = pd.concat(df_attrs_lst,axis = 1)
    
    if not os.path.exists("log_analyze_specs"):
        os.makedirs("log_analyze_specs")
    all_table.to_csv(os.path.join("./log_analyze_specs",f"data_{args.dataset_split}_{args.model}.csv"),index=False)  
    with open(os.path.join("./log_analyze_specs",f"data_{args.dataset_split}_{args.model}.pkl"),"wb") as f:
        pickle.dump(all_attr_analyze, f)  
    with open(save_path + ".pkl", "wb") as f:
        pickle.dump(records, f)


    # df_attr = df_attr.fillna(0)
  

def flatten(l):
    return [item for sublist in l for item in sublist]

def append_metrics(key, metrics, true_label, predicted):
    FPR, FNR, ACC = calculate_metrics(predicted, true_label)
    if not np.isnan(FPR):
        metrics[key]["FPR"].append(FPR)
    if not np.isnan(FNR):
        metrics[key]["FNR"].append(FNR)
    if not np.isnan(ACC):
        metrics[key]["ACC"].append(ACC)


if __name__ == "__main__":
    args = get_arg_parser()
    main(args)
