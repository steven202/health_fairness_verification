from collections import OrderedDict
import os
import pickle
import pandas as pd
from cbig.Nguyen2020.misc2 import REVERSE_CONVERTERS
import numpy as np
import torch


def get_pnr_fpr(model, dataset_split, mode="disc"):
    load_path = f"run7_analyze_spec/model_{model}_split_{dataset_split}_log_{mode}"
    with open(load_path + ".pkl", "rb") as f:
        attrs_metrics = pickle.load(f)
    return attrs_metrics


trans = {
    "AGE": "age",
    "PTGENDER": "gender",
    "PTEDUCAT": "education",
    "PTETHCAT": "ethnicity",
    "PTRACCAT": "race",
    "PTMARRY": "marriage",
}
inv_trans = {v: k for k, v in trans.items()}

ATTRS = ["AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY"]

dataset_split = 0
n_split = 0
splits_num = 10

k_fold_data = torch.load(
    f"./k_fold_rent/data_{dataset_split}_fold_{n_split}.pth",
)
train_idx, val_idx, test_idx, patients, labels = (
    k_fold_data["train_idx"],
    k_fold_data["val_idx"],
    k_fold_data["test_idx"],
    k_fold_data["patients"],
    k_fold_data["labels"],
)
new_df = pd.read_csv(f"new_df_split_{dataset_split}.csv", index_col=0)
assert not (test_idx > new_df.shape[0]).any()
dataframes = [
    new_df.loc[train_idx],
    new_df.loc[val_idx],
    new_df.loc[test_idx],
]
names = ["train", "validation", "test"]
counter = 0
column_needed = OrderedDict()
for df, name in zip(dataframes, names):
    df_rids = df.sort_values(by=["RID", "VISCODE"], ascending=False).drop_duplicates(subset=["RID"])
    unchanged = df_rids.loc[(np.abs(df_rids["CDX"]) <= 1e-14)]
    changed = df_rids.loc[(np.abs(df_rids["CDX"] - 1.0) <= 1e-14)]

    for rids_, name_ in zip([df_rids, unchanged, changed], ["all rids", "unchanged rids", "changed rids"]):
        for attr in ATTRS:
            attr_ = trans[attr]
            counts = dict(rids_[attr].apply(REVERSE_CONVERTERS[attr]).value_counts())
            if attr_ not in column_needed:
                column_needed[attr_] = []
            for k in counts.keys():
                if k not in column_needed[attr_]:
                    column_needed[attr_].append(k)
                    counter += 1
            column_needed[attr_] = sorted(column_needed[attr_])

# columns_ = [item for sublist in column_needed.values() for item in sublist]
columns_ = []
models = ["logistic", "mlp3", "mlp6"]
for attr_, categories in column_needed.items():
    if attr_ in ["gender", "ethnicity", "race", "marriage"]:
        for model in models:
            columns_.append(model + "_" + attr_ + "_c_" + "fpr")
            columns_.append(model + "_" + attr_ + "_c_" + "fnr")
            columns_.append(model + "_" + attr_ + "_c_" + "acc")
            columns_.append(model + "_" + attr_ + "_p_" + "fpr")
            columns_.append(model + "_" + attr_ + "_p_" + "fnr")
            columns_.append(model + "_" + attr_ + "_p_" + "acc")
        columns_ += categories
modes = ["disc"]

evals = [
    "fnr",
    "acc",
    "fpr",
]
# evals_ = []
# for model in models:
#     for eval in evals:
#         evals_.append(model + "_" + eval)
columns_ = (
    ["data_split", "n_fold", "mode", "tot_visits", "tot_rids", "tot_unchanged", "tot_changed", "status", "rids"]
    # + evals_
    + columns_
)
table = pd.DataFrame(columns=columns_)
counter = 0
dataset_split = 0
n_split = 0
splits_num = 10
for dataset_split in [0, 1]:
    for n_split in range(10):
        k_fold_data = torch.load(
            f"./k_fold_rent/data_{dataset_split}_fold_{n_split}.pth",
        )
        train_idx, val_idx, test_idx, patients, labels = (
            k_fold_data["train_idx"],
            k_fold_data["val_idx"],
            k_fold_data["test_idx"],
            k_fold_data["patients"],
            k_fold_data["labels"],
        )
        new_df = pd.read_csv(f"new_df_split_{dataset_split}.csv", index_col=0)
        assert not (test_idx > new_df.shape[0]).any()
        dataframes = [
            new_df.loc[train_idx],
            new_df.loc[val_idx],
            new_df.loc[test_idx],
        ]
        names = ["train", "validation", "test"]
        for df, name in zip(dataframes, names):
            df_rids = df.sort_values(by=["RID", "VISCODE"], ascending=False).drop_duplicates(subset=["RID"])
            unchanged = df_rids.loc[(np.abs(df_rids["CDX"]) <= 1e-14)]
            changed = df_rids.loc[(np.abs(df_rids["CDX"] - 1.0) <= 1e-14)]

            for rids_, name_ in zip([df_rids, unchanged, changed], ["all_rids", "unchange_rids", "change_rids"]):
                table.loc[counter, "data_split"] = dataset_split
                table.loc[counter, "n_fold"] = n_split
                table.loc[counter, "mode"] = name
                table.loc[counter, "tot_visits"] = df.shape[0]
                table.loc[counter, "tot_rids"] = df_rids.shape[0]
                table.loc[counter, "tot_unchanged"] = unchanged.shape[0]
                table.loc[counter, "tot_changed"] = changed.shape[0]
                table.loc[counter, "status"] = name_
                table.loc[counter, "rids"] = rids_.shape[0]
                for attr in ATTRS:
                    for model in models:
                        with open(os.path.join("./log_analyze_specs",f"data_{dataset_split}_{model}.pkl"),"rb") as f:
                            all_attr_analyze = pickle.load(f) 
                        metrics = get_pnr_fpr(model, dataset_split, mode="disc")
                        for k, v in metrics.items():
                            if (
                                len(k) in [1]
                                and k[0][3] == attr
                                and trans[attr] in ["gender", "ethnicity", "race", "marriage"]
                            ):
                                table.loc[counter, model + "_" + trans[attr] + "_c_" + "fpr"] = v["clean"]["FPR"][
                                    n_split
                                ]
                                table.loc[counter, model + "_" + trans[attr] + "_c_" + "fnr"] = v["clean"]["FNR"][
                                    n_split
                                ]
                                table.loc[counter, model + "_" + trans[attr] + "_c_" + "acc"] = v["clean"]["ACC"][
                                    n_split
                                ]
                                table.loc[counter, model + "_" + trans[attr] + "_p_" + "fpr"] = v["verify"]["FPR"][
                                    n_split
                                ]
                                table.loc[counter, model + "_" + trans[attr] + "_p_" + "fnr"] = v["verify"]["FNR"][
                                    n_split
                                ]
                                table.loc[counter, model + "_" + trans[attr] + "_p_" + "acc"] = v["verify"]["ACC"][
                                    n_split
                                ]
                                if len(all_attr_analyze[k[0][3]][name_][n_split])>0:
                                    current_column = table.columns.get_loc(model + "_" + trans[k[0][3]] + "_p_" + "acc")+1
                                    for k_,v_ in all_attr_analyze[k[0][3]][name_][n_split].items():
                                        k_=k_.replace("_","_to_")
                                        if k_ not in table.columns.to_list():
                                            table.insert(current_column, k_,np.NaN)
                                            current_column+=1
                                        table.loc[counter,k_]=v_
                            
                    counts = dict(rids_[attr].apply(REVERSE_CONVERTERS[attr]).value_counts())
                    for k, v in counts.items():
                        table.loc[counter, k] = v
                counter += 1
table.replace(0.0, np.nan, inplace=True)
prefix = "_"+os.getcwd().split("/")[-1].replace("health_fairness_","")
if not os.path.exists("tables_analyze_spec"+prefix):
    os.makedirs("tables_analyze_spec"+prefix)
table.to_csv("tables_analyze_spec"+prefix+"/all_table.csv")
for name in names:
    table.loc[table["mode"] == name].to_csv(f"tables_analyze_spec"+prefix+f"/{name}_table.csv")
for status in ["all_rids", "unchange_rids", "change_rids"]:
    table.loc[table["status"] == status].to_csv(f"tables_analyze_spec"+prefix+f"/{status}_table.csv")
# print("done")
