import numpy as np
from sklearn.model_selection import GroupKFold
import torch
import cbig.Nguyen2020.misc as misc
from torch.utils.data import DataLoader, TensorDataset
import functools
import pandas as pd
import pickle


def intialize_generator_mlp(args):
    # preprocessing data
    # spreadsheet = "./fairness_data/TADPOLE_D1_D2.csv"
    # features = "./fairness_data/features"

    # columns = ["RID", "DXCHANGE", "EXAMDATE"]
    # columns = "RID	PTID	VISCODE	SITE	D1	D2	COLPROT	ORIGPROT	EXAMDATE	DX_bl	DXCHANGE DX".split()
    # columns = "RID VISCODE AGE	PTGENDER	PTEDUCAT	PTETHCAT	PTRACCAT	PTMARRY Month_bl DX".split()
    # features = misc.load_feature(features)
    # features = "CDRSB ADAS11 ADAS13 MMSE RAVLT_immediate RAVLT_learning RAVLT_forgetting RAVLT_perc_forgetting MOCA FAQ Entorhinal Fusiform Hippocampus ICV MidTemp Ventricles WholeBrain AV45 FDG ABETA_UPENNBIOMK9_04_19_17 TAU_UPENNBIOMK9_04_19_17 PTAU_UPENNBIOMK9_04_19_17".split()

    # frame = misc.load_table(spreadsheet, columns + features)
    # frame["Vent"] = frame.Ventricles / frame.ICV
    # features.append("Vent")
    # frame = frame.replace({"VISCODE": "bl"}, 0.0)
    # frame["VISCODE"] = np.where(
    #     frame["VISCODE"].astype(str).str.startswith("m0"),
    #     (frame["VISCODE"].astype(str).str[2:]),
    #     frame["VISCODE"],
    # )
    # frame["VISCODE"] = np.where(
    #     frame["VISCODE"].astype(str).str.startswith("m"),
    #     (frame["VISCODE"].astype(str).str[1:]),
    #     frame["VISCODE"],
    # )
    # frame["VISCODE"] = frame["VISCODE"].astype(float)
    # frame["VISCODE"] = frame["VISCODE"].div(12.0)
    # frame["AGE"] = frame["AGE"] + frame["VISCODE"]
    # frame = frame.sort_values(by=["RID", "VISCODE"])
    # with open("frame_indices.pkl", "rb") as fp:  # Unpickling
    #     frame_indices = pickle.load(fp)
    # assert frame_indices == frame.index.tolist()
    ######################################
    frame = pd.read_csv("processed_data.csv", index_col=0)
    features = "./fairness_data/features"
    features = misc.load_feature(features)
    features = "CDRSB ADAS11 ADAS13 MMSE RAVLT_immediate RAVLT_learning RAVLT_forgetting RAVLT_perc_forgetting MOCA FAQ Entorhinal Fusiform Hippocampus ICV MidTemp Ventricles WholeBrain AV45 FDG ABETA_UPENNBIOMK9_04_19_17 TAU_UPENNBIOMK9_04_19_17 PTAU_UPENNBIOMK9_04_19_17".split()
    features.append("Vent")
    ######################################
    # load indices
    with open("saved_pd_indices.pkl", "rb") as f:
        selected_dict = pickle.load(f)
    frame_indices_unchanged = selected_dict[args.dataset_split]["unchanged"]
    frame_indices_changed = selected_dict[args.dataset_split]["changed"]
    frame_changed = frame.loc[list(frame_indices_changed.keys())]
    frame_unchanged = frame.loc[list(frame_indices_unchanged.keys())]

    frame_changed["CDX"] = 1.0
    frame_unchanged["CDX"] = 0.0
    features.append("CDX")
    df = pd.concat([frame_changed, frame_unchanged], axis=0, ignore_index=True)
    df = df.sample(frac=1,random_state=0).reset_index(drop=True)
    #####################################################
    # frame_changed_keys = list(frame_indices_changed.keys())
    # frame_unchanged_keys = list(frame_indices_unchanged.keys())
    # frame["CDX"] = np.nan
    # frame.loc[frame_changed_keys, "CDX"] = 1.0
    # frame.loc[frame_unchanged_keys, "CDX"] = 0.0
    # features.append("CDX")
    # df = frame
    ######################################################
    # df[features] = (df[features] - mean) / std

    # only keep the first occurence of every RID
    # df = df.sort_values(by=["RID", "VISCODE"]).drop_duplicates(subset=["RID"])
    df.to_csv(f"sampled_data_split_{args.dataset_split}.csv")
    # if args.dataset_split == 0:
    #     df = df.loc[(np.abs(df["DX"]) <= 1e-14)]  #  | (np.abs(df["DX"] - 1.0) <= 1e-14)]
    #     pass
    # elif args.dataset_split == 1:
    #     df = df.loc[(np.abs(df["DX"] - 1.0) <= 1e-14)]  # | (np.abs(df["DX"] - 2.0) <= 1e-14)]
    #     df["DX"] = df["DX"] - 1.0
    #     pass

    # iloc or loc, no difference
    new_df = df.reset_index(drop=True)
    new_df.to_csv(f"new_df_split_{args.dataset_split}.csv")
    new_df.sort_values(by=["RID", "VISCODE"], ascending=False).drop_duplicates(subset=["RID"]).reset_index(
        drop=True
    ).to_csv(f"unique_rids_split_{args.dataset_split}.csv")
    feature_names = new_df.columns.values.tolist()
    feature_names.remove("DX")
    feature_names.remove("CDX")
    feature_names.remove("Month_bl")
    # feature_names.remove("DX_bl")
    assert "Unnamed: 0" not in feature_names
    # feature_names.remove("Unnamed: 0")
    feature_names.remove("RID")
    feature_names.remove("VISCODE")
    # feature_names.remove("MMSE") dx
    # feature_names.remove("CDRSB") dx
    # mean = new_df.loc[:, feature_names].mean()
    # std = new_df.loc[:, feature_names].std()
    # new_df[feature_names] = (new_df[feature_names] - mean) / std
    patients_ = new_df[feature_names].to_numpy()
    labels_ = new_df[["CDX"]].to_numpy()
    groups_ = new_df[["RID"]].to_numpy()
    patients_[np.isnan(patients_)] = 0.0
    valid = (~np.isnan(labels_)).squeeze()

    patients = torch.tensor(patients_[valid], dtype=torch.float32)
    print(patients.shape, functools.reduce(lambda x, y: x * y, patients.shape))
    labels = torch.tensor(labels_[valid], dtype=torch.long).squeeze()
    groups = torch.tensor(groups_[valid], dtype=torch.long).squeeze()
    group_kfold = GroupKFold(n_splits=10)
    splits_num = group_kfold.get_n_splits(patients, labels, groups)
    generator = group_kfold.split(X=patients, y=labels, groups=groups)

    # test_dataset = TensorDataset(patients[test_idx], labels[test_idx])
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return (
        patients,
        labels,
        groups,
        generator,
        splits_num,
        new_df,
    )


def iterate_through_generator(patients, labels, groups, generator):
    other_idx, test_idx = next(generator)
    group_kfold2 = GroupKFold(n_splits=9)
    # group_kfold2.split(patients[other_idx], labels[other_idx], groups[other_idx])
    generator2 = group_kfold2.split(X=patients[other_idx], y=labels[other_idx], groups=groups[other_idx])
    train_idx, val_idx = next(generator2)
    assert np.intersect1d(other_idx[train_idx], test_idx).size == 0
    assert np.intersect1d(other_idx[val_idx], test_idx).size == 0
    train_idx, val_idx = other_idx[train_idx], other_idx[val_idx]
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert np.intersect1d(val_idx, test_idx).size == 0
    assert np.intersect1d(train_idx, val_idx).size == 0
    return train_idx, val_idx, test_idx
