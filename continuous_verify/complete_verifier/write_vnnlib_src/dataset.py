import numpy as np
from sklearn.model_selection import GroupKFold
import torch
import functools
from write_vnnlib_src import misc

def intialize_generator_mlp(args):
    columns = ["RID", "DXCHANGE", "EXAMDATE"]
    columns = "RID	PTID	VISCODE	SITE	D1	D2	COLPROT	ORIGPROT	EXAMDATE	DX_bl	DXCHANGE DX".split()
    columns = "RID VISCODE AGE	PTGENDER	PTEDUCAT	PTETHCAT	PTRACCAT	PTMARRY Month_bl DX".split()
    features = misc.load_feature(args.features)
    features = "CDRSB ADAS11 ADAS13 MMSE RAVLT_immediate RAVLT_learning RAVLT_forgetting RAVLT_perc_forgetting MOCA FAQ Entorhinal Fusiform Hippocampus ICV MidTemp Ventricles WholeBrain AV45 FDG ABETA_UPENNBIOMK9_04_19_17 TAU_UPENNBIOMK9_04_19_17 PTAU_UPENNBIOMK9_04_19_17".split()

    frame = misc.load_table(args.spreadsheet, columns + features)
    frame["Vent"] = frame.Ventricles / frame.ICV
    features.append("Vent")
    frame = frame.replace({"VISCODE": "bl"}, 0.0)
    frame["VISCODE"] = np.where(
        frame["VISCODE"].astype(str).str.startswith("m0"),
        (frame["VISCODE"].astype(str).str[2:]),
        frame["VISCODE"],
    )
    frame["VISCODE"] = np.where(
        frame["VISCODE"].astype(str).str.startswith("m"),
        (frame["VISCODE"].astype(str).str[1:]),
        frame["VISCODE"],
    )
    frame["VISCODE"] = frame["VISCODE"].astype(float)
    frame["VISCODE"] = frame["VISCODE"].div(12.0)

    # a backup plan for normalizing all columns
    special_columns = " AGE	PTGENDER	PTEDUCAT	PTETHCAT	PTRACCAT	PTMARRY Month_bl ".split()
    frame["AGE"] = frame["AGE"] + frame["VISCODE"]
    # mean = frame.loc[:, features].mean()
    # std = frame.loc[:, features].std()
    df = frame.copy()

    # df[features] = (df[features] - mean) / std

    # only keep the first occurence of every RID
    df = df.sort_values(by=["RID", "VISCODE"]).drop_duplicates(subset=["RID"])

    if args.dataset_split == 0:
        df = df.loc[(np.abs(df["DX"]) <= 1e-14) | (np.abs(df["DX"] - 1.0) <= 1e-14)]
    elif args.dataset_split == 1:
        df = df.loc[(np.abs(df["DX"] - 1.0) <= 1e-14) | (np.abs(df["DX"] - 2.0) <= 1e-14)]
        df["DX"] = df["DX"] - 1.0
    else: 
        pass

    new_df = df
    new_df.to_csv("fairness_data/fairness.csv")
    feature_names = new_df.columns.values.tolist()
    feature_names.remove("DX")
    feature_names.remove("RID")
    feature_names.remove("VISCODE")
    # feature_names.remove("MMSE.bl")
    # feature_names.remove("CDRSB.bl")
    feature_names.remove("MMSE")
    feature_names.remove("CDRSB")
    # mean = new_df.loc[:, feature_names].mean()
    # std = new_df.loc[:, feature_names].std()
    # new_df[feature_names] = (new_df[feature_names] - mean) / std
    patients_ = new_df[feature_names].to_numpy()
    labels_ = new_df[["DX"]].to_numpy()
    groups_ = new_df[["RID"]].to_numpy()
    patients_[np.isnan(patients_)] = 0.0
    valid = (~np.isnan(labels_)).squeeze()

    patients = torch.tensor(patients_[valid], dtype=torch.float32)
    # print(patients.shape, functools.reduce(lambda x, y: x * y, patients.shape))
    print(f"Number of features: {patients.shape[1]}")
    print(f"Number of samples: {patients.shape[0]}")
    print(f"Number of classes: {len(np.unique(labels_[valid]))}")
    # print(f"Number of groups: {len(np.unique(groups_[valid]))}")
    labels = torch.tensor(labels_[valid], dtype=torch.long).squeeze()
    groups = torch.tensor(groups_[valid], dtype=torch.long).squeeze()
    group_kfold = GroupKFold(n_splits=10)
    splits_num = group_kfold.get_n_splits(patients, labels, groups)
    generator = group_kfold.split(X=patients, y=labels, groups=groups)

    # test_dataset = TensorDataset(patients[test_idx], labels[test_idx])
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return patients, labels, groups, generator, splits_num


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
