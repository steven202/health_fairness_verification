import argparse
import numpy as np
import pandas as pd
import pickle
import itertools
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
import cbig.Nguyen2020.misc as misc
import matplotlib.pyplot as plt

from utils import set_seed

def preprocess_data(spreadsheet="./fairness_data/TADPOLE_D1_D2.csv", features_file="./fairness_data/features", generated_data_dir="generated_data"):
    if generated_data_dir != "":
        os.makedirs(generated_data_dir, exist_ok=True)
    
    columns = "RID VISCODE AGE PTGENDER PTEDUCAT PTETHCAT PTRACCAT PTMARRY Month_bl DX".split()
    features = misc.load_feature(features_file)
    features = "CDRSB ADAS11 ADAS13 MMSE RAVLT_immediate RAVLT_learning RAVLT_forgetting RAVLT_perc_forgetting MOCA FAQ Entorhinal Fusiform Hippocampus ICV MidTemp Ventricles WholeBrain AV45 FDG ABETA_UPENNBIOMK9_04_19_17 TAU_UPENNBIOMK9_04_19_17 PTAU_UPENNBIOMK9_04_19_17".split()

    frame = misc.load_table(spreadsheet, columns + features)
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
    frame["AGE"] = frame["AGE"] + frame["VISCODE"]
    frame = frame.sort_values(by=["RID", "VISCODE"])
    frame_indices = frame.index.tolist()
    frame.to_csv(os.path.join(generated_data_dir, "processed_data.csv"))

    with open(os.path.join(generated_data_dir, "frame_indices.pkl"), "wb") as fp:
        pickle.dump(frame_indices, fp)

    return frame

def analyze_data(dataset_split, selected_year, frame, generated_data_dir="generated_data"):
    if generated_data_dir != "":
        os.makedirs(generated_data_dir, exist_ok=True)
    
    selected_dict = {
        0: {"unchanged": None, "changed": None},
        1: {"unchanged": None, "changed": None},
    }
    years = {}
    months = {}
    years_unchanged = {}
    months_unchanged = {}

    for num_of_years in list(np.arange(0.5, 10.0 + 0.1, 0.5)):
        frame = pd.read_csv(os.path.join(generated_data_dir, "processed_data.csv"), index_col=0)
        raw_data = frame[
            [
                "RID",
                "DX",
                "VISCODE",
            ]
        ].to_numpy()
        pd_indices = np.expand_dims(frame.index.to_numpy(), axis=1)
        raw_data = np.concatenate((raw_data, pd_indices), axis=1)

        rid_backward = dict()
        data_dic_tmp_2 = {}
        for row in raw_data:
            if np.isnan((row[0])) or np.isnan((row[1])):
                continue
            if int(row[0]) not in data_dic_tmp_2:
                data_dic_tmp_2[int(row[0])] = dict()
            if int(row[1]) not in data_dic_tmp_2[int(row[0])]:
                data_dic_tmp_2[int(row[0])][int(row[1])] = []
            data_dic_tmp_2[int(row[0])][int(row[1])].append(row[2])

        for k, v in data_dic_tmp_2.items():
            comb_tmp = []
            if 0 in v and 1 in v:
                comb_tmp.append(list(itertools.product(v[0], v[1])))
            if 1 in v and 2 in v:
                comb_tmp.append(list(itertools.product(v[1], v[2])))
            if 0 in v and 2 in v:
                comb_tmp.append(list(itertools.product(v[0], v[2])))
            for combinations in comb_tmp:
                for slot in combinations:
                    if slot[0] > slot[1]:
                        if k in rid_backward:
                            if rid_backward[k] >= slot[0]:
                                rid_backward[k] = slot[0]
                        else:
                            rid_backward[k] = slot[0]

        if dataset_split == 0:
            start_dx = 0
            end_dx = 1
        elif dataset_split == 1:
            start_dx = 1
            end_dx = 2

        print("length of rid backward", len(set(rid_backward.keys())))

        classes = f"{'NC=0, MCI=1' if dataset_split == 0 else 'MCI=0, AD=1'}"
        data = []
        if dataset_split == 0:
            for row in raw_data:
                if np.isnan(row[0]) or np.isnan((row[1])) or np.isnan((row[2])): 
                    continue
                if int(row[1]) != 2:
                    data.append(row)
        elif dataset_split == 1:
            for row in raw_data:
                if np.isnan(row[0]) or np.isnan((row[1])) or np.isnan((row[2])): 
                    continue
                if int(row[1]) != 0:
                    data.append(row)

        data_dic_tmp = dict()

        for row in data:
            if np.isnan((row[0])) or np.isnan((row[1])):
                continue
            if int(row[0]) not in data_dic_tmp:
                data_dic_tmp[int(row[0])] = dict()
            if int(row[1]) not in data_dic_tmp[int(row[0])]:
                data_dic_tmp[int(row[0])][int(row[1])] = dict()
            data_dic_tmp[int(row[0])][int(row[1])][row[2]] = row[3]

        data_dic_change = dict()
        data_dic_unchange = dict()

        counter_changed = 0
        counter_unchanged = 0

        for k, v in data_dic_tmp.items():
            if len(set(v.keys())) == 1 and list(v.keys())[0] != end_dx:
                sorted_year = sorted(v[start_dx].keys())
                v[start_dx] = dict(sorted(v[start_dx].items()))
                assert sorted_year == list(v[start_dx].keys())
                for idx, start_point in enumerate(sorted_year):
                    for idx2, end_point in enumerate(sorted_year):
                        if start_point + 1e-14 < end_point and end_point - start_point + 1e-14 >= num_of_years:
                            data_dic_unchange[int(v[start_dx][start_point])] = (start_point, end_point)
                            counter_unchanged += 1
                            break
            elif len(set(v.keys())) == 2:
                v[start_dx] = dict(sorted(v[start_dx].items()))
                v[end_dx] = dict(sorted(v[end_dx].items()))
                start_year = sorted(v[start_dx].keys())
                end_year = sorted(v[end_dx].keys())

                if start_year + end_year != sorted(start_year + end_year):
                    start_year = sorted(i for i in start_year if i < rid_backward[k])
                    end_year = sorted(i for i in end_year if i < rid_backward[k])
                    v[start_dx] = {key: v[start_dx][key] for key in start_year}
                    v[end_dx] = {key: v[end_dx][key] for key in end_year}
                else:
                    pass
                assert list(v[start_dx].keys()) == start_year
                assert list(v[end_dx].keys()) == end_year
                total_dx = {**v[start_dx], **v[end_dx]}
                total_year = start_year + end_year
                assert start_year + end_year == total_year
                assert start_year + end_year == sorted(total_dx.keys())
                for idx, start_point in enumerate(total_year):
                    tmp_dict_change_unchanged = dict()
                    for idx2, end_point in enumerate(total_year):
                        if start_point + 1e-14 < end_point:
                            start_stage = 0 if start_point in start_year else 1
                            end_stage = 0 if end_point in start_year else 1
                            is_changed = 0 if start_stage == end_stage else 1
                            if not is_changed and start_stage == 1:
                                continue
                            if is_changed and start_stage == 1:
                                raise Exception("should not have reverse case here")
                            pd_id = int(v[start_dx][start_point])
                            tmp_dict_change_unchanged[end_point] = is_changed
                    tmp_dict_change_unchanged = dict(sorted(tmp_dict_change_unchanged.items()))
                    if len(tmp_dict_change_unchanged) == 0:
                        continue
                    elif start_point + num_of_years in tmp_dict_change_unchanged:
                        if tmp_dict_change_unchanged[start_point + num_of_years] == 0:
                            data_dic_unchange[pd_id] = (start_point, start_point + num_of_years)
                            counter_unchanged += 1
                        else:
                            data_dic_change[pd_id] = (start_point, start_point + num_of_years)
                            counter_changed += 1
                    else:
                        left_set = sorted(i for i in tmp_dict_change_unchanged.keys() if i <= start_point + num_of_years)
                        right_set = sorted(i for i in tmp_dict_change_unchanged.keys() if i >= start_point + num_of_years)
                        if len(left_set) == 0 and tmp_dict_change_unchanged[np.min(right_set)] == 0:
                            data_dic_unchange[pd_id] = (start_point, np.min(right_set))
                            counter_unchanged += 1
                        elif len(left_set) == 0 and tmp_dict_change_unchanged[np.min(right_set)] == 1:
                            data_dic_change[pd_id] = (start_point, np.min(right_set))
                            counter_changed += 1
                        elif len(right_set) == 0 and tmp_dict_change_unchanged[np.max(left_set)] == 0:
                            continue
                        elif len(right_set) == 0 and tmp_dict_change_unchanged[np.max(left_set)] == 1:
                            data_dic_change[pd_id] = (start_point, np.max(left_set))
                            counter_changed += 1
                        else:
                            is_left_changed = tmp_dict_change_unchanged[np.max(left_set)]
                            is_right_changed = tmp_dict_change_unchanged[np.min(right_set)]
                            if is_left_changed == is_right_changed and is_right_changed == 0:
                                data_dic_unchange[pd_id] = (start_point, np.min(right_set))
                                counter_unchanged += 1
                            elif is_left_changed == is_right_changed and is_right_changed == 1:
                                data_dic_change[pd_id] = (start_point, np.max(left_set))
                                counter_changed += 1

        assert len(set(list(data_dic_change.keys())).intersection(set(list(data_dic_unchange.keys())))) == 0
        assert counter_unchanged == len(data_dic_unchange)
        assert counter_changed == len(data_dic_change)
        if num_of_years == selected_year:
            selected_dict[dataset_split]["unchanged"] = data_dic_unchange
            selected_dict[dataset_split]["changed"] = data_dic_change

        rids_0 = []
        rids_1 = []
        rids_2 = []
        rids_0_1 = []
        rids_1_2 = []

        if dataset_split == 0:
            rids_0_1 = data_dic_change
            rids_0 = data_dic_unchange
        else:
            rids_1_2 = data_dic_change
            rids_1 = data_dic_unchange

        if dataset_split == 0:
            assert len(rids_2) == 0 and len(rids_1_2) == 0
            print(
                "NL unchanged: {}, NL to MCI: {}".format(
                    len(rids_0),
                    len(rids_0_1),
                )
            )
        elif dataset_split == 1:
            assert len(rids_0) == 0 and len(rids_0_1) == 0
            print(
                "MCI unchanged: {}, MCI to AD: {}".format(
                    len(rids_1),
                    len(rids_1_2),
                )
            )

        if dataset_split == 0:
            rids_changed = rids_0_1
            rids_unchanged = rids_0
        elif dataset_split == 1:
            rids_changed = rids_1_2
            rids_unchanged = rids_1
        years[num_of_years] = len(rids_changed)
        months[int(num_of_years * 12)] = len(rids_changed)
        years_unchanged[num_of_years] = len(rids_unchanged)
        months_unchanged[int(num_of_years * 12)] = len(rids_unchanged)

    return selected_dict, years, months, years_unchanged, months_unchanged

def plot_years(years, years_unchanged, classes, selected_year, dataset_split, plots_dir="plots"):
    if plots_dir != "":
        os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in years.keys():
        if k not in years_unchanged:
            years_unchanged[k] = 0
    for k in years_unchanged.keys():
        if k not in years:
            years[k] = 0
    years = dict(sorted(years.items()))
    years_unchanged = dict(sorted(years_unchanged.items()))

    x = list(years.keys())
    y = list(years.values())
    x2 = list(years_unchanged.keys())
    y2 = list(years_unchanged.values())
    assert x == x2
    X_axis = np.arange(len(x2))
    plt.grid()
    plt.bar(X_axis - 0.2, y2, color = "cornflowerblue", width = 0.4, label = "unchanged", alpha = 0.7)
    plt.bar(X_axis + 0.2, y, color ="darkgoldenrod", width = 0.4, label = "progressed", alpha = 0.7)
    plt.xticks(X_axis, x2)

    plt.ylabel("No. of people for the conversion", fontsize=18)
    plt.xlabel("No. of years for the conversion", fontsize=18)
    # plt.title(classes, y=-0.3, fontsize=18)
    min_val = np.min([np.min(y), np.min(y2)])
    max_val = np.max([np.max(y), np.max(y2)])
    plt.legend(fontsize=18)
    ax.set_yticks(np.arange(min_val, max_val + 1, (max_val - min_val) // 20))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"year_{selected_year}_split_{dataset_split}_class_{classes}.pdf"),dpi=300,bbox_inches="tight")
    plt.close()

def plot_months(months, months_unchanged, classes, selected_year, dataset_split, plots_dir="plots"):
    if plots_dir != "":
        os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in months.keys():
        if k not in months_unchanged:
            months_unchanged[k] = 0
    for k in months_unchanged.keys():
        if k not in months:
            months[k] = 0
    months = dict(sorted(months.items()))
    months_unchanged = dict(sorted(months_unchanged.items()))

    x = list(months.keys())
    y = list(months.values())
    x2 = list(months_unchanged.keys())
    y2 = list(months_unchanged.values())
    assert x == x2
    X_axis = np.arange(len(x2))
    plt.grid()
    plt.bar(X_axis - 0.2, y2, color = 'cornflowerblue', width = 0.4, label = "unchanged", alpha = 0.7)
    plt.bar(X_axis + 0.2, y, color = 'darkgoldenrod', width = 0.4, label = "progressed", alpha = 0.7)
    plt.xticks(X_axis, x2)

    plt.ylabel("No. of people for the conversion", fontsize=18)
    plt.xlabel("No. of months for the conversion", fontsize=18)
    # plt.title(classes, y=-0.3, fontsize=18)
    min_val = np.min([np.min(y), np.min(y2)])
    max_val = np.max([np.max(y), np.max(y2)])
    plt.legend(fontsize=18)
    ax.set_yticks(np.arange(min_val, max_val + 1, (max_val - min_val) // 20))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"month_{selected_year}_split_{dataset_split}_class_{classes}.pdf"),dpi=300,bbox_inches="tight")
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(description='Data Preprocessing and Analysis')
    parser.add_argument('--spreadsheet', type=str, default="./fairness_data/TADPOLE_D1_D2.csv", help='Path to the spreadsheet file')
    parser.add_argument('--features', type=str, default="./fairness_data/features", help='Path to the features file')
    parser.add_argument('--selected-year', type=float, choices=[3.0, 5.0], required=True, help='Selected year value')
    parser.add_argument('--plots', type=str, default="plots", help='Directory to save plots')
    parser.add_argument('--data-dir', type=str, default="generated_data", help='Directory to save generated data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    set_seed(args.seed)    
    frame = preprocess_data(args.spreadsheet, args.features, args.data_dir)

    all_selected_dict = {
        0: {"unchanged": None, "changed": None},
        1: {"unchanged": None, "changed": None},
    }
    for split in [0, 1]:
        selected_dict, years, months, years_unchanged, months_unchanged = analyze_data(split, args.selected_year, frame, args.data_dir)
        all_selected_dict[split]["unchanged"] = selected_dict[split]["unchanged"]
        all_selected_dict[split]["changed"] = selected_dict[split]["changed"]
        classes = f"{'NC=0, MCI=1' if split == 0 else 'MCI=0, AD=1'}"
        plot_years(years, years_unchanged, classes, args.selected_year, split, args.plots)
        plot_months(months, months_unchanged, classes, args.selected_year, split, args.plots)

    print(len(all_selected_dict[0]['unchanged']),
          len(all_selected_dict[0]['changed']), 
          len(all_selected_dict[1]['unchanged']),
          len(all_selected_dict[1]['changed']))

    with open(os.path.join(args.data_dir, 'saved_pd_indices.pkl'), 'wb') as f:
        pickle.dump(all_selected_dict, f)
    # verify the saved data
    with open(os.path.join(args.data_dir, 'saved_pd_indices.pkl'), 'rb') as f:
        loaded_dict = pickle.load(f)
    assert len(loaded_dict[0]['unchanged']) == len(all_selected_dict[0]['unchanged'])
    assert len(loaded_dict[0]['changed']) == len(all_selected_dict[0]['changed'])
    assert len(loaded_dict[1]['unchanged']) == len(all_selected_dict[1]['unchanged'])
    assert len(loaded_dict[1]['changed']) == len(all_selected_dict[1]['changed'])
    
if __name__ == '__main__':
    main()
