import argparse
import json
import os

import numpy as np
import torch

from write_vnnlib_src.utils import seed_func
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_2

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--indices-dir", default="/home/cw3344@drexel.edu/fairness_verification_continuous_training/checkpoints_03062024")
    parser.add_argument("--vnnlib-result-dir", default="write_vnnlib_verified_result_dir")
    parser.add_argument("--vnnlib-clean-result-dir", default="/home/cw3344@drexel.edu/fairness_verification_continuous_training/checkpoints_03062024/write_clean_verified_result_dir")
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args
def main(args):
    seed_func(args.seed)
    with open(os.path.join(args.indices_dir, "data_split_indices.json"), "r") as f:
        data_splits = json.load(f)
    with open(os.path.join(args.indices_dir, "labels.json"), "r") as f:
        labels = json.load(f)
    
    for model in sorted(os.listdir(args.vnnlib_result_dir)):
        for combination in sorted(os.listdir(os.path.join(args.vnnlib_result_dir, model))):
            mn_results = {"statisic": [], "p-value": []}
            save_dir = os.path.join(args.vnnlib_result_dir, model, combination)
            # if model == "mlp6" and combination == "PTEDUCAT_PTETHCAT":
                # print()
            all_mn_results = {"mean-statisic": [], "mean-p-value": [], "clean_accuracy": [], "verified_accuracy": []}
            for fold in range(args.folds):
                test_indices = data_splits[str(fold)]["test"]
                original_labels = np.array(labels)[test_indices]
                with open(os.path.join(save_dir, f"fold_{fold}.json"), "r") as f:
                    data_temp = json.load(f)
                    perturbed_verified_biased_idx = data_temp["verified_biased"]
                    perturbed_verified_fair_idx = data_temp["verified_fair"]
                with open(os.path.join(args.vnnlib_clean_result_dir, model, f"fold_{fold}.json"), "r") as f:
                    clean_data_temp = json.load(f)
                    clean_verified_biased_idx = clean_data_temp["verified_biased"]
                    clean_verified_fair_idx = clean_data_temp["verified_fair"]
                assert test_indices == sorted(perturbed_verified_biased_idx + perturbed_verified_fair_idx)
                assert test_indices == sorted(clean_verified_biased_idx + clean_verified_fair_idx)
                # change the indices to boolean result
                # Initialize the boolean result arrays
                perturbed_verified_bool = -1 * np.ones(len(test_indices))
                clean_verified_bool = -1 * np.ones(len(test_indices))
                
                # Map the perturbed indices to boolean results
                perturbed_verified_bool[np.isin(test_indices, perturbed_verified_biased_idx)] = 0
                perturbed_verified_bool[np.isin(test_indices, perturbed_verified_fair_idx)] = 1
                
                # Map the clean indices to boolean results
                clean_verified_bool[np.isin(test_indices, clean_verified_biased_idx)] = 0
                clean_verified_bool[np.isin(test_indices, clean_verified_fair_idx)] = 1
                
                clean_accuracy = sum(clean_verified_bool)/(clean_verified_bool.size)
                verified_accuracy = sum(perturbed_verified_bool)/(perturbed_verified_bool.size)
                all_mn_results["clean_accuracy"].append(clean_accuracy)
                all_mn_results["verified_accuracy"].append(verified_accuracy)
                # verify no -1 in the result
                assert -1 not in perturbed_verified_bool and -1 not in clean_verified_bool

                # Create contingency table
                # follow the guide from https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
                n00 = ((clean_verified_bool != 1) & (perturbed_verified_bool != 1)).sum()
                n01 = ((clean_verified_bool != 1) & (perturbed_verified_bool == 1)).sum()
                n10 = ((clean_verified_bool == 1) & (perturbed_verified_bool != 1)).sum()
                n11 = ((clean_verified_bool == 1) & (perturbed_verified_bool == 1)).sum()
                table = [[n11, n10], [n01, n00]]
                
                # Use McNemar test
                b = table[0][1]
                c = table[1][0]
                if (b + c) < 25:
                    result = mcnemar_2(table, exact=True)
                else:
                    result = mcnemar_2(table, exact=False, correction=True)
                
                stat, p2 = result.statistic, result.pvalue
                mn_results["statisic"].append(stat)
                mn_results["p-value"].append(p2)
            assert len(all_mn_results["verified_accuracy"]) == args.folds and len(all_mn_results["clean_accuracy"]) == args.folds and len(mn_results["statisic"]) == args.folds and len(mn_results["p-value"]) == args.folds
            print(f"Results of model {model} combination {combination} statistic={np.mean(mn_results['statisic'])}, p-value={np.mean(mn_results['p-value'])}, clean_accuracy={np.mean(all_mn_results['clean_accuracy'])*100:.4f}%, verified_accuracy={np.mean(all_mn_results['verified_accuracy'])*100:.4f}%")
            # print(f"Contingency table: {table}")
            all_mn_results["mean-statisic"].append(np.mean(mn_results['statisic']))
            all_mn_results["mean-p-value"].append(np.mean(mn_results['p-value']))
        # print(f"Results for model {model}, combination {combination}:")
        # print(f"Minimum McNemar Test Result: min-p-value={np.min(all_mn_results['mean-p-value'])}")
        # combinations = sorted(os.listdir(os.path.join(args.vnnlib_result_dir, model)))
        # sorted_combinations = [(p_value, combination) for p_value, combination in sorted(zip(all_mn_results["mean-p-value"], combinations), key=lambda x: x[0])]
        # print("Sorted combinations:")
        # for p_value, combination in sorted_combinations:
        #     print(f"Combination {combination}: p-value={p_value}")
        print()
if __name__ == "__main__":
    args = get_arg_parser()
    main(args)