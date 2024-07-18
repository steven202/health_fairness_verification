import argparse
import json
import os

import numpy as np

from write_vnnlib_src.utils import seed_func


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--indices-dir", default="/home/cw3344@drexel.edu/fairness_verification_continuous_training/checkpoints_03062024")
    parser.add_argument("--vnnlib-execute-dir", default="write_vnnlib_execute_dir")
    parser.add_argument("--vnnlib-result-dir", default="write_vnnlib_verified_result_dir")
    parser.add_argument("--seed", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

def main(args):
    seed_func(args.seed)
    with open(os.path.join(args.indices_dir, "data_split_indices.json"), "r") as f:
        data_splits = json.load(f)
    for model in sorted(os.listdir(args.vnnlib_execute_dir)):
        for combination in sorted(os.listdir(os.path.join(args.vnnlib_execute_dir, model))):
            whole_test_set = set()
            whole_verified_biased = set() # unsafe
            whole_verified_fair = set() # safe
            save_dir = os.path.join(args.vnnlib_result_dir, model, combination)
            os.makedirs(save_dir, exist_ok=True)
            verified_accuracies = []
            for fold in range(args.folds):
                execute_log_path = os.path.join(args.vnnlib_execute_dir, model, combination, f"fold_{fold}.txt")
                assert os.path.exists(execute_log_path), f"{execute_log_path} does not exist"
                with open(execute_log_path, "r") as f:
                    lines = f.readlines()
                    # make sure unknown is not in any one of the last three lines
                    last_num = -5
                    # assert np.all(["unknown" not in temp for temp in lines[last_num:]]), f"unknown should not be in the last three lines"
                    results = lines[last_num:]
                    # make sure the last line is safe and the second last line is unsafe
                    # assert "unsafe" in results[0], f"Second last line should be unsafe"
                    # assert "safe" in results[1], f"Last line should be safe"
                    # print(f"Model: {model}, Combination: {combination}, Fold: {fold}")
                    # select the line with "unsafe" and "safe from the last two lines
                    line_with_unsafe = [temp for temp in results if "unsafe" in temp]
                    line_with_safe = [temp for temp in results if ("safe" in temp and "unsafe" not in temp) or "unknown" in temp]
                    # make sure there is only one line with "unsafe" and "safe" and they are different lines and they are not empty
                    # assert len(line_with_unsafe) == 1 and len(line_with_safe) == 1 and line_with_unsafe[0] != line_with_safe[0], f"There should be only one line with 'unsafe' and 'safe' and they should be different lines"
                    # assert len(line_with_unsafe[0].split("index: ")) == 2 and len(line_with_safe[0].split("index: ")) == 2, f"Line with 'unsafe' and 'safe' should have 'index: '"
                    unsafe_pgd = [[int(temp) for temp in line_with_unsafe[i].split("index: ")[1].strip().replace("[", "").replace("]", "").split(", ")] for i in range(len(line_with_unsafe))]
                    safe_incomplete = [[int(temp) for temp in line_with_safe[i].split("index: ")[1].strip().replace("[", "").replace("]", "").split(", ")] for i in range(len(line_with_safe))]
                    unsafe_pgd, safe_incomplete = [item for sublist in unsafe_pgd for item in sublist], [item for sublist in safe_incomplete for item in sublist]
                    # convert index to test index
                    unsafe_pgd = [data_splits[str(fold)]["test"][temp] for temp in unsafe_pgd]
                    safe_incomplete = [data_splits[str(fold)]["test"][temp] for temp in safe_incomplete]
                    if fold != 9:
                        assert len(unsafe_pgd) + len(safe_incomplete) == 173, f"Length of unsafe_pgd and safe_incomplete should be 173"
                    else:
                        assert len(unsafe_pgd) + len(safe_incomplete) == 172, f"Length of unsafe_pgd and safe_incomplete should be 172"
                    # print(f"fold {fold}", sorted(unsafe_pgd+safe_incomplete))
                    # before that, make sure whole_test_set doesn't have any intersection with unsafe_pgd and safe_incomplete
                    assert len(whole_test_set.intersection(set(unsafe_pgd))) == 0
                    assert len(whole_test_set.intersection(set(safe_incomplete))) == 0
                    # append the unsafe_pgd and safe_incomplete to whole_test_set
                    whole_test_set = whole_test_set.union(set(unsafe_pgd))
                    whole_test_set = whole_test_set.union(set(safe_incomplete))
                    # append the unsafe_pgd and safe_incomplete to whole_verified_biased and whole_verified_safe
                    whole_verified_biased = whole_verified_biased.union(set(unsafe_pgd))
                    whole_verified_fair = whole_verified_fair.union(set(safe_incomplete))
                    # save the unsafe_pgd and safe_incomplete to a json file, sorted
                    with open(os.path.join(save_dir, f"fold_{fold}.json"), "w") as f:
                        json.dump({"verified_biased": sorted(list(unsafe_pgd)), "verified_fair": sorted(list(safe_incomplete))}, f)
                    verified_accuracy = len(safe_incomplete)/(len(unsafe_pgd) + len(safe_incomplete))
                    verified_accuracies.append(verified_accuracy)
            print(f"model: {model} combination {combination} average verified_accuracy={np.mean(verified_accuracies)*100:.4f}%")
            # make sure the length of whole_test_set is equal to the length of data_splits[i]["test"] for all i in range(10)
            assert len(whole_test_set) == sum([len(data_splits[str(i)]["test"]) for i in range(10)]), f"Length of whole_test_set should be equal to the length of data_splits[i]['test'] for all i in range(10)"
            
            
if __name__ == "__main__":
    args = get_arg_parser()
    main(args)