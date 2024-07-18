import pickle
import os
if not os.path.exists("log_fnr_fpr"):
    os.makedirs("log_fnr_fpr")

modes = ["disc"]
models = ["logistic", "mlp3", "mlp6"]
dataset_splits = [0, 1]
for model in models:
    for mode in modes:
        for dataset_split in dataset_splits:
            load_path = f"run7/model_{model}_split_{dataset_split}_log_{mode}"
            save_path = f"log_fnr_fpr/model_{model}_split_{dataset_split}_log_{mode}.txt"
            with open(load_path + ".pkl", "rb") as f:
                attrs_metrics = pickle.load(f)
            clean_metrics = list(attrs_metrics.items())[0][1]["clean"]
            with open(save_path, "w") as f:
                f.write(f"for model_{model}_split_{dataset_split}_log_{mode}:\n")
                for metric, values in clean_metrics.items():
                    f.write(f"metric {metric}:\n")
                    f.write("| ")
                    for i in range(len(values)):
                        f.write(f"{(i+1):10d} | ")
                    f.write("\n| ")
                    for value in values:
                        f.write(f"{value:.8f} | ")
                    f.write("\n")
                f.write("end of file\n")
