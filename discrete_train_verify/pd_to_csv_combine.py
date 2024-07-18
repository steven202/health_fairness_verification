

import os
import pickle
import numpy as np
import pandas as pd

models = ["logistic", "mlp3", "mlp6"]
df_h = []
for dataset_split in [0,1]:
    df_v = []
    for model in models:
        path = os.path.join("./log_analyze_specs",f"data_{dataset_split}_{model}")
        df_tmp = pd.read_csv(path+".csv",index_col=0)
        df_v.append(df_tmp)
    df_tmp2 = pd.concat(df_v,axis=0)
    df_h.append(df_tmp2)
df_tot = pd.concat(df_h,axis=0)
df_tot.insert(0,"model_split",np.NaN)
df_tot = df_tot.reset_index(drop=True)
counter = 0
for dataset_split in [0,1]:
    for model in models:
        for k in range(0,30):
            df_tot.loc[counter*30+k,"model_split"]=model+"_split_"+str(dataset_split)
            # print(counter*30+k)
        counter+=1
df_tot.replace(0.0, np.nan, inplace=True)
prefix = "_"+os.getcwd().split("/")[-1].replace("health_fairness_","")
if not os.path.exists("tables_analyze_spec"+prefix):
    os.makedirs("tables_analyze_spec"+prefix)
df_tot.to_csv("tables_analyze_spec"+prefix+"/combine_vertical.csv", index=False)
# print("done")
        # with open(os.path.join("./log_analyze_specs",f"data_{dataset_split}_{model}.csv"),"rb") as f:
            # all_attr_analyze = pickle.load(f)