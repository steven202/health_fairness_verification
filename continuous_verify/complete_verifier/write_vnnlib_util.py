import numpy as np
import os.path as path

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from warnings import filterwarnings

filterwarnings(action="ignore", category=DeprecationWarning, message="`np.long` is a deprecated alias")

def gen_ref_frame(data, test_timepoint_mask):
    """Create reference frame which is used to evalute models' prediction"""
    columns = ["RID", "CognitiveAssessmentDate", "Diagnosis", "ADAS13", "ScanDate"]
    # ret = pd.DataFrame(
    #     np.nan, index=range(len(test_timepoint_mask)), columns=columns)
    # ret[columns] = data[['RID', 'EXAMDATE', 'DXCHANGE', 'ADAS13', 'EXAMDATE']]

    ret = data[["RID", "EXAMDATE", "DXCHANGE", "ADAS13", "EXAMDATE"]]
    ret.columns = columns
    ret["Ventricles"] = data["Ventricles"] / data["ICV"]
    ret = ret[test_timepoint_mask == 1]

    # map diagnosis from numeric categories back to labels
    mapping = {
        1: "CN",
        7: "CN",
        9: "CN",
        2: "MCI",
        4: "MCI",
        8: "MCI",
        3: "AD",
        5: "AD",
        6: "AD",
    }
    ret.replace({"Diagnosis": mapping}, inplace=True)
    ret.reset_index(drop=True, inplace=True)

    return ret


def calculate_metrics(y_pred, y_test):
    """calculate metrics according to https://vitalflux.com/accuracy-precision-recall-f1-score-python-example/"""
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    if isinstance(y_pred, np.ndarray):
        y_test = y_test.cpu().detach().clone()
    else:
        y_pred, y_test = y_pred.cpu().detach().clone(), y_test.cpu().detach().clone()
    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
    TN, FP, FN, TP = tn, fp, fn, tp
    # metric = BinaryConfusionMatrix().to(y_pred.device)
    # conf_matrix = metric(y_pred, y_test)
    # TN, FN, FP, TP = (
    #     conf_matrix[0][0],
    #     conf_matrix[0][1],
    #     conf_matrix[1][0],
    #     conf_matrix[1][1],
    # )
    # https://stackoverflow.com/questions/45053238/how-to-get-all-confusion-matrix-terminologies-tpr-fpr-tnr-fnr-for-a-multi-c
    # Fall out or false positive rate
    assert (FP + TN).item() != 0  # all y_pred==1
    assert (TP + FN).item() != 0  # all y_pred==0
    if (FP + TN).item() == 0 and FP.item() == 0:
        # FPR = np.nan
        FPR = 0
    else:
        FPR = 1.0 * FP / (FP + TN)  # 100 nan
        FPR = FPR.item()
    # False negative rate
    if (TP + FN).item() == 0 and FN.item() == 0:
        # FNR = np.nan
        FNR = 0
    else:
        FNR = 1.0 * FN / (TP + FN)
        FNR = FNR.item()
    # Overall accuracy
    if (TP + FP + FN + TN).item() == 0 and (TP + TN).item() == 0:
        # ACC = np.nan
        ACC = 0
    else:
        ACC = 1.0 * (TP + TN) / (TP + FP + FN + TN)
        ACC = ACC.item()
    return FPR, FNR, ACC


def mcnemar(y_target, y_model1, y_model2):
    # http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
    # main source:
    # https://machinelearningmastery.com/mcnemars-test-for-machine-learning/#:~:text=dataset%20is%20large.-,McNemar's%20Test%20in%20Python,on%20the%20amount%20of%20data.
    # https://towardsdatascience.com/mcnemars-test-to-evaluate-machine-learning-classifiers-with-python-9f26191e1a6b
    from mlxtend.evaluate import mcnemar_table
    from statsmodels.stats.contingency_tables import mcnemar as mcnemar_2

    y_target, y_model1, y_model2 = (
        y_target.cpu().detach().clone().numpy(),
        y_model1.cpu().detach().clone().numpy(),
        y_model2.cpu().detach().clone().numpy(),
    )
    tb = mcnemar_table(y_target=y_target, y_model1=y_model1, y_model2=y_model2)
    b = tb[0][1]
    c = tb[1][0]
    n00 = ((y_target != y_model1) & (y_target != y_model2)).sum()
    n01 = ((y_target != y_model1) & (y_target == y_model2)).sum()
    n10 = ((y_target == y_model1) & (y_target != y_model2)).sum()
    n11 = ((y_target == y_model1) & (y_target == y_model2)).sum()
    table = [[n11, n10], [n01, n00]]

    if (b + c) < 25:
        # chi2, p = mcnemar(ary=tb, exact=True)
        result = mcnemar_2(table, exact=True)
        stat, p2 = result.statistic, result.pvalue
    else:
        result = mcnemar_2(table, exact=False, correction=True)
        stat, p2 = result.statistic, result.pvalue

    return (stat, p2)

