import numpy as np
import os.path as path

import pandas as pd
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
# from torchmetrics.classification import BinaryConfusionMatrix
from warnings import filterwarnings
from itertools import combinations

filterwarnings(action="ignore", category=DeprecationWarning, message="`np.long` is a deprecated alias")

# copy from https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def set_seed(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def flatten_combinations(l):
    """flatten all sublist in a list"""
    return [item for sublist in l for item in sublist]


def split_by_median_date(data, subjects):
    """
    Split timepoints in two halves, use first half to predict second half
    Args:
        data (Pandas data frame): input data
        subjects: list of subjects
    Return:
        first_half (ndarray): boolean mask, rows used as input
        second_half (ndarray): boolean mask, rows to predict
    """
    first_half = np.zeros(data.shape[0], int)
    second_half = np.zeros(data.shape[0], int)
    for rid in subjects:
        subj_mask = (data.RID == rid) & data.has_data
        median_date = np.sort(data.EXAMDATE[subj_mask])[subj_mask.sum() // 2]
        first_half[subj_mask & (data.EXAMDATE < median_date)] = 1
        second_half[subj_mask & (data.EXAMDATE >= median_date)] = 1
    return first_half, second_half


def gen_fold(data, nb_folds, outdir):
    """Generate *nb_folds* cross-validation folds from *data"""
    subjects = np.unique(data.RID)
    has_2tp = np.array([np.sum(data.RID == rid) >= 2 for rid in subjects])

    potential_targets = np.random.permutation(subjects[has_2tp])
    folds = np.array_split(potential_targets, nb_folds)

    leftover = [subjects[~has_2tp]]

    for test_fold in range(nb_folds):
        val_fold = (test_fold + 1) % nb_folds
        train_folds = [i for i in range(nb_folds) if (i != test_fold and i != val_fold)]

        train_subj = np.concatenate([folds[i] for i in train_folds] + leftover, axis=0)
        val_subj = folds[val_fold]
        test_subj = folds[test_fold]

        train_timepoints = (np.in1d(data.RID, train_subj) & data.has_data).astype(int)
        val_in_timepoints, val_out_timepoints = split_by_median_date(data, val_subj)
        test_in_timepoints, test_out_timepoints = split_by_median_date(data, test_subj)

        mask_frame = gen_mask_frame(data, train_timepoints, val_in_timepoints, test_in_timepoints)
        mask_frame.to_csv(path.join(outdir, "fold%d_mask.csv" % test_fold), index=False)

        val_frame = gen_ref_frame(data, val_out_timepoints)
        val_frame.to_csv(path.join(outdir, "fold%d_val.csv" % test_fold), index=False)

        test_frame = gen_ref_frame(data, test_out_timepoints)
        test_frame.to_csv(path.join(outdir, "fold%d_test.csv" % test_fold), index=False)


def gen_mask_frame(data, train, val, test):
    """
    Create a frame with 3 masks:
        train: timepoints used for training model
        val: timepoints used for validation
        test: timepoints used for testing model
    """
    col = ["RID", "EXAMDATE"]
    ret = pd.DataFrame(data[col], index=range(train.shape[0]))
    ret["train"] = train
    ret["val"] = val
    ret["test"] = test

    return ret


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
    # assert (FP + TN).item() != 0  # all y_pred==1 # when we have masks, this may happen
    # assert (TP + FN).item() != 0  # all y_pred==0 # when we have masks, this may happen
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


# def mcnemar(y_pred, y_test):
#     """calculate metrics according to https://vitalflux.com/accuracy-precision-recall-f1-score-python-example/"""
#     # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#     # http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
#     if isinstance(y_pred, np.ndarray):
#         y_test = y_test.cpu().detach().clone()
#     else:
#         y_pred, y_test = y_pred.cpu().detach().clone(), y_test.cpu().detach().clone()
#     conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#     tn, fp, fn, tp = conf_matrix.ravel()
#     TN, FP, FN, TP = tn, fp, fn, tp
#     # https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5
#     mcnemar_data = [[TP, FP], [FN, TN]]
#     from statsmodels.stats.contingency_tables import mcnemar

#     print(mcnemar(mcnemar_data, exact=False))

#     print(mcnemar(mcnemar_data, exact=False, correction=False))

#     b = FP
#     c = FN
#     chi_squared = (b - c) ** 2 / (b + c)
#     return chi_squared


def mcnemar(y_target, y_model1, y_model2):
    # http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
    # main source:
    # https://machinelearningmastery.com/mcnemars-test-for-machine-learning/#:~:text=dataset%20is%20large.-,McNemar's%20Test%20in%20Python,on%20the%20amount%20of%20data.
    # https://towardsdatascience.com/mcnemars-test-to-evaluate-machine-learning-classifiers-with-python-9f26191e1a6b
    from mlxtend.evaluate import mcnemar_table
    # from mlxtend.evaluate import mcnemar
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
    # statistic = (n10 - n01) ** 2 / (n10 + n01)
    # chi_squared = np.divide(np.square(b - c), (b + c))
    # chi_squared2 = np.divide(np.square(np.abs(b - c) - 1), (b + c))
    if (b + c) < 25:
        # chi2, p = mcnemar(ary=tb, exact=True)
        result = mcnemar_2(table, exact=True)
        stat, p2 = result.statistic, result.pvalue
    else:
        # chi2, p = mcnemar(ary=tb, corrected=True)
        result = mcnemar_2(table, exact=False, correction=True)
        stat, p2 = result.statistic, result.pvalue
    # alpha = 0.05
    # print("statistic=%.3f, p-value=%.3f" % (result.statistic, result.pvalue))
    # if result.pvalue > alpha:
    #     print("Same proportions of errors (fail to reject H0)")
    # else:
    #     print("Different proportions of errors (reject H0)")
    # significance_value = 0.05
    # if result.pvalue < significance_value:
    #     print("Reject Null hypotesis")
    # else:
    #     print("Fail to reject Null hypotesis")
    # print("chi-squared:", chi2)
    # print("p-value:", p)
    return (stat, p2)


# https://github.com/imrahulr/adversarial_robustness_pytorch.git
class CosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing LR schedule (used in Carmon et al, 2019).
    """

    def __init__(self, optimizer, max_lr, epochs, last_epoch=-1):
        self.max_lr = max_lr
        self.epochs = epochs
        self._reset()
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def _reset(self):
        self.current_lr = self.max_lr
        self.current_epoch = 1

    def step(self):
        self.current_lr = self.max_lr * 0.5 * (1 + np.cos((self.current_epoch - 1) / self.epochs * np.pi))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr
        self.current_epoch += 1
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_lr(self):
        return self.current_lr

# https://www.geeksforgeeks.org/itertools-combinations-module-python-print-possible-combinations/
def rSubset(arr, r):

    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))