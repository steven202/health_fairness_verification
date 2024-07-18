import itertools
import numpy as np

import torch

from test import test_model_mlp


def discrete_method(image, true_label, model, num_classes, column_bounds, clf=None, mean=None, std=None):
    num = len(column_bounds)
    all_discrete_bounds = []

    for column_bound in column_bounds:
        tmp = list(range(int(np.floor(column_bound[1])), int(np.ceil(column_bound[2]) + 1)))
        if column_bound[3] == "PTGENDER":
            pass
        elif column_bound[3] == "PTETHCAT":
            tmp = tmp[0:2]
        elif column_bound[3] == "PTRACCAT":
            tmp = tmp[0:3]
        elif column_bound[3] == "PTMARRY":
            tmp = tmp[0:4]
        tmp = (np.array(tmp) - mean[column_bound[0]].item()) / (std[column_bound[0]].item() + 1e-6)
        all_discrete_bounds.append(tmp.tolist())
    all_combinations = list(itertools.product(*all_discrete_bounds))
    loss_fn = torch.nn.CrossEntropyLoss()
    images_ = []
    discrete_accs = []
    outputs_ = []
    predicted_ = []
    correct = 0
    total = 0
    for combination in all_combinations:  # age =0 marriage = 0
        tmp = image.detach().clone().to(image.device)
        for i in range(num):
            tmp[:, column_bounds[i][0]] = combination[i]
        images_.append(tmp)
    images__ = torch.stack(images_)
    labels__ = true_label.detach().clone().unsqueeze(dim=0).repeat(len(all_combinations), 1).to(true_label.device)
    for batch in range(images__.shape[1]):
        image_tmp = images__[:, batch, :]
        label_tmp = labels__[:, batch]
        if clf != None:
            predicted = clf.predict(image_tmp.cpu().detach().numpy())
            predicted = torch.from_numpy(predicted).to(image_tmp.device)
        else:
            outputs, predicted, discrete_acc = test_model_mlp(
                model,
                image_tmp,
                label_tmp,
            )  # universal perturbation
        # for any testing sample, for example, a sample with perturbed age =1,2,3,4 marrigage=1,2,3
        # take any combination of age and marriage, check the prediction of the model
        # discrete_accs.append(discrete_acc)
        # outputs_.append(outputs)
        # verified true
        # (predicted == label_tmp).all()
        # if not (predicted != label_tmp).any():
        if (predicted == label_tmp).all():
            predicted_.append(true_label[batch])
            correct += 1
        # verified false
        else:
            predicted_.append(1 - true_label[batch])
        total += 1

    return None, torch.stack(predicted_), 1.0 * correct / total
