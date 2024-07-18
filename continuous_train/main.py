#!/usr/bin/env python
# write function get_verified_biased_fair_indices_torch
import argparse
import copy
import pickle
from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import sk2torch
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from src.dataset import intialize_generator_mlp, iterate_through_generator
from src.models import MLP3Layer, MLP6Layer
from src.train_val_test import test, train, validate
from src.utils import seed_func
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

def get_arg_parser():
    """Get command-line arguments."""
    parser = argparse.ArgumentParser(description="Continuous verification model training")
    parser.add_argument("--batch-size", "-bs", type=int, default=64, help="Batch size")
    parser.add_argument("--model", "-m", type=str, default="mlp3", choices=["mlp3", "mlp6", "logistic"], help="Model type")
    parser.add_argument("--dataset-split", "-ds", type=int, default=-1, choices=[0, 1, -1], help="Dataset split option")
    parser.add_argument("--save-path", "-s", type=str, default="checkpoints_tmp", help="Path to save model checkpoints")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--spreadsheet", type=str, default="./fairness_data/TADPOLE_D1_D2.csv", help="Path to spreadsheet")
    parser.add_argument("--features", type=str, default="./fairness_data/features", help="Path to features")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds for cross-validation")
    parser.add_argument("--outdir", type=str, default="output", help="Output directory")
    parser.add_argument("--eps", type=float, default=0.3, help="Epsilon value")
    parser.add_argument("--train-epoch", type=int, default=1700, help="Number of training epochs")
    parser.add_argument("--split-data", action="store_true", help="Flag to split data")
    parser.add_argument("--val", action="store_true", help="Flag to enable validation")
    parser.add_argument("--init", action="store_true", default=True, help="Flag to initialize weights")
    parser.add_argument("--comb", action="store_true", help="Flag for combination")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument('--learning-rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--beta1', type=float, default=0.95, help="Beta1 for Adam optimizer")
    parser.add_argument('--beta2', type=float, default=0.98, help="Beta2 for Adam optimizer")
    parser.add_argument('--T-max', type=int, default=100, help="Maximum number of iterations for scheduler")
    parser.add_argument('--eta-min', type=float, default=1e-5, help="Minimum learning rate for scheduler")
    parser.add_argument('--train',action="store_true", help="Flag to train model")
    parser.add_argument('--L1',action="store_true", help="Flag to L1 regularization")
    parser.add_argument("--verified-result-dir", default="write_clean_verified_result_dir")
    return parser.parse_args()

def get_verified_biased_fair_indices_torch(torch_model, patients, labels, test_idx):
    torch_model.eval()
    test_data = patients[test_idx]
    test_labels = labels[test_idx]
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.tensor(test_data, dtype=torch.float32)
    if not isinstance(test_labels, torch.Tensor):
        test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    device = next(torch_model.parameters()).device
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    
    with torch.no_grad():
        outputs = torch_model(test_data)
        _, predicted = torch.max(outputs, 1)
    
    correct = (predicted == test_labels).cpu().numpy()
    incorrect = (predicted != test_labels).cpu().numpy()
    # test_idx is list
    test_idx = np.array(test_idx)
    verified_biased_indices = test_idx[incorrect]
    verified_fair_indices = test_idx[correct]
    # convert to list, sorted
    verified_biased_indices = sorted(verified_biased_indices.tolist())
    verified_fair_indices = sorted(verified_fair_indices.tolist())
    return verified_biased_indices, verified_fair_indices
def save_verified_indices(args, model, patients, labels, test_idx, n_split):
    verified_biased, verified_fair = get_verified_biased_fair_indices_torch(model, patients, labels, test_idx)
    os.makedirs(os.path.join(args.save_path, args.verified_result_dir, args.model), exist_ok=True)
    with open(os.path.join(args.save_path, args.verified_result_dir, args.model, f"fold_{n_split}.json"), "w") as f:
        json.dump({"verified_biased": sorted(verified_biased), "verified_fair": sorted(verified_fair)}, f)

def main(args):
    seed_func(args.seed)
    num_classes = 3 if args.dataset_split == -1 else 2
    print(f"Model: {args.model}, Dataset Split: {args.dataset_split}, Num of Classes: {num_classes}, Seed: {args.seed}")

    patients, labels, groups, generator, splits_num = intialize_generator_mlp(args)
    # save labels to json file
    with open(os.path.join(args.save_path, "labels.json"), "w") as f:
        json.dump(labels.tolist(), f)
    accuracies = []
    os.makedirs(args.save_path, exist_ok=True)

    if args.split_data:
        data_splits = {}
        for n_split in tqdm(range(splits_num)):
            train_idx, val_idx, test_idx = iterate_through_generator(patients, labels, groups, generator)
            train_idx, val_idx, test_idx = train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
            data_splits[n_split] = {"train": train_idx, "val": val_idx, "test": test_idx}
        with open(args.save_path + "/data_split_indices.json", "w") as f:
            json.dump(data_splits, f)
        return
    
    os.makedirs(os.path.join(args.save_path, f"{args.model}"), exist_ok=True)
    with open(os.path.join(args.save_path, "data_split_indices.json"), "r") as f:
        data_splits = json.load(f)
    for n_split in tqdm(range(splits_num)):
        train_idx = data_splits[str(n_split)]["train"]
        val_idx = data_splits[str(n_split)]["val"]
        test_idx = data_splits[str(n_split)]["test"]
        if args.model == "mlp3":
            model = MLP3Layer(num_classes=num_classes, input_dim=patients.shape[1], hidden_size=args.hidden_dim)
        elif args.model == "mlp6":
            model = MLP6Layer(num_classes=num_classes, input_dim=patients.shape[1], hidden_size=args.hidden_dim)
        elif args.model == "logistic":
            if args.train:
                logistic = LogisticRegression(max_iter=args.train_epoch, random_state=42, penalty='l1', C=0.001, solver='liblinear')
                logistic.fit(patients[train_idx], labels[train_idx])
                with open(os.path.join(args.save_path, f"{args.model}", f"split_{n_split}.pkl"), "wb") as f:
                    pickle.dump(logistic, f)
                test_acc = 100 * logistic.score(patients[test_idx], labels[test_idx])
            with open(os.path.join(args.save_path, f"{args.model}", f"split_{n_split}.pkl"), "rb") as f:
                logistic = pickle.load(f)
            test_acc_2 = 100 * logistic.score(patients[test_idx], labels[test_idx])
            ###################### convert sklearn model to torch model ############################
            torch_logistic = LogisticRegressionTorch(patients.shape[1], num_classes)
            torch_logistic.linear.weight.data = torch.tensor(logistic.coef_, dtype=torch.float32)
            torch_logistic.linear.bias.data = torch.tensor(logistic.intercept_, dtype=torch.float32)
            torch_logistic.eval()
            test_acc_3 = test(torch_logistic, DataLoader(TensorDataset(patients[test_idx], labels[test_idx]), batch_size=args.batch_size, shuffle=False), 'cpu')
            torch.save(torch_logistic, os.path.join(args.save_path, f"{args.model}", f"split_{n_split}.pth"))
            save_verified_indices(args, torch_logistic, patients, labels, test_idx, n_split)
            # check acc3 and acc2 are same
            assert np.isclose(test_acc_2, test_acc_3), f"Model conversion failed for fold {n_split}"
            # save logistic_torch model to onnx
            torch_logistic.to('cpu')
            input_tensor = patients[test_idx].to('cpu')[0].reshape(1, patients.shape[1])
            torch.onnx.export(torch_logistic, input_tensor, os.path.join(args.save_path, f"{args.model}", f"torch_split_{n_split}.onnx"))
            if not args.train:
                print(f"Fold {n_split}, Test Accuracy: {test_acc_3:.4f}")
                accuracies.append(test_acc_3)
            if args.train:
                assert np.isclose(test_acc, test_acc_2), f"Model loading failed for fold {n_split}"
                print(f"Final Fold {n_split}, Test Accuracy: {test_acc:.4f}")
                accuracies.append(test_acc)
                # initial_type = [('float_input', FloatTensorType([None, patients.shape[1]]))]
                # onx = convert_sklearn(logistic, initial_types=initial_type)
                # with open(os.path.join(args.save_path, f"{args.model}", f"split_{n_split}.onnx"), "wb") as f:
                #     f.write(onx.SerializeToString())         
            # torch_logistic = sk2torch.wrap(logistic)
            # torch.save(torch_logistic, os.path.join(args.save_path, f"{args.model}", f"sk_torch_split_{n_split}.pth"))
            # torch_logistic.to('cpu')
            # input_tensor = patients[test_idx].to('cpu')
            # torch.onnx.export(torch_logistic, input_tensor, os.path.join(args.save_path, f"{args.model}", f"sk_torch_split_{n_split}.onnx"))
            continue
        ############## model training #########################
        model_ = copy.deepcopy(model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model_ = model_.to(device)

        os.makedirs(os.path.join(args.save_path, f"{args.model}"), exist_ok=True)

        train_dataset = TensorDataset(patients[train_idx], labels[train_idx])
        if args.comb:
            train_dataset = TensorDataset(patients[train_idx+val_idx], labels[train_idx+val_idx])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        val_dataset = TensorDataset(patients[val_idx], labels[val_idx])
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = TensorDataset(patients[test_idx], labels[test_idx])
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

        best_acc = 0
        best_model_state = None
        best_acc_history = []

        if args.init:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
        if args.train:
            for epoch in range(args.train_epoch - 1):
                train_acc = train(model, train_loader, criterion, optimizer, device, args.L1)
                val_acc = validate(model, val_loader, criterion, device)
                scheduler.step()
                best_acc_history.append(train_acc)

                if args.val:
                    if val_acc > best_acc:# and epoch != len(range(args.train_epoch)) - 1:
                        best_acc = val_acc
                        model.eval()
                        best_model_state = model.state_dict()
                        best_epoch = epoch
                        torch.save(model.state_dict(), os.path.join(args.save_path, f"{args.model}", f"split_{n_split}.pth"))
                else:  
                    if True:# and epoch != len(range(args.train_epoch)) - 1: # train_acc > best_acc
                        best_acc = train_acc
                        model.eval()
                        best_model_state = model.state_dict()
                        best_epoch = epoch
                        torch.save(model.state_dict(), os.path.join(args.save_path, f"{args.model}", f"split_{n_split}.pth"))
                ######### model testing ##########
                model_.load_state_dict(model.state_dict())
                test_acc = test(model_, test_loader, device)
                print(f'Fold {n_split}, Epoch {epoch+1}, Train Accuracy: {train_acc:.4f}, Valid Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}, lr: {scheduler.get_last_lr()}')
        if args.train:
            model.load_state_dict(best_model_state)
            test_acc = test(model, test_loader, device)
            print(f"Final Fold {n_split}, Best Epoch {best_epoch + 1}, Test Accuracy: {test_acc:.4f}\n")
            accuracies.append(test_acc)
            torch.save(best_model_state, os.path.join(args.save_path, f"{args.model}", f"split_{n_split}.pth"))
        model.load_state_dict(torch.load(os.path.join(args.save_path, f"{args.model}", f"split_{n_split}.pth")))
        test_acc_2 = test(model, test_loader, device)
        save_verified_indices(args, model, patients, labels, test_idx, n_split)
        if not args.train:
                print(f"Fold {n_split}, Test Accuracy: {test_acc_2:.4f}")
                accuracies.append(test_acc_2)
        if args.train:
            assert np.isclose(test_acc, test_acc_2), f"Model loading failed for fold {n_split}"
        model.to('cpu')
        input_tensor = patients[test_idx].to('cpu')[0].reshape(1, patients.shape[1])
        torch.onnx.export(model, input_tensor, os.path.join(args.save_path, f"{args.model}", f"split_{n_split}.onnx"))
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")

if __name__ == "__main__":
    args = get_arg_parser()
    main(args)
