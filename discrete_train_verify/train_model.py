#!/usr/bin/env python
import argparse
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from models import LogisticRegressionModel, mlp_3_layer, mlp_4_layer, mlp_6_layer
from test import test_model_mlp
from train import train_model_logistic, train_model_mlp3, train_model_mlp6
from utils import set_seed



pd.options.mode.chained_assignment = None  # default='warn'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_arg_parser():
    """get arguments"""
    parser = argparse.ArgumentParser(description="fanirness verification with mlp models")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="mlp3",
        choices=["mlp3", "mlp4", "mlp6", "logistic", "rnn1", "rnn2"],
        help="model type",
    )
    parser.add_argument("--save-path", "-s", type=str, default="ckpt.pth", help="save path")
    parser.add_argument(
        "--discrete",
        "-d",
        action="store_true",
        help="use discrete verification instead of continuous",
    )
    parser.add_argument(
        "--dataset-split",
        "-ds",
        type=int,
        default=0,
        choices=[0, 1, -1],
        help="0 for NL=0, MCI=1, 1 for MCI=0, Dementia=1, -1 means no dataset split",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spreadsheet", default="./fairness_data/TADPOLE_D1_D2.csv")
    parser.add_argument("--features", default="./fairness_data/features")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--train-epoch", type=int, default=100)
    parser.add_argument("--batch-size", "-bs", type=int, default=31, help="batch size")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--repeat", type=str2bool, default=False, help="oversample minority class to balance dataset or not") 
    args, _ = parser.parse_known_args()
    args.model = f"{args.model}_ckpt.pth" if args.model == "ckpt.pth" else args.model
    # args.debug = True
    return args


def main(args):
    args.discrete = True
    args.debug = True
    set_seed(args.seed)
    # records = {}
    # today = datetime.now().strftime("%m%d%Y_%H%M%S")
    # mode = "disc" if args.discrete else "cont"
    # save_path = f"run5/{today}_model_{args.model}_split_{args.dataset_split}_log_{mode}"
    print(f"Model: {args.model}")
    print(f"Verification Mode: {'Discrete' if args.discrete else 'Continuous'}")
    # NL=0, MCI=1, Dementia=2
    print(f"classes: {'NL=0, MCI=1' if args.dataset_split==0 else 'MCI=0, Dementia=1'}")
    num_classes = 3 if args.dataset_split == -1 else 2

    # patients, labels, groups, generator, splits_num = intialize_generator_mlp(args)
    ######################### set input range ####################################

    # reintilize generator
    # patients, labels, groups, generator, splits_num = intialize_generator_mlp(args)
    splits_num = 10

    for n_split in tqdm(range(splits_num)):
        k_fold_data = torch.load(
            f"./k_fold_rent/data_{args.dataset_split}_fold_{n_split}.pth",
        )
        train_idx, val_idx, test_idx, patients, labels = (
            k_fold_data["train_idx"],
            k_fold_data["val_idx"],
            k_fold_data["test_idx"],
            k_fold_data["patients"],
            k_fold_data["labels"],
        )
        # train_idx, val_idx, test_idx = iterate_through_generator(patients, labels, groups, generator)
        ################ prepare test image and labels ###############################
        # transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=mean, std=std)])
        if args.repeat:
            # oversample minority class to balance dataset
            unchanged_mask,changed_mask = labels[train_idx]==0,labels[train_idx]==1
            ratio_changed = int(np.round(unchanged_mask.sum()/changed_mask.sum()))
            ratio_unchanged = int(np.round(changed_mask.sum()/unchanged_mask.sum()))
            if ratio_changed>1:
                train_idx = np.append(np.repeat(train_idx[changed_mask],ratio_changed), train_idx[unchanged_mask])
                np.random.shuffle(train_idx)
            elif ratio_unchanged>1:
                train_idx = np.append(np.repeat(train_idx[unchanged_mask],ratio_unchanged), train_idx[changed_mask])
                np.random.shuffle(train_idx)           
        mean, std = torch.mean(patients[train_idx], dim=0), torch.std(patients[train_idx], dim=0)
        patients = (patients - mean) / (std + 1e-6)
        # train_dataset = TensorDataset((patients[train_idx] - mean) / (std + 1e-6), labels[train_idx])
        # transforms.normalize
        train_dataset = TensorDataset(patients[train_idx], labels[train_idx])

        # class weighting
        # labels_unique, counts = torch.unique(labels[train_idx], return_counts=True)
        # print("Unique labels : {}".format(labels_unique))
        # class_weights = [sum(counts) / c for c in counts]
        # # assign weight to each input sample
        # example_weights = [class_weights[e] for e in labels[train_idx]]
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(
        #     example_weights, len(labels[train_idx]), replacement=True
        # )

        class_sample_count = torch.tensor(
            [(labels[train_idx] == t).sum() for t in torch.unique(labels[train_idx], sorted=True)]
        )
        weight = 1.0 / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in labels[train_idx]])

        # Create sampler, dataset, loader
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # from torchsampler import ImbalancedDatasetSampler
        # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ImbalancedDatasetSampler(train_dataset))
        # image = (patients[test_idx] - mean) / (std + 1e-6)
        image = patients[test_idx]
        true_label = labels[test_idx]
        if torch.cuda.is_available():
            image = image.to("cuda")
            true_label = true_label.to("cuda")
        # print("Running on", image.device)
        ############ prepare model ###############
        if args.model == "mlp3":
            model = mlp_3_layer(num_classes=num_classes)
            criterion = torch.nn.CrossEntropyLoss()
        elif args.model == "mlp4":
            model = mlp_4_layer(num_classes=num_classes)
            criterion = torch.nn.CrossEntropyLoss()
        elif args.model == "mlp6":
            model = mlp_6_layer(num_classes=num_classes)
            criterion = torch.nn.CrossEntropyLoss()
        elif args.model == "logistic":
            model = LogisticRegressionModel(1)
            criterion = torch.nn.BCELoss()
        ############## model test #########################
        if torch.cuda.is_available():
            model = model.to("cuda")

        if args.model == "logistic":  # args.train:
            model, training_epoch = train_model_logistic(
                args,
                model,
                train_dataloader,
                patients[train_idx],
                labels[train_idx],
                patients[val_idx],
                labels[val_idx],
                image,
                true_label,
                criterion,
            )
        elif args.model == "mlp3":
            model, training_epoch = train_model_mlp3(
                args,
                model,
                train_dataloader,
                patients[val_idx],
                labels[val_idx],
                image,
                true_label,
                criterion,
            )
        elif args.model == "mlp6":
            model, training_epoch = train_model_mlp6(
                args,
                model,
                train_dataloader,
                patients[val_idx],
                labels[val_idx],
                image,
                true_label,
                criterion,
            )
        # model.load_state_dict(torch.load(args.save_path))
        model.eval()
        outputs, predicted, acc = test_model_mlp(model, image, true_label)
        print(f"Accuracy of the network on test images: {100 * acc:.4f} %")
        ckpt = {
            "model": model.state_dict(),
            "acc": acc,
            "outputs": outputs,
            "predicted": predicted,
            "test_x": image,
            "test_y": true_label,
            "mean": mean,
            "std": std,
        }

        torch.save(
            ckpt,
            f"./ckpt2/model_{args.model}_data_{args.dataset_split}_fold_{n_split}.pth",
        )
        # break


if __name__ == "__main__":
    args = get_arg_parser()
    main(args)
