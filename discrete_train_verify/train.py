import copy
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from test import test_model_mlp

from utils import CosineLR
from datetime import datetime
import os
import numpy as np
import shutil


def train_model_logistic(args, model, train_dataloader, x, y, validate_x, validate_y, image, true_label, criterion):
    model.train()
    # args.train_epoch = 50
    # optimizer = optim.SGD(
    #     model.parameters(), lr=0.4, weight_decay=5e-4, momentum=0.9, nesterov=True
    # )  # lr scheduler 0.99 lr decay
    # for mlp earlier, it is max_iter = 1000， args.train_epoch = 1000
    # optimizer = torch.optim.LBFGS(model.parameters(), max_iter=20, lr=0.01)  # args.lr)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=1, lr=0.01)  # args.lr)
    # scheduler = CosineLR(optimizer, max_lr=0.1, epochs=30)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    # print("Start training")
    validate_loss_min = float("inf")
    training_epoch = 0
    val_acc_lst = []
    log_idx_lst = []
    log_step = 10
    today = datetime.now().strftime("%m%d%Y_%H%M%S%f")
    tmp_path = f"./tmp/{today}_tmp_ckpt"
    # model.float()
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    def loss_closure():
        optimizer.zero_grad()
        oupt = model(x)
        loss_val = criterion(oupt, y)
        loss_val.backward()
        return loss_val

    for epoch in range(args.train_epoch):
        # for x, y in train_dataloader:
        model.train()
        if torch.cuda.is_available():
            x = x.to("cuda")
            y = y.to("cuda")
        if isinstance(criterion, nn.BCELoss):
            y = y.to(torch.float32)
        # optimizer.zero_grad()
        # outputs = model(x)
        # loss = criterion(outputs, y)
        # loss.backward()
        optimizer.step(loss_closure)

        scheduler.step()
        train_acc = (model.predict(x) == y).sum().item() / y.shape[0]
        # if (epoch + 1) % 10 == 0 and (epoch + 1) >= int(args.train_epoch // 2):
        if args.debug or (epoch + 1) >= int(args.train_epoch // 2):
            model.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    validate_x = validate_x.to("cuda")
                    validate_y = validate_y.to("cuda")
                validate_pred = model(validate_x)
                if isinstance(criterion, nn.BCELoss):
                    validate_pred = validate_pred
                    validate_y = validate_y.to(torch.float32)
                validate_loss = criterion(validate_pred, validate_y)
                if validate_loss < validate_loss_min:
                    validate_loss_min = validate_loss
                    previous_model = copy.deepcopy(model)
                if isinstance(criterion, nn.BCELoss):
                    validate_acc = (model.predict(validate_x) == validate_y).sum().item() / validate_y.shape[0]
                _, _, test_acc = test_model_mlp(model, image, true_label)
                val_acc_lst.append(validate_acc)
                log_idx_lst.append(epoch + 1)
                torch.save(model.state_dict(), os.path.join(tmp_path, f"{epoch+1}.pt"))
                if args.debug:
                    print(
                        f"train_acc {train_acc*100:.2f} valid_acc {validate_acc*100:.2f} test_acc {test_acc*100:.2f} lr {scheduler.get_last_lr()[0]:.4f}"
                    )
        training_epoch = epoch + 1
    # print("Finished training")
    idx = np.argmax(val_acc_lst)
    best_epoch = log_idx_lst[idx]
    model.load_state_dict(torch.load(os.path.join(tmp_path, f"{best_epoch}.pt")))
    shutil.rmtree(tmp_path)
    torch.save(model.state_dict(), args.save_path)
    print(f"training_epoch: {training_epoch}")
    return model, training_epoch


def train_model_mlp3(args, model, train_dataloader, validate_x, validate_y, image, true_label, criterion):
    model.train()
    # args.train_epoch = 1000
    # for mlp earlier, args.train_epoch = 1000
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # , momentum=0.9, nesterov=True
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)  # , momentum=0.9, nesterov=True)
    # )  # lr scheduler 0.99 lr decay
    # scheduler = CosineLR(optimizer, max_lr=0.1, epochs=int(args.train_epoch))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)
    # print("Start training")
    training_epoch = 0
    val_acc_lst = []
    log_idx_lst = []
    log_step = 10
    today = datetime.now().strftime("%m%d%Y_%H%M%S%f")
    tmp_path = f"./tmp/{today}_tmp_ckpt"
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    for epoch in range(args.train_epoch):
        model.train()
        correct = 0
        total = 0
        for x, y in train_dataloader:
            if torch.cuda.is_available():
                x = x.to("cuda")
                y = y.to("cuda")
            optimizer.zero_grad()
            outputs = model(x)
            if isinstance(criterion, nn.BCELoss):
                outputs = outputs
                y = y.to(torch.float32)
            loss = criterion(outputs, y)
            l2_lambda = 0.005
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

            loss = loss # + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            if isinstance(criterion, nn.BCELoss):
                correct = (model.predict(x) == y).sum().item()
            else:
                correct += (outputs.max(1)[1] == y).sum().item()
            total += outputs.shape[0]
            train_acc = correct / total
        scheduler.step()
        # acc*3, lr

        # if (epoch + 1) % 10 == 0 and (epoch + 1) >= int(args.train_epoch // 2):
        if args.debug or (epoch + 1) >= int(args.train_epoch // 2):
            model.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    validate_x = validate_x.to("cuda")
                    validate_y = validate_y.to("cuda")
                validate_pred = model(validate_x)
                if isinstance(criterion, nn.BCELoss):
                    validate_pred = validate_pred
                    validate_y = validate_y.to(torch.float32)
                validate_loss = criterion(validate_pred, validate_y)
                # print("validate_loss", validate_loss)
                if isinstance(criterion, nn.BCELoss):
                    validate_acc = (model.predict(validate_x) == validate_y).sum().item() / validate_y.shape[0]
                else:
                    validate_acc = (validate_pred.max(1)[1] == validate_y).sum().item() / validate_y.shape[0]
                _, _, test_acc = test_model_mlp(model, image, true_label)
                val_acc_lst.append(validate_acc)
                log_idx_lst.append(epoch + 1)
                torch.save(model.state_dict(), os.path.join(tmp_path, f"{epoch+1}.pt"))
                if args.debug:
                    print(
                        f"train_acc {train_acc*100:.2f} valid_acc {validate_acc*100:.2f} test_acc {test_acc*100:.2f} lr {scheduler.get_last_lr()[0]:.4f}"
                    )
        training_epoch = epoch + 1
    # print("Finished training")
    idx = np.argmax(val_acc_lst)
    best_epoch = log_idx_lst[idx]
    model.load_state_dict(torch.load(os.path.join(tmp_path, f"{best_epoch}.pt")))
    shutil.rmtree(tmp_path)
    torch.save(model.state_dict(), args.save_path)
    print(f"training_epoch: {training_epoch}")
    return model, training_epoch


def train_model_mlp6(args, model, train_dataloader, validate_x, validate_y, image, true_label, criterion):
    model.train()
    # args.train_epoch = 1000
    # for mlp earlier, args.train_epoch = 1000
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # , momentum=0.9, nesterov=True
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # , momentum=0.9, nesterov=True)
    # )  # lr scheduler 0.99 lr decay
    # scheduler = CosineLR(optimizer, max_lr=0.1, epochs=int(args.train_epoch))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.99)
    # print("Start training")
    training_epoch = 0
    val_acc_lst = []
    log_idx_lst = []
    log_step = 10
    today = datetime.now().strftime("%m%d%Y_%H%M%S%f")
    tmp_path = f"./tmp/{today}_tmp_ckpt"
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    for epoch in range(args.train_epoch):
        model.train()
        correct = 0
        total = 0
        for x, y in train_dataloader:
            if torch.cuda.is_available():
                x = x.to("cuda")
                y = y.to("cuda")
            optimizer.zero_grad()
            outputs = model(x)
            if isinstance(criterion, nn.BCELoss):
                outputs = outputs
                y = y.to(torch.float32)
            loss = criterion(outputs, y)
            l2_lambda = 0.005
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

            loss = loss # + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            if isinstance(criterion, nn.BCELoss):
                correct = (model.predict(x) == y).sum().item()
            else:
                correct += (outputs.max(1)[1] == y).sum().item()
            total += outputs.shape[0]
            train_acc = correct / total
        scheduler.step()
        # acc*3, lr

        # if (epoch + 1) % 10 == 0 and (epoch + 1) >= int(args.train_epoch // 2):
        if args.debug or (epoch + 1) >= int(args.train_epoch // 2):
            model.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    validate_x = validate_x.to("cuda")
                    validate_y = validate_y.to("cuda")
                validate_pred = model(validate_x)
                if isinstance(criterion, nn.BCELoss):
                    validate_pred = validate_pred
                    validate_y = validate_y.to(torch.float32)
                validate_loss = criterion(validate_pred, validate_y)
                # print("validate_loss", validate_loss)
                if isinstance(criterion, nn.BCELoss):
                    validate_acc = (model.predict(validate_x) == validate_y).sum().item() / validate_y.shape[0]
                else:
                    validate_acc = (validate_pred.max(1)[1] == validate_y).sum().item() / validate_y.shape[0]
                _, _, test_acc = test_model_mlp(model, image, true_label)
                val_acc_lst.append(validate_acc)
                log_idx_lst.append(epoch + 1)
                torch.save(model.state_dict(), os.path.join(tmp_path, f"{epoch+1}.pt"))
                if args.debug:
                    print(
                        f"train_acc {train_acc*100:.2f} valid_acc {validate_acc*100:.2f} test_acc {test_acc*100:.2f} lr {scheduler.get_last_lr()[0]:.4f}"
                    )
        training_epoch = epoch + 1
    # print("Finished training")
    idx = np.argmax(val_acc_lst)
    best_epoch = log_idx_lst[idx]
    model.load_state_dict(torch.load(os.path.join(tmp_path, f"{best_epoch}.pt")))
    shutil.rmtree(tmp_path)
    torch.save(model.state_dict(), args.save_path)
    print(f"training_epoch: {training_epoch}")
    return model, training_epoch
