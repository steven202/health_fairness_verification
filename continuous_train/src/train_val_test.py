import torch


def train(model, train_loader, criterion, optimizer, device, L1_regularization=False):
    model.train()
    total_correct = 0
    total_samples = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if L1_regularization:
            # add l1 regularization with lambda = 0.001
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + l1 * 0.00005
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == target).sum().item()
        total_samples += target.size(0)
    return 100 * total_correct / total_samples


def validate(model, valid_loader, criterion, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
    return 100 * total_correct / total_samples


def test(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
    return 100 * total_correct / total_samples