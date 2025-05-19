import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils.prune as prune
import pandas as pd

def load_dataset(batch_size=32, num_workers=4, pin_memory=True):
    normalize_scratch = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        normalize_scratch,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize_scratch,
    ])
    rootdir = '/opt/img/effdl-cifar10/'
    import os
    if not os.path.exists(rootdir):
        rootdir = './effdl-cifar10/'
    c10train = torchvision.datasets.CIFAR10(rootdir, train=True, download=True, transform=transform_train)
    c10test = torchvision.datasets.CIFAR10(rootdir, train=False, download=True, transform=transform_test)
    trainloader = DataLoader(
        c10train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    testloader = DataLoader(
        c10test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return trainloader, testloader

def evaluate(net, dataloader, device):
    net.eval()
    correct, total, test_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100. * correct / total
    return test_loss / len(dataloader), acc

def train(model, trainloader, testloader, optimizer, scheduler, device, epochs=10, log_file=None):
    criterion = nn.CrossEntropyLoss()
    log = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
    }
    for epoch in range(epochs):
        model.train()
        total, correct, loss_total = 0, 0, 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss_total += loss.item()
        train_acc = 100. * correct / total
        test_loss, test_acc = evaluate(model, testloader, device)
        lr = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch+1:2d}] LR: {lr:.5f} | "
              f"Train Loss: {loss_total:.3f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")
        log["epoch"].append(epoch+1)
        log["train_loss"].append(loss_total)
        log["train_acc"].append(train_acc)
        log["test_loss"].append(test_loss)
        log["test_acc"].append(test_acc)
        log["lr"].append(lr)
        scheduler.step()
    if log_file:
        df = pd.DataFrame(log)
        df.to_csv(log_file, index=False)
        print(f"Training log saved to {log_file}")
    return log

def log_model_results(logfile, model_name, prune_type, prune_ratio, test_acc, test_loss, score):
    import os
    log_entry = {
        "model": model_name,
        "prune_type": prune_type,
        "prune_ratio": prune_ratio,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "score": score
    }
    df = pd.DataFrame([log_entry])
    if not os.path.exists(logfile):
        df.to_csv(logfile, index=False)
    else:
        df.to_csv(logfile, mode='a', header=False, index=False)

def prune_and_evaluate(model, dataloader, device, ratios, logfile="model_results.csv"):
    import copy
    model_name = model.__class__.__name__
    for amount in ratios:
        # Unstructured pruning
        pruned_model = copy.deepcopy(model)
        parameters_to_prune = [(m, 'weight') for m in pruned_model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        test_loss, test_acc = evaluate(pruned_model, dataloader, device)
        score = compute_score(pruned_model)
        print(f"Prune {amount*100:.0f}% (Unstructured) | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}% | Score: {score:.4f}")
        log_model_results(logfile, model_name, "unstructured", amount, test_acc, test_loss, score)
        # Structured pruning (filters for Conv2d, output neurons for Linear)
        pruned_structured = copy.deepcopy(model)
        for m in pruned_structured.modules():
            if isinstance(m, nn.Conv2d):
                prune.ln_structured(m, name='weight', amount=amount, n=2, dim=0)  # Prune filters
            elif isinstance(m, nn.Linear):
                prune.ln_structured(m, name='weight', amount=amount, n=2, dim=0)  # Prune output neurons
        test_loss_s, test_acc_s = evaluate(pruned_structured, dataloader, device)
        score_s = compute_score(pruned_structured)
        print(f"Prune {amount*100:.0f}% (Structured) | Test Loss: {test_loss_s:.3f} | Test Acc: {test_acc_s:.2f}% | Score: {score_s:.4f}")
        log_model_results(logfile, model_name, "structured", amount, test_acc_s, test_loss_s, score_s)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def compute_score(model, input_size=(1,3,32,32), q_w=None, q_a=None):
    w = sum(p.numel() for p in model.parameters())
    zero_w = sum(int((p == 0).sum()) for p in model.parameters())
    p_u = zero_w / w
    total_filters, zero_filters = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight = m.weight.data.abs()
            axes = (1,2,3) if isinstance(m, nn.Conv2d) else (1,)
            sums = weight.sum(dim=axes)
            zero_filters += (sums == 0).sum().item()
            total_filters += sums.numel()
    p_s = zero_filters / total_filters if total_filters>0 else 0
    bits = torch.finfo(next(model.parameters()).dtype).bits
    if q_w is None:
        q_w = bits
    if q_a is None:
        q_a = bits
    from thop import profile
    if profile is None:
        raise ImportError("thop not installed, cannot compute flops automatically")
    dummy = torch.randn(input_size).to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    was_training = model.training
    model.eval()
    profile_result = profile(model, inputs=(dummy,), verbose=False)
    if was_training:
        model.train()
    if len(profile_result) == 2:
        flops, _ = profile_result
    else:
        flops, _, _ = profile_result
    macs = flops
    param_score = (1 - (p_s + p_u)) * (q_w / 32) * w / 5.6e6
    ops_score = (1 - p_s) * (max(q_w, q_a) / 32) * macs / 2.8e8
    print(f"Parameters: {w} | MACs: {macs:.2e} | "
          f"Pruning Ratio (unstruct): {p_u:.2f} | "
          f"Pruning Ratio (struct): {p_s:.2f} | "
          f"Bitwidth (weights): {q_w} | Bitwidth (activations): {q_a}")
    print(f"Score (params): {param_score:.4f} | "
          f"Score (ops): {ops_score:.4f}")
    print(f"Efficiency Score (total): {param_score + ops_score:.4f}")
    return param_score + ops_score
