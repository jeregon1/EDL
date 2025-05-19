import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models import ResNet50, EfficientResNet18
from utils import evaluate
import pandas as pd
import os


def distillation_loss(student_logits, teacher_logits, true_labels, T=4.0, alpha=0.7):
    """
    Distillation loss: KL divergence (soft labels) + CrossEntropy (hard labels)
    """
    soft_loss = nn.KLDivLoss(reduction="batchmean")(
        nn.functional.log_softmax(student_logits / T, dim=1),
        nn.functional.softmax(teacher_logits / T, dim=1)
    ) * (T * T)

    hard_loss = nn.CrossEntropyLoss()(student_logits, true_labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


def train_student(teacher_path="models/teacher_resnet50_96.00acc.pth",
                  acc_target=91.0, max_epochs=150, batch_size=128, lr=0.1,
                  momentum=0.9, weight_decay=5e-4, warmup_epochs=5,
                  T=4.0, alpha=0.7, patience=10,
                  log_path="log_student.csv"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    rootdir = '/opt/img/effdl-cifar10/'
    if not os.path.exists(rootdir):
        rootdir = './effdl-cifar10/'

    # Datasets and loaders
    trainset = datasets.CIFAR10(rootdir, train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(rootdir, train=False, download=True, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Models
    teacher = ResNet50(groups=1).to(device)
    state_dict = torch.load(teacher_path, weights_only=True)
    teacher.load_state_dict(state_dict)
    teacher.eval()

    student = EfficientResNet18().to(device)

    # Optim & schedulers
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = SequentialLR(optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )

    scaler = GradScaler()
    best_acc = 0.0
    best_state = None
    epochs_since_improvement = 0

    log = {
        "epoch": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
    }

    for epoch in range(1, max_epochs + 1):
        student.train()
        total, correct = 0, 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                with autocast():
                    teacher_outputs = teacher(inputs)

            optimizer.zero_grad()

            with autocast():
                student_outputs = student(inputs)
                loss = distillation_loss(student_outputs, teacher_outputs, labels, T=T, alpha=alpha)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = student_outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        test_loss, test_acc = evaluate(student, testloader, device)
        lr_now = scheduler.get_last_lr()[0]

        print(f"[*] [Epoch {epoch:3d}] LR: {lr_now:.5f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        log["epoch"].append(epoch)
        log["train_acc"].append(train_acc)
        log["test_loss"].append(test_loss)
        log["test_acc"].append(test_acc)
        log["lr"].append(lr_now)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = student.state_dict().copy()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if test_acc >= acc_target:
            print(f"[*] Student reached {acc_target}% test accuracy at epoch {epoch}")
            break

        if epochs_since_improvement >= patience:
            print(f"[!] Early stopping: no improvement for {patience} epochs (Best Acc: {best_acc:.2f}%)")
            break

        scheduler.step()

    # Save log and best model
    pd.DataFrame(log).to_csv(log_path, index=False)
    print(f"[+] Training finished. Best Acc: {best_acc:.2f}%. Log saved to {log_path}.")

    if best_state:
        os.makedirs("models", exist_ok=True)
        torch.save(best_state, "models/student_resnet18_best.pth")
        print(f"[+] Best model saved to models/student_resnet18_best.pth")


if __name__ == "__main__":
    train_student()
