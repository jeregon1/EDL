import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import prepare_qat, convert, get_default_qat_qconfig
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import ResNet18, ResNet50
from utils import evaluate
import pandas as pd
import os

def distillation_loss(student_logits, teacher_logits, true_labels, T=4.0, alpha=0.7):
    soft_loss = nn.KLDivLoss(reduction="batchmean")(
        nn.functional.log_softmax(student_logits / T, dim=1),
        nn.functional.softmax(teacher_logits / T, dim=1)
    ) * (T * T)
    hard_loss = nn.CrossEntropyLoss()(student_logits, true_labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def train_student_qat(teacher_path="models/teacher_resnet50_96.00acc.pth",
                      acc_target=91.0, max_epochs=150, batch_size=128, lr=0.01,
                      momentum=0.9, weight_decay=5e-4, warmup_epochs=5,
                      T=4.0, alpha=0.7, log_path="log_student_qat.csv"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    rootdir = '/opt/img/effdl-cifar10/'

    if not os.path.exists(rootdir):
        rootdir = './effdl-cifar10/'

    trainset = datasets.CIFAR10(rootdir, train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(rootdir, train=False, download=True, transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Teacher
    teacher = ResNet50().to(device)
    state_dict = torch.load(teacher_path)
    teacher.load_state_dict(state_dict)
    teacher.eval()

    # Student
    student = ResNet18()
    student.qconfig = get_default_qat_qconfig("fbgemm")
    student_prepared = prepare_qat(student.train())

    student_prepared = student_prepared.to(device)

    optimizer = optim.SGD(student_prepared.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = SequentialLR(optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )

    best_acc = 0.0
    best_state = None
    log = {k: [] for k in ["epoch", "train_acc", "test_loss", "test_acc", "lr"]}

    for epoch in range(1, max_epochs + 1) :

        student_prepared.train()
        total, correct = 0, 0

        for inputs, labels in trainloader :
            
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            optimizer.zero_grad()
            student_outputs = student_prepared(inputs)
            loss = distillation_loss(student_outputs, teacher_outputs, labels, T=T, alpha=alpha)
            loss.backward()
            optimizer.step()

            _, predicted = student_outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        test_loss, test_acc = evaluate(student_prepared, testloader, device)
        lr_now = scheduler.get_last_lr()[0]

        print(f"[*] [Epoch {epoch:3d}] LR: {lr_now:.5f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        log["epoch"].append(epoch)
        log["train_acc"].append(train_acc)
        log["test_loss"].append(test_loss)
        log["test_acc"].append(test_acc)
        log["lr"].append(lr_now)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = student_prepared.state_dict()

        if test_acc >= acc_target:
            print(f"[*] Student QAT reached {acc_target}% test accuracy at epoch {epoch}")
            break

        scheduler.step()

    # Convert to quantized model
    student_prepared.cpu()
    quantized_model = convert(student_prepared.eval(), inplace=False)

    os.makedirs("models", exist_ok=True)
    torch.save(quantized_model.state_dict(), "models/student_resnet18_qat_best.pth")
    pd.DataFrame(log).to_csv(log_path, index=False)

    print(f"[+] QAT Training finished. Best Acc: {best_acc:.2f}%")
    print("[+] Quantized model saved to models/student_resnet18_qat_best.pth")
    print(f"[+] Training log saved to {log_path}")

if __name__ == "__main__":
    train_student_qat()

