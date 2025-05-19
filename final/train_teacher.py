import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from models import ResNet50
from utils import load_dataset


def train_teacher(acc_target=96.0, max_epochs=200, batch_size=128, lr=0.1, momentum=0.9, weight_decay=5e-4, num_workers=4, pin_memory=True) :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data augmentation: crop, flip, RandAugment, RandomErasing
    normalize = transforms.Normalize(
        (0.4914,0.4822,0.4465),
        (0.2023,0.1994,0.2010)
    )
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Override load_dataset for teacher
    rootdir = '/opt/img/effdl-cifar10/'
    
    import os
    if not os.path.exists(rootdir):
        rootdir = './effdl-cifar10/'
    
    trainset = torchvision.datasets.CIFAR10(rootdir, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(rootdir, train=False, download=True, transform=test_transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=pin_memory)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    model = ResNet50(groups=1).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_acc = 0.0

    for epoch in range(1, max_epochs+1) :
    
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader :

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            
            total += labels.size(0)

        scheduler.step()
        train_acc = 100.*correct/total
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad() :
        
            for inputs, labels in testloader :
        
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                
                total += labels.size(0)

        test_acc = 100.*correct / total

        print(f"Epoch {epoch:3d}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc :
        
            best_acc = test_acc
        
            if best_acc >= acc_target/2 :
                torch.save(model.state_dict(), f"models/teacher_resnet50_{best_acc:.2f}acc.pth")
        
        if test_acc >= acc_target :
            print(f"Reached target accuracy {acc_target}% at epoch {epoch}.")
            break

    print(f"Training finished. Best Test Acc: {best_acc:.2f}%")


if __name__ == '__main__':
    train_teacher()
