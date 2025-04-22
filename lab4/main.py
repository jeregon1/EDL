'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from numpy import save
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import copy
import torch.nn.utils.prune as prune

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

# CIFAR10 data loading and preprocessing
normalize_scratch = transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2023, 0.1994, 0.2010)
)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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
trainloader = DataLoader(c10train, batch_size=32, shuffle=True)
testloader = DataLoader(c10test, batch_size=32)

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

def train(model, trainloader, testloader, optimizer, scheduler, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
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
        print(f"[Epoch {epoch+1:2d}] LR: {scheduler.get_last_lr()[0]:.5f} | "
              f"Train Loss: {loss_total:.3f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")
        scheduler.step()

def prune_and_evaluate(model, dataloader, device, ratios):
    for amount in ratios:
        pruned_model = copy.deepcopy(model)
        parameters_to_prune = [(m, 'weight') for m in pruned_model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        test_loss, test_acc = evaluate(pruned_model, dataloader, device)
        print(f"Prune {amount*100:.0f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")

def save_model(model, path):
    """Save the model's state_dict to the given path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

from thop import profile

def compute_score(model, input_size=(1,3,32,32)):
    """Auto-compute efficiency score: parameters, MACs, bitwidths, pruning ratios."""
    # total weights and unstructured pruning ratio
    w = sum(p.numel() for p in model.parameters())
    zero_w = sum(int((p == 0).sum()) for p in model.parameters())
    p_u = zero_w / w
    # structured pruning ratio: fraction of completely zeroed filters/neurons
    total_filters, zero_filters = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight = m.weight.data.abs()
            axes = (1,2,3) if isinstance(m, nn.Conv2d) else (1,)
            sums = weight.sum(dim=axes)
            zero_filters += (sums == 0).sum().item()
            total_filters += sums.numel()
    p_s = zero_filters / total_filters if total_filters>0 else 0
    # bitwidth from dtype
    bits = torch.finfo(next(model.parameters()).dtype).bits
    q_w = q_a = bits
    # compute MACs
    if profile is None:
        raise ImportError("thop not installed, cannot compute flops automatically")
    dummy = torch.randn(input_size).to(next(model.parameters()).device)
    model.eval()
    profile_result = profile(model, inputs=(dummy,), verbose=False)
    if len(profile_result) == 2:
        flops, _ = profile_result
    else:
        flops, _, _ = profile_result
    macs = flops / 2
    # compute scores
    param_score = (1 - (p_s + p_u)) * (q_w / 32) * w / 5.6e6
    ops_score = (1 - p_s) * (q_a / 32) * macs / 2.8e8
    return param_score + ops_score

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ResNet18().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    train(net, trainloader, testloader, optimizer, scheduler, device, epochs=20)
    save_model(net, 'resnet18_cifar10_20.pth')
    prune_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
    prune_and_evaluate(net, testloader, device, prune_ratios)
