import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from torch.quantization import quantize_dynamic

# ========== Model Definition ==========

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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

# ========== Dataset Loading ==========

def load_dataset():
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    rootdir = './effdl-cifar10/'
    c10train = torchvision.datasets.CIFAR10(rootdir, train=True, download=True, transform=transform_train)
    c10test = torchvision.datasets.CIFAR10(rootdir, train=False, download=True, transform=transform_test)
    trainloader = DataLoader(c10train, batch_size=32, shuffle=True)
    testloader = DataLoader(c10test, batch_size=32)
    return trainloader, testloader

# ========== Evaluation ==========

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
    p_s = zero_filters / total_filters if total_filters > 0 else 0

    bits = torch.finfo(next(model.parameters()).dtype).bits
    if q_w is None:
        q_w = bits
    if q_a is None:
        q_a = bits

    from thop import profile
    dummy = torch.randn(input_size).to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    was_training = model.training
    model.eval()
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    if was_training:
        model.train()
    macs = flops

    param_score = (1 - (p_s + p_u)) * (q_w / 32) * w / 5.6e6
    ops_score = (1 - p_s) * (max(q_w, q_a) / 32) * macs / 2.8e8
    return param_score + ops_score

# ========== Training ==========

def train(model, trainloader, testloader, optimizer, scheduler, device, epochs=10):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        test_loss, test_acc = evaluate(model, testloader, device)
        scheduler.step()

        print(f"[Epoch {epoch+1}] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

# ========== Pruning ==========

def prune_and_evaluate(model, dataloader, device, ratios):
    import copy
    results = []
    for amount in ratios:
        pruned_model = copy.deepcopy(model)
        parameters_to_prune = [(m, 'weight') for m in pruned_model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        test_loss, test_acc = evaluate(pruned_model, dataloader, device)
        score = compute_score(pruned_model)
        print(f"Prune {amount*100:.0f}% | Acc: {test_acc:.2f}% | Score: {score:.4f}")
        results.append({'name': f'Pruned {int(amount*100)}%', 'score': score, 'acc': test_acc})
    return results

# ========== Main ==========

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ResNet18().to(device).half()

    trainloader, testloader = load_dataset()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    epochs = 20

    train(net, trainloader, testloader, optimizer, scheduler, device, epochs=epochs)

    # Baseline
    baseline_loss, baseline_acc = evaluate(net, testloader, device)
    baseline_score = compute_score(net)
    print(f"Baseline model | Acc: {baseline_acc:.2f}% | Score: {baseline_score:.4f}")

    # Pruning
    prune_ratios = [0.2, 0.4, 0.6, 0.8]
    prune_results = prune_and_evaluate(net, testloader, device, prune_ratios)

    # Quantized
    net_quantized = quantize_dynamic(net, {nn.Linear}, dtype=torch.qint8)
    quantized_loss, quantized_acc = evaluate(net_quantized, testloader, device)
    quantized_score = compute_score(net_quantized)
    print(f"Quantized model | Acc: {quantized_acc:.2f}% | Score: {quantized_score:.4f}")

    # Pruned+Quantized
    net_pruned_q = quantize_dynamic(net, {nn.Linear}, dtype=torch.qint8)
    pruned_quantized_loss, pruned_quantized_acc = evaluate(net_pruned_q, testloader, device)
    pruned_quantized_score = compute_score(net_pruned_q)
    print(f"Pruned+Quantized model | Acc: {pruned_quantized_acc:.2f}% | Score: {pruned_quantized_score:.4f}")

    # Résumé
    architectures = [
        {'name': 'Baseline', 'score': baseline_score, 'acc': baseline_acc},
        *prune_results,
        {'name': 'Quantized', 'score': quantized_score, 'acc': quantized_acc},
        {'name': 'Pruned+Quantized', 'score': pruned_quantized_score, 'acc': pruned_quantized_acc},
    ]

    # Plot
    scores = [arch['score'] for arch in architectures]
    accuracies = [arch['acc'] for arch in architectures]
    labels = [arch['name'] for arch in architectures]

    plt.figure(figsize=(10, 6))
    plt.scatter(scores, accuracies)
    for i, label in enumerate(labels):
        plt.annotate(label, (scores[i], accuracies[i]))
    plt.axhline(90, color='r', linestyle='--', label='90% accuracy limit')
    plt.xlabel('Efficiency Score')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Efficiency Score')
    plt.grid()
    plt.legend()
    plt.show()

