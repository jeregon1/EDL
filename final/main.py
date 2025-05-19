from models import ResNet18
from utils import load_dataset, train, evaluate, prune_and_evaluate, save_model, compute_score
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ResNet18(groups=1).to(device)
    # Convert model to half precision
    net = net.half()
    # test compute_score function for half precision
    score = compute_score(net)
    print(f"Efficiency score (half precision): {score:.4f}")
    trainloader, testloader = load_dataset()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    epochs = 100
    train(net, trainloader, testloader, optimizer, scheduler, device, epochs=epochs)
    # Automatic model save name
    save_path = f"./pretrained/resnet18_e{epochs}.pth"
    save_model(net, save_path)
    prune_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
    prune_and_evaluate(net, testloader, device, prune_ratios)
