import torch
from models import ResNet18
from utils import prune_and_evaluate, load_dataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle distillé ResNet18
model = ResNet18().to(device)
model.load_state_dict(torch.load("models/student_resnet18_distillation_best.pth"))
model.eval()

# Dataset CIFAR10 (test uniquement ici)
_, testloader = load_dataset(batch_size=128)

# Taux de pruning à tester
ratios = [0.2, 0.4, 0.6]

# Fichier de log spécifique ResNet18
logfile = "log/prune_resnet.csv"

# Exécution
prune_and_evaluate(model, dataloader=testloader, device=device, ratios=ratios, logfile=logfile)
