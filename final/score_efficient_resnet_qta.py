import torch
from models import EfficientResNet18
from utils import compute_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle
model = EfficientResNet18()
model.eval()

# Charger les poids quantifiés
state_dict = torch.load("models/student_efficient_resnet18_qat_best.pth", map_location=device)
model.load_state_dict(state_dict)

# Envoyer sur le bon device
model = model.to(device)

# Calculer le score
score = compute_score(model, input_size=(1, 3, 32, 32))
print(f"Efficiency Score (QAT - EfficientResNet18): {score:.4f}")
