from models import ResNet18, EfficientResNet18
from utils import compute_score
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientResNet18().to(device)
model.eval()

score = compute_score(model, input_size=(1,3,32,32))
print(f"Efficiency Score (non-trained): {score:.4f}")
