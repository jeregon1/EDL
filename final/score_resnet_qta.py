import torch
from utils import compute_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("models/student_resnet18_qat_best.pt", map_location=device)
model.eval()

score = compute_score(model, input_size=(1, 3, 32, 32), q_w=8, q_a=8)
print(f"Efficiency Score (QAT, scripted): {score:.4f}")
