import torch
from models import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Recréer le modèle et le quantifier
model_fp = ResNet18()
model_fp.eval()
model_fp.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model_fp, inplace=True)

# 2. Charger les poids entraînés (quantifiés)
state_dict = torch.load("models/student_resnet18_qat_best.pth", map_location=device)
model_fp.load_state_dict(state_dict)

# 3. Convertir en modèle quantifié
model_quantized = torch.quantization.convert(model_fp.eval(), inplace=False).to(device)

# 4. TorchScript : transformer le modèle en un modèle optimisé pour l’inférence
scripted_model = torch.jit.script(model_quantized)

# 5. Sauvegarde dans un fichier .pt
torch.jit.save(scripted_model, "models/student_resnet18_qat_best.pt")
print("[*] TorchScript model saved to: models/student_resnet18_qat_best.pt")
