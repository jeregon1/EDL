import torch
from models import ResNet18

# 1. Charger le modèle en mode entraînement
model_fp = ResNet18()
model_fp.load_state_dict(torch.load("models/student_resnet18_qat_best.pth", map_location="cpu"))
model_fp.train()  # Nécessaire pour prepare_qat

# 2. Configuration de la quantization
model_fp.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

# (Optionnel) Ajouter model_fp.fuse_model() ici si ton modèle le supporte
# model_fp.fuse_model()

# 3. Préparer pour QAT
torch.quantization.prepare_qat(model_fp, inplace=True)

# 4. Convertir en modèle quantifié (inference-ready)
model_qat = torch.quantization.convert(model_fp.eval(), inplace=False)

# 5. Compiler avec TorchScript
traced = torch.jit.script(model_qat)
traced.save("models/resnet18_qat_final_jit.pt")

print("[+] Modèle quantifié et exporté avec succès vers models/resnet18_qat_final_jit.pt")
