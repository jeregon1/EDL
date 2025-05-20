import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/log_student_resnet18_distillation.csv")

plt.figure()
plt.plot(df["epoch"], df["test_acc"], label="ResNet18 Distillation")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy (%)")
plt.title("ResNet18 Distillation - Accuracy vs Epochs")
plt.grid(True)
plt.legend()
plt.savefig("logs/acc_vs_epochs_resnet18_distillation.png")
plt.show()
