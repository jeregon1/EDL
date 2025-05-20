import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy(csv_files, labels, title="Accuracy vs Epoch"):
    plt.figure(figsize=(10, 6))
    
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        plt.plot(df["epoch"], df["test_acc"], label=f"{label} (Test)", linestyle='-')
        if "train_acc" in df.columns:
            plt.plot(df["epoch"], df["train_acc"], label=f"{label} (Train)", linestyle='--')

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Exemple d’utilisation :
csv_files = [
    "logs/log_student_resnet18_distillation.csv",
    "logs/log_student_qat.csv",
    "logs/log_efficient_qat.csv",
    "logs/log_efficient_resnet18_student.csv"
]
labels = ["ResNet18 + Distill", "ResNet18 + QAT", "Efficient + QAT", "Efficient Standard"]

plot_accuracy(csv_files, labels)

def plot_loss(csv_files, labels, title="Test Loss vs Epoch"):
    plt.figure(figsize=(10, 6))
    
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        plt.plot(df["epoch"], df["test_loss"], label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_loss(csv_files, labels)


def plot_lr(csv_files, labels, title="Learning Rate Schedule"):
    plt.figure(figsize=(10, 6))
    
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        plt.plot(df["epoch"], df["lr"], label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_lr(csv_files, labels)


import pandas as pd
import matplotlib.pyplot as plt

# Charger les deux fichiers
resnet_df = pd.read_csv("logs/prune_resnet.csv")
efficient_df = pd.read_csv("logs/prune_efficient_resnet.csv")

# Ajout d'une colonne pour identifier le modèle
resnet_df["model_type"] = "ResNet18"
efficient_df["model_type"] = "EfficientResNet18"

# Concaténer les deux
df = pd.concat([resnet_df, efficient_df])

# Ne garder que structured pruning
df = df[df["prune_type"] == "structured"]

# Plot
plt.figure(figsize=(10, 6))
for model_type in df["model_type"].unique():
    subset = df[df["model_type"] == model_type]
    plt.bar(
        subset["prune_ratio"] + (0.02 if model_type == "EfficientResNet18" else -0.02), 
        subset["score"], 
        width=0.04, 
        label=model_type
    )

plt.title("Efficiency Score vs Prune Ratio (Structured Pruning)")
plt.xlabel("Prune Ratio")
plt.ylabel("Efficiency Score (lower is better)")
plt.xticks([0.2, 0.4, 0.6])
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plots/pruning_score_comparison.png", dpi=300)
plt.show()

