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
plt.savefig("logs/pruning_score_comparison.png", dpi=300)
plt.show()
