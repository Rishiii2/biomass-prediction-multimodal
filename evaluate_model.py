import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==================================================
# ARGUMENT PARSER
# ==================================================

parser = argparse.ArgumentParser(description="Evaluate LTN Biomass Models")
parser.add_argument("--csv",      type=str, default="train.csv")
parser.add_argument("--images",   type=str, default="images")
parser.add_argument("--results",  type=str, default="results")
args = parser.parse_args()

TARGET_COLUMNS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ==================================================
# DATA
# ==================================================

df = pd.read_csv(args.csv)
pivot_df = df.pivot_table(
    index=["image_path", "Pre_GSHH_NDVI", "Height_Ave_cm"],
    columns="target_name",
    values="target"
).reset_index()
pivot_df.columns.name = None
pivot_df = pivot_df.dropna(subset=TARGET_COLUMNS)

_, valid_df = train_test_split(pivot_df, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

class BiomassDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(args.images, str(row["image_path"]))
        image    = Image.open(img_path).convert("RGB")
        image    = transform(image)
        tabular  = torch.tensor([row["Pre_GSHH_NDVI"], row["Height_Ave_cm"]], dtype=torch.float32)
        targets  = torch.tensor([row[t] for t in TARGET_COLUMNS], dtype=torch.float32)
        return image, tabular, targets

dataset = BiomassDataset(valid_df)
loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# ==================================================
# MODEL
# ==================================================

class BiomassModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        feat = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.tabular_net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(feat + 128, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, image, tabular):
        img_feat = self.backbone(image)
        tab_feat = self.tabular_net(tabular)
        return self.head(torch.cat([img_feat, tab_feat], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================
# HELPER: run inference on validation set
# ==================================================

def get_predictions(model_path):
    model = BiomassModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, tabular, targets in loader:
            images  = images.to(device)
            tabular = tabular.to(device)
            outputs = model(images, tabular)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return preds, targets

# ==================================================
# HELPER: compute per-target metrics
# ==================================================

def compute_metrics(preds, targets):
    rows = []
    for i, name in enumerate(TARGET_COLUMNS):
        mae  = mean_absolute_error(targets[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))
        r2   = r2_score(targets[:, i], preds[:, i])
        rows.append({"Target": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    return pd.DataFrame(rows)

# ==================================================
# LOAD BOTH MODELS
# ==================================================

ltn_path  = os.path.join(args.results, "best_model_ltn.pth")
base_path = os.path.join(args.results, "best_model_baseline.pth")

print("Loading LTN model...")
preds_ltn, truth = get_predictions(ltn_path)

print("Loading baseline model...")
preds_base, _    = get_predictions(base_path)

metrics_ltn  = compute_metrics(preds_ltn,  truth)
metrics_base = compute_metrics(preds_base, truth)

# ==================================================
# PRINT COMPARISON TABLE
# ==================================================

print("\n" + "="*70)
print("EVALUATION RESULTS: LTN vs Neural-Only Baseline")
print("="*70)

print("\n--- LTN Model (with logical axioms) ---")
print(metrics_ltn.to_string(index=False, float_format="%.4f"))

print("\n--- Baseline Model (MSE only, no axioms) ---")
print(metrics_base.to_string(index=False, float_format="%.4f"))

comparison = pd.DataFrame({
    "Target":      TARGET_COLUMNS,
    "LTN_R2":      metrics_ltn["R2"].values,
    "Base_R2":     metrics_base["R2"].values,
    "Delta_R2":    metrics_ltn["R2"].values - metrics_base["R2"].values,
    "LTN_MAE":     metrics_ltn["MAE"].values,
    "Base_MAE":    metrics_base["MAE"].values,
    "LTN_RMSE":    metrics_ltn["RMSE"].values,
    "Base_RMSE":   metrics_base["RMSE"].values,
})

print("\n--- Comparison (positive Delta_R2 means LTN is better) ---")
print(comparison.to_string(index=False, float_format="%.4f"))

print(f"\nMean LTN  R²: {metrics_ltn['R2'].mean():.4f}")
print(f"Mean Base R²: {metrics_base['R2'].mean():.4f}")
print(f"Mean ΔR²:     {(metrics_ltn['R2'] - metrics_base['R2']).mean():+.4f}")

metrics_ltn.to_csv(os.path.join(args.results, "metrics_ltn.csv"), index=False)
metrics_base.to_csv(os.path.join(args.results, "metrics_baseline.csv"), index=False)
comparison.to_csv(os.path.join(args.results, "metrics_comparison.csv"), index=False)

# ==================================================
# AXIOM VIOLATION ANALYSIS
# Checks whether the LTN actually enforced the biological rules
# ==================================================

IDX = {name: i for i, name in enumerate(TARGET_COLUMNS)}

def axiom_violations(preds, label):
    total  = preds[:, IDX["Dry_Total_g"]]
    green  = preds[:, IDX["Dry_Green_g"]]
    clover = preds[:, IDX["Dry_Clover_g"]]
    dead   = preds[:, IDX["Dry_Dead_g"]]

    v1 = np.mean(total < green)          # total should >= green
    v2 = np.mean(total < clover)         # total should >= clover
    v3 = np.mean(total < dead)           # total should >= dead
    v4 = np.mean(preds < 0)              # all should be non-negative
    v5 = np.mean(green < clover * 0.5)   # green >= clover*0.5

    print(f"\n  [{label}] Axiom Violation Rates:")
    print(f"    A1 total >= green  : {v1*100:.1f}% violated")
    print(f"    A2 total >= clover : {v2*100:.1f}% violated")
    print(f"    A3 total >= dead   : {v3*100:.1f}% violated")
    print(f"    A4 all >= 0        : {v4*100:.1f}% violated")
    print(f"    A5 green >= 0.5*cl : {v5*100:.1f}% violated")
    return [v1, v2, v3, v4, v5]

print("\n" + "="*70)
print("AXIOM VIOLATION ANALYSIS")
print("="*70)
violations_ltn  = axiom_violations(preds_ltn,  "LTN")
violations_base = axiom_violations(preds_base, "Baseline")

# ==================================================
# PLOT 1: Predicted vs Actual scatter (LTN, per target)
# ==================================================

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]

for i, (ax, name) in enumerate(zip(axes, TARGET_COLUMNS)):
    ax.scatter(truth[:, i], preds_ltn[:, i], alpha=0.6, s=20, color=colors[i], edgecolors="none")
    mn = min(truth[:, i].min(), preds_ltn[:, i].min())
    mx = max(truth[:, i].max(), preds_ltn[:, i].max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.5, label="Perfect")
    r2  = metrics_ltn.loc[metrics_ltn["Target"] == name, "R2"].values[0]
    mae = metrics_ltn.loc[metrics_ltn["Target"] == name, "MAE"].values[0]
    ax.set_title(f"{name}\nR²={r2:.3f}  MAE={mae:.1f}g", fontsize=9, fontweight="bold")
    ax.set_xlabel("Actual (g)", fontsize=8)
    ax.set_ylabel("Predicted (g)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.suptitle("LTN Model: Predicted vs Actual Biomass", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(args.results, "scatter_ltn.png"), dpi=150, bbox_inches="tight")
plt.close()

# ==================================================
# PLOT 2: LTN vs Baseline R² comparison bar chart
# ==================================================

x     = np.arange(len(TARGET_COLUMNS))
width = 0.35
labels = [t.replace("_", "\n") for t in TARGET_COLUMNS]

fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - width/2, metrics_base["R2"], width, label="Baseline (MSE only)", color="#e74c3c", alpha=0.85)
b2 = ax.bar(x + width/2, metrics_ltn["R2"],  width, label="LTN (with axioms)",   color="#2ecc71", alpha=0.85)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("R² Score")
ax.set_title("LTN vs Neural-Only Baseline: R² per Target", fontweight="bold", fontsize=13)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(min(0, metrics_base["R2"].min() - 0.05), 1.05)
plt.tight_layout()
plt.savefig(os.path.join(args.results, "comparison_r2.png"), dpi=150, bbox_inches="tight")
plt.close()

# ==================================================
# PLOT 3: Axiom violation comparison bar chart
# ==================================================

axiom_labels = ["A1: total≥green", "A2: total≥clover", "A3: total≥dead", "A4: all≥0", "A5: green≥0.5cl"]
x2 = np.arange(len(axiom_labels))

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x2 - width/2, [v*100 for v in violations_base], width, label="Baseline", color="#e74c3c", alpha=0.85)
ax.bar(x2 + width/2, [v*100 for v in violations_ltn],  width, label="LTN",      color="#2ecc71", alpha=0.85)
ax.set_xticks(x2)
ax.set_xticklabels(axiom_labels, fontsize=9)
ax.set_ylabel("Violation Rate (%)")
ax.set_title("Axiom Violation Rate: LTN vs Baseline\n(lower is better)", fontweight="bold", fontsize=12)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.results, "axiom_violations.png"), dpi=150, bbox_inches="tight")
plt.close()

# ==================================================
# PLOT 4: Training curves (LTN)
# ==================================================

hist_path = os.path.join(args.results, "training_history_ltn.csv")
if os.path.exists(hist_path):
    hist = pd.read_csv(hist_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(hist["epoch"], hist["train_loss"],  "b-",  label="Train")
    axes[0].plot(hist["epoch"], hist["valid_loss"],  "r-",  label="Val")
    axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(hist["epoch"], hist["train_data"],  "b--", label="Train data")
    axes[1].plot(hist["epoch"], hist["train_axiom"], "g--", label="Train axiom")
    axes[1].plot(hist["epoch"], hist["valid_data"],  "b-",  label="Val data")
    axes[1].plot(hist["epoch"], hist["valid_axiom"], "g-",  label="Val axiom")
    axes[1].set_title("Data vs Axiom Loss"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    axes[2].plot(hist["epoch"], hist["sat_score"], color="purple")
    axes[2].set_title("Axiom Satisfaction Score\n(higher = rules more satisfied)")
    axes[2].set_ylim(0, 1.05); axes[2].axhline(1.0, ls="--", color="gray", lw=0.8)
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Epoch")

    plt.suptitle("LTN Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

print("\n" + "="*70)
print("All plots saved to:", args.results)
print("  scatter_ltn.png       — predicted vs actual per target")
print("  comparison_r2.png     — LTN vs baseline R² bar chart")
print("  axiom_violations.png  — how well each model obeys the rules")
print("  training_curves.png   — loss and satisfaction over epochs")
print("="*70)
print("DONE")
