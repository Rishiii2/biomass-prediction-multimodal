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
from tqdm import tqdm

# ==================================================
# ARGUMENT PARSER  (no more hardcoded paths)
# ==================================================

parser = argparse.ArgumentParser(description="Train LTN Biomass Model")
parser.add_argument("--csv",        type=str, default="train.csv",   help="Path to train.csv")
parser.add_argument("--images",     type=str, default="images",      help="Root folder containing images")
parser.add_argument("--save_dir",   type=str, default="results",     help="Where to save outputs")
parser.add_argument("--epochs",     type=int, default=150)
parser.add_argument("--patience",   type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--lam",        type=float, default=0.3,         help="LTN axiom loss weight")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# ==================================================
# TARGET COLUMNS
# ==================================================

TARGET_COLUMNS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g"
]

# ==================================================
# LOAD AND PIVOT CSV
# ==================================================

df = pd.read_csv(args.csv)
print(f"Original rows: {len(df)}")

pivot_df = df.pivot_table(
    index=["image_path", "Pre_GSHH_NDVI", "Height_Ave_cm"],
    columns="target_name",
    values="target"
).reset_index()
pivot_df.columns.name = None

print(f"Pivoted samples: {len(pivot_df)}")

# Drop rows missing any target
pivot_df = pivot_df.dropna(subset=TARGET_COLUMNS)
print(f"After dropping NaN targets: {len(pivot_df)}")

# ==================================================
# TRAIN / VAL SPLIT
# ==================================================

train_df, valid_df = train_test_split(pivot_df, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)}   Val: {len(valid_df)}")

# ==================================================
# TRANSFORMS  (ImageNet normalization — required for pretrained EfficientNet)
# ==================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)   # <-- was missing before
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ==================================================
# DATASET
# ==================================================

class BiomassDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(args.images, str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        tabular = torch.tensor(
            [row["Pre_GSHH_NDVI"], row["Height_Ave_cm"]],
            dtype=torch.float32
        )

        targets = torch.tensor(
            [row[t] for t in TARGET_COLUMNS],
            dtype=torch.float32
        )

        return image, tabular, targets

train_dataset = BiomassDataset(train_df, train_transform)
valid_dataset = BiomassDataset(valid_df, valid_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# ==================================================
# MODEL  (unchanged architecture — your original design)
# ==================================================

class BiomassModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        image_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.tabular_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(image_features + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, image, tabular):
        img_feat  = self.backbone(image)
        tab_feat  = self.tabular_net(tabular)
        x         = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(x)

# ==================================================
# LTN LOSS  ← THE CORE NEW ADDITION
#
# What is this?
# Instead of only minimising prediction error (MSE),
# we also penalise violations of biological rules:
#
#   Rule 1:  Dry_Total_g  >=  Dry_Green_g   (total can't be less than one part)
#   Rule 2:  Dry_Total_g  >=  Dry_Clover_g  (same)
#   Rule 3:  Dry_Total_g  >=  Dry_Dead_g    (same)
#   Rule 4:  GDM_g        >=  0             (biomass is non-negative)
#   Rule 5:  If NDVI is high, Dry_Green_g should be relatively large
#
# Each rule is expressed as a fuzzy truth value in [0, 1].
# fuzzy_geq(a, b) ≈ 1 when a > b  (rule satisfied)
#                 ≈ 0 when a < b  (rule violated)
#
# Total loss = MSE  +  lam * (1 - mean satisfaction)
# ==================================================

class LTNLoss(nn.Module):
    """
    Neuro-Symbolic loss combining data MSE with soft logical axiom satisfaction.

    Axioms encoded:
      A1: Dry_Total_g >= Dry_Green_g
      A2: Dry_Total_g >= Dry_Clover_g
      A3: Dry_Total_g >= Dry_Dead_g
      A4: GDM_g >= 0  (non-negativity)
      A5: Dry_Green_g >= Dry_Clover_g * 0.5  (green dominates clover fraction)

    Args:
        lam (float): weight on axiom satisfaction loss (0 = pure neural, 1 = equal weight)
        sharpness (float): how steep the fuzzy boundary is (higher = harder constraint)
    """

    # Index map matching TARGET_COLUMNS order
    IDX = {
        "Dry_Clover_g": 0,
        "Dry_Dead_g":   1,
        "Dry_Green_g":  2,
        "Dry_Total_g":  3,
        "GDM_g":        4
    }

    def __init__(self, lam=0.3, sharpness=5.0):
        super().__init__()
        self.lam       = lam
        self.sharpness = sharpness
        self.mse       = nn.MSELoss()

    def fuzzy_geq(self, a, b):
        """
        Differentiable approximation of truth(a >= b).
        Returns a tensor of shape [batch] with values in (0, 1).
        sigmoid(sharpness * (a - b)) → 1 when a >> b, 0 when a << b
        """
        return torch.sigmoid(self.sharpness * (a - b))

    def p_mean(self, truths, p=2):
        """
        Generalised mean (pMean) over a list of scalar tensors.
        p=2 is the standard LTN choice — punishes violations harder than arithmetic mean.
        Formula: ( mean(v_i ^ p) ) ^ (1/p)
        """
        stacked = torch.stack(truths)          # [num_axioms]
        return torch.pow(torch.mean(torch.pow(stacked, p)), 1.0 / p)

    def forward(self, preds, targets):
        """
        preds:   [batch, 5]  — raw model outputs (grams, may be negative before clamp)
        targets: [batch, 5]  — ground truth grams

        Returns:
            total_loss  — backpropagate this
            data_loss   — for logging
            axiom_loss  — for logging
            sat_score   — satisfaction in [0,1], for logging
        """
        I = self.IDX

        # ── Data loss (supervised regression) ─────────────────────────────
        data_loss = self.mse(preds, targets)

        # ── Axiom truth values ─────────────────────────────────────────────
        total  = preds[:, I["Dry_Total_g"]]   # shape [batch]
        green  = preds[:, I["Dry_Green_g"]]
        clover = preds[:, I["Dry_Clover_g"]]
        dead   = preds[:, I["Dry_Dead_g"]]
        gdm    = preds[:, I["GDM_g"]]

        # A1: total >= green
        a1 = self.fuzzy_geq(total, green).mean()

        # A2: total >= clover
        a2 = self.fuzzy_geq(total, clover).mean()

        # A3: total >= dead
        a3 = self.fuzzy_geq(total, dead).mean()

        # A4: all predictions non-negative  (biomass can't be negative)
        # Vectorised: check all 5 targets at once
        a4 = torch.sigmoid(self.sharpness * preds).mean()

        # A5: green >= clover * 0.5  (green fraction dominates clover fraction)
        a5 = self.fuzzy_geq(green, clover * 0.5).mean()

        # ── Aggregate satisfaction with pMean (p=2) ───────────────────────
        sat_score  = self.p_mean([a1, a2, a3, a4, a5], p=2)
        axiom_loss = 1.0 - sat_score

        # ── Combined loss ──────────────────────────────────────────────────
        total_loss = data_loss + self.lam * axiom_loss

        return total_loss, data_loss, axiom_loss, sat_score.item()

# ==================================================
# SETUP
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

def run_training(lam, save_name):
    """Train one model with given lam. Returns history list."""

    model     = BiomassModel().to(device)
    criterion = LTNLoss(lam=lam)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_loss       = 1e9
    patience_counter = 0
    history          = []

    print(f"\n{'='*60}")
    print(f"Training: {save_name}  (lam={lam})")
    print(f"{'='*60}")

    for epoch in range(args.epochs):

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_total = train_data = train_ax = 0.0

        for images, tabular, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            images  = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, tabular)
            loss, dl, al, _ = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_total += loss.item()
            train_data  += dl.item()
            train_ax    += al.item()

        n = len(train_loader)
        train_total /= n
        train_data  /= n
        train_ax    /= n

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        valid_total = valid_data = valid_ax = valid_sat = 0.0

        with torch.no_grad():
            for images, tabular, targets in valid_loader:
                images  = images.to(device)
                tabular = tabular.to(device)
                targets = targets.to(device)

                outputs = model(images, tabular)
                loss, dl, al, sat = criterion(outputs, targets)

                valid_total += loss.item()
                valid_data  += dl.item()
                valid_ax    += al.item()
                valid_sat   += sat

        m = len(valid_loader)
        valid_total /= m
        valid_data  /= m
        valid_ax    /= m
        valid_sat   /= m

        scheduler.step()

        print(
            f"Epoch {epoch+1:>3}/{args.epochs} | "
            f"Train {train_total:.4f} (data={train_data:.4f} ax={train_ax:.4f}) | "
            f"Val {valid_total:.4f} | Sat={valid_sat:.3f}"
        )

        history.append({
            "epoch":        epoch + 1,
            "train_loss":   train_total,
            "train_data":   train_data,
            "train_axiom":  train_ax,
            "valid_loss":   valid_total,
            "valid_data":   valid_data,
            "valid_axiom":  valid_ax,
            "sat_score":    valid_sat
        })

        if valid_total < best_loss:
            best_loss = valid_total
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{save_name}.pth"))
            print("  ✓ BEST MODEL SAVED")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return history

# ==================================================
# TRAIN LTN MODEL  (with axiom loss)
# ==================================================

history_ltn = run_training(lam=args.lam, save_name="best_model_ltn")

pd.DataFrame(history_ltn).to_csv(
    os.path.join(args.save_dir, "training_history_ltn.csv"), index=False
)

# ==================================================
# TRAIN BASELINE  (neural only, lam=0 — no axioms)
# Needed for the comparative analysis the assignment requires
# ==================================================

history_base = run_training(lam=0.0, save_name="best_model_baseline")

pd.DataFrame(history_base).to_csv(
    os.path.join(args.save_dir, "training_history_baseline.csv"), index=False
)

print("\nDONE — both models trained and saved.")
print(f"LTN model:      {args.save_dir}/best_model_ltn.pth")
print(f"Baseline model: {args.save_dir}/best_model_baseline.pth")
