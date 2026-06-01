"""
predict.py — Run inference on a single image.

Usage:
    python predict.py --image path/to/image.jpg --ndvi 0.65 --height 45.0

Outputs the predicted biomass (grams) for all 5 targets.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ==================================================
# ARGUMENT PARSER
# ==================================================

parser = argparse.ArgumentParser(description="Predict biomass from a single image")
parser.add_argument("--image",      type=str,   required=True,  help="Path to .jpg image")
parser.add_argument("--ndvi",       type=float, required=True,  help="NDVI value (e.g. 0.65)")
parser.add_argument("--height",     type=float, required=True,  help="Average height in cm (e.g. 45.0)")
parser.add_argument("--model",      type=str,   default=os.path.join("results", "best_model_ltn.pth"),
                    help="Path to trained model weights (.pth)")
args = parser.parse_args()

TARGET_COLUMNS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g"
]

# ==================================================
# MODEL  (must match train_model.py exactly)
# ==================================================

class BiomassModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        feat = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.tabular_net = nn.Sequential(
            nn.Linear(2, 64),  nn.ReLU(),
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

# ==================================================
# LOAD MODEL
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.model):
    print(f"Error: model file not found at '{args.model}'")
    print("Train first with:  python train_model.py")
    sys.exit(1)

model = BiomassModel().to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()

# ==================================================
# PREPROCESS  (same transform as training — with ImageNet normalisation)
# ==================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

if not os.path.exists(args.image):
    print(f"Error: image not found at '{args.image}'")
    sys.exit(1)

image   = Image.open(args.image).convert("RGB")
image   = transform(image).unsqueeze(0).to(device)

tabular = torch.tensor([[args.ndvi, args.height]], dtype=torch.float32).to(device)

# ==================================================
# INFERENCE
# ==================================================

with torch.no_grad():
    pred = model(image, tabular).cpu().numpy()[0]

# ==================================================
# OUTPUT
# ==================================================

print("\n" + "="*40)
print("  BIOMASS PREDICTIONS")
print("="*40)
print(f"  Image  : {args.image}")
print(f"  NDVI   : {args.ndvi}")
print(f"  Height : {args.height} cm")
print("-"*40)
for name, val in zip(TARGET_COLUMNS, pred):
    val = max(0.0, float(val))   # clip negatives — biomass >= 0
    print(f"  {name:<16}: {val:>8.2f} g")
print("="*40)

# Quick sanity check: print a warning if ordering axioms are violated
total  = max(0, float(pred[3]))
green  = max(0, float(pred[2]))
clover = max(0, float(pred[0]))
dead   = max(0, float(pred[1]))

if total < green or total < clover or total < dead:
    print("\n  ⚠  Note: some ordering axioms not perfectly satisfied.")
    print("     This can occur in edge cases; consider retraining with higher --lam.")
