# Neuro-Symbolic Biomass Prediction
### AIMS DTU Research Intern 2026

Multi-modal Logical Tensor Network for predicting grassland biomass from RGB imagery and agronomic metadata.

---

## What This Project Does

Predicts 5 grassland biomass components (grams) from:
- RGB pasture images
- NDVI (vegetation index)
- Average canopy height (cm)

Using a **Neuro-Symbolic** approach: a multi-modal neural network (EfficientNet-B0 + MLP) whose training is guided by **Logical Tensor Network (LTN)** axioms that encode biological knowledge about how biomass components relate to each other.

---

## Targets

| Target | Description |
|---|---|
| Dry_Clover_g | Dry clover biomass in grams |
| Dry_Dead_g | Dry dead material biomass |
| Dry_Green_g | Dry green leaf biomass |
| Dry_Total_g | Total dry biomass |
| GDM_g | Grass dry matter |

---

## Model Architecture

```
Image (224x224x3)                Tabular [NDVI, Height]
    ↓                                   ↓
EfficientNet-B0                  MLP: Linear(2→64)→Linear(64→128)
(pretrained ImageNet)                   ↓
    ↓                              128-d features
1280-d features
    ↓
    └──────── Concat [1408-d] ──────────┘
                    ↓
          Linear(1408→512) → ReLU → Dropout(0.3)
                    ↓
          Linear(512→128)  → ReLU
                    ↓
          Linear(128→5)    → 5 biomass predictions
```

---

## LTN Axioms (Soft Logical Rules)

The LTN loss adds a **satisfaction penalty** whenever predictions violate:

| Axiom | Rule | Biology |
|---|---|---|
| A1 | Dry_Total_g ≥ Dry_Green_g | Total can't be less than one component |
| A2 | Dry_Total_g ≥ Dry_Clover_g | Same |
| A3 | Dry_Total_g ≥ Dry_Dead_g | Same |
| A4 | All predictions ≥ 0 | Biomass is non-negative |
| A5 | Dry_Green_g ≥ 0.5 × Dry_Clover_g | Green fraction dominates clover |

**Total Loss = MSE Loss + λ × (1 − Satisfaction Score)**

where λ (default 0.3) controls the trade-off between data fitting and rule compliance.

---

## Results

| Target | LTN R² | Baseline R² | Δ R² |
|---|---|---|---|
| Dry_Clover_g | — | — | — |
| Dry_Dead_g | — | — | — |
| Dry_Green_g | — | — | — |
| Dry_Total_g | — | — | — |
| GDM_g | — | — | — |

*(Fill in after running `evaluate_model.py`)*

---

## Reproduce This Work

### 1. Clone and install

```bash
git clone https://github.com/Rishiii2/biomass-prediction-multimodal.git
cd biomass-prediction-multimodal
pip install -r requirements.txt
```

### 2. Prepare dataset

Place your dataset files as:
```
biomass-prediction-multimodal/
├── train.csv
├── images/
│   ├── image_001.jpg
│   └── ...
```

### 3. Train (both LTN and baseline)

```bash
python train_model.py --csv train.csv --images images --save_dir results
```

Optional flags:
```bash
python train_model.py \
    --csv train.csv \
    --images images \
    --save_dir results \
    --epochs 150 \
    --patience 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --lam 0.3
```

`--lam` is the LTN axiom weight. `0` = pure neural baseline, `0.3` = recommended.

### 4. Evaluate and compare models

```bash
python evaluate_model.py --csv train.csv --images images --results results
```

Generates:
- `results/metrics_ltn.csv` — LTN per-target MAE / RMSE / R²
- `results/metrics_baseline.csv` — Baseline metrics
- `results/metrics_comparison.csv` — Side-by-side comparison
- `results/scatter_ltn.png` — Predicted vs actual plots
- `results/comparison_r2.png` — R² bar chart
- `results/axiom_violations.png` — Rule violation rates
- `results/training_curves.png` — Loss and satisfaction curves

### 5. Predict on a new image

```bash
python predict.py --image path/to/image.jpg --ndvi 0.65 --height 45.0
```

---

## Repository Structure

```
biomass-prediction-multimodal/
├── train_model.py       # Training: LTN model + baseline
├── evaluate_model.py    # Evaluation + plots + comparison
├── predict.py           # Single-image inference
├── requirements.txt     # Python dependencies
├── report/
│   └── Project_Report.md.txt   # Technical report
└── results/
    ├── best_model_ltn.pth       # Trained LTN model weights
    ├── best_model_baseline.pth  # Trained baseline weights
    ├── metrics_ltn.csv
    ├── metrics_baseline.csv
    ├── metrics_comparison.csv
    ├── training_history_ltn.csv
    ├── training_history_baseline.csv
    ├── scatter_ltn.png
    ├── comparison_r2.png
    ├── axiom_violations.png
    └── training_curves.png
```

---

## Assumptions

1. Images are correctly linked via the `image_path` column in `train.csv`.
2. NDVI and height measurements represent ground-truth field conditions.
3. Biomass target values are destructive-sampling ground truth in grams.
4. The train/val split (80/20, seed=42) is fixed for reproducibility.
5. The LTN axioms reflect general botanical relationships; they are soft constraints, not hard boundaries.

---

## Limitations

1. Small dataset (357 samples) — overfitting risk is real; augmentation and early stopping mitigate this.
2. Only 2 tabular features; species, date, and soil metadata could improve predictions.
3. LTN axioms are manually designed — automatically mined rules could be stronger.
4. No temporal modelling — growth trajectories over time are not captured.
5. Dead material (Dry_Dead_g) is visually similar to soil, making it the hardest target.

---

## External Resources

- [PyTorch](https://pytorch.org/) — deep learning framework
- [EfficientNet-B0](https://pytorch.org/vision/stable/models/efficientnet.html) — pretrained on ImageNet (Tan & Le, 2019)
- [LTN Paper](https://arxiv.org/abs/2012.13635) — Badreddine et al., 2022
- [scikit-learn](https://scikit-learn.org/) — metrics and data splitting
