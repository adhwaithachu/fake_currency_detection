"""
evaluate.py — Evaluate trained ResNet50 model
Run: python src/evaluate.py
"""

import os
import sys

import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

# ── Resolve paths relative to project root ───────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
IMG_SIZE   = 224
BATCH_SIZE = 8
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load class names ──────────────────────────────────────────────────────
class_names_path = os.path.join(MODEL_DIR, "class_names.txt")
if not os.path.exists(class_names_path):
    print(f"ERROR: {class_names_path} not found.\nRun 'python src/train.py' first.")
    sys.exit(1)

with open(class_names_path) as f:
    class_names = [line.strip() for line in f if line.strip()]
print(f"Classes: {class_names}")

# ── Val dataset ───────────────────────────────────────────────────────────
val_dir = os.path.join(DATA_DIR, "val")
if not os.path.isdir(val_dir):
    print(f"ERROR: Val directory not found: {val_dir}\nRun 'python augment_dataset.py' first.")
    sys.exit(1)

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
if len(val_dataset) == 0:
    print("ERROR: Val set is empty. Run 'python augment_dataset.py' first.")
    sys.exit(1)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Val samples: {len(val_dataset)}")


# ── Load model ────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}\nRun 'python src/train.py' first.")
        sys.exit(1)

    try:
        model = models.resnet50(weights=None)
    except TypeError:
        model = models.resnet50(pretrained=False)

    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(class_names)),
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


model = load_model()
print("Model loaded.")

# ── Inference ─────────────────────────────────────────────────────────────
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs.to(DEVICE))
        probs   = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_probs = np.array(all_probs)

# ── Metrics ───────────────────────────────────────────────────────────────
acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print("\n" + "═" * 45)
print("         EVALUATION RESULTS")
print("═" * 45)
print(f"  Accuracy  : {acc * 100:.2f}%")
print(f"  Precision : {prec * 100:.2f}%")
print(f"  Recall    : {rec * 100:.2f}%")
print(f"  F1-Score  : {f1 * 100:.2f}%")
print("─" * 45)
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
)
plt.title("Confusion Matrix", fontweight="bold")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
print("Confusion matrix saved.")

# ── ROC curve ─────────────────────────────────────────────────────────────
if len(class_names) == 2:
    pos_label = class_names.index("genuine") if "genuine" in class_names else 1
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, pos_label], pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="navy")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "roc_curve.png"), dpi=150)
    print(f"ROC-AUC: {roc_auc:.4f}  — saved roc_curve.png")

print("\nEvaluation complete!")
