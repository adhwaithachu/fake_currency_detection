"""
train.py — Train ResNet50 for Fake Currency Detection
Optimized for small datasets with aggressive augmentation + class weighting.
Run: python src/train.py
"""

import os
import sys
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────
# Resolve paths relative to project root (works whether you run from root or src/)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "model")

BATCH_SIZE  = 16
NUM_EPOCHS  = 40
LR          = 0.0001
IMG_SIZE    = 224
NUM_CLASSES = 2

os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

# ── VALIDATE DATA DIR ─────────────────────────────────────────────────────
for split in ["train", "val"]:
    split_path = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_path):
        print(f"\nERROR: Missing directory: {split_path}")
        if split == "val":
            print("  → Run 'python augment_dataset.py' first to create the val split.")
        sys.exit(1)
    classes_found = [
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    ]
    if not classes_found:
        print(f"\nERROR: No class sub-folders found in {split_path}")
        sys.exit(1)

# ── TRANSFORMS ────────────────────────────────────────────────────────────
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ]),
    "val": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}

# ── DATASETS ──────────────────────────────────────────────────────────────
image_datasets = {
    s: datasets.ImageFolder(os.path.join(DATA_DIR, s), transform=data_transforms[s])
    for s in ["train", "val"]
}
dataloaders = {
    s: DataLoader(
        image_datasets[s],
        batch_size=BATCH_SIZE,
        shuffle=(s == "train"),
        num_workers=0,
        pin_memory=False,
    )
    for s in ["train", "val"]
}

dataset_sizes = {s: len(image_datasets[s]) for s in ["train", "val"]}
class_names   = image_datasets["train"].classes
print(f"Classes : {class_names}")
print(f"Train   : {dataset_sizes['train']}  |  Val: {dataset_sizes['val']}")

if dataset_sizes["val"] == 0:
    print("\nERROR: Val set is empty. Run 'python augment_dataset.py' first.")
    sys.exit(1)

# ── CLASS WEIGHTS (handle imbalance) ─────────────────────────────────────
counts = [0] * NUM_CLASSES
for _, lbl in image_datasets["train"].samples:
    counts[lbl] += 1

# Guard against zero counts
counts = [max(c, 1) for c in counts]
weights   = [max(counts) / c for c in counts]
class_wts = torch.FloatTensor(weights).to(DEVICE)
print(f"Class weights: { {class_names[i]: round(weights[i], 2) for i in range(NUM_CLASSES)} }")


# ── MODEL ─────────────────────────────────────────────────────────────────
def build_model():
    # Use weights= API (torchvision >= 0.13); falls back gracefully
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except AttributeError:
        # Older torchvision
        model = models.resnet50(pretrained=True)

    # Freeze early layers, fine-tune later ones
    for name, param in model.named_parameters():
        param.requires_grad = any(
            name.startswith(layer) for layer in ["layer3", "layer4", "fc"]
        )

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return model.to(DEVICE)


# ── TRAINING LOOP ─────────────────────────────────────────────────────────
def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}  {'─' * 35}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = running_correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss    / dataset_sizes[phase]
            epoch_acc  = running_correct.double() / dataset_sizes[phase]
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())
            print(f"  {phase.upper():5s} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
                print(f"  ✓ Best model saved (acc={best_acc:.4f})")

    print(f"\nBest val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model, history


# ── PLOT ──────────────────────────────────────────────────────────────────
def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val")
    axes[0].set_title("Loss");     axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train")
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
    print("Training curves saved.")


# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model     = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_wts)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-3,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    model, history = train_model(model, criterion, optimizer, scheduler)
    plot_history(history)

    with open(os.path.join(MODEL_DIR, "class_names.txt"), "w") as f:
        f.write("\n".join(class_names))

    print("\nDone! Files in model/")
    print("  best_model.pth | training_curves.png | class_names.txt")
