"""
predict.py — Predict a single currency image
Usage: python src/predict.py --image path/to/note.jpg
Also imported by app.py
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Resolve paths relative to project root ───────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names():
    path = os.path.join(MODEL_DIR, "class_names.txt")
    if not os.path.exists(path):
        return ["fake", "genuine"]
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_model_arch(num_classes):
    """Build the same architecture used in train.py."""
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
        nn.Linear(256, num_classes),
    )
    return model


def load_model(num_classes):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            "Run 'python augment_dataset.py' then 'python src/train.py' first."
        )
    model = build_model_arch(num_classes)
    # weights_only=False keeps compatibility with older PyTorch versions
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def preprocess(image_path):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0)


# ── Lazy-loaded singletons (efficient for the Flask app) ──────────────────
_model       = None
_class_names = None


def predict(image_path):
    global _model, _class_names
    if _class_names is None:
        _class_names = load_class_names()
    if _model is None:
        _model = load_model(len(_class_names))

    tensor = preprocess(image_path).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]

    idx        = torch.argmax(probs).item()
    label      = _class_names[idx]
    confidence = probs[idx].item()
    all_probs  = {_class_names[i]: round(probs[i].item(), 4) for i in range(len(_class_names))}
    return label, confidence, all_probs


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fake/genuine for a currency image.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"File not found: {args.image}")
        sys.exit(1)

    label, conf, probs = predict(args.image)
    print("\n" + "═" * 38)
    print("   CURRENCY DETECTION RESULT")
    print("═" * 38)
    print(f"  File       : {args.image}")
    print(f"  Prediction : {label.upper()}")
    print(f"  Confidence : {conf * 100:.2f}%")
    for cls, p in probs.items():
        print(f"  {cls:10s} : {p * 100:.2f}%")
    print("═" * 38)
