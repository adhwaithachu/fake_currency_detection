# Fake Currency Detection — ResNet50

## Dataset Summary
- Genuine: 42 images
- Fake: 12 images
- After augmentation: ~300 images per class

---

## Run Order (follow exactly)

### 1. Install packages
```
pip install -r requirements.txt
```

### 2. Augment the dataset  ← run BEFORE training
```
python augment_dataset.py
```
Expands your 54 images to ~600 using flips, rotations, colour jitter etc.
Also creates the `dataset/val/` split automatically.

### 3. Train the model
```
python src/train.py
```
Takes ~10–20 min on CPU. Saves `model/best_model.pth` when done.

### 4. Evaluate (optional)
```
python src/evaluate.py
```
Prints accuracy, precision, recall, F1. Saves confusion matrix + ROC curve to `model/`.

### 5. Run the web app
```
python app.py
```
Open: http://localhost:5000

### 6. Test a single image from the command line
```
python src/predict.py --image dataset/val/genuine/genuine_1.jpg
python src/predict.py --image dataset/val/fake/fake_1.jpg
```

---

## Folder Structure
```
fake_currency_detection/
├── dataset/
│   ├── train/
│   │   ├── genuine/        ← 42 original genuine images
│   │   └── fake/           ← 12 original fake images
│   └── val/                ← created automatically by augment_dataset.py
├── model/                  ← best_model.pth saved here after training
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── templates/
│   └── index.html
├── static/
│   └── uploads/
├── app.py
├── augment_dataset.py
└── requirements.txt
```
