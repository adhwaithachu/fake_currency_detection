"""
augment_dataset.py
Expands the small dataset using heavy augmentation.
Run FIRST before train.py:  python augment_dataset.py
"""

import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

random.seed(42)

SRC_TRAIN        = os.path.join("dataset", "train")
VAL_DIR          = os.path.join("dataset", "val")
TARGET_PER_CLASS = 300   # how many images to generate per class


# ── helpers ──────────────────────────────────────────────────────────────
def augment_image(img):
    """Apply a random chain of augmentations and return the new image."""
    ops = []

    if random.random() < 0.5:
        ops.append(("flip_h",))
    if random.random() < 0.3:
        ops.append(("flip_v",))
    if random.random() < 0.7:
        ops.append(("rotate", random.uniform(-25, 25)))
    if random.random() < 0.6:
        factor = random.uniform(0.6, 1.5)
        ops.append(("brightness", factor))
    if random.random() < 0.6:
        factor = random.uniform(0.6, 1.5)
        ops.append(("contrast", factor))
    if random.random() < 0.4:
        factor = random.uniform(0.7, 1.4)
        ops.append(("saturation", factor))
    if random.random() < 0.3:
        ops.append(("blur",))
    if random.random() < 0.3:
        ops.append(("sharpen",))
    if random.random() < 0.4:
        ops.append(("crop", random.uniform(0.75, 0.95)))

    result = img.copy().convert("RGB")

    for op in ops:
        if op[0] == "flip_h":
            result = ImageOps.mirror(result)
        elif op[0] == "flip_v":
            result = ImageOps.flip(result)
        elif op[0] == "rotate":
            result = result.rotate(op[1], expand=False, fillcolor=(200, 200, 200))
        elif op[0] == "brightness":
            result = ImageEnhance.Brightness(result).enhance(op[1])
        elif op[0] == "contrast":
            result = ImageEnhance.Contrast(result).enhance(op[1])
        elif op[0] == "saturation":
            result = ImageEnhance.Color(result).enhance(op[1])
        elif op[0] == "blur":
            result = result.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        elif op[0] == "sharpen":
            result = result.filter(ImageFilter.SHARPEN)
        elif op[0] == "crop":
            w, h   = result.size
            new_w  = int(w * op[1])
            new_h  = int(h * op[1])
            left   = random.randint(0, max(0, w - new_w))
            top    = random.randint(0, max(0, h - new_h))
            result = result.crop((left, top, left + new_w, top + new_h))
            result = result.resize((w, h), Image.LANCZOS)

    return result


def expand_class(class_name, target):
    src_dir   = os.path.join(SRC_TRAIN, class_name)
    originals = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not originals:
        print(f"  No images found in {src_dir}")
        return

    existing = len(originals)
    needed   = max(0, target - existing)
    print(f"  {class_name}: {existing} originals → generating {needed} augmented copies …")

    aug_dir = src_dir + "_aug"
    os.makedirs(aug_dir, exist_ok=True)

    # Copy originals into aug dir
    for f in originals:
        shutil.copy(os.path.join(src_dir, f), os.path.join(aug_dir, f))

    count = 0
    while count < needed:
        src_name = random.choice(originals)
        img      = Image.open(os.path.join(src_dir, src_name))
        aug      = augment_image(img)
        out_name = f"aug_{class_name}_{count:04d}.jpg"
        aug.save(os.path.join(aug_dir, out_name), "JPEG", quality=92)
        count += 1

    # Replace src with aug dir
    shutil.rmtree(src_dir)
    os.rename(aug_dir, src_dir)
    print(f"  {class_name}: done → {len(os.listdir(src_dir))} images total")


def create_val_split(class_name, val_ratio=0.2):
    src_dir = os.path.join(SRC_TRAIN, class_name)
    val_dir = os.path.join(VAL_DIR, class_name)
    os.makedirs(val_dir, exist_ok=True)

    images = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(images)
    n_val    = max(1, int(len(images) * val_ratio))
    val_imgs = images[:n_val]

    for f in val_imgs:
        shutil.move(os.path.join(src_dir, f), os.path.join(val_dir, f))

    print(f"  {class_name}: {len(images) - n_val} train | {n_val} val")


# ── main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Step 1: Augmenting images ──")
    for cls in ["genuine", "fake"]:
        expand_class(cls, TARGET_PER_CLASS)

    print("\n── Step 2: Creating val split ──")
    for cls in ["genuine", "fake"]:
        create_val_split(cls, val_ratio=0.2)

    print("\n✓ Dataset ready!")
    for split in ["train", "val"]:
        for cls in ["genuine", "fake"]:
            d = os.path.join("dataset", split, cls)
            n = len(os.listdir(d)) if os.path.exists(d) else 0
            print(f"  dataset/{split}/{cls}: {n} images")
