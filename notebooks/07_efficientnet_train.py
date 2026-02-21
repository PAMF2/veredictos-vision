# %%
import subprocess
subprocess.run(["pip", "install", "-q", "timm", "albumentations"], check=False)

# %%
import os
import csv
import glob
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.cuda.amp as amp
import timm
from sklearn.metrics import cohen_kappa_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

# %%
APTOS_BASE    = "/kaggle/input/datasets/mariaherrerot/aptos2019"
MESSIDOR_BASE = "/kaggle/input/datasets/mariaherrerot/messidor2preprocess"
DDR_BASE      = "/kaggle/input/datasets/mariaherrerot/ddrdataset"
OUTPUT_DIR    = "/kaggle/working/outputs/efficientnet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

GPU_ID = 1
MULTI_GPU = False
if torch.cuda.is_available():
    GPU_ID = int(os.getenv("GPU_ID", "0"))
    USE_MULTI_GPU = os.getenv("USE_MULTI_GPU", "0") == "1"
    if torch.cuda.device_count() > GPU_ID:
        DEVICE = torch.device(f"cuda:{GPU_ID}")
    else:
        DEVICE = torch.device("cuda:0")
    MULTI_GPU = torch.cuda.device_count() > 1 and USE_MULTI_GPU
else:
    DEVICE = torch.device("cpu")

print(f"GPUs available: {torch.cuda.device_count()} | Using: {DEVICE} | Multi-GPU: {MULTI_GPU}")

MODEL_NAME = os.getenv("MODEL_NAME", "efficientnet_b3")
IMG_SIZE   = int(os.getenv("IMG_SIZE", "300"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
EPOCHS     = int(os.getenv("EPOCHS", "40"))
LR         = 1e-4
NUM_CLASSES = 5
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "1"))

# %%
print("=== APTOS2019 Structure ===")
for root, dirs, files in os.walk(APTOS_BASE):
    depth = root.replace(APTOS_BASE, "").count(os.sep)
    if depth <= 2:
        imgs = sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
        csvs = [f for f in files if f.endswith(".csv")]
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/  [{imgs} imgs, csvs={csvs}]")

print("\n=== Messidor-2 Structure ===")
for root, dirs, files in os.walk(MESSIDOR_BASE):
    depth = root.replace(MESSIDOR_BASE, "").count(os.sep)
    if depth <= 2:
        imgs = sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif")))
        csvs = [f for f in files if f.endswith(".csv")]
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/  [{imgs} imgs, csvs={csvs}]")

# %%
def discover_image_dirs(base, extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")):
    dirs = []
    for root, _, files in os.walk(base):
        if any(f.lower().endswith(extensions) for f in files):
            dirs.append(root)
    dirs = sorted(set(dirs), key=lambda p: (len(p), p))
    return dirs


def build_image_index(img_dirs):
    index = {}
    for d in img_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"):
            for p in glob.glob(os.path.join(d, ext)):
                index.setdefault(os.path.basename(p).lower(), p)
    return index


# %%
def load_aptos(base):
    csv_candidates = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
    train_csv = None
    for c in csv_candidates:
        name = os.path.basename(c).lower()
        if "train" in name or "label" in name or "data" in name:
            train_csv = c
            break
    if train_csv is None and csv_candidates:
        train_csv = csv_candidates[0]

    if train_csv is None:
        print("APTOS: no CSV found, skipping")
        return [], []

    df = pd.read_csv(train_csv)
    print(f"APTOS CSV: {train_csv}")
    print(df.head(3))
    print(f"Columns: {df.columns.tolist()}")

    img_col = next((c for c in df.columns if "image" in c.lower() or "id" in c.lower()), df.columns[0])
    lbl_col = next((c for c in df.columns if "diagnos" in c.lower() or "label" in c.lower()
                    or "grade" in c.lower() or "level" in c.lower()), None)

    if lbl_col is None:
        print("APTOS: label column not found, skipping")
        return [], []

    img_dirs = discover_image_dirs(base)
    image_index = build_image_index(img_dirs)
    print(f"APTOS image dirs discovered: {len(img_dirs)}")

    paths, labels = [], []
    for _, row in df.iterrows():
        stem = str(row[img_col]).replace(".jpg", "").replace(".png", "")
        found = None
        for d in img_dirs:
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(d, stem + ext)
                if os.path.exists(candidate):
                    found = candidate
                    break
            if found:
                break
        if found is None:
            # Some datasets keep already full filename in CSV
            key = str(row[img_col]).strip().lower()
            found = image_index.get(key)
        if found is None:
            for ext in (".jpg", ".jpeg", ".png"):
                found = image_index.get((stem + ext).lower())
                if found:
                    break
        if found:
            lbl = int(row[lbl_col])
            if 0 <= lbl <= 4:
                paths.append(found)
                labels.append(lbl)

    print(f"APTOS loaded: {len(paths)} samples")
    return paths, labels


# %%
def load_messidor(base):
    csv_candidates = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
    csv_file = None
    for c in csv_candidates:
        name = os.path.basename(c).lower()
        if "messidor" in name or "data" in name or "label" in name:
            csv_file = c
            break
    if csv_file is None and csv_candidates:
        csv_file = csv_candidates[0]

    if csv_file is None:
        print("Messidor: no CSV found, skipping")
        return [], []

    df = pd.read_csv(csv_file)
    print(f"\nMessidor CSV: {csv_file}")
    print(df.head(3))
    print(f"Columns: {df.columns.tolist()}")

    img_col = next((c for c in df.columns if "image" in c.lower() or "name" in c.lower()
                    or "file" in c.lower()), df.columns[0])

    lbl_col = next((c for c in df.columns if "diagnos" in c.lower()), None)
    if lbl_col is None:
        grade_map_cols = [c for c in df.columns if "retinopathy" in c.lower() or
                          "grade" in c.lower() or "level" in c.lower() or "adjudicated" in c.lower()]
        lbl_col = grade_map_cols[0] if grade_map_cols else None

    if lbl_col is None:
        print("Messidor: label column not found, skipping")
        return [], []

    explicit_dirs = [
        os.path.join(base, "messidor-2", "messidor-2", "preprocess"),
        os.path.join(base, "messidor-2", "preprocess"),
    ]
    img_dirs = [d for d in explicit_dirs if os.path.isdir(d)] + discover_image_dirs(base)
    img_dirs = sorted(set(img_dirs), key=lambda p: (len(p), p))
    image_index = build_image_index(img_dirs)
    print(f"Messidor image dirs discovered: {len(img_dirs)}")

    paths, labels = [], []
    for _, row in df.iterrows():
        fname = str(row[img_col])
        if not fname.lower().endswith((".jpg", ".png", ".tif", ".jpeg")):
            fname = fname + ".jpg"
        found = None
        for d in img_dirs:
            candidate = os.path.join(d, fname)
            if os.path.exists(candidate):
                found = candidate
                break
            candidate2 = os.path.join(d, os.path.basename(fname))
            if os.path.exists(candidate2):
                found = candidate2
                break
        if found is None:
            found = image_index.get(os.path.basename(fname).lower())
        if found:
            raw_lbl = row[lbl_col]
            try:
                lbl = int(float(raw_lbl))
                if lbl > 4:
                    lbl = min(lbl, 4)
                if 0 <= lbl <= 4:
                    paths.append(found)
                    labels.append(lbl)
            except (ValueError, TypeError):
                pass

    print(f"Messidor loaded: {len(paths)} samples")
    return paths, labels


# %%
aptos_paths, aptos_labels = load_aptos(APTOS_BASE)
messidor_paths, messidor_labels = load_messidor(MESSIDOR_BASE)

all_paths = aptos_paths + messidor_paths
all_labels = aptos_labels + messidor_labels

print(f"\nCombined dataset: {len(all_paths)} samples")
if len(all_paths) == 0:
    raise RuntimeError("No data loaded. Check dataset paths and CSV structure.")

label_counts = [0] * NUM_CLASSES
for l in all_labels:
    label_counts[l] += 1
print(f"Label distribution: {label_counts}")

# %%
indices = list(range(len(all_paths)))
random.shuffle(indices)
split = int(0.85 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

train_paths  = [all_paths[i] for i in train_idx]
train_labels = [all_labels[i] for i in train_idx]
val_paths    = [all_paths[i] for i in val_idx]
val_labels   = [all_labels[i] for i in val_idx]

print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

# %%
train_tf = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(p=0.2),
    A.CLAHE(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# %%
class DRDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        except Exception:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.labels[idx]


# %%
train_counts = [0] * NUM_CLASSES
for l in train_labels:
    train_counts[l] += 1

class_weights = [1.0 / (c + 1) for c in train_counts]
total_w = sum(class_weights)
class_weights = [w / total_w * NUM_CLASSES for w in class_weights]
weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {[round(w, 3) for w in class_weights]}")

# %%
train_ds = DRDataset(train_paths, train_labels, train_tf)
val_ds   = DRDataset(val_paths,   val_labels,   val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=(NUM_WORKERS > 0))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=(NUM_WORKERS > 0))

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | workers={NUM_WORKERS}")

# %%
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)
if MULTI_GPU:
    model = nn.DataParallel(model)
print(f"Model: {MODEL_NAME} | on {DEVICE} | batch={BATCH_SIZE} | img={IMG_SIZE} | accum={GRAD_ACCUM_STEPS}")

# %%
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = amp.GradScaler()


# %%
def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (imgs, labels) in enumerate(loader, start=1):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        with amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels) / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        if step % GRAD_ACCUM_STEPS == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item()
        if step % 20 == 0 or step == len(loader):
            print(f"  train step {step}/{len(loader)} | loss={(loss.item() * GRAD_ACCUM_STEPS):.4f}")
    return total_loss / len(loader)


def val_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            with amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic",
                               labels=list(range(NUM_CLASSES)))
    return avg_loss, acc, kappa


# %%
best_kappa = -1.0
history = []

print("Starting training loop...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, scaler)
    val_loss, val_acc, val_kappa = val_epoch(model, val_loader)
    scheduler.step()

    improved = val_kappa > best_kappa
    if improved:
        best_kappa = val_kappa
        ckpt_path = os.path.join(OUTPUT_DIR, "efficientnet_dr_best.pth")
        state = model.module.state_dict() if MULTI_GPU else model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": state,
            "val_acc": val_acc,
            "val_kappa": val_kappa,
            "val_loss": val_loss,
            "num_classes": NUM_CLASSES,
        }, ckpt_path)

    history.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 5),
        "val_loss": round(val_loss, 5),
        "val_acc": round(val_acc, 5),
        "val_kappa": round(val_kappa, 5),
    })

    tag = " *" if improved else ""
    print(f"Ep {epoch:03d}/{EPOCHS} | train_loss={train_loss:.4f} | "
          f"val_loss={val_loss:.4f} | acc={val_acc:.4f} | kappa={val_kappa:.4f}{tag}")

# %%
hist_path = os.path.join(OUTPUT_DIR, "history.csv")
with open(hist_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_kappa"])
    writer.writeheader()
    writer.writerows(history)

print(f"\nBest Quadratic Kappa: {best_kappa:.4f}")
print(f"Target Kappa >= 0.90: {'PASSED' if best_kappa >= 0.90 else 'NOT YET'}")
print(f"Checkpoint: {os.path.join(OUTPUT_DIR, 'efficientnet_dr_best.pth')}")
print(f"History: {hist_path}")
