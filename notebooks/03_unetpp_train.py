# %%
import subprocess
subprocess.run(["pip", "install", "-q", "segmentation-models-pytorch", "albumentations", "timm"], check=False)

# %%
import os
import sys
import csv
import glob
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# %%
DATASET_CANDIDATES = [
    os.getenv("DATASET_BASE", "").strip(),
    "/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction",
    "/kaggle/input/datasets/vidheeshnacode/stare-dataset",
    "/kaggle/input/datasets/srinjoybhuiya/drive-retinal-vessel-segmentation-pixelwise",
]


def has_image_files(base):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")
    for root, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith(exts):
                return True
    return False


def pick_dataset_base(candidates):
    for p in candidates:
        if p and os.path.exists(p) and has_image_files(p):
            return p
    return None


DATASET_BASE = pick_dataset_base(DATASET_CANDIDATES)
if not DATASET_BASE:
    raise FileNotFoundError(
        "No valid segmentation dataset found (with image files). "
        "Set DATASET_BASE to DRIVE/STARE path."
    )

DRIVE_ROOT = os.getenv(
    "DRIVE_ROOT",
    "/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction",
).strip()
TRAIN_IMAGE_DIR = os.getenv("TRAIN_IMAGE_DIR", os.path.join(DRIVE_ROOT, "DRIVE", "training", "images"))
TRAIN_MASK_DIR = os.getenv("TRAIN_MASK_DIR", os.path.join(DRIVE_ROOT, "DRIVE", "training", "1st_manual"))
VAL_IMAGE_DIR = os.getenv("VAL_IMAGE_DIR", os.path.join(DRIVE_ROOT, "DRIVE", "test", "images"))
VAL_MASK_DIR = os.getenv("VAL_MASK_DIR", os.path.join(DRIVE_ROOT, "DRIVE", "test", "1st_manual"))

OUTPUT_DIR = "/kaggle/working/outputs/unetpp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MULTI_GPU = torch.cuda.device_count() > 1
print(f"GPUs available: {torch.cuda.device_count()} | Multi-GPU: {MULTI_GPU}")

IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-4
EARLY_STOP_PATIENCE = 12
VAL_RATIO = 0.2
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# %%
print(f"Dataset base: {DATASET_BASE}")
print("=== Dataset Structure ===")
for root, dirs, files in os.walk(DATASET_BASE):
    depth = root.replace(DATASET_BASE, "").count(os.sep)
    indent = "  " * depth
    print(f"{indent}{os.path.basename(root)}/")
    if depth <= 2:
        img_count = sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")))
        if img_count > 0:
            print(f"{indent}  [{img_count} images]")

# %%
def sample_id(path):
    stem = Path(path).stem.lower()
    return stem.split("_")[0]


def collect_pairs_from_dirs(img_dir, msk_dir):
    pairs = []
    if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
        return pairs

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    mask_files = sorted(glob.glob(os.path.join(msk_dir, "*.*")))
    mask_by_id = {sample_id(m): m for m in mask_files}
    for img in imgs:
        sid = sample_id(img)
        m = mask_by_id.get(sid)
        if m:
            pairs.append((img, m))
    return pairs


def collect_drive_pairs(base):
    return collect_pairs_from_dirs(
        os.path.join(base, "DRIVE", "training", "images"),
        os.path.join(base, "DRIVE", "training", "1st_manual"),
    )


def is_mask_file(path):
    name = Path(path).name.lower()
    full = str(path).lower().replace("\\", "/")
    # In DRIVE, '/mask/' is FOV mask; do not use as vessel label.
    if "/drive/" in full and "/mask/" in full:
        return False
    return (
        "manual1" in name
        or "manual" in name
        or "label" in name
        or "_gt" in name
        or "/masks" in full
        or "/1st_manual" in full
        or "/manual" in full
        or "/ground" in full
        or "/label" in full
        or "/labels" in full
        or "/vessel" in full
    )


def collect_pairs(base):
    drive_pairs = collect_drive_pairs(base)
    if drive_pairs:
        return drive_pairs

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")
    files = []
    for root, _, fs in os.walk(base):
        for f in fs:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))

    image_files = [p for p in files if not is_mask_file(p)]
    mask_files = [p for p in files if is_mask_file(p)]
    mask_by_id = {}
    for m in mask_files:
        sid = sample_id(m)
        if sid not in mask_by_id:
            mask_by_id[sid] = m

    pairs = []
    for img in sorted(image_files):
        sid = sample_id(img)
        m = mask_by_id.get(sid)
        if m:
            pairs.append((img, m))
    return pairs


pairs = collect_pairs_from_dirs(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
if not pairs:
    pairs = collect_pairs(DATASET_BASE)
print(f"Paired samples (train): {len(pairs)}")

if len(pairs) < 10:
    raise RuntimeError(
        f"Found only {len(pairs)} training pairs. "
        "Use a segmentation dataset with masks (DRIVE/STARE)."
    )

all_images = [p[0] for p in pairs]
all_masks = [p[1] for p in pairs]

train_imgs, train_msks = all_images, all_masks
val_pairs = collect_pairs_from_dirs(VAL_IMAGE_DIR, VAL_MASK_DIR)
if val_pairs:
    val_imgs = [p[0] for p in val_pairs]
    val_msks = [p[1] for p in val_pairs]
    print(f"Paired samples (val): {len(val_pairs)}")
    print(f"Val source: {VAL_IMAGE_DIR} | {VAL_MASK_DIR}")
else:
    rand = random.Random(SEED)
    idx = list(range(len(train_imgs)))
    rand.shuffle(idx)
    train_imgs = [train_imgs[i] for i in idx]
    train_msks = [train_msks[i] for i in idx]
    split = int((1.0 - VAL_RATIO) * len(train_imgs))
    val_imgs, val_msks = train_imgs[split:], train_msks[split:]
    train_imgs, train_msks = train_imgs[:split], train_msks[:split]
    print(f"Paired samples (val fallback split): {len(val_imgs)}")

NUM_CLASSES = 1
MODE = "segmentation"

# %%
train_tf = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.Affine(scale=(0.92, 1.08), translate_percent=(-0.04, 0.04), rotate=(-12, 12), p=0.6),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03, p=0.5),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# %%
class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        msk = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        msk = (msk > 127).astype(np.float32)

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]

        return img, msk.unsqueeze(0)


class ClsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        label = self.labels[idx] if self.labels else 0
        return img, label


# %%
if MODE == "segmentation":
    train_ds = SegDataset(train_imgs, train_msks, train_tf)
    val_ds = SegDataset(val_imgs, val_msks, val_tf)
else:
    raise RuntimeError("This script is for segmentation only.")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
print("Pair samples:")
for i in range(min(3, len(train_imgs))):
    print(f"  {Path(train_imgs[i]).name} -> {Path(train_msks[i]).name}")

dbg_imgs, dbg_msks = next(iter(train_loader))
print(f"Sanity | img.shape={tuple(dbg_imgs.shape)} | mask.shape={tuple(dbg_msks.shape)}")
print(f"Sanity | mask unique={torch.unique(dbg_msks)}")

# %%
if MODE == "segmentation":
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
else:
    raise RuntimeError("This script is for segmentation only.")

model = model.to(DEVICE)

if MULTI_GPU:
    model = nn.DataParallel(model)
    print(f"DataParallel on {torch.cuda.device_count()} GPUs")

# %%
if MODE == "segmentation":
    tversky_loss_fn = smp.losses.TverskyLoss(
        mode="binary", from_logits=True, alpha=0.7, beta=0.3
    )
    bce_loss_fn = nn.BCEWithLogitsLoss()

    def criterion(logits, targets):
        d = tversky_loss_fn(logits, targets)
        b = bce_loss_fn(logits, targets)
        return 0.7 * d + 0.3 * b

else:
    raise RuntimeError("This script is for segmentation only.")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = amp.GradScaler()


# %%
def dice_score(pred_logits, targets, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * targets).sum()
    return (2.0 * intersection / (pred.sum() + targets.sum() + 1e-6)).item()


# %%
DEBUG_BATCH_ONCE = True


def train_epoch(model, loader, optimizer, scaler):
    global DEBUG_BATCH_ONCE
    model.train()
    total_loss = 0.0
    for batch_idx, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        with amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)

        if DEBUG_BATCH_ONCE and batch_idx == 0:
            probs = torch.sigmoid(outputs)
            print(
                "Debug | logits[min,max]=({:.4f}, {:.4f}) | prob_mean={:.4f} | target_mean={:.4f}".format(
                    outputs.min().item(),
                    outputs.max().item(),
                    probs.mean().item(),
                    targets.mean().item(),
                )
            )
            DEBUG_BATCH_ONCE = False

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    dice_by_th = {th: 0.0 for th in THRESHOLDS}
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            with amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            total_loss += loss.item()
            if MODE == "segmentation":
                for th in THRESHOLDS:
                    dice_by_th[th] += dice_score(outputs, targets, threshold=th)
    avg_loss = total_loss / len(loader)
    if MODE != "segmentation":
        return avg_loss, 0.0, 0.5
    avg_dice_by_th = {th: (dice_by_th[th] / len(loader)) for th in THRESHOLDS}
    best_th = max(avg_dice_by_th, key=avg_dice_by_th.get)
    return avg_loss, avg_dice_by_th[best_th], best_th


# %%
best_metric = 0.0
history = []
epochs_without_improve = 0
best_threshold = 0.5

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, scaler)
    val_loss, val_dice, val_th = val_epoch(model, val_loader)
    scheduler.step()

    metric = val_dice if MODE == "segmentation" else (1.0 - val_loss)
    improved = metric > best_metric

    if improved:
        best_metric = metric
        best_threshold = val_th
        epochs_without_improve = 0
        ckpt_path = os.path.join(OUTPUT_DIR, "unet_r34_drive_best.pth")
        state = model.module.state_dict() if MULTI_GPU else model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": state,
            "val_dice": val_dice,
            "val_loss": val_loss,
            "mode": MODE,
            "num_classes": NUM_CLASSES,
            "arch": "unet_resnet34",
            "best_threshold": float(best_threshold),
        }, ckpt_path)
    else:
        epochs_without_improve += 1

    history.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 5),
        "val_loss": round(val_loss, 5),
        "val_dice": round(val_dice, 5),
    })

    tag = " *" if improved else ""
    print(f"Ep {epoch:03d}/{EPOCHS} | train_loss={train_loss:.4f} | "
          f"val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | th={val_th:.2f}{tag}")

    if epochs_without_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch} (no val_dice improvement for {EARLY_STOP_PATIENCE} epochs)")
        break

# %%
hist_path = os.path.join(OUTPUT_DIR, "history.csv")
with open(hist_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_dice"])
    writer.writeheader()
    writer.writerows(history)

print(f"\nBest val_dice: {best_metric:.4f}")
print(f"Best threshold: {best_threshold:.2f}")
print(f"Checkpoint: {os.path.join(OUTPUT_DIR, 'unet_r34_drive_best.pth')}")
print(f"History: {hist_path}")
print(f"Target mDice >= 0.75: {'PASSED' if best_metric >= 0.75 else 'NOT YET'}")
