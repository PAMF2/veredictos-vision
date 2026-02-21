# %%
import os
import glob
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
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

CHECKPOINT = "/kaggle/working/outputs/unetpp/unet_r34_drive_best.pth"
OUTPUT_DIR = "/kaggle/working/outputs/unetpp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DRIVE_ROOT = os.getenv(
    "DRIVE_ROOT",
    "/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction",
).strip()
TEST_IMAGE_DIR = os.getenv("TEST_IMAGE_DIR", os.path.join(DRIVE_ROOT, "DRIVE", "test", "images"))
TEST_MASK_DIR = os.getenv("TEST_MASK_DIR", os.path.join(DRIVE_ROOT, "DRIVE", "test", "1st_manual"))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
N_SAMPLES = 10
SEED = 42
random.seed(SEED)

# %%
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found at {CHECKPOINT}. "
        "Run 03_unetpp_train.py first."
    )

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
print(f"Checkpoint loaded | epoch={ckpt.get('epoch')} | "
      f"val_dice={ckpt.get('val_dice', 'N/A'):.4f} | "
      f"mode={ckpt.get('mode')} | num_classes={ckpt.get('num_classes')}")

MODE = ckpt.get("mode", "segmentation")
NUM_CLASSES = ckpt.get("num_classes", 1)
BEST_THRESHOLD = float(ckpt.get("best_threshold", 0.5))

# %%
arch = ckpt.get("arch", "unet_resnet34")
if arch == "unet_resnet34":
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
else:
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()
print("Model loaded and in eval mode.")
print(f"Using threshold: {BEST_THRESHOLD:.2f}")

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
    drive_pairs = collect_pairs_from_dirs(TEST_IMAGE_DIR, TEST_MASK_DIR)
    if drive_pairs:
        return drive_pairs

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


paired_samples = collect_pairs(DATASET_BASE)

print(f"Paired samples available: {len(paired_samples)}")
print(f"Eval source: {TEST_IMAGE_DIR} | {TEST_MASK_DIR}")

if len(paired_samples) == 0:
    raise RuntimeError(f"No paired image-mask samples found in {DATASET_BASE}.")

samples = random.sample(paired_samples, min(N_SAMPLES, len(paired_samples)))

# %%
tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

unnorm_mean = np.array([0.485, 0.456, 0.406])
unnorm_std = np.array([0.229, 0.224, 0.225])


def unnormalize(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * unnorm_std + unnorm_mean
    return np.clip(img, 0, 1)


# %%
all_dice = []

IS_BINARY = (NUM_CLASSES == 1)
GRID_COLS = 3 if IS_BINARY else 4

fig, axes = plt.subplots(len(samples), GRID_COLS,
                          figsize=(GRID_COLS * 4, len(samples) * 4))
if len(samples) == 1:
    axes = axes[np.newaxis, :]

with torch.no_grad():
    for i, (img_path, msk_path) in enumerate(samples):
        raw_img = np.array(Image.open(img_path).convert("RGB"))
        raw_msk = np.array(Image.open(msk_path).convert("L"))
        gt_binary = (raw_msk > 127).astype(np.float32)

        aug = tf(image=raw_img)
        inp = aug["image"].unsqueeze(0).to(DEVICE)

        with amp.autocast():
            logits = model(inp)

        if IS_BINARY:
            pred_prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
            pred_binary = (pred_prob > BEST_THRESHOLD).astype(np.float32)

            gt_resized = np.array(Image.fromarray(gt_binary).resize(
                (pred_binary.shape[1], pred_binary.shape[0]),
                Image.NEAREST
            ))

            intersection = (pred_binary * gt_resized).sum()
            dice = 2.0 * intersection / (pred_binary.sum() + gt_resized.sum() + 1e-6)
            all_dice.append(dice)

            display_img = np.array(Image.fromarray(raw_img).resize((IMG_SIZE, IMG_SIZE)))
            overlay = display_img.copy().astype(np.float32) / 255.0
            overlay[pred_binary > 0.5] = [1.0, 0.0, 0.0]

            axes[i, 0].imshow(display_img)
            axes[i, 0].set_title(f"Original\n{Path(img_path).stem[:20]}", fontsize=8)
            axes[i, 1].imshow(gt_resized, cmap="gray")
            axes[i, 1].set_title(f"Ground Truth", fontsize=8)
            axes[i, 2].imshow(pred_binary, cmap="gray")
            axes[i, 2].set_title(f"Prediction (th={BEST_THRESHOLD:.2f})\nDice={dice:.3f}", fontsize=8)

        else:
            pred_cls = torch.argmax(logits[0], dim=0).cpu().numpy()
            class_names = ["EX", "HE", "MA", "SE"] if NUM_CLASSES == 4 else \
                          [f"C{c}" for c in range(NUM_CLASSES)]

            display_img = np.array(Image.fromarray(raw_img).resize((IMG_SIZE, IMG_SIZE)))
            axes[i, 0].imshow(display_img)
            axes[i, 0].set_title(f"Original\n{Path(img_path).stem[:20]}", fontsize=8)
            axes[i, 1].imshow(gt_binary, cmap="gray")
            axes[i, 1].set_title("Ground Truth (bin)", fontsize=8)
            axes[i, 2].imshow(pred_cls, cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1)
            axes[i, 2].set_title("Prediction (classes)", fontsize=8)

            gt_resized = np.array(Image.fromarray(gt_binary).resize(
                (pred_cls.shape[1], pred_cls.shape[0]), Image.NEAREST))
            pred_bin = (pred_cls > 0).astype(np.float32)
            intersection = (pred_bin * gt_resized).sum()
            dice = 2.0 * intersection / (pred_bin.sum() + gt_resized.sum() + 1e-6)
            all_dice.append(dice)

            overlay = display_img.copy().astype(np.float32) / 255.0
            overlay[pred_cls > 0] = [1.0, 0.0, 0.0]
            axes[i, 3].imshow(np.clip(overlay, 0, 1))
            axes[i, 3].set_title("Overlay", fontsize=8)

        for ax in axes[i]:
            ax.axis("off")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "unetpp_test_results.png"), dpi=100, bbox_inches="tight")
plt.show()
print(f"Saved: {os.path.join(OUTPUT_DIR, 'unetpp_test_results.png')}")

# %%
mdice = float(np.mean(all_dice)) if all_dice else 0.0
print(f"\n=== Test Results ({len(samples)} samples) ===")
for idx, (d, (ip, _)) in enumerate(zip(all_dice, samples)):
    print(f"  [{idx+1:02d}] {Path(ip).name:<40} Dice={d:.4f}")
print(f"\nmDice: {mdice:.4f}")
print(f"Target mDice >= 0.75: {'PASSED' if mdice >= 0.75 else 'BELOW TARGET'}")
