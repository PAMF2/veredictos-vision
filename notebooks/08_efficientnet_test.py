# %%
import os
import glob
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter

import torch
import torch.cuda.amp as amp
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
import albumentations as A
from albumentations.pytorch import ToTensorV2

# %%
APTOS_BASE    = "/kaggle/input/datasets/mariaherrerot/aptos2019"
MESSIDOR_BASE = "/kaggle/input/datasets/mariaherrerot/messidor2preprocess"
CHECKPOINT    = "/kaggle/working/outputs/efficientnet/efficientnet_dr_best.pth"
OUTPUT_DIR    = "/kaggle/working/outputs/efficientnet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = int(os.getenv("IMG_SIZE", "300"))
NUM_CLASSES = 5
N_EXAMPLES  = 10
SEED        = 42
random.seed(SEED)

CLASS_NAMES = ["Grade 0\n(No DR)", "Grade 1\n(Mild)", "Grade 2\n(Moderate)",
               "Grade 3\n(Severe)", "Grade 4\n(Proliferative)"]

# %%
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found at {CHECKPOINT}. "
        "Run 07_efficientnet_train.py first."
    )

try:
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
except Exception as e:
    print(f"Standard torch.load failed ({type(e).__name__}). Retrying with weights_only=False...")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
print(f"Checkpoint loaded | epoch={ckpt.get('epoch')} | "
      f"val_acc={ckpt.get('val_acc', 0):.4f} | "
      f"val_kappa={ckpt.get('val_kappa', 0):.4f}")

NUM_CLASSES = ckpt.get("num_classes", 5)
MODEL_NAME = ckpt.get("model_name", os.getenv("MODEL_NAME", "efficientnet_b3"))

# %%
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
try:
    model.load_state_dict(ckpt["model_state_dict"])
except RuntimeError:
    fallback = "efficientnet_b4" if MODEL_NAME != "efficientnet_b4" else "efficientnet_b3"
    print(f"Load failed for {MODEL_NAME}, trying {fallback}...")
    model = timm.create_model(fallback, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    MODEL_NAME = fallback
model = model.to(DEVICE)
model.eval()
print(f"Model loaded and in eval mode: {MODEL_NAME}")

# %%
def discover_image_dirs(base, extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")):
    dirs = []
    for root, _, files in os.walk(base):
        if any(f.lower().endswith(extensions) for f in files):
            dirs.append(root)
    return sorted(set(dirs), key=lambda p: (len(p), p))


def build_image_index(img_dirs):
    index = {}
    for d in img_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"):
            for p in glob.glob(os.path.join(d, ext)):
                index.setdefault(os.path.basename(p).lower(), p)
    return index


def load_aptos(base):
    csv_candidates = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
    train_csv = None
    for c in csv_candidates:
        if "train" in os.path.basename(c).lower():
            train_csv = c
            break
    if train_csv is None and csv_candidates:
        train_csv = csv_candidates[0]
    if train_csv is None:
        return [], []

    df = pd.read_csv(train_csv)
    img_col = next((c for c in df.columns if "image" in c.lower() or "id" in c.lower()), df.columns[0])
    lbl_col = next((c for c in df.columns if "diagnos" in c.lower() or "label" in c.lower() or "grade" in c.lower()), None)
    if lbl_col is None:
        return [], []

    img_dirs = discover_image_dirs(base)
    image_index = build_image_index(img_dirs)

    paths, labels = [], []
    for _, row in df.iterrows():
        stem = str(row[img_col]).replace(".jpg", "").replace(".png", "")
        found = None
        for ext in (".jpg", ".jpeg", ".png"):
            found = image_index.get((stem + ext).lower())
            if found:
                break
        if found is None:
            found = image_index.get(str(row[img_col]).strip().lower())
        if found:
            lbl = int(row[lbl_col])
            if 0 <= lbl <= 4:
                paths.append(found)
                labels.append(lbl)
    return paths, labels


def load_messidor(base):
    csv_candidates = glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
    csv_file = None
    for c in csv_candidates:
        if "messidor" in os.path.basename(c).lower():
            csv_file = c
            break
    if csv_file is None and csv_candidates:
        csv_file = csv_candidates[0]
    if csv_file is None:
        return [], []

    df = pd.read_csv(csv_file)
    img_col = next((c for c in df.columns if "image" in c.lower() or "name" in c.lower() or "file" in c.lower()), df.columns[0])
    lbl_col = next((c for c in df.columns if "diagnos" in c.lower()), None)
    if lbl_col is None:
        return [], []

    explicit_dirs = [
        os.path.join(base, "messidor-2", "messidor-2", "preprocess"),
        os.path.join(base, "messidor-2", "preprocess"),
    ]
    img_dirs = [d for d in explicit_dirs if os.path.isdir(d)] + discover_image_dirs(base)
    img_dirs = sorted(set(img_dirs), key=lambda p: (len(p), p))
    image_index = build_image_index(img_dirs)

    paths, labels = [], []
    for _, row in df.iterrows():
        fname = str(row[img_col]).strip()
        found = image_index.get(os.path.basename(fname).lower())
        if found:
            lbl = int(float(row[lbl_col]))
            lbl = min(lbl, 4)
            if 0 <= lbl <= 4:
                paths.append(found)
                labels.append(lbl)
    return paths, labels


def reconstruct_split(base_aptos, base_messidor, seed=42):
    aptos_paths, aptos_labels = load_aptos(base_aptos)
    messidor_paths, messidor_labels = load_messidor(base_messidor)
    all_paths = aptos_paths + messidor_paths
    all_labels = aptos_labels + messidor_labels
    if not all_paths:
        raise RuntimeError("No samples found. Check dataset paths.")
    rng = random.Random(seed)
    indices = list(range(len(all_paths)))
    rng.shuffle(indices)
    split = int(0.85 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_paths = [all_paths[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]
    return train_paths, train_labels, val_paths, val_labels


train_paths, train_labels, val_paths, val_labels = reconstruct_split(APTOS_BASE, MESSIDOR_BASE, seed=SEED)
val_samples = list(zip(val_paths, val_labels))

overlap = set(train_paths).intersection(set(val_paths))
print(f"Leakage check | overlap train/val: {len(overlap)}")
print(f"Train class dist: {dict(sorted(Counter(train_labels).items()))}")
print(f"Val class dist:   {dict(sorted(Counter(val_labels).items()))}")
if overlap:
    print("Warning: train/val overlap detected.")
else:
    print("Split check passed: no overlap.")
print(f"Val samples available: {len(val_samples)}")

# %%
tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def predict_batch(paths):
    tensors = []
    for p in paths:
        try:
            img = np.array(Image.open(p).convert("RGB"))
        except Exception:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        tensors.append(tf(image=img)["image"])
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad(), amp.autocast():
        logits = model(batch)
    return logits.argmax(dim=1).cpu().numpy()


# %%
all_gt, all_pred = [], []
EVAL_BATCH = 32

for start in range(0, len(val_samples), EVAL_BATCH):
    batch = val_samples[start:start + EVAL_BATCH]
    paths  = [s[0] for s in batch]
    labels = [s[1] for s in batch]
    preds  = predict_batch(paths)
    all_gt.extend(labels)
    all_pred.extend(preds.tolist())

all_gt   = np.array(all_gt)
all_pred = np.array(all_pred)

# %%
accuracy = (all_gt == all_pred).mean()
kappa = cohen_kappa_score(all_gt, all_pred, weights="quadratic",
                           labels=list(range(NUM_CLASSES)))

per_class_acc = []
for cls in range(NUM_CLASSES):
    mask = all_gt == cls
    if mask.sum() > 0:
        per_class_acc.append((all_gt[mask] == all_pred[mask]).mean())
    else:
        per_class_acc.append(float("nan"))

print(f"\n=== Evaluation Results ({len(all_gt)} samples) ===")
print(f"Overall Accuracy:       {accuracy:.4f}")
print(f"Quadratic Kappa:        {kappa:.4f}")
print(f"Target Acc >= 0.85:     {'PASSED' if accuracy >= 0.85 else 'BELOW TARGET'}")
print(f"Target Kappa >= 0.90:   {'PASSED' if kappa >= 0.90 else 'BELOW TARGET'}")
print("\nPer-class Accuracy:")
for cls, (name, acc_c) in enumerate(zip(CLASS_NAMES, per_class_acc)):
    n = (all_gt == cls).sum()
    name_clean = name.replace("\n", " ")
    print(f"  {name_clean:<30} acc={acc_c:.4f}  (n={n})")

prec, rec, f1, sup = precision_recall_fscore_support(
    all_gt, all_pred, labels=list(range(NUM_CLASSES)), zero_division=0
)
print("\nPer-class Precision / Recall / F1:")
for i, cname in enumerate(CLASS_NAMES):
    c = cname.replace("\n", " ")
    print(f"  {c:<30} P={prec[i]:.4f} R={rec[i]:.4f} F1={f1[i]:.4f} (n={sup[i]})")

# %%
cm = confusion_matrix(all_gt, all_pred, labels=list(range(NUM_CLASSES)))
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

tick_labels = [n.replace("\n", " ") for n in CLASS_NAMES]
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(tick_labels, fontsize=9)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True", fontsize=11)
ax.set_title(f"Confusion Matrix (normalized)\nAcc={accuracy:.3f}  Kappa={kappa:.3f}", fontsize=12)

thresh = 0.5
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        color = "white" if cm_norm[i, j] > thresh else "black"
        ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                ha="center", va="center", color=color, fontsize=8)

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
fig.savefig(cm_path, dpi=100, bbox_inches="tight")
plt.show()
print(f"Saved: {cm_path}")

# %%
correct_samples = [(p, g, pr) for (p, g), pr in
                   zip([(s[0], s[1]) for s in val_samples[:len(all_gt)]], all_pred)
                   if g == pr]
wrong_samples = [(p, g, pr) for (p, g), pr in
                 zip([(s[0], s[1]) for s in val_samples[:len(all_gt)]], all_pred)
                 if g != pr]

n_show = min(N_EXAMPLES // 2, len(correct_samples), len(wrong_samples))
n_show = max(n_show, 1)

rng = random.Random(SEED)
correct_show = rng.sample(correct_samples, n_show)
wrong_show   = rng.sample(wrong_samples,   n_show)

fig2, axes = plt.subplots(2, n_show, figsize=(n_show * 3, 7))
if n_show == 1:
    axes = axes[:, np.newaxis]

for col, (path, gt, pred) in enumerate(correct_show):
    try:
        img = np.array(Image.open(path).convert("RGB").resize((224, 224)))
    except Exception:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
    axes[0, col].imshow(img)
    axes[0, col].set_title(f"GT={gt} Pred={pred}", fontsize=8, color="green")
    axes[0, col].axis("off")

for col, (path, gt, pred) in enumerate(wrong_show):
    try:
        img = np.array(Image.open(path).convert("RGB").resize((224, 224)))
    except Exception:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
    axes[1, col].imshow(img)
    axes[1, col].set_title(f"GT={gt} Pred={pred}", fontsize=8, color="red")
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Correct", fontsize=10, color="green")
axes[1, 0].set_ylabel("Wrong",   fontsize=10, color="red")

plt.suptitle("EfficientNet-B4 DR Grading: Sample Predictions", fontsize=12, y=1.01)
plt.tight_layout()
examples_path = os.path.join(OUTPUT_DIR, "prediction_examples.png")
fig2.savefig(examples_path, dpi=100, bbox_inches="tight")
plt.show()
print(f"Saved: {examples_path}")

# %%
metrics_df = pd.DataFrame({
    "class_id": list(range(NUM_CLASSES)),
    "class_name": [c.replace("\n", " ") for c in CLASS_NAMES],
    "support": sup,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "accuracy": per_class_acc,
})
metrics_csv = os.path.join(OUTPUT_DIR, "per_class_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(NUM_CLASSES)
w = 0.25
axes3[0].bar(x - w, prec, width=w, label="Precision")
axes3[0].bar(x, rec, width=w, label="Recall")
axes3[0].bar(x + w, f1, width=w, label="F1")
axes3[0].set_ylim(0, 1.0)
axes3[0].set_xticks(x)
axes3[0].set_xticklabels([str(i) for i in range(NUM_CLASSES)])
axes3[0].set_title("Per-class Metrics")
axes3[0].set_xlabel("Class")
axes3[0].legend()

gt_counts = np.array([(all_gt == i).sum() for i in range(NUM_CLASSES)])
pred_counts = np.array([(all_pred == i).sum() for i in range(NUM_CLASSES)])
axes3[1].bar(x - 0.2, gt_counts, width=0.4, label="GT")
axes3[1].bar(x + 0.2, pred_counts, width=0.4, label="Pred")
axes3[1].set_xticks(x)
axes3[1].set_xticklabels([str(i) for i in range(NUM_CLASSES)])
axes3[1].set_title("Class Distribution (GT vs Pred)")
axes3[1].set_xlabel("Class")
axes3[1].legend()

plt.tight_layout()
metrics_plot_path = os.path.join(OUTPUT_DIR, "per_class_metrics.png")
fig3.savefig(metrics_plot_path, dpi=100, bbox_inches="tight")
plt.show()
print(f"Saved: {metrics_plot_path}")
print(f"Saved: {metrics_csv}")

# %%
print("\n=== Final Summary ===")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Val samples evaluated: {len(all_gt)}")
print(f"Overall Accuracy:   {accuracy:.4f}  (target >= 0.85)")
print(f"Quadratic Kappa:    {kappa:.4f}  (target >= 0.90)")
print(f"Confusion matrix:   {cm_path}")
print(f"Sample predictions: {examples_path}")
print(f"Per-class metrics:  {metrics_plot_path}")
print(f"Metrics CSV:        {metrics_csv}")
