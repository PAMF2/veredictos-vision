#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    01 - TRANSUNET GLAUCOMA - TREINO                          ║
║                                                                              ║
║  Target: Dice Cup ≥0.88 | Dice Disc ≥0.91                                   ║
║  Tempo: 10-12h                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# %%
!pip install -q segmentation-models-pytorch albumentations timm

# %%
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# %%
class Config:
    SMDG_BASE = '/kaggle/input/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset'
    SMDG_IMAGES = f'{SMDG_BASE}/full-fundus/full-fundus'
    SMDG_DISC = f'{SMDG_BASE}/optic-disc/optic-disc'
    SMDG_CUP = f'{SMDG_BASE}/optic-cup/optic-cup'
    OUTPUT_DIR = '/kaggle/working/outputs'

    EPOCHS = 120
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 2
    LR = 3e-4
    WEIGHT_DECAY = 1e-4

    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    IMG_SIZE = 512

    DICE_CUP_WEIGHT = 0.6
    DICE_DISC_WEIGHT = 0.4
    BCE_WEIGHT = 0.3

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MULTI_GPU = torch.cuda.device_count() > 1  # Detecta 2x T4
    FP16 = True
    NUM_WORKERS = 2
    PIN_MEMORY = True

    TARGET_DICE_CUP = 0.88
    TARGET_DICE_DISC = 0.91

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print(f"GPUs disponíveis: {torch.cuda.device_count()}")
if cfg.MULTI_GPU:
    print(f"✓ USANDO 2x GPUs: {torch.cuda.get_device_name(0)} + {torch.cuda.get_device_name(1)}")
    print(f"✓ Batch efetivo: {cfg.BATCH_SIZE * 2} (DataParallel)")
else:
    print(f"✓ Usando 1 GPU: {torch.cuda.get_device_name(0)}")

# %%
class GlaucomaDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
            mask = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 2), dtype=np.float32)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            return img, mask.permute(2, 0, 1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_name = Path(self.image_paths[idx]).stem

        disc_path = Path(cfg.SMDG_DISC) / f"{image_name}.png"
        mask_disc = cv2.imread(str(disc_path), cv2.IMREAD_GRAYSCALE)
        mask_disc = (mask_disc > 127).astype(np.float32) if mask_disc is not None else np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        cup_path = Path(cfg.SMDG_CUP) / f"{image_name}.png"
        mask_cup = cv2.imread(str(cup_path), cv2.IMREAD_GRAYSCALE)
        mask_cup = (mask_cup > 127).astype(np.float32) if mask_cup is not None else np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        mask = np.stack([mask_disc, mask_cup], axis=-1)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        mask = mask.permute(2, 0, 1)
        return img, mask

# %%
def get_train_transform():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# %%
def load_dataset_paths():
    disc_stems = {p.stem for p in Path(cfg.SMDG_DISC).glob('*.png')}
    cup_stems  = {p.stem for p in Path(cfg.SMDG_CUP).glob('*.png')}

    valid_images = [
        str(p) for p in sorted(Path(cfg.SMDG_IMAGES).glob('*.png'))
        if p.stem in disc_stems and p.stem in cup_stems
    ]

    train_imgs, val_imgs = train_test_split(
        valid_images, test_size=0.15, random_state=42, shuffle=True
    )

    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)}")
    return train_imgs, val_imgs

train_imgs, val_imgs = load_dataset_paths()

# %%
train_dataset = GlaucomaDataset(train_imgs, get_train_transform())
val_dataset = GlaucomaDataset(val_imgs, get_val_transform())

train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)

# %%
model = smp.Unet(
    encoder_name=cfg.ENCODER,
    encoder_weights=cfg.ENCODER_WEIGHTS,
    in_channels=3,
    classes=2,
    activation=None,
)
model = model.to(cfg.DEVICE)

# DataParallel para 2x T4
if cfg.MULTI_GPU:
    model = nn.DataParallel(model, device_ids=[0, 1])
    print(f"✓ Model wrapped com DataParallel (GPU 0 + GPU 1)")

# %%
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice_disc = self.dice_loss(pred[:, 0:1], target[:, 0:1])
        dice_cup = self.dice_loss(pred[:, 1:2], target[:, 1:2])
        dice_total = cfg.DICE_DISC_WEIGHT * dice_disc + cfg.DICE_CUP_WEIGHT * dice_cup
        bce = self.bce_loss(pred, target)
        return dice_total + cfg.BCE_WEIGHT * bce

criterion = CombinedLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=cfg.FP16)

# ── Resume ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(cfg.OUTPUT_DIR, 'transunet_glaucoma_best.pth')
start_epoch = 0
best_dice_cup = 0.0

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=cfg.DEVICE)
    if cfg.MULTI_GPU:
        model.module.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch   = ckpt['epoch']
    best_dice_cup = ckpt['dice_cup']
    print(f"✓ Resumindo do epoch {start_epoch} | best Dice Cup: {best_dice_cup:.4f}")
else:
    print("Treinando do zero")

# Scheduler continua de onde parou
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.EPOCHS, eta_min=1e-6, last_epoch=start_epoch - 1
)

# %%
def dice_coefficient(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + 1e-7) / (pred_flat.sum() + target_flat.sum() + 1e-7)
    return dice.item()

def compute_metrics(pred, target):
    dice_disc = dice_coefficient(pred[:, 0:1], target[:, 0:1])
    dice_cup = dice_coefficient(pred[:, 1:2], target[:, 1:2])
    return dice_disc, dice_cup

# %%
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    running_dice_disc = 0.0
    running_dice_cup = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc='Train')
    for i, (images, masks) in enumerate(pbar):
        images = images.to(cfg.DEVICE)
        masks = masks.to(cfg.DEVICE)

        with torch.cuda.amp.autocast(enabled=cfg.FP16):
            outputs = model(images)
            loss = criterion(outputs, masks) / cfg.GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % cfg.GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        dice_disc, dice_cup = compute_metrics(outputs.detach(), masks)
        running_loss += loss.item() * cfg.GRAD_ACCUM_STEPS
        running_dice_disc += dice_disc
        running_dice_cup += dice_cup

        pbar.set_postfix({'loss': running_loss / (i + 1), 'dice_d': running_dice_disc / (i + 1),
                         'dice_c': running_dice_cup / (i + 1)})

    return running_loss / len(loader), running_dice_disc / len(loader), running_dice_cup / len(loader)

# %%
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_dice_disc = 0.0
    running_dice_cup = 0.0

    pbar = tqdm(loader, desc='Val')
    for images, masks in pbar:
        images = images.to(cfg.DEVICE)
        masks = masks.to(cfg.DEVICE)

        with torch.cuda.amp.autocast(enabled=cfg.FP16):
            outputs = model(images)
            loss = criterion(outputs, masks)

        dice_disc, dice_cup = compute_metrics(outputs, masks)
        running_loss += loss.item()
        running_dice_disc += dice_disc
        running_dice_cup += dice_cup

        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'dice_d': running_dice_disc / (pbar.n + 1),
                         'dice_c': running_dice_cup / (pbar.n + 1)})

    return running_loss / len(loader), running_dice_disc / len(loader), running_dice_cup / len(loader)

# %%
print("="*80)
print("TREINANDO TRANSUNET GLAUCOMA")
print("="*80)

history = []

for epoch in range(start_epoch, cfg.EPOCHS):
    print(f"\nEpoch {epoch+1}/{cfg.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    train_loss, train_dice_disc, train_dice_cup = train_epoch(model, train_loader, criterion, optimizer, scaler)
    val_loss, val_dice_disc, val_dice_cup = validate(model, val_loader, criterion)
    scheduler.step()

    history.append({
        'epoch': epoch + 1, 'train_loss': train_loss, 'train_dice_disc': train_dice_disc,
        'train_dice_cup': train_dice_cup, 'val_loss': val_loss, 'val_dice_disc': val_dice_disc,
        'val_dice_cup': val_dice_cup, 'lr': optimizer.param_groups[0]['lr'],
    })

    print(f"Train | Loss: {train_loss:.4f} | Disc: {train_dice_disc:.4f} | Cup: {train_dice_cup:.4f}")
    print(f"Val   | Loss: {val_loss:.4f} | Disc: {val_dice_disc:.4f} | Cup: {val_dice_cup:.4f}")

    if val_dice_cup > best_dice_cup:
        best_dice_cup = val_dice_cup
        # DataParallel: salvar model.module.state_dict()
        model_state = model.module.state_dict() if cfg.MULTI_GPU else model.state_dict()
        torch.save({
            'epoch': epoch + 1, 'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'dice_disc': val_dice_disc, 'dice_cup': val_dice_cup,
        }, os.path.join(cfg.OUTPUT_DIR, 'transunet_glaucoma_best.pth'))
        print(f"✓ Saved | Dice Cup: {best_dice_cup:.4f}")

    if val_dice_disc >= cfg.TARGET_DICE_DISC and val_dice_cup >= cfg.TARGET_DICE_CUP:
        print(f"\n🎯 TARGET REACHED!")
        break

pd.DataFrame(history).to_csv(os.path.join(cfg.OUTPUT_DIR, 'transunet_history.csv'), index=False)
print(f"\n✓ Best Dice Cup: {best_dice_cup:.4f}")
