#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    01 - TRANSUNET GLAUCOMA - TESTE                           ║
║                                                                              ║
║  Carrega modelo treinado e testa em imagens                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# %%
import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import matplotlib.pyplot as plt

# %%
SMDG_BASE = '/kaggle/input/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset'
SMDG_IMAGES = f'{SMDG_BASE}/full-fundus/full-fundus'
SMDG_DISC = f'{SMDG_BASE}/optic-disc/optic-disc'
SMDG_CUP = f'{SMDG_BASE}/optic-cup/optic-cup'
CHECKPOINT = '/kaggle/working/outputs/transunet_glaucoma_best.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint não encontrado: {CHECKPOINT}\n"
        f"Execute 01_transunet_train.py primeiro!"
    )

# %%
model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights=None,
    in_channels=3,
    classes=2,
    activation=None,
)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

print(f"✓ Modelo carregado | Dice Cup: {checkpoint['dice_cup']:.4f} | Dice Disc: {checkpoint['dice_disc']:.4f}")

# %%
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# %%
def predict_image(image_path):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    augmented = transform(image=img_rgb)
    img_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(img_tensor)
            output = torch.sigmoid(output)

    pred_disc = output[0, 0].cpu().numpy()
    pred_cup = output[0, 1].cpu().numpy()

    pred_disc_mask = (pred_disc > 0.5).astype(np.uint8) * 255
    pred_cup_mask = (pred_cup > 0.5).astype(np.uint8) * 255

    return img_rgb, pred_disc_mask, pred_cup_mask

# %%
disc_stems = {p.stem for p in Path(SMDG_DISC).glob('*.png')}
cup_stems  = {p.stem for p in Path(SMDG_CUP).glob('*.png')}
test_images = [p for p in sorted(Path(SMDG_IMAGES).glob('*.png'))
               if p.stem in disc_stems and p.stem in cup_stems][:10]

for img_path in test_images:
    img, pred_disc, pred_cup = predict_image(img_path)

    # Load GT masks
    name = img_path.stem
    gt_disc = cv2.imread(str(Path(SMDG_DISC) / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    gt_cup = cv2.imread(str(Path(SMDG_CUP) / f"{name}.png"), cv2.IMREAD_GRAYSCALE)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_disc, cmap='gray')
    axes[0, 1].set_title('GT Disc')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pred_disc, cmap='gray')
    axes[0, 2].set_title('Pred Disc')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(img)
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gt_cup, cmap='gray')
    axes[1, 1].set_title('GT Cup')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pred_cup, cmap='gray')
    axes[1, 2].set_title('Pred Cup')
    axes[1, 2].axis('off')

    plt.suptitle(f'{name}')
    plt.tight_layout()
    plt.show()

    # Compute CDR
    disc_area = (pred_disc > 127).sum()
    cup_area = (pred_cup > 127).sum()
    cdr = np.sqrt(cup_area / disc_area) if disc_area > 0 else 0

    gt_disc_area = (gt_disc > 127).sum()
    gt_cup_area = (gt_cup > 127).sum()
    gt_cdr = np.sqrt(gt_cup_area / gt_disc_area) if gt_disc_area > 0 else 0

    print(f"{name} | CDR Pred: {cdr:.3f} | CDR GT: {gt_cdr:.3f} | Diff: {abs(cdr - gt_cdr):.3f}")
    print("-" * 80)
