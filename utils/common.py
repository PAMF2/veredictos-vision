#!/usr/bin/env python3
# Funções comuns reutilizadas em todos os dias

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_fundus(image, target_size=512):
    """
    Preprocessing padrão para imagens de fundo de olho:
    1. Resize mantendo aspect ratio
    2. Center crop
    3. CLAHE para equalização
    4. Circle mask para remover bordas
    """
    h, w = image.shape[:2]

    # Resize mantendo aspect ratio
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    resized = cv2.resize(image, (new_w, new_h))

    # Center crop
    start_h = (new_h - target_size) // 2
    start_w = (new_w - target_size) // 2
    cropped = resized[start_h:start_h+target_size, start_w:start_w+target_size]

    # CLAHE em canal LAB
    lab = cv2.cvtColor(cropped, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Circle mask (remove bordas pretas)
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    center = (target_size // 2, target_size // 2)
    radius = int(target_size * 0.45)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply mask
    for i in range(3):
        enhanced[:, :, i] = cv2.bitwise_and(enhanced[:, :, i], enhanced[:, :, i], mask=mask)

    return enhanced


def ben_graham_preprocessing(image):
    """
    Ben Graham's preprocessing para DR detection:
    - Subtract local average color
    - Clip extremes
    - Scale to [0, 255]
    """
    image = image.astype(np.float32)

    # Gaussian blur para média local
    ksize = int(image.shape[0] * 0.1) | 1  # Ensure odd
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # Subtract
    preprocessed = image - blurred

    # Clip extremes
    preprocessed = np.clip(preprocessed, -1, 1)

    # Scale to [0, 255]
    preprocessed = ((preprocessed + 1) * 127.5).astype(np.uint8)

    return preprocessed


# =============================================================================
# METRICS
# =============================================================================

def dice_coefficient(pred, target, threshold=0.5, smooth=1e-7):
    """
    Calcula Dice coefficient
    pred: tensor (B, C, H, W) logits ou probabilities
    target: tensor (B, C, H, W) binary
    """
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    pred = (pred > threshold).float()

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return dice.item()


def iou_score(pred, target, threshold=0.5, smooth=1e-7):
    """IoU (Jaccard Index)"""
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    pred = (pred > threshold).float()

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def sensitivity_specificity(pred, target, threshold=0.5):
    """Sensitivity (Recall) e Specificity"""
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    pred = (pred > threshold).float()

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    tp = ((pred_flat == 1) & (target_flat == 1)).sum().float()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum().float()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().float()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().float()

    sensitivity = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)

    return sensitivity.item(), specificity.item()


# =============================================================================
# LOSSES
# =============================================================================

class DiceLoss(nn.Module):
    """Differentiable Dice Loss"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss para desequilíbrio de classes
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        focal_loss = focal_weight * bce

        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalização do Dice
    α > β → penaliza mais false negatives (melhor recall)
    α < β → penaliza mais false positives (melhor precision)
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1 - tversky


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_segmentation(image, mask_true, mask_pred, save_path=None):
    """
    Visualiza comparação segmentação
    image: (H, W, 3) numpy array
    mask_true: (H, W) numpy array
    mask_pred: (H, W) numpy array
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(mask_true, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(mask_pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Overlay
    overlay = image.copy()
    overlay[mask_pred > 0.5] = [255, 0, 0]  # Red for prediction
    overlay[mask_true > 0.5] = [0, 255, 0]  # Green for GT
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Red=Pred, Green=GT)')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save checkpoint (suporta DataParallel)"""
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, filepath)


def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """Load checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['metrics']


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping para evitar overfitting"""
    def __init__(self, patience=10, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


# =============================================================================
# DATA UTILITIES
# =============================================================================

def get_train_val_split(image_paths, mask_paths, val_ratio=0.15, seed=42):
    """Train/val split mantendo correspondência"""
    from sklearn.model_selection import train_test_split

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=val_ratio, random_state=seed
    )

    return train_imgs, val_imgs, train_masks, val_masks


def count_parameters(model):
    """Conta parâmetros treináveis"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6,
    }
