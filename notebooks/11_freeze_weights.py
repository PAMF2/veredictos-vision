#!/usr/bin/env python3
"""
11 - Freeze Weights Pack

Objetivo:
- Carregar checkpoints treinados
- Extrair apenas os pesos (state_dict)
- Salvar pesos "congelados" para inferencia
- Gerar manifest consolidado
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


OUT_DIR = Path("/kaggle/working/outputs/final_submission/frozen_weights")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        # Alguns checkpoints já são o próprio state_dict.
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore[return-value]
    raise RuntimeError("Checkpoint sem state_dict reconhecivel.")


def _num_params(state_dict: Dict[str, torch.Tensor]) -> int:
    total = 0
    for _, t in state_dict.items():
        if isinstance(t, torch.Tensor):
            total += t.numel()
    return total


def freeze_one(name: str, src: str, dst_name: str) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"{name}: checkpoint nao encontrado: {src}")

    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)

    dst_path = OUT_DIR / dst_name
    torch.save(state_dict, dst_path)

    meta: Dict[str, Any] = {
        "name": name,
        "source_checkpoint": str(src_path),
        "frozen_weights": str(dst_path),
        "num_tensors": len(state_dict),
        "num_params": _num_params(state_dict),
    }

    # Captura métricas se existirem no checkpoint
    if isinstance(ckpt, dict):
        for k in ("epoch", "dice_disc", "dice_cup", "val_dice", "val_loss", "val_acc", "val_kappa", "best_threshold"):
            if k in ckpt:
                v = ckpt[k]
                try:
                    meta[k] = float(v) if isinstance(v, (int, float)) else v
                except Exception:
                    meta[k] = str(v)
    return meta, state_dict


def main() -> None:
    models = [
        {
            "name": "transunet_glaucoma",
            "src": "/kaggle/working/outputs/transunet_glaucoma_best.pth",
            "dst": "transunet_glaucoma_state_dict.pth",
        },
        {
            "name": "unet_vessel_drive",
            "src": "/kaggle/working/outputs/unetpp/unet_r34_drive_best.pth",
            "dst": "unet_vessel_drive_state_dict.pth",
        },
        {
            "name": "efficientnet_dr_grading",
            "src": "/kaggle/working/outputs/efficientnet/efficientnet_dr_best.pth",
            "dst": "efficientnet_dr_grading_state_dict.pth",
        },
    ]

    manifest: Dict[str, Any] = {"models": []}
    for m in models:
        meta, _ = freeze_one(m["name"], m["src"], m["dst"])
        manifest["models"].append(meta)
        print(f"[OK] {m['name']} -> {meta['frozen_weights']}")

    manifest_path = OUT_DIR / "frozen_weights_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print("-", manifest_path)
    for m in manifest["models"]:
        print("-", m["frozen_weights"])


if __name__ == "__main__":
    main()
