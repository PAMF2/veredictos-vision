# Weights

Large `.pth` files are intentionally excluded from git history (`*.pth` in `.gitignore`).

Use the release assets to download model weights:
- https://github.com/PAMF2/veredictos-vision/releases/tag/v1.0-weights

Expected files:
- `transunet_glaucoma_best.pth`
- `efficientnet_dr_best.pth`
- `unet_r34_drive_best.pth`

Integrity verification:
- check `weights_manifest.json` (SHA256 + size)

Notes:
- Keep local copies in this folder for inference.
- For distribution, prefer GitHub Releases or Kaggle Dataset artifacts.
