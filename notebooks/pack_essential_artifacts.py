#!/usr/bin/env python3
"""
Pack essential Veredictos Vision artifacts from /kaggle/working into one zip.
Run in Kaggle after training/inference completes.
"""

from pathlib import Path
import zipfile

BASE = Path('/kaggle/working')
ZIP_NAME = 'veredictos_essential_artifacts.zip'

ESSENTIAL_FILES = [
    # Final model weights (for your GitHub weights/ folder)
    'outputs/transunet_glaucoma_best.pth',
    'outputs/efficientnet/efficientnet_dr_best.pth',
    'outputs/unetpp/unet_r34_drive_best.pth',

    # Metrics and key plots
    'outputs/final_submission/summary_metrics.json',
    'outputs/efficientnet/per_class_metrics.csv',
    'outputs/efficientnet/confusion_matrix.png',
    'outputs/efficientnet/per_class_metrics.png',
    'outputs/efficientnet/prediction_examples.png',
    'outputs/unetpp/unetpp_test_results.png',
]


def main() -> None:
    zip_path = BASE / ZIP_NAME
    found = []
    missing = []

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel in ESSENTIAL_FILES:
            src = BASE / rel
            if src.exists():
                zf.write(src, arcname=rel)
                found.append(rel)
            else:
                missing.append(rel)

    print('ZIP created:', zip_path)
    print('Size (MB):', round(zip_path.stat().st_size / (1024 * 1024), 2))
    print('\nIncluded files:')
    for f in found:
        print(' -', f)

    if missing:
        print('\nMissing files (not added):')
        for f in missing:
            print(' -', f)


if __name__ == '__main__':
    main()
