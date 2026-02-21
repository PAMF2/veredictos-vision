#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    00 - SETUP E VERIFICAÇÃO                                  ║
║                                                                              ║
║  Instala dependências e verifica datasets                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# %%
print("="*80)
print("INSTALANDO DEPENDÊNCIAS...")
print("="*80)

!pip install -q timm segmentation-models-pytorch albumentations
!pip install -q opencv-python-headless scikit-learn

print("\n✓ Instalação completa!")

# %%
import os
from pathlib import Path

# %%
print("\n" + "="*80)
print("TODOS OS DATASETS DISPONÍVEIS")
print("="*80)

# Lista simples de todos os datasets (user/dataset-name)
print("\n--- /kaggle/input/ ---")
!find /kaggle/input/datasets -mindepth 2 -maxdepth 2 -type d | sort

# Conta imagens por dataset
print("\n--- Contagem de imagens por dataset ---")
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.ppm', '.bmp', '.tif', '.tiff'}
dataset_counts = {}
for root, dirs, files in os.walk('/kaggle/input/datasets'):
    imgs = [f for f in files if Path(f).suffix.lower() in IMG_EXTS]
    if imgs:
        parts = Path(root).parts
        if len(parts) >= 6:
            key = f"{parts[4]}/{parts[5]}"
        else:
            key = str(Path(root).relative_to('/kaggle/input'))
        dataset_counts[key] = dataset_counts.get(key, 0) + len(imgs)

print(f"\n{'Dataset':<60} {'Imagens':>8}")
print("-"*70)
for ds, count in sorted(dataset_counts.items(), key=lambda x: -x[1]):
    print(f"  {ds:<58} {count:>8}")

# %%
print("\n" + "="*80)
print("DATASETS CONHECIDOS E SEUS PATHS")
print("="*80)

KNOWN_DATASETS = {
    'SMDG (Glaucoma)'         : '/kaggle/input/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset',
    'DDR Dataset'             : '/kaggle/input/datasets/mariaherrerot/ddrdataset',
    'APTOS 2019'              : '/kaggle/input/datasets/mariaherrerot/aptos2019',
    'APTOS 2019 JPG'          : '/kaggle/input/datasets/subhajeetdas/aptos-2019-jpg',
    'Messidor-2'              : '/kaggle/input/datasets/mariaherrerot/messidor2preprocess',
    'DR 224x224 Gaussian'     : '/kaggle/input/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered',
    'STARE'                   : '/kaggle/input/datasets/vidheeshnacode/stare-dataset',
    'Retinal Vessel Seg'      : '/kaggle/input/datasets/ipythonx/retinal-vessel-segmentation',
    'Eye Diseases Class'      : '/kaggle/input/datasets/gunavenkatdoddi/eye-diseases-classification',
    'DRIVE pixelwise'         : '/kaggle/input/datasets/srinjoybhuiya/drive-retinal-vessel-segmentation-pixelwise',
    'DRIVE original'          : '/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction',
}

for name, path in KNOWN_DATASETS.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗ NÃO ENCONTRADO"
    print(f"  {status}  {name:<25} {path}")

# %%
print("\n" + "="*80)
print("SMDG - DATASET PRINCIPAL (GLAUCOMA)")
print("="*80)

SMDG_BASE   = '/kaggle/input/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset'
SMDG_IMAGES = f'{SMDG_BASE}/full-fundus/full-fundus'
SMDG_DISC   = f'{SMDG_BASE}/optic-disc/optic-disc'
SMDG_CUP    = f'{SMDG_BASE}/optic-cup/optic-cup'

for label, path in [('Imagens', SMDG_IMAGES), ('Disc masks', SMDG_DISC), ('Cup masks', SMDG_CUP)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path não encontrado: {path}")
    print(f"  ✓ {label}: {path}")

images_all = sorted(Path(SMDG_IMAGES).glob('*.png'))
disc_stems  = {p.stem for p in Path(SMDG_DISC).glob('*.png')}
cup_stems   = {p.stem for p in Path(SMDG_CUP).glob('*.png')}
valid       = [p for p in images_all if p.stem in disc_stems and p.stem in cup_stems]

print(f"\n✓ Contagem SMDG:")
print(f"  - Total full-fundus   : {len(images_all)}")
print(f"  - Disc masks          : {len(disc_stems)}")
print(f"  - Cup masks           : {len(cup_stems)}")
print(f"  - VÁLIDAS (treino)    : {len(valid)}")

if len(valid) == 0:
    raise ValueError("Nenhuma imagem válida!")

# %%
import torch

print(f"\n✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

print("\n" + "="*80)
print(f"✓ SETUP COMPLETO | {len(valid)} imagens válidas para treino")
print("="*80)
