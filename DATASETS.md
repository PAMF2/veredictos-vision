# Datasets and Data Usage

This document lists the datasets used in Veredictos Vision, with usage scope and references.

## 1) Glaucoma structural branch (disc/cup -> CDR)

### Dataset
- **SMDG / Multichannel Glaucoma Benchmark Dataset** (Kaggle mirror used in experiments)
- Kaggle path used in code:
  - `/kaggle/input/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset`

### How it was used
- Input fundus images from `full-fundus/full-fundus`.
- Training and evaluation of the TransUNet-style segmentation pipeline.
- Derived outputs: disc/cup masks -> CDR -> glaucoma structural risk category.

### Citation / source
- Kaggle dataset page (mirror used in this project):
  - `https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset`
- Original benchmark citation should be used according to the dataset card/license.

---

## 2) Diabetic Retinopathy grading branch

### Datasets
- **APTOS 2019 Blindness Detection** (preprocessed Kaggle mirror)
  - code path: `/kaggle/input/datasets/mariaherrerot/aptos2019`
- **Messidor-2** (preprocessed Kaggle mirror)
  - code path: `/kaggle/input/datasets/mariaherrerot/messidor2preprocess`

### How they were used
- Harmonized into a unified DR training pool.
- Label schema aligned to 5-grade DR classification.
- Leakage checks applied before train/validation evaluation.
- EfficientNet-B3 trained for ordinal DR staging.
- Primary endpoint: Quadratic Weighted Kappa (QWK).

### Citation / source
- APTOS challenge/data page:
  - `https://www.kaggle.com/competitions/aptos2019-blindness-detection`
- Messidor-2 Kaggle mirror used:
  - `https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess`
- If publishing, also cite the original Messidor-2 source as required by its terms.

---

## 3) Vessel segmentation branch

### Primary dataset
- **DRIVE: Digital Retinal Images for Vessel Extraction**
  - code path candidates include:
    - `/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction`

### Fallback/auxiliary dataset candidates in code
- **STARE dataset**
  - `/kaggle/input/datasets/vidheeshnacode/stare-dataset`
- **DRIVE pixelwise segmentation mirror**
  - `/kaggle/input/datasets/srinjoybhuiya/drive-retinal-vessel-segmentation-pixelwise`

### How they were used
- UNet/UNet++ vessel segmentation training and testing.
- Mask supervision for vessel map learning.
- Post-processing to derive vessel-density biomarker.

### Citation / source
- DRIVE original reference site:
  - `https://drive.grand-challenge.org/`
- Kaggle DRIVE mirror used in experiments:
  - `https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction`
- STARE mirror:
  - `https://www.kaggle.com/datasets/vidheeshnacode/stare-dataset`

---

## 4) MedGemma language layer

### Note
- The language layer does **not** use image datasets directly for prediction.
- It consumes structured outputs from the vision branches:
  - `CDR`, `glaucoma_risk`, `DR_grade`, `DR_confidence`, `vessel_density`.

### Model source
- `google/medgemma-4b-it` (gated model access required).

---

## Licensing and compliance

- This repository does **not** redistribute original dataset files.
- Users must download datasets from official sources/Kaggle pages under their licenses.
- For external publication/submission, cite both Kaggle mirror pages used in execution and original dataset publications/cards when required.
