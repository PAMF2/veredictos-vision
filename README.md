# Veredictos Vision - Sprint Final

Multi-pathology retinal screening pipeline with MedGemma clinical report generation.

## What is included
- `notebooks/01_transunet_train.py` / `01_transunet_test.py`: glaucoma segmentation (disc/cup)
- `notebooks/03_unetpp_train.py` / `04_unetpp_test.py`: retinal vessel segmentation (DRIVE)
- `notebooks/07_efficientnet_train.py` / `08_efficientnet_test.py`: DR grading
- `notebooks/09_medgemma_integration.py`: MedGemma clinical report integration
- `notebooks/11_submission_pack.py`: final pack generation (metrics + multi-case reports)
- `notebooks/12_demo_upload_pipeline.py`: upload demo UI for video recording
- `utils/medgemma_report.py`: report prompt + fallback logic
- `streamlit_app.py`: lightweight app entrypoint

## Final metrics
- TransUNet Glaucoma: Disc Dice `0.9551`, Cup Dice `0.8683`
- U-Net Vessel (DRIVE): Dice `0.7172`
- EfficientNet DR Grading: Accuracy `0.9616`, QWK `0.9793`

## Quick run (Kaggle GPU)
```bash
python /kaggle/working/sprintfinal/notebooks/11_submission_pack.py
```

## Demo run (upload image UI)
```bash
python /kaggle/working/sprintfinal/notebooks/12_demo_upload_pipeline.py
```

## Notes
- This project is AI-assisted screening support and does not replace clinical diagnosis.
- Keep secrets (e.g., `HF_TOKEN`) in Kaggle Secrets, not in source code.
