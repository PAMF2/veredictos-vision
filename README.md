# Veredictos Vision - Sprint Final

Multi-pathology retinal screening pipeline for fundus images:
- Glaucoma segmentation (disc/cup)
- Vessel segmentation
- DR grading
- Clinical report generation with MedGemma

## Repository structure
- `notebooks/01_transunet_train.py`, `notebooks/01_transunet_test.py`: glaucoma model
- `notebooks/03_unetpp_train.py`, `notebooks/04_unetpp_test.py`: vessel model (DRIVE)
- `notebooks/07_efficientnet_train.py`, `notebooks/08_efficientnet_test.py`: DR grading model
- `notebooks/09_medgemma_integration.py`: MedGemma report integration
- `notebooks/11_submission_pack.py`: final package generation
- `notebooks/12_demo_upload_pipeline.py`: upload-based live demo UI
- `utils/medgemma_report.py`: prompt/report utilities
- `streamlit_app.py`: lightweight app entrypoint

## Final metrics
- TransUNet Glaucoma: Disc Dice `0.9551`, Cup Dice `0.8683`
- U-Net Vessel (DRIVE): Dice `0.7172`
- EfficientNet DR Grading: Accuracy `0.9616`, QWK `0.9793`

## Kaggle setup
1. Set accelerator to GPU (T4 recommended for MedGemma live demo).
2. Add Hugging Face token in Kaggle Secrets as `HF_TOKEN`.
3. Place repository under `/kaggle/working/sprintfinal` (or adjust paths accordingly).

## Run final package
```bash
python /kaggle/working/sprintfinal/notebooks/11_submission_pack.py
```

Outputs:
- `/kaggle/working/outputs/final_submission/summary_metrics.json`
- `/kaggle/working/outputs/final_submission/clinical_report_case*.txt`
- `/kaggle/working/outputs/final_submission/clinical_report_case*_meta.json`
- `/kaggle/working/outputs/final_submission/FINAL_REPORT.md`

## Run upload demo (video recording)
```bash
python /kaggle/working/sprintfinal/notebooks/12_demo_upload_pipeline.py
```

The demo provides:
- image upload
- pipeline signal estimation for showcase
- MedGemma clinical report generation

## Troubleshooting
- `ModuleNotFoundError: No module named 'utils'`:
  - fixed in `12_demo_upload_pipeline.py` by auto-injecting project root into `sys.path`
  - ensure you run the script from the repository copy (`/kaggle/working/sprintfinal`)
- If MedGemma fails to load:
  - verify `HF_TOKEN` in Secrets
  - confirm GPU is enabled

## Clinical safety note
This project is AI-assisted screening support and does not replace clinical diagnosis by an ophthalmologist.
