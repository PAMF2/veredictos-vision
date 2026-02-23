# Sprint Final - Project Context (Veredictos Vision)

## Objective
Build a reproducible multi-pathology retinal screening pipeline with:
- Glaucoma structural signal (CDR)
- Diabetic retinopathy grading (0-4)
- Vessel density biomarker
- Clinical report generation via MedGemma

All execution targeted Kaggle free infrastructure.

## What Was Built
- `01_transunet_train.py` + `01_transunet_test.py`: optic disc/cup segmentation pipeline and glaucoma-related structural signal extraction.
- `03_unetpp_train.py` + `04_unetpp_test.py`: vessel segmentation branch and vessel-density estimate.
- `07_efficientnet_train.py` + `08_efficientnet_test.py`: DR grading branch (EfficientNet-B3).
- `09_medgemma_integration.py`: MedGemma clinical-report integration with guardrails.
- `11_submission_pack.py`: final packaging pipeline for report artifacts.
- `12_demo_upload_pipeline.py`: live demo UI (path-based on Kaggle) with 3 real checkpoints + MedGemma.

## Trained Checkpoints
- Glaucoma: `/kaggle/working/outputs/transunet_glaucoma_best.pth`
- Vessel: `/kaggle/working/outputs/unetpp/unet_r34_drive_best.pth`
- DR grading: `/kaggle/working/outputs/efficientnet/efficientnet_dr_best.pth`

## Final Metrics (Best Reported)
- DR grading (QWK): **0.979**
- Glaucoma cup Dice: **0.868**
- Vessel score: **0.717**
- DR validation snapshot: accuracy **0.9616**, strong per-class precision/recall/F1

## MedGemma Integration Notes
Main issues encountered and solved:
- Gated model access (`google/medgemma-4b-it`): required HF access + `HF_TOKEN` in Kaggle secrets.
- TPU instability for this workload in Kaggle: moved to GPU-first for reliable generation.
- Prompt echo/template contamination: introduced generation retries + output sanitation.
- Final demo branch in `12_demo_upload_pipeline.py` includes post-processing and versioned diagnostics.

## Current Stable Demo State
- 3 checkpoints load and infer correctly.
- Device path confirms GPU (`cuda:0`) when session is healthy.
- MedGemma generation runs with retries and cleaned output.
- Diagnostic signals include `pipeline_version` and per-branch output values.

## Artifacts Generated for Submission
- Final report markdown and clinical cases under `/kaggle/working/outputs/final_submission`
- MedGemma clinical report files under `/kaggle/working/outputs/medgemma`
- Technical overview HTML: `11_technical_overview.html`

## Practical Constraints Documented
- Kaggle Gradio upload had event-loop instability in some sessions.
- Path-based image input was kept as robust fallback for live demo continuity.
- CUDA context occasionally becomes poisoned after repeated mixed runs; restart session and run demo notebook first.

## Delivery Status
Core technical deliverables are complete:
- Vision models trained and exported
- MedGemma integrated and running
- End-to-end demo available
- Technical report and final packaging script available

Remaining work is presentation polish (video narration, final PDF formatting, and submission assembly).
