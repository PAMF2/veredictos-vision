# Veredictos Vision

A multi-agent retinal screening system that fuses three specialist vision models with MedGemma to produce clinically structured, signal-grounded reports.

## Why this project

Most retinal AI pipelines optimize one task at a time. Veredictos Vision is designed as an integrated screening stack:

- structural glaucoma signal (CDR)
- diabetic retinopathy staging (5-grade)
- retinal vessel biomarker extraction
- controlled clinical-language synthesis

The objective is not only high benchmark scores, but reproducibility, traceability, and deployment robustness in low-resource environments.

## Methodological framing

Veredictos Vision is formulated as a constrained multi-objective system:

- maximize branch-specific clinical performance
- preserve interpretability through explicit intermediate variables
- enforce signal-text consistency in the language layer
- maintain execution stability under limited infrastructure

Formally, the pipeline exposes a structured state tuple
`z = (CDR, glaucoma_risk, DR_grade, DR_confidence, vessel_density)`,
and report generation is treated as a constrained mapping `r = G(z)`, not an unconstrained free-form diagnosis process.

## System architecture

Veredictos Vision uses four coordinated agents:

1. **Glaucoma Agent (TransUNet-style)**
- Disc/cup segmentation
- CDR estimation
- Structural glaucoma risk category

2. **DR Agent (EfficientNet-B3)**
- 5-class diabetic retinopathy grading
- Confidence output
- Ordinal-aware evaluation (QWK)

3. **Vessel Agent (UNet++)**
- Retinal vessel segmentation
- Vessel-density biomarker

4. **Clinical Report Agent (MedGemma)**
- Consumes structured outputs from agents 1-3
- Generates clinician-facing narrative
- Constrained by consistency checks and output sanitation

## Final results

- **DR grading**: Accuracy **0.9616**, QWK **0.9793**
- **Glaucoma branch**: Disc Dice **0.9551**, Cup Dice **0.8683**
- **Vessel branch**: Dice/score **0.7172**

## Experimental protocol (summary)

- Branches were trained independently with endpoint-aligned objectives.
- DR model selection prioritized **QWK** (ordinal consistency), not only flat accuracy.
- Data leakage checks were enforced before final metric reporting.
- Inference outputs were exported via deterministic artifact paths for reproducibility.
- MedGemma outputs were gated by consistency/sanitation checks to reduce instruction echo and numeric drift.

## Repository layout

```text
notebooks/
  01_*                # glaucoma training/eval scripts
  03_* / 04_*         # vessel training/eval scripts
  07_* / 08_*         # DR training/eval scripts
  09_medgemma_integration.py
  11_submission_pack.py
  12_demo_upload_pipeline.py
  pack_essential_artifacts.py

artifacts/            # metrics and key figures
weights/              # README + manifest (hashes/sizes)
utils/
  medgemma_report.py
PROJECT_DESCRIPTION.md
README.md
```


## Datasets and citations

See [`DATASETS.md`](DATASETS.md) for:
- dataset sources and links
- branch-by-branch usage in this pipeline
- citation and licensing notes
## Model weights

Weights are provided as GitHub Release assets (not tracked directly in git history).

- Release: **v1.0-weights**
- URL: `https://github.com/PAMF2/veredictos-vision/releases/tag/v1.0-weights`

Included files:
- `transunet_glaucoma_best.pth`
- `efficientnet_dr_best.pth`
- `unet_r34_drive_best.pth`
- `weights_manifest.json` (SHA256 + file sizes)

## Quickstart (Kaggle)

### 1) Environment
- Enable **GPU** (T4 recommended).
- Add `HF_TOKEN` in Kaggle Secrets (with access to `google/medgemma-4b-it`).
- Place repository at `/kaggle/working/sprintfinal` (or adjust paths).

### 2) Generate final submission package

```bash
python /kaggle/working/sprintfinal/notebooks/11_submission_pack.py
```

Output:
- `/kaggle/working/outputs/final_submission/summary_metrics.json`
- `/kaggle/working/outputs/final_submission/clinical_report_case*.txt`
- `/kaggle/working/outputs/final_submission/clinical_report_case*_meta.json`
- `/kaggle/working/outputs/final_submission/FINAL_REPORT.md`

### 3) Run interactive demo pipeline

```bash
python /kaggle/working/sprintfinal/notebooks/12_demo_upload_pipeline.py
```

The demo provides:
- model loading checks
- deterministic signal extraction (`cdr`, `dr_grade`, `dr_conf`, `vessel_density`)
- MedGemma report generation with consistency controls

## Artifact packaging

To package key files into a zip on Kaggle:

```bash
python /kaggle/working/sprintfinal/notebooks/pack_essential_artifacts.py
```

## Design principles

- **Deterministic fusion**: no hidden late-stage decision layer
- **Signal-first reporting**: language grounded in authoritative numeric outputs
- **Reproducibility-first**: deterministic output paths and checkpoint contracts
- **Operational resilience**: robust handling of runtime and generation failures

## Scientific scope and limitations

This repository reports challenge-grade, retrospective evaluation and engineering reproducibility.  
It does **not** claim prospective clinical validation or replacement of specialist judgment.  
Cross-site domain-shift analysis, uncertainty calibration, and clinician-rated report utility remain active extension directions.

## Troubleshooting

### `ModuleNotFoundError: utils`
Use the repository script directly (`/kaggle/working/sprintfinal/...`) so path injection resolves correctly.

### MedGemma access errors
- verify `HF_TOKEN`
- verify account access to gated model
- restart session after token or dependency changes

### CUDA device-side assert
- restart Kaggle session
- run only the target notebook first (clean GPU context)

## Clinical safety

This project is intended for **AI-assisted screening support** only and does not replace diagnosis, treatment planning, or specialist medical judgment.

## Citation

If this repository is useful, cite it as:

```bibtex
@misc{veredictosvision2026,
  title={Veredictos Vision: Multi-Agent Retinal Screening with Controlled Clinical Report Generation},
  author={Pedro Afonso M.F. and Gabriel Maia},
  year={2026},
  howpublished={\url{https://github.com/PAMF2/veredictos-vision}}
}
```

