# Veredictos Vision

## Project Description

Veredictos Vision is a **modular, multi-agent retinal screening framework** designed for integrated fundus interpretation under constrained compute environments. The system combines specialized deep-learning perception agents with a controlled medical language layer to produce clinically structured outputs that remain numerically grounded and operationally reproducible.

Unlike monolithic pipelines that optimize a single endpoint, Veredictos Vision explicitly models retinal screening as a **multi-objective systems problem**: diabetic retinopathy staging, glaucoma structural risk estimation, and vascular-biomarker extraction must be solved jointly while preserving traceability and consistency in final communication.

---

## Clinical Problem and Motivation

Fundus-based triage in real practice is multi-factorial. Clinicians simultaneously evaluate lesion burden, optic-disc/cup morphology, and vascular condition. Most benchmark systems still isolate these tasks, which creates a translational gap between leaderboard performance and practical screening utility.

Our motivation was to design a pipeline that:
- Preserves explicit disease-relevant signals rather than opaque latent decisions.
- Supports reproducibility in free-tier infrastructure.
- Produces clinically coherent text without allowing uncontrolled language-model behavior.

---

## Multi-Agent Architecture

Veredictos Vision uses four coordinated agents:

### 1) Glaucoma Structural Agent
- **Model family:** TransUNet-style segmentation.
- **Objective:** segment optic disc/cup and derive cup-to-disc ratio (CDR).
- **Output:** CDR + structural glaucoma risk category.

### 2) Diabetic Retinopathy Agent
- **Model family:** EfficientNet-B3 classifier.
- **Objective:** classify DR into 5 ordinal grades.
- **Output:** DR grade + calibrated confidence.

### 3) Vascular Agent
- **Model family:** UNet++ segmentation.
- **Objective:** segment retinal vasculature and estimate vessel density.
- **Output:** vessel mask-derived density biomarker.

### 4) Clinical Language Agent
- **Model:** MedGemma.
- **Objective:** convert structured outputs into clinically readable report text.
- **Constraint:** language generation is grounded in authoritative numeric signals.

This is a true multi-agent design because each module has a distinct task objective, output schema, and validation criterion, and integration happens through deterministic orchestration rather than hidden ensemble voting.

---

## Deterministic Fusion and Traceability

The fused state is explicitly represented as:

- `CDR`
- `glaucoma_risk`
- `DR_grade`
- `DR_confidence`
- `vessel_density`

All downstream report logic uses this typed tuple. This enables full traceability:
- from checkpoint -> to numerical signal,
- from numerical signal -> to final clinical narrative.

By design, no language output is accepted without consistency checks against these structured values.

---

## Methodological Principles

### Task-specialized optimization
Each agent is trained and selected against endpoint-specific metrics, avoiding cross-task compromises that often occur in shared-backbone training.

### Ordinal-aware DR evaluation
Model selection for DR prioritizes **Quadratic Weighted Kappa (QWK)** to respect severity ordering, rather than relying only on flat accuracy.

### Reliability as a first-class objective
Runtime instability (accelerator state, gated model access, generation drift) is treated as an experimental dimension, not a post-hoc engineering issue.

### Controlled language generation
MedGemma operates under consistency gating, sanitation, and staged refinement to prevent instruction echo, prompt leakage, and numeric mismatch.

---

## Runtime Strategy and Safety Controls

To support stable operation in notebook-based infrastructure, we implemented:

- Preflight dependency checks.
- GPU/CPU-aware loading logic.
- Defensive generation settings and input sanitization.
- Multi-pass MedGemma flow (generation, audit, formatting).
- Numeric consistency validation before accepting final text.
- Deterministic self-healing mode for operational continuity.

These controls reduce fragile demo behavior and improve reproducibility under real challenge constraints.

---

## Quantitative Performance

Under free Kaggle compute, final branch outcomes were:

- **DR grading:** QWK = **0.979**
- **Cup segmentation:** Dice = **0.868**
- **Vessel segmentation score:** **0.717**

These results were obtained with explicit artifact contracts and model checkpoint provenance, allowing reproducible reruns.

---

## Ablation and Integration Findings

Ablation analysis indicated:

- Removing branch signals degrades triage coherence and recommendation quality.
- Language strictness alone can increase rejection rates; constrained refinement is more robust.
- Deterministic signal anchoring is critical to prevent narrative drift.
- Structured numeric consistency checks meaningfully reduce clinically incoherent outputs.

Overall, the best operational frontier was achieved through specialized perception + deterministic fusion + controlled narrative refinement.

---

## Reproducibility and Deliverables

The project includes reproducible outputs for challenge submission:

- Trained checkpoints for all three vision agents.
- Evaluation metrics and plots.
- Clinical report generation pipeline.
- Demo interface for path-based fundus inference.
- Submission-oriented technical manuscript assets.

All artifacts are exported through deterministic paths to simplify external validation and packaging.

---

## Scientific and Practical Contribution

The primary contribution of Veredictos Vision is not a single benchmark number, but a **reproducible systems methodology** for low-resource medical AI:

1. Decompose heterogeneous clinical tasks into specialist agents.
2. Preserve explicit intermediate signals for interpretability.
3. Fuse deterministically to avoid hidden end-stage decisions.
4. Constrain language generation to structured evidence.
5. Engineer runtime resilience as part of the scientific protocol.

This framework is especially relevant for teams without enterprise compute budgets who still require auditable and clinically meaningful screening outputs.

---

## Limitations and Next Steps

Current limitations include:
- Challenge-grade retrospective validation (no prospective clinical trial in this phase).
- Limited external domain-shift analysis across devices/sites.
- Need for broader clinician-rated evaluation of report usefulness.

Planned next steps:
- Cross-site external validation.
- Calibration and uncertainty reporting by class.
- Expanded human-in-the-loop report quality assessment.
- Workflow-level studies for triage adoption.

---

## Final Statement

Veredictos Vision demonstrates that high-quality, multi-pathology retinal triage can be implemented with strong methodological rigor in a low-resource setting. By combining specialist perception agents with deterministic fusion and controlled MedGemma reporting, the system provides a practical bridge between benchmark performance and deployment-oriented clinical communication.
