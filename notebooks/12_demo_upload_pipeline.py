#!/usr/bin/env python3
"""
12 - Demo Upload Pipeline (Kaggle/Notebook)

Interface simples para demo:
- Upload da imagem de fundo de olho
- Estimativa rapida de sinais (CDR/DR/vasos) para showcase
- Geração de relatorio clinico via MedGemma (ou fallback)

Uso em notebook:
    !python /kaggle/working/sprintfinal/notebooks/12_demo_upload_pipeline.py
"""

import os
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from utils.medgemma_report import ScreeningResults, generate_clinical_report


def _estimate_from_image(img: np.ndarray) -> Dict[str, float]:
    """
    Estimativa leve para demo de pipeline (nao substitui inferencia real dos checkpoints).
    """
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img.astype(np.float32) / 255.0

    red = float(np.mean(img[..., 0]))
    green = float(np.mean(img[..., 1]))
    blue = float(np.mean(img[..., 2]))

    vessel_density = float(np.clip(0.06 + 0.22 * (green - blue + 0.2), 0.03, 0.35))
    cdr = float(np.clip(0.38 + 0.35 * (red - green + 0.15), 0.25, 0.85))

    dr_score = float(np.clip((red * 1.2 + (1.0 - green) * 0.8), 0.0, 1.0))
    if dr_score < 0.30:
        dr_grade = 0
    elif dr_score < 0.45:
        dr_grade = 1
    elif dr_score < 0.62:
        dr_grade = 2
    elif dr_score < 0.78:
        dr_grade = 3
    else:
        dr_grade = 4
    dr_conf = float(np.clip(0.70 + 0.28 * abs(dr_score - 0.5), 0.70, 0.97))

    glaucoma_risk = "low"
    if cdr >= 0.75:
        glaucoma_risk = "emergent"
    elif cdr >= 0.65:
        glaucoma_risk = "high"
    elif cdr >= 0.55:
        glaucoma_risk = "moderate"

    return {
        "cdr": cdr,
        "glaucoma_risk": glaucoma_risk,
        "dr_grade": int(dr_grade),
        "dr_conf": dr_conf,
        "vessel_density": vessel_density,
    }


def _dr_label(grade: int) -> str:
    return {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative",
    }.get(int(grade), "Unknown")


def run_pipeline(image: Image.Image, use_medgemma: bool, model_id: str) -> Tuple[str, str]:
    if image is None:
        return "No image uploaded.", "{}"

    arr = np.array(image.convert("RGB"))
    est = _estimate_from_image(arr)
    results = ScreeningResults(
        cdr=est["cdr"],
        glaucoma_risk=est["glaucoma_risk"],
        dr_grade=est["dr_grade"],
        dr_label=_dr_label(est["dr_grade"]),
        dr_conf=est["dr_conf"],
        vessel_density=est["vessel_density"],
    )

    output = generate_clinical_report(
        results=results,
        use_medgemma=bool(use_medgemma),
        model_id=model_id,
    )

    metrics_text = (
        f"mode: {output.get('mode', 'unknown')}\n"
        f"cdr: {results.cdr:.3f}\n"
        f"glaucoma_risk: {results.glaucoma_risk}\n"
        f"dr_grade: {results.dr_grade} ({results.dr_label})\n"
        f"dr_conf: {results.dr_conf:.3f}\n"
        f"vessel_density: {results.vessel_density:.3f}\n"
    )
    if output.get("error"):
        metrics_text += f"error: {output['error']}\n"

    return output["report"], metrics_text


def main() -> None:
    # Keep default to GPU flow for Kaggle T4 demo.
    os.environ.setdefault("FORCE_GPU_ONLY", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")

    import gradio as gr

    with gr.Blocks(title="Veredictos Vision Demo Pipeline") as demo:
        gr.Markdown("# Veredictos Vision - Retinal Screening Demo")
        gr.Markdown(
            "Upload de imagem de fundo de olho -> estimativa pipeline -> relatorio clinico MedGemma."
        )

        with gr.Row():
            image = gr.Image(type="pil", label="Fundus image")
            with gr.Column():
                use_medgemma = gr.Checkbox(value=True, label="Use MedGemma")
                model_id = gr.Textbox(
                    value="google/medgemma-4b-it",
                    label="Model ID",
                )
                run_btn = gr.Button("Run Pipeline", variant="primary")

        report = gr.Textbox(label="Clinical Report", lines=18)
        metrics = gr.Textbox(label="Pipeline Signals", lines=10)

        run_btn.click(
            fn=run_pipeline,
            inputs=[image, use_medgemma, model_id],
            outputs=[report, metrics],
        )

    demo.launch(share=False)


if __name__ == "__main__":
    main()
