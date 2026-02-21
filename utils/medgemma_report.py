from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ScreeningResults:
    cdr: float
    glaucoma_risk: str
    dr_grade: int
    dr_label: str
    dr_conf: float
    vessel_density: float


def _overall_risk(results: ScreeningResults) -> str:
    if results.dr_grade >= 4 or results.cdr >= 0.75:
        return "emergent"
    if results.dr_grade >= 3 or results.cdr >= 0.65:
        return "high"
    if results.dr_grade >= 2 or results.cdr >= 0.55:
        return "moderate"
    return "low"


def _follow_up_recommendation(risk: str) -> str:
    if risk == "emergent":
        return "Urgent referral to ophthalmology within 24-72 hours."
    if risk == "high":
        return "Comprehensive ophthalmology evaluation within 1-2 weeks."
    if risk == "moderate":
        return "Specialist follow-up in 1-3 months with repeat fundus imaging."
    return "Routine annual eye screening and metabolic risk control."


def build_medgemma_prompt(results: ScreeningResults) -> str:
    return f"""You are a clinical ophthalmology AI assistant. Based on the following automated retinal screening results, generate a structured clinical report.

PATIENT SCREENING RESULTS:
- Glaucoma Assessment: CDR = {results.cdr:.3f} | Risk: {results.glaucoma_risk}
- Diabetic Retinopathy: Grade {results.dr_grade} ({results.dr_label}) | Confidence: {results.dr_conf:.1%}
- Vascular Analysis: Vessel density = {results.vessel_density:.1%} | Segmentation available

Generate a report with:
1. FINDINGS: Describe each pathology finding in clinical language
2. RISK ASSESSMENT: Overall risk level (low/moderate/high/emergent)
3. RECOMMENDATIONS: Next steps (follow-up interval, referrals, exams)
4. DISCLAIMER: This is AI-assisted screening, not a diagnosis.
"""


def generate_rule_based_report(results: ScreeningResults) -> str:
    risk = _overall_risk(results)
    follow_up = _follow_up_recommendation(risk)

    findings = [
        f"Glaucoma screening: cup-to-disc ratio (CDR) {results.cdr:.3f}, categorized as {results.glaucoma_risk} risk.",
        f"Diabetic retinopathy grading: grade {results.dr_grade} ({results.dr_label}) with model confidence {results.dr_conf:.1%}.",
        f"Retinal vascular analysis: estimated vessel density {results.vessel_density:.1%}.",
    ]

    return (
        "1. FINDINGS\n"
        + "\n".join([f"- {f}" for f in findings])
        + "\n\n2. RISK ASSESSMENT\n"
        + f"- Overall triage risk level: {risk}.\n\n"
        + "3. RECOMMENDATIONS\n"
        + f"- {follow_up}\n"
        + "- Correlate with visual acuity, IOP, OCT, and clinical examination as available.\n"
        + "- Repeat imaging if image quality is suboptimal or findings are discordant.\n\n"
        + "4. DISCLAIMER\n"
        + "- This is AI-assisted retinal screening support and does not replace clinical diagnosis."
    )


class MedGemmaGenerator:
    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        max_new_tokens: int = 420,
        temperature: float = 0.2,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tokenizer = None
        self._model = None
        self._load_error = None

    def _lazy_load(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        except Exception as exc:
            self._load_error = exc

    def generate(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        self._lazy_load()
        if self._load_error is not None:
            return None, f"MedGemma load/generation failed: {self._load_error}"
        try:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text, None
        except Exception as exc:
            return None, f"MedGemma inference failed: {exc}"


def generate_clinical_report(
    results: ScreeningResults,
    use_medgemma: bool = True,
    model_id: str = "google/medgemma-4b-it",
) -> Dict[str, str]:
    prompt = build_medgemma_prompt(results)
    rule_report = generate_rule_based_report(results)

    if not use_medgemma:
        return {
            "mode": "rule_based_only",
            "report": rule_report,
            "prompt": prompt,
            "error": "",
        }

    generator = MedGemmaGenerator(model_id=model_id)
    llm_report, error = generator.generate(prompt)
    if llm_report is None:
        return {
            "mode": "fallback_rule_based",
            "report": rule_report,
            "prompt": prompt,
            "error": error or "Unknown MedGemma error.",
        }
    return {
        "mode": "medgemma",
        "report": llm_report,
        "prompt": prompt,
        "error": "",
    }
