#!/usr/bin/env python3
"""
09 - MedGemma Integration (script)

Gera relatorio clinico estruturado com fallback rule-based.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
import json
import os
import re
import warnings

# Reduce framework-noise in Kaggle runtime before importing transformers stack.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

warnings.filterwarnings(
    "ignore",
    message=".*Transparent hugepages are not enabled.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*np\\.object.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*tensorflow can conflict with `torch-xla`.*",
)


@dataclass
class ScreeningResults:
    cdr: float
    glaucoma_risk: str
    dr_grade: int
    dr_label: str
    dr_conf: float
    vessel_density: float


def build_prompt(r: ScreeningResults) -> str:
    return f"""You are a clinical ophthalmology AI assistant. Based on the following automated retinal screening results, generate a structured clinical report.

PATIENT SCREENING RESULTS:
- Glaucoma Assessment: CDR = {r.cdr:.3f} | Risk: {r.glaucoma_risk}
- Diabetic Retinopathy: Grade {r.dr_grade} ({r.dr_label}) | Confidence: {r.dr_conf:.1%}
- Vascular Analysis: Vessel density = {r.vessel_density:.1%} | Segmentation available

Generate a report with:
1. FINDINGS
2. RISK ASSESSMENT (low/moderate/high/emergent)
3. RECOMMENDATIONS (follow-up interval, referrals, exams)
4. DISCLAIMER (AI-assisted screening, not diagnosis)

Output strictly with sections 1-4 only. Do not add extra sections, headers, or numbering.
"""


def overall_risk(r: ScreeningResults) -> str:
    if r.dr_grade >= 4 or r.cdr >= 0.75:
        return "emergent"
    if r.dr_grade >= 3 or r.cdr >= 0.65:
        return "high"
    if r.dr_grade >= 2 or r.cdr >= 0.55:
        return "moderate"
    return "low"


def rule_based_report(r: ScreeningResults) -> str:
    risk = overall_risk(r)
    follow = {
        "emergent": "Urgent ophthalmology referral within 24-72 hours.",
        "high": "Specialist assessment within 1-2 weeks.",
        "moderate": "Follow-up in 1-3 months with repeat retinal imaging.",
        "low": "Routine annual screening and risk-factor control.",
    }[risk]

    return (
        "1. FINDINGS\n"
        f"- Glaucoma screening: CDR {r.cdr:.3f}, risk category {r.glaucoma_risk}.\n"
        f"- DR grading: Grade {r.dr_grade} ({r.dr_label}), confidence {r.dr_conf:.1%}.\n"
        f"- Vascular analysis: estimated vessel density {r.vessel_density:.1%}.\n\n"
        "2. RISK ASSESSMENT\n"
        f"- Overall triage risk: {risk}.\n\n"
        "3. RECOMMENDATIONS\n"
        f"- {follow}\n"
        "- Correlate with clinical exam, IOP, OCT, and visual acuity.\n\n"
        "4. DISCLAIMER\n"
        "- This is AI-assisted retinal screening support and not a definitive diagnosis."
    )


def medgemma_generate(
    prompt: str,
    model_id: str = "google/medgemma-4b-it",
    max_new_tokens: int = 180,
) -> Tuple[Optional[str], Optional[str]]:
    import torch
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")
        except Exception:
            hf_token = ""

    if hf_token:
        login(token=hf_token)

    is_tpu = bool(os.getenv("TPU_NAME") or os.getenv("COLAB_TPU_ADDR") or os.getenv("TPU_WORKER_ID"))
    # Attempt chain: MedGemma first, then a much smaller instruct model for guaranteed runtime.
    candidate_models: List[str] = [model_id, os.getenv("SMALL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")]
    errors: List[str] = []

    for mid in candidate_models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(mid)

            if is_tpu:
                import torch_xla
                device = torch_xla.device()
                model = AutoModelForCausalLM.from_pretrained(
                    mid,
                    dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
                model = model.to(device)
                target_device = device
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    mid,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
                target_device = model.device

            model.eval()
            messages = [{"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                plain = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {"input_ids": plain["input_ids"], "attention_mask": plain.get("attention_mask")}
            inputs = {k: v.to(target_device) for k, v in inputs.items() if v is not None}

            with torch.no_grad():
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(inputs["input_ids"])
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=attention_mask,
                    max_new_tokens=120 if mid != model_id else max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=3,
                    max_time=35,
                    use_cache=False if is_tpu else True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            idx = text.find("1. FINDINGS")
            if idx >= 0:
                text = text[idx:].strip()
            if mid != model_id:
                text = "[Fallback LLM used: " + mid + "]\n\n" + text
            return text, None
        except Exception as e:
            errors.append(f"{mid}: {e}")
            continue

    return None, " | ".join(errors)


def clean_report_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t

    # Keep only content starting at the structured report section.
    idx = t.find("1. FINDINGS")
    if idx >= 0:
        t = t[idx:]

    # Remove common leaked prompt/template fragments.
    drop_patterns = [
        r"^Output strictly with sections 1-4 only\..*$",
        r"^Generate a report with:.*$",
        r"^1\.\s*FINDINGS\s*2\.\s*RISK ASSESSMENT.*$",
        r"^model\s*$",
    ]
    cleaned_lines: List[str] = []
    for line in t.splitlines():
        line_strip = line.strip()
        should_drop = any(re.match(p, line_strip, flags=re.IGNORECASE) for p in drop_patterns)
        if should_drop:
            continue
        cleaned_lines.append(line)

    t = "\n".join(cleaned_lines).strip()
    # Remove inline template chunks accidentally concatenated into one line.
    t = re.sub(
        r"(?is)1\.\s*FINDINGS\s*2\.\s*RISK ASSESSMENT.*?AI-assisted screening,\s*not diagnosis\)\s*",
        "",
        t,
    )
    # Normalize common markdown/header variants to canonical sections.
    t = re.sub(r"(?im)^\s*1\.\s*\**\s*FINDINGS\s*\**\s*:?\s*$", "1. FINDINGS", t)
    t = re.sub(r"(?im)^\s*2\.\s*\**\s*RISK\s+ASSESSMENT\s*\**\s*:?\s*$", "2. RISK ASSESSMENT", t)
    t = re.sub(r"(?im)^\s*3\.\s*\**\s*RECOMMENDATIONS\s*\**\s*:?\s*$", "3. RECOMMENDATIONS", t)
    t = re.sub(r"(?im)^\s*4\.\s*\**\s*DISCLAIMER\s*\**\s*:?\s*$", "4. DISCLAIMER", t)
    return t


def is_valid_structured_report(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    required = [
        "1. FINDINGS",
        "2. RISK ASSESSMENT",
        "3. RECOMMENDATIONS",
        "4. DISCLAIMER",
    ]
    if any(k not in t for k in required):
        return False

    # Reject template-only outputs with no real content.
    if "Output strictly with sections 1-4 only" in t:
        return False

    # Ensure there is substantive text after each section header.
    for i, sec in enumerate(required):
        start = t.find(sec)
        end = t.find(required[i + 1]) if i + 1 < len(required) else len(t)
        if start < 0:
            return False
        body = t[start + len(sec): end].strip()
        if len(body) < 8:
            return False
        # Reject degenerate section bodies like "1 FINDINGS".
        body_norm = re.sub(r"[\s\-\*\.:_]+", " ", body).strip().upper()
        if body_norm in {"1 FINDINGS", "FINDINGS", "2 RISK ASSESSMENT", "RISK ASSESSMENT"}:
            return False
    return True


def coerce_to_structured_report(text: str, r: ScreeningResults) -> str:
    t = clean_report_text(text)
    if is_valid_structured_report(t):
        return t
    # Soft recovery: keep usable medgemma content inside required 4-section skeleton.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    # Remove residual instruction-like fragments.
    lines = [
        ln for ln in lines
        if "Generate a report with" not in ln
        and "Output strictly with sections 1-4 only" not in ln
        and "RISK ASSESSMENT (low/moderate/high/emergent)" not in ln
    ]
    brief = " ".join(lines)[:400] if lines else ""
    # Remove nested/duplicated section tokens and leftover markdown markers.
    brief = re.sub(r"(?i)\b1\.\s*FINDINGS\b", "", brief).strip()
    brief = re.sub(r"(?i)\b2\.\s*RISK\s+ASSESSMENT\b", "", brief).strip()
    brief = re.sub(r"(?i)\b3\.\s*RECOMMENDATIONS\b", "", brief).strip()
    brief = re.sub(r"(?i)\b4\.\s*DISCLAIMER\b", "", brief).strip()
    brief = re.sub(r"\*\*|\*", "", brief).strip()
    brief = re.sub(r"\s{2,}", " ", brief).strip(" -:\n\t")
    # If MedGemma answer is too weak, return a robust clinical fallback.
    alpha_count = len(re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", brief))
    if alpha_count < 25 or brief.upper() in {"1 FINDINGS", "FINDINGS"}:
        return rule_based_report(r)
    if not brief:
        return rule_based_report(r)
    return (
        "1. FINDINGS\n"
        f"- {brief}\n\n"
        "2. RISK ASSESSMENT\n"
        f"- Overall triage risk: {overall_risk(r)}.\n\n"
        "3. RECOMMENDATIONS\n"
        "- Correlate with clinical exam, IOP, OCT, and visual acuity.\n\n"
        "4. DISCLAIMER\n"
        "- This is AI-assisted retinal screening support and not a definitive diagnosis."
    )


def _load_medgemma_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _distance(a: ScreeningResults, b: Dict[str, Any]) -> float:
    # Simple weighted distance over the generated medgemma dataset metadata.
    return (
        abs(a.cdr - float(b.get("cdr", 0.0))) * 4.0
        + abs(a.dr_grade - int(b.get("dr_grade", 0))) * 1.5
        + abs(a.dr_conf - float(b.get("dr_conf", 0.0))) * 1.0
        + abs(a.vessel_density - float(b.get("vessel_density", 0.0))) * 2.0
        + (0.5 if a.glaucoma_risk != str(b.get("glaucoma_risk", "")) else 0.0)
    )


def medgemma_from_dataset(
    r: ScreeningResults,
    jsonl_path: str = "/kaggle/working/outputs/medgemma/distill_train.jsonl",
) -> Tuple[Optional[str], Optional[str]]:
    rows = _load_medgemma_jsonl(jsonl_path)
    if not rows:
        return None, f"dataset not found/empty: {jsonl_path}"

    best_text: Optional[str] = None
    best_score = float("inf")
    for row in rows:
        meta = row.get("meta", {}) or {}
        msgs = row.get("messages", []) or []
        if len(msgs) < 2:
            continue
        assistant = clean_report_text(msgs[1].get("content", ""))
        if not assistant:
            continue
        assistant = coerce_to_structured_report(assistant, r)
        score = _distance(r, meta)
        if score < best_score:
            best_score = score
            best_text = assistant

    if not best_text:
        return None, "no valid assistant samples in dataset"
    return best_text, None


def generate_report(
    r: ScreeningResults,
    use_medgemma: bool = True,
    use_dataset_cache: bool = False,
) -> Dict[str, str]:
    prompt = build_prompt(r)
    fallback = rule_based_report(r)
    if not use_medgemma:
        return {"mode": "rule_based_only", "report": fallback, "prompt": prompt, "error": ""}
    dataset_err = ""
    out, err = medgemma_generate(prompt)
    if out is not None:
        cleaned = coerce_to_structured_report(out, r)
        return {"mode": "medgemma_live", "report": cleaned, "prompt": prompt, "error": ""}

    if use_dataset_cache:
        dataset_out, dataset_err = medgemma_from_dataset(r)
        if dataset_out is not None:
            return {
                "mode": "medgemma_dataset",
                "report": coerce_to_structured_report(dataset_out, r),
                "prompt": prompt,
                "error": "",
            }

    if out is None:
        full_err = err or "unknown"
        if dataset_err:
            full_err = f"{full_err} | dataset_error: {dataset_err}"
        return {"mode": "fallback_rule_based", "report": fallback, "prompt": prompt, "error": full_err}
    return {"mode": "fallback_rule_based", "report": fallback, "prompt": prompt, "error": "unknown"}


def main() -> None:
    # Substitua com outputs reais dos modelos
    results = ScreeningResults(
        cdr=0.612,
        glaucoma_risk="high",
        dr_grade=2,
        dr_label="Moderate",
        dr_conf=0.9616,
        vessel_density=0.12,
    )

    output = generate_report(
        results,
        use_medgemma=True,
        use_dataset_cache=(os.getenv("USE_MEDGEMMA_DATASET_CACHE", "0") == "1"),
    )
    print("MODE:", output["mode"])
    if output["error"]:
        print("ERROR:", output["error"])
    print("\nREPORT:\n")
    print(output["report"])

    out_dir = "/kaggle/working/outputs/medgemma"
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "clinical_report.txt")
    meta_path = os.path.join(out_dir, "clinical_report_meta.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output["report"])
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": output["mode"],
                "error": output["error"],
                "inputs": results.__dict__,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved:")
    print("-", txt_path)
    print("-", meta_path)


if __name__ == "__main__":
    main()
