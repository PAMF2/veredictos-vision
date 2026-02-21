#!/usr/bin/env python3
"""
10 - MedGemma Only

Pipeline objetivo para Kaggle:
1) Monta prompts estruturados de triagem retinal
2) Gera respostas SOMENTE com MedGemma
3) Salva resultados em JSONL
"""

import argparse
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

warnings.filterwarnings("ignore", message=".*Transparent hugepages are not enabled.*")
warnings.filterwarnings("ignore", message=".*np\\.object.*")


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

Output strictly with sections 1-4 only.
"""


def overall_risk(r: ScreeningResults) -> str:
    if r.dr_grade >= 4 or r.cdr >= 0.75:
        return "emergent"
    if r.dr_grade >= 3 or r.cdr >= 0.65:
        return "high"
    if r.dr_grade >= 2 or r.cdr >= 0.55:
        return "moderate"
    return "low"


def rule_report(r: ScreeningResults) -> str:
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


def make_synthetic_samples(n: int, seed: int = 42) -> List[ScreeningResults]:
    rng = random.Random(seed)
    out: List[ScreeningResults] = []
    labels = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
    risks = ["low", "moderate", "high", "emergent"]
    for _ in range(n):
        dr_grade = rng.choice([0, 1, 2, 3, 4])
        cdr = round(rng.uniform(0.3, 0.85), 3)
        vessel_density = round(rng.uniform(0.06, 0.22), 3)
        dr_conf = round(rng.uniform(0.75, 0.995), 3)
        glaucoma_risk = risks[min(3, int((cdr - 0.3) / 0.14))]
        out.append(
            ScreeningResults(
                cdr=cdr,
                glaucoma_risk=glaucoma_risk,
                dr_grade=dr_grade,
                dr_label=labels[dr_grade],
                dr_conf=dr_conf,
                vessel_density=vessel_density,
            )
        )
    return out


def init_teacher(model_id: str, hf_token: str = "") -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    try:
        import torch
        from huggingface_hub import login
        from transformers import AutoModelForCausalLM, AutoTokenizer
        is_tpu = bool(os.getenv("TPU_NAME") or os.getenv("COLAB_TPU_ADDR") or os.getenv("TPU_WORKER_ID"))

        if hf_token:
            login(token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if is_tpu:
            import torch_xla.core.xla_model as xm  # type: ignore
            device = xm.xla_device()
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


def try_teacher(prompt: str, tokenizer: Any, model: Any) -> Optional[str]:
    try:
        import torch
        is_tpu = bool(os.getenv("TPU_NAME") or os.getenv("COLAB_TPU_ADDR") or os.getenv("TPU_WORKER_ID"))

        msgs = [{"role": "user", "content": prompt}]
        inp = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inp = {k: v.to(model.device) for k, v in inp.items()}
        attention_mask = inp.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inp["input_ids"])
        with torch.no_grad():
            out = model.generate(
                input_ids=inp["input_ids"],
                attention_mask=attention_mask,
                max_new_tokens=40 if is_tpu else 64,
                do_sample=False,
                max_time=8 if is_tpu else 15,
                use_cache=False if is_tpu else True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        k = text.find("1. FINDINGS")
        return text[k:].strip() if k >= 0 else text.strip()
    except Exception:
        return None


def build_distill_jsonl(
    output_jsonl: str,
    n_samples: int,
    teacher_model: str,
    use_teacher: bool,
    allow_rule_fallback: bool,
    hf_token: str = "",
) -> None:
    rows = []
    tokenizer = None
    model = None
    teacher_error = None
    is_tpu = bool(os.getenv("TPU_NAME") or os.getenv("COLAB_TPU_ADDR") or os.getenv("TPU_WORKER_ID"))
    if is_tpu and use_teacher and os.getenv("FORCE_TEACHER_ON_TPU", "0") != "1":
        print("TPU + teacher detected. Forcing real teacher generation on TPU.")
        os.environ["FORCE_TEACHER_ON_TPU"] = "1"

    if use_teacher:
        tokenizer, model, teacher_error = init_teacher(teacher_model, hf_token=hf_token)
        if teacher_error:
            if allow_rule_fallback:
                print(f"Teacher init failed: {teacher_error}")
                print("Falling back to rule-based teacher because --allow_rule_fallback is enabled.")
                use_teacher = False
            else:
                raise RuntimeError(
                    f"Teacher init failed and fallback is disabled. Error: {teacher_error}"
                )

    medgemma_failures = 0

    for s in make_synthetic_samples(n_samples):
        prompt = build_prompt(s)
        teacher = None
        if use_teacher:
            print(f"[medgemma] generating sample {len(rows)+1}/{n_samples}...")
        if use_teacher and tokenizer is not None and model is not None:
            teacher = try_teacher(prompt, tokenizer, model)
        if teacher is None:
            medgemma_failures += 1
            if allow_rule_fallback:
                teacher = rule_report(s)
            else:
                raise RuntimeError(
                    "Teacher generation failed for at least one sample and fallback is disabled. "
                    "Use --allow_rule_fallback if you want to continue with mixed data."
                )
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": teacher},
                ],
                "meta": asdict(s),
            }
        )
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved distill dataset: {output_jsonl} ({len(rows)} samples)")
    if use_teacher:
        print(f"MedGemma failures during generation: {medgemma_failures}/{len(rows)}")


def run_sft_lora(
    dataset_jsonl: str,
    base_model: str,
    out_dir: str,
    epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 8,
) -> None:
    from typing import Any
    import subprocess
    import sys
    import torch
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError:
        print("Installing missing dependency: peft")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft", "accelerate"])
        from peft import LoraConfig, get_peft_model

    class JsonlCausalDataset(Dataset):
        def __init__(self, path: str, tokenizer: Any, max_len: int = 1024) -> None:
            self.items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    m = obj["messages"]
                    text = f"<|user|>\n{m[0]['content']}\n<|assistant|>\n{m[1]['content']}"
                    toks = tokenizer(
                        text,
                        truncation=True,
                        max_length=max_len,
                        padding=False,
                        return_tensors=None,
                    )
                    self.items.append(
                        {
                            "input_ids": toks["input_ids"],
                            "attention_mask": toks.get("attention_mask"),
                        }
                    )

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self.items[idx]

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = JsonlCausalDataset(dataset_jsonl, tok, max_len=1024)

    is_tpu = bool(os.getenv("TPU_NAME") or os.getenv("COLAB_TPU_ADDR") or os.getenv("TPU_WORKER_ID"))
    if is_tpu:
        import torch_xla.core.xla_model as xm  # type: ignore
        device = xm.xla_device()
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    model = model.to(device)
    print(f"Training device: {device} | TPU: {is_tpu} | CUDA: {torch.cuda.is_available()}")

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        for x in batch:
            ids = x["input_ids"]
            mask = x["attention_mask"] if x.get("attention_mask") is not None else [1] * len(ids)
            pad_n = max_len - len(ids)
            input_ids.append(ids + [tok.pad_token_id] * pad_n)
            attention_mask.append(mask + [0] * pad_n)
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        attn_t = torch.tensor(attention_mask, dtype=torch.long)
        labels_t = input_ids_t.clone()
        labels_t[attn_t == 0] = -100
        return {"input_ids": input_ids_t, "attention_mask": attn_t, "labels": labels_t}

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    use_cuda_amp = torch.cuda.is_available() and (not is_tpu)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
    model.train()

    global_step = 0
    for ep in range(1, epochs + 1):
        running = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            amp_ctx = torch.cuda.amp.autocast(enabled=use_cuda_amp) if use_cuda_amp else nullcontext()
            with amp_ctx:
                out = model(**batch)
                loss = out.loss / grad_accum
            if use_cuda_amp:
                scaler.scale(loss).backward()
                if step % grad_accum == 0 or step == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if step % grad_accum == 0 or step == len(loader):
                    if is_tpu:
                        xm.optimizer_step(optimizer)  # type: ignore[name-defined]
                        xm.mark_step()  # type: ignore[name-defined]
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
            running += loss.item() * grad_accum
            global_step += 1
            if step % 10 == 0 or step == len(loader):
                print(f"Epoch {ep}/{epochs} | step {step}/{len(loader)} | loss={running/step:.4f}")

        os.makedirs(out_dir, exist_ok=True)
        ep_dir = os.path.join(out_dir, f"epoch_{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        model.save_pretrained(ep_dir)
        tok.save_pretrained(ep_dir)
        print(f"Saved epoch checkpoint: {ep_dir}")

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved distilled student adapter/model: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_jsonl", default="/kaggle/working/outputs/medgemma/distill_train.jsonl")
    p.add_argument("--out_dir", default="/kaggle/working/outputs/medgemma")
    p.add_argument("--medgemma_model", default="google/medgemma-4b-it")
    p.add_argument("--student_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    # Defaults tuned for "click Run cell" in Kaggle GPU notebooks.
    p.add_argument("--n_samples", type=int, default=120)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--allow_rule_fallback", action="store_true")
    # In notebooks (IPython/Jupyter), argv includes kernel flags like "-f ...json".
    # parse_known_args keeps script usable both in terminal and notebook "Run" mode.
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"Ignoring notebook/runtime args: {unknown}")

    # MedGemma-only: force MedGemma path and disable fallback.
    args.allow_rule_fallback = False

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            hf_token = ""

    is_tpu = bool(os.getenv("TPU_NAME") or os.getenv("COLAB_TPU_ADDR") or os.getenv("TPU_WORKER_ID"))
    if is_tpu:
        print("TPU runtime detected. MedGemma mode is enabled.")
        os.environ.setdefault("PJRT_DEVICE", "TPU")

    print("=== Distillation Config ===")
    print(f"medgemma_model: {args.medgemma_model}")
    print(f"student_model: {args.student_model} (ignored in MedGemma-only mode)")
    print(f"dataset_jsonl: {args.dataset_jsonl}")
    print(f"out_dir: {args.out_dir}")
    print(f"n_samples: {args.n_samples}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"grad_accum: {args.grad_accum}")
    print(f"use_medgemma: True")
    print(f"allow_rule_fallback: {args.allow_rule_fallback}")
    print("==========================")
    print("MedGemma-only mode: fallback disabled, student training disabled.")

    build_distill_jsonl(
        output_jsonl=args.dataset_jsonl,
        n_samples=args.n_samples,
        teacher_model=args.medgemma_model,
        use_teacher=True,
        allow_rule_fallback=args.allow_rule_fallback,
        hf_token=hf_token,
    )
    print("MedGemma-only mode complete. Student training is disabled.")


if __name__ == "__main__":
    main()
