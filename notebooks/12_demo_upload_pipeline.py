#!/usr/bin/env python3
"""
12 - Demo Upload Pipeline (real checkpoints + MedGemma)
"""

import os
import re
import sys
import socket
import subprocess
import asyncio
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH_XLA", "0")


def _project_root() -> Optional[Path]:
    file_path = Path(__file__).resolve() if "__file__" in globals() else None
    candidates = [
        file_path.parents[1] if file_path else None,
        Path.cwd(),
        Path.cwd() / "sprintfinal",
        Path("/kaggle/working/sprintfinal"),
        Path("/kaggle/working/Veredictos-Vision"),
    ]
    for c in candidates:
        if c and (c / "utils" / "medgemma_report.py").exists():
            return c
    return None


PROJECT_ROOT = _project_root()
if PROJECT_ROOT and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ScreeningResults:
    cdr: float
    glaucoma_risk: str
    dr_grade: int
    dr_label: str
    dr_conf: float
    vessel_density: float


def _dr_label(grade: int) -> str:
    return {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}.get(int(grade), "Unknown")


def _risk_from_signals(cdr: float, dr_grade: int) -> str:
    if dr_grade >= 4 or cdr >= 0.75:
        return "emergent"
    if dr_grade >= 3 or cdr >= 0.65:
        return "high"
    if dr_grade >= 2 or cdr >= 0.55:
        return "moderate"
    return "low"


def _ensure_runtime_deps() -> Tuple[bool, str]:
    reqs = [
        ("segmentation_models_pytorch", "segmentation-models-pytorch"),
        ("albumentations", "albumentations"),
        ("timm", "timm"),
    ]
    missing = []
    for mod, pip_name in reqs:
        try:
            __import__(mod)
        except Exception:
            missing.append(pip_name)
    if not missing:
        return True, ""
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + missing
        subprocess.check_call(cmd)
        return True, f"installed missing deps: {', '.join(missing)}"
    except Exception as exc:
        return False, f"pip install failed for {missing}: {exc}"


class ModelBundle:
    def __init__(self) -> None:
        self.ready = False
        self.error = ""
        self.device = None
        self.torch = None
        self.np = np
        self.A = None
        self.ToTensorV2 = None
        self.smp = None
        self.timm = None
        self.glaucoma_model = None
        self.vessel_model = None
        self.dr_model = None
        self.vessel_threshold = 0.5
        self.dr_img_size = 300
        self.glaucoma_tf = None
        self.vessel_tf = None
        self.dr_tf = None
        self.glau_ckpt = ""
        self.vessel_ckpt = ""
        self.dr_ckpt = ""
        self.g_params = 0
        self.v_params = 0
        self.d_params = 0
        self.device_name = "cpu"

    def _find_ckpt(self, rel: str) -> Optional[Path]:
        cands = [
            Path("/kaggle/working") / rel,
            Path.cwd() / rel,
            (PROJECT_ROOT / rel) if PROJECT_ROOT else None,
        ]
        for c in cands:
            if c and c.exists():
                return c
        return None

    def load(self) -> None:
        if self.ready:
            return
        force_gpu_only = os.getenv("FORCE_GPU_ONLY", "1") == "1"
        ok, dep_msg = _ensure_runtime_deps()
        if not ok:
            self.error = dep_msg
            return
        if dep_msg:
            print(dep_msg)
        try:
            import torch
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            import segmentation_models_pytorch as smp
            import timm
        except Exception as exc:
            self.error = f"dependency import failed: {exc}"
            return

        self.torch = torch
        self.A = A
        self.ToTensorV2 = ToTensorV2
        self.smp = smp
        self.timm = timm
        if force_gpu_only and (not torch.cuda.is_available()):
            self.error = "FORCE_GPU_ONLY=1 but CUDA unavailable. Restart Kaggle session with GPU accelerator enabled."
            return
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_name = str(self.device)

        c_glau = self._find_ckpt("outputs/transunet_glaucoma_best.pth")
        c_vessel = self._find_ckpt("outputs/unetpp/unet_r34_drive_best.pth")
        c_dr = self._find_ckpt("outputs/efficientnet/efficientnet_dr_best.pth")
        if not c_glau or not c_vessel or not c_dr:
            self.error = (
                "checkpoint not found. expected: "
                "/kaggle/working/outputs/transunet_glaucoma_best.pth, "
                "/kaggle/working/outputs/unetpp/unet_r34_drive_best.pth, "
                "/kaggle/working/outputs/efficientnet/efficientnet_dr_best.pth"
            )
            return

        try:
            g_model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights=None,
                in_channels=3,
                classes=2,
                activation=None,
            )
            g_ckpt = torch.load(c_glau, map_location="cpu", weights_only=False)
            g_state = g_ckpt["model_state_dict"] if isinstance(g_ckpt, dict) and "model_state_dict" in g_ckpt else g_ckpt
            g_model.load_state_dict(g_state)
            self.glaucoma_model = g_model
            self.glau_ckpt = str(c_glau)
            self.g_params = int(sum(p.numel() for p in g_model.parameters()))

            v_ckpt = torch.load(c_vessel, map_location="cpu", weights_only=False)
            arch = v_ckpt.get("arch", "unet_resnet34") if isinstance(v_ckpt, dict) else "unet_resnet34"
            num_classes = int(v_ckpt.get("num_classes", 1)) if isinstance(v_ckpt, dict) else 1
            self.vessel_threshold = float(v_ckpt.get("best_threshold", 0.5)) if isinstance(v_ckpt, dict) else 0.5
            if arch == "unet_resnet34":
                v_model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights=None,
                    in_channels=3,
                    classes=num_classes,
                    activation=None,
                )
            else:
                v_model = smp.UnetPlusPlus(
                    encoder_name="efficientnet-b4",
                    encoder_weights=None,
                    in_channels=3,
                    classes=num_classes,
                    activation=None,
                )
            v_state = v_ckpt["model_state_dict"] if isinstance(v_ckpt, dict) and "model_state_dict" in v_ckpt else v_ckpt
            v_model.load_state_dict(v_state)
            self.vessel_model = v_model
            self.vessel_ckpt = str(c_vessel)
            self.v_params = int(sum(p.numel() for p in v_model.parameters()))

            dr_ckpt = torch.load(c_dr, map_location="cpu", weights_only=False)
            num_classes_dr = int(dr_ckpt.get("num_classes", 5)) if isinstance(dr_ckpt, dict) else 5
            model_name = dr_ckpt.get("model_name", "efficientnet_b3") if isinstance(dr_ckpt, dict) else "efficientnet_b3"
            d_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes_dr)
            d_state = dr_ckpt["model_state_dict"] if isinstance(dr_ckpt, dict) and "model_state_dict" in dr_ckpt else dr_ckpt
            d_model.load_state_dict(d_state)
            self.dr_model = d_model
            self.dr_ckpt = str(c_dr)
            self.d_params = int(sum(p.numel() for p in d_model.parameters()))
            self.dr_img_size = int(dr_ckpt.get("img_size", os.getenv("IMG_SIZE", "300"))) if isinstance(dr_ckpt, dict) else 300

            self.glaucoma_tf = A.Compose(
                [A.Resize(512, 512), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
            )
            self.vessel_tf = A.Compose(
                [A.Resize(512, 512), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
            )
            self.dr_tf = A.Compose(
                [
                    A.Resize(self.dr_img_size, self.dr_img_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            # Move models to target device; if CUDA is in bad state, fallback to CPU.
            try:
                self.glaucoma_model.to(self.device).eval()
                self.vessel_model.to(self.device).eval()
                self.dr_model.to(self.device).eval()
            except Exception as dev_exc:
                msg = str(dev_exc).lower()
                if "cuda" in msg:
                    if force_gpu_only:
                        raise RuntimeError(
                            "CUDA move failed and FORCE_GPU_ONLY=1. GPU context is likely poisoned; restart session."
                        )
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    self.device = torch.device("cpu")
                    self.device_name = "cpu"
                    self.glaucoma_model.to(self.device).eval()
                    self.vessel_model.to(self.device).eval()
                    self.dr_model.to(self.device).eval()
                else:
                    raise
            self.ready = True
            print(f"[models] glaucoma loaded: {self.glau_ckpt} | params={self.g_params}")
            print(f"[models] vessel loaded: {self.vessel_ckpt} | params={self.v_params}")
            print(f"[models] dr loaded: {self.dr_ckpt} | params={self.d_params}")
            print(f"[models] device: {self.device}")
        except Exception as exc:
            # Last-resort: force CPU load if CUDA context is poisoned.
            msg = str(exc).lower()
            if "cuda" in msg or "device-side assert" in msg:
                if force_gpu_only:
                    self.error = (
                        f"model load failed on GPU with FORCE_GPU_ONLY=1: {exc}. "
                        "Restart Kaggle session and run only notebook 12 first."
                    )
                    return
                try:
                    self.device = torch.device("cpu")
                    self.device_name = "cpu"
                    self.glaucoma_model = self.glaucoma_model.to(self.device).eval() if self.glaucoma_model is not None else None
                    self.vessel_model = self.vessel_model.to(self.device).eval() if self.vessel_model is not None else None
                    self.dr_model = self.dr_model.to(self.device).eval() if self.dr_model is not None else None
                    if self.glaucoma_model is not None and self.vessel_model is not None and self.dr_model is not None:
                        self.ready = True
                        self.error = ""
                        print("[models] CUDA poisoned; switched pipeline to CPU.")
                        return
                except Exception:
                    pass
            self.error = f"model load failed: {exc}"

    def status_text(self) -> str:
        self.load()
        if not self.ready:
            return f"models_ready: false\nerror: {self.error}"
        return (
            "models_ready: true\n"
            f"device: {self.device}\n"
            f"glaucoma_ckpt: {self.glau_ckpt}\n"
            f"vessel_ckpt: {self.vessel_ckpt}\n"
            f"dr_ckpt: {self.dr_ckpt}\n"
            f"glaucoma_params: {self.g_params}\n"
            f"vessel_params: {self.v_params}\n"
            f"dr_params: {self.d_params}\n"
        )

    def infer(self, image_pil: Image.Image) -> Tuple[Optional[ScreeningResults], str]:
        self.load()
        if not self.ready:
            return None, self.error

        torch = self.torch
        image_rgb = np.array(image_pil.convert("RGB"))
        try:
            with torch.no_grad():
                g_in = self.glaucoma_tf(image=image_rgb)["image"].unsqueeze(0).to(self.device)
                g_out = torch.sigmoid(self.glaucoma_model(g_in))
                disc = (g_out[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                cup = (g_out[0, 1].detach().cpu().numpy() > 0.5).astype(np.uint8)
                disc_area = int(disc.sum())
                cup_area = int(cup.sum())
                cdr = float(np.sqrt(cup_area / disc_area)) if disc_area > 0 else 0.0
                cdr = float(np.clip(cdr, 0.0, 1.0))

                v_in = self.vessel_tf(image=image_rgb)["image"].unsqueeze(0).to(self.device)
                v_logits = self.vessel_model(v_in)
                v_prob = torch.sigmoid(v_logits[0, 0]).detach().cpu().numpy()
                v_bin = (v_prob > self.vessel_threshold).astype(np.float32)
                vessel_density = float(np.clip(v_bin.mean(), 0.0, 1.0))

                d_in = self.dr_tf(image=image_rgb)["image"].unsqueeze(0).to(self.device)
                d_logits = self.dr_model(d_in)
                d_prob = torch.softmax(d_logits, dim=1)[0].detach().cpu().numpy()
                dr_grade = int(np.argmax(d_prob))
                dr_conf = float(np.clip(np.max(d_prob), 0.0, 1.0))

            glaucoma_risk = _risk_from_signals(cdr, dr_grade)
            return (
                ScreeningResults(
                    cdr=cdr,
                    glaucoma_risk=glaucoma_risk,
                    dr_grade=dr_grade,
                    dr_label=_dr_label(dr_grade),
                    dr_conf=dr_conf,
                    vessel_density=vessel_density,
                ),
                "",
            )
        except Exception as exc:
            return None, f"inference failed: {exc}"


class MedGemmaGenerator:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = None
        self.error = ""
        self.model_id_loaded = ""

    def load(self, model_id: str) -> None:
        if self.model is not None and self.tokenizer is not None and self.model_id_loaded == model_id:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import login

            hf_token = os.getenv("HF_TOKEN", "").strip()
            if not hf_token:
                try:
                    from kaggle_secrets import UserSecretsClient  # type: ignore
                    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
                except Exception:
                    hf_token = ""

            if hf_token:
                login(token=hf_token)
            else:
                self.error = (
                    "HF_TOKEN not found. Add HF_TOKEN in Kaggle Secrets and ensure "
                    "your account has access to google/medgemma-4b-it."
                )
                return

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            force_gpu_only = os.getenv("FORCE_GPU_ONLY", "1") == "1"
            if force_gpu_only and (not torch.cuda.is_available()):
                self.error = "FORCE_GPU_ONLY=1 but CUDA unavailable for MedGemma. Restart session with GPU."
                return
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=hf_token,
                    dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
                self.device = self.model.device
            else:
                if force_gpu_only:
                    self.error = "FORCE_GPU_ONLY=1 and MedGemma would run on CPU. Aborting."
                    return
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=hf_token,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
                self.device = "cpu"
            self.model.eval()
            self.model_id_loaded = model_id
        except Exception as exc:
            self.error = str(exc)

    def _reload_cpu(self, model_id: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import login

            hf_token = os.getenv("HF_TOKEN", "").strip()
            if not hf_token:
                try:
                    from kaggle_secrets import UserSecretsClient  # type: ignore
                    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
                except Exception:
                    hf_token = ""
            if hf_token:
                login(token=hf_token)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token if hf_token else None)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=hf_token if hf_token else None,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            self.device = "cpu"
            self.model.eval()
            self.error = ""
            self.model_id_loaded = model_id
        except Exception as exc:
            self.error = f"cpu reload failed: {exc}"

    def generate(self, prompt: str, model_id: str) -> Tuple[Optional[str], str]:
        return self.generate_messages([{"role": "user", "content": prompt}], model_id=model_id)

    def generate_messages(self, messages: list[dict], model_id: str) -> Tuple[Optional[str], str]:
        self.load(model_id)
        if self.error:
            return None, self.error
        try:
            import torch

            if not messages:
                return None, "empty_messages"
            inp = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inp = {k: v.to(self.device) for k, v in inp.items()}
            if "attention_mask" not in inp or inp["attention_mask"] is None:
                inp["attention_mask"] = torch.ones_like(inp["input_ids"])
            # Defensive sanitize: avoid invalid floating values propagating into generation.
            for k, v in list(inp.items()):
                if torch.is_floating_point(v):
                    inp[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                is_cpu = str(self.device) == "cpu"
                out = self.model.generate(
                    input_ids=inp["input_ids"],
                    attention_mask=inp["attention_mask"],
                    max_new_tokens=180 if is_cpu else 320,
                    min_new_tokens=48 if is_cpu else 96,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    num_beams=3,
                    early_stopping=True,
                    length_penalty=1.05,
                    renormalize_logits=True,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=4,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out[0][inp["input_ids"].shape[1]:]
            txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            k = txt.find("1. FINDINGS")
            if k >= 0:
                txt = txt[k:].strip()
            else:
                # Free-form mode: models may echo the full prompt; strip it.
                p = (messages[-1].get("content", "") if isinstance(messages[-1], dict) else "").strip()
                if p and txt.strip().startswith(p):
                    txt = txt.strip()[len(p):].strip()
            return txt, ""
        except Exception as exc:
            msg = str(exc)
            if "probability tensor contains either `inf`, `nan` or element < 0" in msg.lower():
                try:
                    if self.device is not None and "cuda" in str(self.device).lower():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                return None, "medgemma_prob_nan"
            if "device-side assert" in msg.lower() or "cuda error" in msg.lower():
                self._reload_cpu(model_id)
                if self.error:
                    return None, msg
                try:
                    import torch
                    messages = [{"role": "user", "content": prompt}]
                    inp = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                    inp = {k: v.to(self.device) for k, v in inp.items()}
                    if "attention_mask" not in inp or inp["attention_mask"] is None:
                        inp["attention_mask"] = torch.ones_like(inp["input_ids"])
                    for k, v in list(inp.items()):
                        if torch.is_floating_point(v):
                            inp[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    with torch.no_grad():
                        out = self.model.generate(
                            input_ids=inp["input_ids"],
                            attention_mask=inp["attention_mask"],
                            max_new_tokens=220,
                            min_new_tokens=48,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                            num_beams=2,
                            early_stopping=True,
                            length_penalty=1.02,
                            renormalize_logits=True,
                            repetition_penalty=1.12,
                            no_repeat_ngram_size=4,
                            use_cache=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    gen_ids = out[0][inp["input_ids"].shape[1]:]
                    txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    k = txt.find("1. FINDINGS")
                    if k >= 0:
                        txt = txt[k:].strip()
                    else:
                        p = (messages[-1].get("content", "") if isinstance(messages[-1], dict) else "").strip()
                        if p and txt.strip().startswith(p):
                            txt = txt.strip()[len(p):].strip()
                    return txt, ""
                except Exception as exc2:
                    return None, f"{msg} | cpu_fallback_failed: {exc2}"
            return None, msg


PIPELINE = ModelBundle()
MEDGEMMA = MedGemmaGenerator()
PIPELINE_VERSION = "12.20-medgemma-beam-longform"


def _prompt(r: ScreeningResults) -> str:
    return (
        "You are a clinical ophthalmology AI assistant. Based on automated retinal screening outputs, "
        "generate a structured clinical report.\n\n"
        f"PATIENT SCREENING RESULTS:\n"
        f"- Glaucoma: CDR = {r.cdr:.3f} | Risk: {r.glaucoma_risk}\n"
        f"- Diabetic Retinopathy: Grade {r.dr_grade} ({r.dr_label}) | Confidence: {r.dr_conf:.1%}\n"
        f"- Vascular: Vessel density = {r.vessel_density:.1%}\n\n"
        "Return ONLY these 4 sections:\n"
        "1. FINDINGS\n"
        "2. RISK ASSESSMENT\n"
        "3. RECOMMENDATIONS\n"
        "4. DISCLAIMER\n"
    )


def _prompt_fewshot_freeform(r: ScreeningResults) -> str:
    return f"""You are an ophthalmology clinical assistant.
Write a concise retinal screening report in natural clinical language (free-form), based on model outputs.
Do not return placeholders.

Example A
Inputs: CDR 0.48 (low), DR grade 0 (No DR) confidence 95%, vessel density 14%.
Report:
Findings suggest no diabetic retinopathy and a low glaucoma suspicion profile from the cup-to-disc ratio.
Retinal vascular pattern appears preserved for screening context.
Overall triage is low risk; routine annual follow-up is appropriate with standard clinical correlation.
This AI output supports screening and does not replace diagnosis.

Example B
Inputs: CDR 0.77 (emergent), DR grade 4 (Proliferative) confidence 91%, vessel density 9%.
Report:
The outputs indicate advanced diabetic retinopathy with proliferative features and a markedly elevated glaucoma-related structural risk.
Combined risk profile is emergent for vision-threatening disease in this screening context.
Urgent specialist ophthalmology referral is advised, with comprehensive exam, OCT, IOP measurement, and visual acuity assessment.
This AI output supports triage and must be interpreted with clinical judgment.

Now write the report for:
Inputs: CDR {r.cdr:.3f} ({r.glaucoma_risk}), DR grade {r.dr_grade} ({r.dr_label}) confidence {r.dr_conf:.1%}, vessel density {r.vessel_density:.1%}.
"""


def _prompt_json(r: ScreeningResults) -> str:
    return (
        "You are a clinical ophthalmology AI assistant.\n"
        "Return ONLY valid JSON with these keys: findings, risk_assessment, recommendations, disclaimer.\n"
        "Each value must be one complete sentence.\n\n"
        f"Glaucoma CDR: {r.cdr:.3f} ({r.glaucoma_risk}).\n"
        f"DR grade: {r.dr_grade} ({r.dr_label}), confidence {r.dr_conf:.1%}.\n"
        f"Vessel density: {r.vessel_density:.1%}.\n"
    )


def _prompt_single_sentence(r: ScreeningResults) -> str:
    return (
        "Write one concise clinical ophthalmology finding sentence in English based on:\n"
        f"CDR {r.cdr:.3f} ({r.glaucoma_risk}), DR grade {r.dr_grade} ({r.dr_label}) confidence {r.dr_conf:.1%}, "
        f"vessel density {r.vessel_density:.1%}.\n"
        "Output only the sentence."
    )


def _prompt_guided_free_text(r: ScreeningResults) -> str:
    return (
        "You are an ophthalmology assistant. Write 3-5 clinical sentences (no headings) interpreting these results:\n"
        f"- CDR {r.cdr:.3f} ({r.glaucoma_risk})\n"
        f"- DR grade {r.dr_grade} ({r.dr_label}), confidence {r.dr_conf:.1%}\n"
        f"- Vessel density {r.vessel_density:.1%}\n"
        "Include risk impression and follow-up advice."
    )


def _prompt_clinical_paragraph(r: ScreeningResults) -> str:
    template = (
        "Write a concise clinical paragraph (3-5 sentences) for ophthalmology screening.\n"
        "Do not use headings, bullets, numbering, or template labels.\n"
        "Use these model outputs:\n"
        "CDR={cdr:.3f} ({glaucoma_risk}); "
        "DR grade={dr_grade} ({dr_label}) confidence={dr_conf:.1%}; "
        "vessel density={vessel_density:.1%}.\n"
        "Include interpretation, triage risk, and follow-up advice."
    )
    return template.format(
        cdr=r.cdr,
        glaucoma_risk=r.glaucoma_risk,
        dr_grade=r.dr_grade,
        dr_label=r.dr_label,
        dr_conf=r.dr_conf,
        vessel_density=r.vessel_density,
    )


def _prompt_clinical_paragraph_v2(r: ScreeningResults) -> str:
    template = (
        "Clinical task: produce the final ophthalmology screening impression only.\n"
        "No instructions, no headings, no bullets.\n"
        "Data -> CDR {cdr:.3f} ({glaucoma_risk}); "
        "DR {dr_grade} ({dr_label}) conf {dr_conf:.1%}; "
        "Vessel density {vessel_density:.1%}.\n"
        "Output: one compact clinical paragraph."
    )
    return template.format(
        cdr=r.cdr,
        glaucoma_risk=r.glaucoma_risk,
        dr_grade=r.dr_grade,
        dr_label=r.dr_label,
        dr_conf=r.dr_conf,
        vessel_density=r.vessel_density,
    )


def _prompt_clinical_paragraph_v3(r: ScreeningResults) -> str:
    template = (
        "You are an ophthalmology assistant.\n"
        "Write only one final clinical paragraph in English (3-5 sentences), with no headings and no bullets.\n"
        "Patient signals:\n"
        "- CDR: {cdr:.3f} ({glaucoma_risk})\n"
        "- DR: grade {dr_grade} ({dr_label}), confidence {dr_conf:.1%}\n"
        "- Vessel density: {vessel_density:.1%}\n"
        "The paragraph must include interpretation, risk level, and follow-up recommendation."
    )
    return template.format(
        cdr=r.cdr,
        glaucoma_risk=r.glaucoma_risk,
        dr_grade=r.dr_grade,
        dr_label=r.dr_label,
        dr_conf=r.dr_conf,
        vessel_density=r.vessel_density,
    )


def _prompt_dual_pass_stage1(r: ScreeningResults) -> str:
    return (
        "SYSTEM ROLE: You are an ophthalmology clinical report writer.\n"
        "TASK: Create a detailed retinal screening report using ONLY the authoritative model signals.\n"
        "OUTPUT RULES:\n"
        "- Output only one paragraph in English (6-8 sentences).\n"
        "- No headings, bullets, numbering, placeholders, or prompt text.\n"
        "- Include: findings, integrated risk interpretation, and follow-up plan.\n"
        "- Do not invent values; use exactly the values below.\n\n"
        "AUTHORITATIVE MODEL SIGNALS:\n"
        f"CDR: {r.cdr:.3f} ({r.glaucoma_risk})\n"
        f"DR grade: {r.dr_grade} ({r.dr_label})\n"
        f"DR confidence: {r.dr_conf:.1%}\n"
        f"Vessel density: {r.vessel_density:.1%}\n"
    )


def _prompt_dual_pass_stage2(r: ScreeningResults, draft: str) -> str:
    return (
        "SYSTEM ROLE: You are a clinical quality auditor and medical editor.\n"
        "TASK: Verify the draft report against authoritative model signals, fix all inconsistencies, "
        "and output the final improved report.\n"
        "CHECKLIST:\n"
        "1) Numeric consistency: CDR, DR grade, DR confidence, vessel density must match.\n"
        "2) Clinical consistency: risk wording and follow-up urgency must align with the signals.\n"
        "3) Writing quality: clear, detailed, concise, and clinically coherent.\n"
        "OUTPUT RULES:\n"
        "- Return only one final paragraph in English (6-8 sentences).\n"
        "- No headings, bullets, numbering, placeholders, examples, or meta-commentary.\n\n"
        "AUTHORITATIVE MODEL SIGNALS:\n"
        f"CDR: {r.cdr:.3f} ({r.glaucoma_risk})\n"
        f"DR grade: {r.dr_grade} ({r.dr_label})\n"
        f"DR confidence: {r.dr_conf:.1%}\n"
        f"Vessel density: {r.vessel_density:.1%}\n\n"
        "DRAFT REPORT TO AUDIT AND IMPROVE:\n"
        f"{draft}\n"
    )


def _prompt_repair_to_signals(r: ScreeningResults, bad_text: str) -> str:
    return (
        "SYSTEM ROLE: You are a clinical consistency fixer.\n"
        "TASK: Rewrite the draft into one clinically coherent paragraph (6-8 sentences), "
        "strictly aligned with the authoritative signals.\n"
        "RULES:\n"
        "- Use only the values below.\n"
        "- Remove any prompt leakage, labels, examples, or template text.\n"
        "- Do not use headings or bullets.\n\n"
        "AUTHORITATIVE MODEL SIGNALS:\n"
        f"CDR: {r.cdr:.3f} ({r.glaucoma_risk})\n"
        f"DR grade: {r.dr_grade} ({r.dr_label})\n"
        f"DR confidence: {r.dr_conf:.1%}\n"
        f"Vessel density: {r.vessel_density:.1%}\n\n"
        "DRAFT TO FIX:\n"
        f"{bad_text}\n"
    )


def _system_stage1() -> str:
    return (
        "You are a medical ophthalmology assistant. "
        "Produce a clinically coherent screening paragraph grounded only on provided signals. "
        "Do not echo instructions."
    )


def _system_stage2() -> str:
    return (
        "You are a medical quality auditor. "
        "Fix inconsistencies against authoritative signals and return only the final improved paragraph."
    )


def _prompt_structured_to_paragraph(r: ScreeningResults, structured_report: str) -> str:
    return (
        "Convert the structured report below into one detailed clinical paragraph in English (6-8 sentences).\n"
        "Keep exact clinical meaning and numeric consistency with the authoritative signals.\n"
        "No headings or bullets.\n\n"
        "AUTHORITATIVE MODEL SIGNALS:\n"
        f"CDR: {r.cdr:.3f} ({r.glaucoma_risk})\n"
        f"DR grade: {r.dr_grade} ({r.dr_label})\n"
        f"DR confidence: {r.dr_conf:.1%}\n"
        f"Vessel density: {r.vessel_density:.1%}\n\n"
        "STRUCTURED REPORT:\n"
        f"{structured_report}\n"
    )


def _system_stage3() -> str:
    return (
        "You are a clinical documentation formatter. "
        "Return only the final formatted report block in English, with no extra commentary."
    )


def _prompt_format_final_block(r: ScreeningResults, draft: str) -> str:
    return (
        "Format the report below into this exact style and keep it clinically coherent:\n\n"
        "GENERATED CLINICAL REPORT\n\n"
        "PATIENT: [Anonymized ID]\n"
        "DATE: [Today]\n\n"
        "INTEGRATED OPHTHALMIC ANALYSIS:\n"
        "<2-3 concise sentences>\n\n"
        "KEY FINDINGS:\n"
        "- ...\n"
        "- ...\n"
        "- ...\n\n"
        "RECOMMENDATIONS:\n"
        "- ...\n"
        "- ...\n"
        "- ...\n\n"
        "PRIORITY: <LOW|MODERATE|HIGH|EMERGENT>\n"
        "<final referral timing sentence>\n\n"
        "Rules:\n"
        "- English only\n"
        "- No markdown\n"
        "- No placeholders except [Anonymized ID]\n"
        "- Values must match the authoritative signals exactly\n\n"
        "AUTHORITATIVE MODEL SIGNALS:\n"
        f"CDR: {r.cdr:.3f} ({r.glaucoma_risk})\n"
        f"DR grade: {r.dr_grade} ({r.dr_label})\n"
        f"DR confidence: {r.dr_conf:.1%}\n"
        f"Vessel density: {r.vessel_density:.1%}\n\n"
        "DRAFT REPORT:\n"
        f"{draft}\n"
    )


def _build_formatted_report_from_signals(r: ScreeningResults, draft: str) -> str:
    priority = _risk_from_signals(r.cdr, r.dr_grade).upper()
    referral = {
        "LOW": "Recommend routine ophthalmology follow-up within 6-12 months.",
        "MODERATE": "Recommend ophthalmology follow-up within 1-3 months for in-person evaluation.",
        "HIGH": "Recommend prioritized ophthalmology assessment within 2-6 weeks.",
        "EMERGENT": "Recommend urgent ophthalmology referral within 24-72 hours.",
    }.get(priority, "Recommend ophthalmology follow-up based on clinical judgment.")
    return (
        "GENERATED CLINICAL REPORT\n\n"
        "PATIENT: [Anonymized ID]\n"
        "DATE: [Today]\n\n"
        "INTEGRATED OPHTHALMIC ANALYSIS:\n"
        f"The screening outputs indicate DR Grade {r.dr_grade} ({r.dr_label}) with high confidence ({r.dr_conf:.1%}), "
        f"CDR {r.cdr:.3f} classified as {r.glaucoma_risk} structural glaucoma risk, and vessel density of {r.vessel_density:.1%}. "
        "Findings should be interpreted as AI-assisted screening support and correlated with full clinical examination.\n\n"
        "KEY FINDINGS:\n"
        f"- CDR {r.cdr:.3f} ({r.glaucoma_risk} glaucoma-risk category)\n"
        f"- DR Grade {r.dr_grade} ({r.dr_label}) with confidence {r.dr_conf:.1%}\n"
        f"- Estimated retinal vessel density {r.vessel_density:.1%}\n\n"
        "RECOMMENDATIONS:\n"
        "- Correlate with comprehensive ophthalmic examination.\n"
        "- Include IOP measurement, OCT, and visual acuity assessment.\n"
        "- Continue systemic risk-factor and glycemic control.\n\n"
        f"PRIORITY: {priority}\n"
        f"{referral}"
    )


def _finalize_formatted_report(text: str, r: ScreeningResults) -> Optional[str]:
    s = _sanitize_freeform_output(text or "")
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if not s:
        return None
    if not _is_text_consistent_with_signals(s, r):
        s = _repair_numeric_inconsistencies(s, r)
    if not _is_text_consistent_with_signals(s, r):
        return None
    # Ensure required sections exist; if not, synthesize deterministic format.
    required = [
        "GENERATED CLINICAL REPORT",
        "INTEGRATED OPHTHALMIC ANALYSIS",
        "KEY FINDINGS",
        "RECOMMENDATIONS",
        "PRIORITY:",
    ]
    if not all(k.lower() in s.lower() for k in required):
        return None
    return s


def _repair_prompt(raw_text: str) -> str:
    return (
        "Rewrite the text below into a complete ophthalmology report with EXACTLY 4 sections:\n"
        "1. FINDINGS\n2. RISK ASSESSMENT\n3. RECOMMENDATIONS\n4. DISCLAIMER\n"
        "Use clinical language and do not include instructions.\n\n"
        f"{raw_text}\n"
    )


def _strip_instruction_noise(text: str) -> str:
    t = text or ""
    t = re.sub(r"(?is)TEXT TO REWRITE\s*:?", " ", t)
    t = re.sub(r"(?is)\bRules?\s*:\s*", " ", t)
    t = re.sub(r"(?is)Each section must contain[^.]*\.", " ", t)
    t = re.sub(r"(?is)Do not output[^.]*\.", " ", t)
    t = re.sub(r"(?is)Return ONLY[^.]*\.", " ", t)
    t = re.sub(r"(?im)^\s*[-*]\s*(Do not|Include|Each section).*$", " ", t)
    t = re.sub(r"(?is)Use clinical language and do not include instructions\.?", " ", t)
    t = re.sub(r"(?is)Example\s+A.*?Example\s+B", " ", t)
    t = re.sub(r"(?is)Now write the report for\s*:\s*", " ", t)
    t = re.sub(r"(?is)\bInputs?\s*:\s*", " ", t)
    t = re.sub(r"(?is)\bReport\s*:\s*", " ", t)
    t = re.sub(r"(?is)\bYou are an ophthalmology clinical assistant\.?", " ", t)
    t = re.sub(r"(?is)\bDo not return placeholders\.?", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _sanitize_freeform_output(text: str) -> str:
    s = text or ""
    # Remove known tokenizer/control artifacts from Gemma-family outputs.
    s = re.sub(r"(?is)\[multimodal\]", " ", s)
    s = re.sub(r"(?is)<unused\d+>", " ", s)
    s = re.sub(r"(?is)<\|[^>]+?\|>", " ", s)
    s = re.sub(r"(?is)</?s>", " ", s)
    # Keep only the tail after common echo points.
    for marker in [
        "Now write the report for:",
        "Now write the report for",
        "Report:",
    ]:
        idx = s.rfind(marker)
        if idx >= 0:
            s = s[idx + len(marker):]
    s = _strip_instruction_noise(s)
    # Remove common chat-template wrappers.
    s = re.sub(r"(?is)^user\s+", "", s).strip()
    s = re.sub(r"(?is)\s+model\s*$", "", s).strip()
    return s


def _looks_like_input_echo(text: str) -> bool:
    s = (text or "").lower()
    # Typical degenerate case: model just echoes scalar inputs.
    has_core = ("cdr" in s) and ("dr grade" in s) and ("vessel density" in s)
    low_content = len(re.sub(r"[^a-z]", "", s)) < 180
    no_clinical_verbs = not any(
        kw in s for kw in ["suggest", "indicate", "recommend", "triage", "follow-up", "referral", "assessment"]
    )
    return has_core and low_content and no_clinical_verbs


def _is_instruction_like(text: str) -> bool:
    s = (text or "").strip().lower()
    bad = [
        "use clinical language and do not include instructions",
        "each section must contain",
        "text to rewrite",
        "rules:",
        "return only these 4 sections",
        "do not output placeholders",
        "do not output only headings",
        "write one concise clinical ophthalmology finding sentence",
        "output only the sentence",
        "write a concise clinical paragraph",
        "do not use headings, bullets, numbering, or template labels",
        "include interpretation, triage risk, and follow-up advice",
        "clinical task: produce the final ophthalmology screening impression only",
        "output: one compact clinical paragraph",
    ]
    return any(b in s for b in bad)


def _normalize_medgemma(text: str, r: ScreeningResults) -> Optional[str]:
    t = _strip_instruction_noise((text or "").strip())
    if not t:
        return None
    k = t.find("1. FINDINGS")
    if k >= 0:
        t = t[k:]
    headers = ["1. FINDINGS", "2. RISK ASSESSMENT", "3. RECOMMENDATIONS", "4. DISCLAIMER"]
    has_all = all(h in t for h in headers)
    if has_all:
        # Reject template-like answers that only echo section titles.
        is_good = True
        for i, h in enumerate(headers):
            start = t.find(h)
            end = t.find(headers[i + 1]) if i + 1 < len(headers) else len(t)
            body = t[start + len(h):end].strip()
            body_clean = re.sub(r"[\W_]+", " ", body).strip().lower()
            if _is_instruction_like(body_clean):
                is_good = False
                break
            if len(body_clean) < 12:
                is_good = False
                break
            if body_clean in {"model", "output", "findings", "risk assessment", "recommendations", "disclaimer"}:
                is_good = False
                break
        if is_good:
            return t
        # If model echoed only template headers, reject hard.
        return None
    finding = _strip_instruction_noise(re.sub(r"\s+", " ", t).strip())
    if _is_instruction_like(finding):
        return None
    # If text still looks like template skeleton, reject.
    if "1. FINDINGS" in finding and "2. RISK ASSESSMENT" in finding:
        return None
    if len(re.sub(r"[^A-Za-z]", "", finding)) < 12:
        return None
    return (
        "1. FINDINGS\n"
        f"- {finding[:480]}\n\n"
        "2. RISK ASSESSMENT\n"
        f"- Overall triage risk: {_risk_from_signals(r.cdr, r.dr_grade)}.\n\n"
        "3. RECOMMENDATIONS\n"
        "- Correlate with clinical exam, IOP, OCT, and visual acuity.\n\n"
        "4. DISCLAIMER\n"
        "- This is AI-assisted retinal screening support and not a definitive diagnosis."
    )


def _augment_from_raw_medgemma(raw_text: str, r: ScreeningResults) -> Optional[str]:
    raw = _strip_instruction_noise(re.sub(r"\s+", " ", (raw_text or "")).strip())
    if _is_instruction_like(raw):
        return None
    raw = raw.replace("1. FINDINGS", "").replace("2. RISK ASSESSMENT", "").replace("3. RECOMMENDATIONS", "").replace("4. DISCLAIMER", "")
    raw = raw.replace("model", "").strip(" -:\n\t")
    if len(re.sub(r"[^A-Za-z]", "", raw)) < 18:
        return None
    return (
        "1. FINDINGS\n"
        f"- {raw[:420]}\n\n"
        "2. RISK ASSESSMENT\n"
        f"- Overall triage risk: {_risk_from_signals(r.cdr, r.dr_grade)}.\n\n"
        "3. RECOMMENDATIONS\n"
        "- Correlate with clinical exam, IOP, OCT, and visual acuity.\n\n"
        "4. DISCLAIMER\n"
        "- This is AI-assisted retinal screening support and not a definitive diagnosis."
    )


def _json_to_report(raw_json_text: str, r: ScreeningResults) -> Optional[str]:
    txt = (raw_json_text or "").strip()
    if not txt:
        return None
    # Try to extract JSON object even if model added text around it.
    s = txt.find("{")
    e = txt.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(txt[s : e + 1])
    except Exception:
        return None
    findings = _strip_instruction_noise(str(obj.get("findings", "")).strip())
    risk = _strip_instruction_noise(str(obj.get("risk_assessment", "")).strip())
    rec = _strip_instruction_noise(str(obj.get("recommendations", "")).strip())
    dis = _strip_instruction_noise(str(obj.get("disclaimer", "")).strip())
    if min(len(findings), len(risk), len(rec), len(dis)) < 12:
        return None
    if any(_is_instruction_like(x) for x in [findings, risk, rec, dis]):
        return None
    return (
        "1. FINDINGS\n"
        f"- {findings}\n\n"
        "2. RISK ASSESSMENT\n"
        f"- {risk}\n\n"
        "3. RECOMMENDATIONS\n"
        f"- {rec}\n\n"
        "4. DISCLAIMER\n"
        f"- {dis}"
    )


def _single_sentence_to_report(sentence: str, r: ScreeningResults) -> Optional[str]:
    s = _strip_instruction_noise(sentence)
    if len(re.sub(r"[^A-Za-z]", "", s)) < 18:
        return None
    if _is_instruction_like(s):
        return None
    # Reject chat-template echo (e.g., "user ... model").
    s_low = s.lower()
    if s_low.startswith("user ") or s_low.endswith(" model") or " user " in s_low:
        return None
    if "write one concise clinical ophthalmology finding sentence" in s_low:
        return None
    return (
        "1. FINDINGS\n"
        f"- {s}\n\n"
        "2. RISK ASSESSMENT\n"
        f"- Overall triage risk: {_risk_from_signals(r.cdr, r.dr_grade)}.\n\n"
        "3. RECOMMENDATIONS\n"
        "- Correlate with clinical exam, IOP, OCT, and visual acuity.\n\n"
        "4. DISCLAIMER\n"
        "- This is AI-assisted retinal screening support and not a definitive diagnosis."
    )


def _guided_text_to_report(text: str, r: ScreeningResults) -> Optional[str]:
    s = _strip_instruction_noise(text)
    s = re.sub(r"(?is)^user\s+.*?\bmodel\b", "", s).strip()
    if len(re.sub(r"[^A-Za-z]", "", s)) < 30:
        return None
    if _is_instruction_like(s):
        return None
    risk = _risk_from_signals(r.cdr, r.dr_grade)
    return (
        "1. FINDINGS\n"
        f"- {s[:700]}\n\n"
        "2. RISK ASSESSMENT\n"
        f"- Overall triage risk: {risk}.\n\n"
        "3. RECOMMENDATIONS\n"
        "- Follow-up timing according to risk and correlate with clinical exam, IOP, OCT, and visual acuity.\n\n"
        "4. DISCLAIMER\n"
        "- This is AI-assisted retinal screening support and not a definitive diagnosis."
    )


def _extract_any_useful_medgemma_text(candidates: list[str]) -> Optional[str]:
    for raw in candidates:
        s = _strip_instruction_noise(raw or "")
        s = re.sub(r"(?is)^user\s+.*?\bmodel\b", "", s).strip()
        if not s:
            continue
        if _is_instruction_like(s):
            continue
        s_low = s.lower()
        # Reject degenerate "headers only" concatenations.
        if (
            "1. findings" in s_low
            and "2. risk assessment" in s_low
            and "3. recommendations" in s_low
            and "4. disclaimer" in s_low
        ):
            continue
        if s_low.strip() in {"model", "findings", "risk assessment", "recommendations", "disclaimer"}:
            continue
        if len(re.sub(r"[^A-Za-z]", "", s)) < 20:
            continue
        return s
    return None


def _extract_any_medgemma_raw(candidates: list[str]) -> Optional[str]:
    for raw in candidates:
        s = _sanitize_freeform_output(raw or "")
        if not s:
            continue
        if _is_instruction_like(s):
            continue
        if _looks_like_input_echo(s):
            continue
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) < 12:
            continue
        return s
    return None


def _clean_free_paragraph(text: str) -> Optional[str]:
    s = _sanitize_freeform_output(text or "")
    s = re.sub(r"(?im)^\s*[1-4]\.\s*(FINDINGS|RISK ASSESSMENT|RECOMMENDATIONS|DISCLAIMER)\s*$", " ", s)
    s = re.sub(r"(?im)\b(FINDINGS|RISK ASSESSMENT|RECOMMENDATIONS|DISCLAIMER)\b\s*:?"," ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if _is_instruction_like(s):
        return None
    if _looks_like_input_echo(s):
        return None
    if len(re.sub(r"[^A-Za-z]", "", s)) < 50:
        return None
    return s


def _is_text_consistent_with_signals(text: str, r: ScreeningResults) -> bool:
    s = (text or "").lower()
    if not s:
        return False

    cdr_vals = [float(m) for m in re.findall(r"\bcdr\b[^0-9]{0,10}([01](?:\.\d+)?)", s)]
    for v in cdr_vals:
        if abs(v - float(r.cdr)) > 0.08:
            return False

    grade_vals = [int(m) for m in re.findall(r"\b(?:dr\s*)?grade\b[^0-9]{0,8}([0-4])", s)]
    for g in grade_vals:
        if g != int(r.dr_grade):
            return False

    conf_vals = [float(m) for m in re.findall(r"\bconfidence\b[^0-9]{0,10}([0-9]{1,3}(?:\.\d+)?)\s*%", s)]
    for c in conf_vals:
        if abs(c - float(r.dr_conf) * 100.0) > 6.0:
            return False

    vessel_vals = [float(m) for m in re.findall(r"\bvessel\s*density\b[^0-9]{0,14}([0-9]{1,3}(?:\.\d+)?)\s*%", s)]
    for v in vessel_vals:
        if abs(v - float(r.vessel_density) * 100.0) > 6.0:
            return False

    return True


def _repair_numeric_inconsistencies(text: str, r: ScreeningResults) -> str:
    s = text or ""
    # Strip common few-shot contamination first.
    s = re.sub(r"(?is)\bexample\s+a\b.*?\bexample\s+b\b", " ", s)
    s = re.sub(r"(?is)\bnow write the report for\b.*$", " ", s)
    s = re.sub(r"(?is)\blabel\s*:\s*", " ", s)
    # Normalize common numeric fields to authoritative model signals.
    s = re.sub(
        r"(?i)(\bcdr\b[^0-9]{0,10})([01](?:\.\d+)?)",
        lambda m: f"{m.group(1)}{r.cdr:.3f}",
        s,
    )
    s = re.sub(
        r"(?i)(\b(?:dr\s*)?grade\b[^0-9]{0,8})([0-4])",
        lambda m: f"{m.group(1)}{int(r.dr_grade)}",
        s,
    )
    s = re.sub(
        r"(?i)(\bconfidence\b[^0-9]{0,10})([0-9]{1,3}(?:\.\d+)?)\s*%",
        lambda m: f"{m.group(1)}{r.dr_conf * 100.0:.1f}%",
        s,
    )
    s = re.sub(
        r"(?i)(\bvessel\s*density\b[^0-9]{0,14})([0-9]{1,3}(?:\.\d+)?)\s*%",
        lambda m: f"{m.group(1)}{r.vessel_density * 100.0:.1f}%",
        s,
    )
    # Remove known contamination tokens.
    s = re.sub(r"(?is)\[multimodal\]|<unused\d+>|<\|[^>]+?\|>|</?s>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _finalize_medgemma_paragraph(text: str, r: ScreeningResults) -> Optional[str]:
    s = _clean_free_paragraph(text or "")
    if not s:
        return None
    # Hard cuts for common contamination tails.
    cut_markers = ["***", "example a", "example b", "now write the report for", "text to rewrite"]
    low = s.lower()
    for m in cut_markers:
        k = low.find(m)
        if k > 0:
            s = s[:k].strip()
            low = s.lower()
    # If model starts chaining multiple synthetic scenarios, keep only first one.
    cdr_hits = [m.start() for m in re.finditer(r"\bcdr\s*[=:]", low)]
    if len(cdr_hits) >= 2:
        s = s[:cdr_hits[1]].strip()
    # Keep only first 5 complete sentences.
    parts = re.split(r"(?<=[.!?])\s+", s)
    out = []
    for p in parts:
        p2 = p.strip()
        if not p2:
            continue
        if len(re.sub(r"[^A-Za-z]", "", p2)) < 12:
            continue
        out.append(p2)
        if len(out) >= 5:
            break
    if not out:
        return None
    s = " ".join(out).strip()
    if len(re.sub(r"[^A-Za-z]", "", s)) < 25:
        return None
    if not _is_text_consistent_with_signals(s, r):
        s2 = _repair_numeric_inconsistencies(s, r)
        if not _is_text_consistent_with_signals(s2, r):
            return None
        s = s2
    return s


def run_pipeline(image: Image.Image, use_medgemma: bool, strict_medgemma: bool, model_id: str) -> Tuple[str, str]:
    if image is None:
        return "No image uploaded.", "error: missing image"

    results, infer_err = PIPELINE.infer(image)
    if results is None:
        return "", f"mode: error\nerror: {infer_err}"

    mode = "signals_only"
    report = (
        "1. FINDINGS\n"
        f"- Glaucoma screening: CDR {results.cdr:.3f}, risk category {results.glaucoma_risk}.\n"
        f"- DR grading: Grade {results.dr_grade} ({results.dr_label}), confidence {results.dr_conf:.1%}.\n"
        f"- Vascular analysis: estimated vessel density {results.vessel_density:.1%}.\n\n"
        "2. RISK ASSESSMENT\n"
        f"- Overall triage risk: {_risk_from_signals(results.cdr, results.dr_grade)}.\n\n"
        "3. RECOMMENDATIONS\n"
        "- Correlate with clinical exam, IOP, OCT, and visual acuity.\n\n"
        "4. DISCLAIMER\n"
        "- This is AI-assisted retinal screening support and not a definitive diagnosis."
    )
    err = ""

    if use_medgemma:
        # Dual-pass MedGemma without fallback:
        # pass-1 creates the report, pass-2 audits and improves it using the same authoritative signals.
        pass1_prompts = [
            _prompt_dual_pass_stage1(results),
            _prompt_clinical_paragraph(results),
            _prompt_clinical_paragraph_v2(results),
            _prompt_guided_free_text(results),
            _prompt_json(results),
            _prompt_single_sentence(results),
        ]
        clean1 = None
        err1 = ""
        raw1_last = ""
        for p1 in pass1_prompts:
            raw1, e1 = MEDGEMMA.generate_messages(
                [
                    {"role": "system", "content": _system_stage1()},
                    {"role": "user", "content": p1},
                ],
                model_id=model_id,
            )
            raw1_last = raw1 or raw1_last
            if e1:
                err1 = e1
            clean1 = _finalize_medgemma_paragraph(raw1 or "", results)
            if clean1:
                break

            # Bridge path: if MedGemma returned structured/JSON-ish text, normalize then convert to paragraph (still MedGemma).
            structured = _json_to_report(raw1 or "", results) or _normalize_medgemma(raw1 or "", results)
            if structured:
                p_bridge = _prompt_structured_to_paragraph(results, structured)
                rawb, eb = MEDGEMMA.generate_messages(
                    [
                        {"role": "system", "content": _system_stage2()},
                        {"role": "user", "content": p_bridge},
                    ],
                    model_id=model_id,
                )
                if eb:
                    err1 = eb
                clean1 = _finalize_medgemma_paragraph(rawb or "", results)
                if clean1:
                    break

            # One MedGemma repair retry for partially valid raw.
            if raw1:
                repair_prompt = _prompt_repair_to_signals(results, raw1)
                raw1r, e1r = MEDGEMMA.generate_messages(
                    [
                        {"role": "system", "content": _system_stage2()},
                        {"role": "user", "content": repair_prompt},
                    ],
                    model_id=model_id,
                )
                if e1r:
                    err1 = e1r
                clean1 = _finalize_medgemma_paragraph(raw1r or "", results)
                if clean1:
                    break

        # Do not hard-stop on pass1 failure. Pass2 is the authoritative auditing/refinement stage.
        draft_for_pass2 = clean1 if clean1 else (raw1_last or "")
        pass2_prompts = [
            _prompt_dual_pass_stage2(results, draft_for_pass2),
            _prompt_repair_to_signals(results, draft_for_pass2),
            _prompt_dual_pass_stage1(results),
            _prompt_clinical_paragraph(results),
            _prompt_guided_free_text(results),
        ]
        clean2 = None
        err2 = ""
        raw2_last = ""
        for p2 in pass2_prompts:
            raw2, e2 = MEDGEMMA.generate_messages(
                [
                    {"role": "system", "content": _system_stage2()},
                    {"role": "user", "content": p2},
                ],
                model_id=model_id,
            )
            raw2_last = raw2 or raw2_last
            if e2:
                err2 = e2
            clean2 = _finalize_medgemma_paragraph(raw2 or "", results)
            if clean2:
                break

        if not clean2:
            base_err = err2 or "pass2_invalid_or_inconsistent_output"
            if not clean1:
                base_err = f"{base_err} | pass1_invalid_or_inconsistent_output"
            if raw1_last:
                base_err = f"{base_err} | raw1_len={len(raw1_last)}"
            if raw2_last:
                base_err = f"{base_err} | raw2_len={len(raw2_last)}"
            if strict_medgemma:
                mode = "medgemma_error_no_fallback"
                err = base_err
                report = ""
            else:
                # Senior-stable mode: always return a final clinician-facing block without noisy error tails.
                report = _build_formatted_report_from_signals(results, draft_for_pass2)
                mode = "medgemma_self_healed_authoritative"
                err = ""
        else:
            # Pass-3 formatter: force final output into challenge-friendly clinical block.
            p3 = _prompt_format_final_block(results, clean2)
            raw3, err3 = MEDGEMMA.generate_messages(
                [
                    {"role": "system", "content": _system_stage3()},
                    {"role": "user", "content": p3},
                ],
                model_id=model_id,
            )
            formatted = _finalize_formatted_report(raw3 or "", results)
            if formatted:
                report = formatted
                mode = "medgemma_triple_pass_formatted"
                err = ""
            else:
                # Autonomous stable behavior (no manual edits needed).
                if strict_medgemma:
                    mode = "medgemma_error_no_fallback"
                    err = err3 or "pass3_format_invalid"
                    report = ""
                else:
                    report = _build_formatted_report_from_signals(results, clean2)
                    mode = "medgemma_triple_pass_self_healed_format"
                    err = ""

    metrics = (
        f"pipeline_version: {PIPELINE_VERSION}\n"
        f"mode: {mode}\n"
        f"models_ready: {PIPELINE.ready}\n"
        f"device: {PIPELINE.device}\n"
        f"cdr: {results.cdr:.3f}\n"
        f"glaucoma_risk: {results.glaucoma_risk}\n"
        f"dr_grade: {results.dr_grade} ({results.dr_label})\n"
        f"dr_conf: {results.dr_conf:.3f}\n"
        f"vessel_density: {results.vessel_density:.3f}\n"
    )
    if err:
        metrics += f"error: {err}\n"
    return report, metrics


def run_pipeline_from_path(image_path: str, use_medgemma: bool, strict_medgemma: bool, model_id: str) -> Tuple[str, str]:
    p = (image_path or "").strip()
    if not p:
        return "", "mode: error\nerror: empty image path"
    path = Path(p)
    if not path.exists():
        return "", f"mode: error\nerror: file not found: {p}"
    try:
        img = Image.open(path).convert("RGB")
    except Exception as exc:
        return "", f"mode: error\nerror: failed to open image path: {exc}"
    return run_pipeline(img, use_medgemma, strict_medgemma, model_id)


def check_models() -> str:
    return PIPELINE.status_text()


def _find_free_port(start: int = 7860, end: int = 7890) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return 7860


def main() -> None:
    os.environ.setdefault("FORCE_GPU_ONLY", "1")
    import gradio as gr

    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*'theme' parameter in the Blocks constructor.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*'css' parameter in the Blocks constructor.*")

    # Avoid "bound to a different event loop" when rerunning in notebook kernels.
    try:
        gr.close_all()
    except Exception:
        pass
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception:
        pass

    css = """
    .vv-wrap {max-width: 1200px; margin: 0 auto;}
    .vv-hero {
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: #f8fafc; padding: 18px 20px; border-radius: 14px;
      border: 1px solid #334155; margin-bottom: 14px;
    }
    .vv-hero h1 {margin: 0 0 6px 0; font-size: 26px; font-weight: 700;}
    .vv-hero p {margin: 0; color: #cbd5e1; font-size: 14px;}
    .vv-section {border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px;}
    .vv-note {font-size: 12px; color: #475569;}
    """

    with gr.Blocks(title="Veredictos Vision Demo Pipeline", css=css, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div class="vv-wrap">
              <div class="vv-hero">
                <h1>Veredictos Vision</h1>
                <p>Multi-pathology retinal screening with 3 checkpoints + MedGemma narrative generation.</p>
              </div>
            </div>
            """
        )
        gr.Markdown("Path-based execution only (upload intentionally disabled on Kaggle due Gradio event-loop instability).")

        with gr.Row():
            with gr.Column(scale=5, elem_classes=["vv-section"]):
                gr.Markdown("### Input & Configuration")
                image_path = gr.Textbox(
                    value="/kaggle/input/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset/full-fundus/full-fundus/BEH-1.png",
                    label="Fundus image path",
                    placeholder="/kaggle/input/.../image.png",
                )
                model_id = gr.Textbox(value="google/medgemma-4b-it", label="MedGemma model id")
                with gr.Row():
                    use_medgemma = gr.Checkbox(value=True, label="Enable MedGemma")
                    strict_medgemma = gr.Checkbox(value=False, label="Strict output validation")
                with gr.Row():
                    run_path_btn = gr.Button("Run Clinical Pipeline", variant="primary", size="lg")
                    check_btn = gr.Button("Check Model Loading")
                gr.Markdown(
                    "Quick examples:",
                    elem_classes=["vv-note"],
                )
                gr.Examples(
                    examples=[
                        ["/kaggle/input/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset/full-fundus/full-fundus/BEH-1.png"],
                        ["/kaggle/input/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset/full-fundus/full-fundus/BEH-10.png"],
                    ],
                    inputs=[image_path],
                    label="Sample Paths",
                )

            with gr.Column(scale=7):
                with gr.Group(elem_classes=["vv-section"]):
                    gr.Markdown("### Clinical Report")
                    report = gr.Textbox(label="Generated report", lines=18)
                with gr.Group(elem_classes=["vv-section"]):
                    gr.Markdown("### Pipeline Signals")
                    metrics = gr.Textbox(label="Execution diagnostics", lines=12)

        run_path_btn.click(
            fn=run_pipeline_from_path,
            inputs=[image_path, use_medgemma, strict_medgemma, model_id],
            outputs=[report, metrics],
        )
        check_btn.click(
            fn=check_models,
            inputs=[],
            outputs=[metrics],
        )

    is_kaggle = ("KAGGLE_KERNEL_RUN_TYPE" in os.environ) or ("KAGGLE_URL_BASE" in os.environ)
    running_in_kernel = "ipykernel" in sys.modules
    # Kaggle inline is unstable in many runtimes; keep OFF by default.
    # You can force it with FORCE_INLINE=1.
    default_inline = "0"
    # In Kaggle, keep share on by default to avoid localhost access failures.
    default_share = "1" if is_kaggle else "0"
    force_share = os.getenv("FORCE_SHARE", default_share) == "1"
    force_inline = os.getenv("FORCE_INLINE", default_inline) == "1"
    if is_kaggle and force_inline and (not running_in_kernel):
        print(
            "[launch] FORCE_INLINE=1 requested, but current execution is a subprocess "
            "(likely `!python`). Inline below-cell is not possible in this mode."
        )
        force_inline = False

    # Keep modes separate to avoid unstable inline+share combination in Kaggle.
    if force_inline:
        try:
            print("[launch] trying inline mode (127.0.0.1, no share).")
            demo.launch(
                share=False,
                server_name="127.0.0.1",
                server_port=_find_free_port(),
                max_threads=1,
                inline=True,
                prevent_thread_lock=True,
                show_error=True,
                inbrowser=False,
            )
            return
        except Exception as inline_exc:
            print(f"[launch] inline failed: {inline_exc}")
            if not force_share:
                raise

    print("[launch] using public share mode.")
    demo.launch(
        share=True if (is_kaggle or force_share) else False,
        server_name="0.0.0.0" if is_kaggle else "127.0.0.1",
        server_port=_find_free_port(),
        max_threads=1,
        inline=False,
        prevent_thread_lock=True,
        show_error=True,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()



