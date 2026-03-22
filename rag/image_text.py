"""
Read text or a description from an image using a **local Hugging Face** model.

Backends (IMAGE_TEXT_BACKEND):
  blip2 (default) — BLIP-2 + OPT; best among allowed HF models for screenshot / UI text.
  llava            — LLaVA 1.5 7B; strong if you have GPU RAM.
  clip_interrogator — CLIP Interrogator (semantic tags; weak for verbatim UI text).
  blip_vqa         — BLIP-1 VQA; often gives useless answers on screenshots.
  blip_caption     — BLIP-1 captioning (scene description, not OCR).
  tesseract        — System Tesseract + pytesseract (classic OCR).

Needs: pip install transformers torch accelerate Pillow
Optional: clip-interrogator, pytesseract + OS Tesseract
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
from typing import Any

from . import config

logger = logging.getLogger(__name__)
_WS = re.compile(r"\s+")

_lock = threading.Lock()
_pipe_vqa: Any = None
_pipe_caption: Any = None
_llava_bundle: tuple[Any, Any, str] | None = None
_blip2_bundle: tuple[Any, Any, str] | None = None
_ci: Any = None

DEFAULT_MODELS = {
    "blip2": "Salesforce/blip2-opt-2.7b",
    "blip_vqa": "Salesforce/blip-vqa-base",
    "blip_caption": "Salesforce/blip-image-captioning-base",
    "llava": "llava-hf/llava-1.5-7b-hf",
}

# BLIP-2 follows this format best when it ends with "Answer:" (Salesforce checkpoint).
_BLIP2_OCR_PROMPT = (
    "Question: This image shows a search bar, text field, or screenshot with typed English words. "
    "Transcribe the complete sentence or query the user typed, word for word, left to right. "
    "Include ordinary words like how, to, make, banana, smoothie. "
    "Output only that text as one line — no labels, no 'the image shows', no guessing. "
    "If you truly see no letters at all, answer NONE. Answer:"
)


def _clean(s: str) -> str:
    return _WS.sub(" ", (s or "").strip()).strip()


def _prepare_image_for_blip2(image):
    """Upscale small crops and boost contrast so thin UI text is easier for BLIP-2 to read."""
    from PIL import Image, ImageEnhance

    img = image
    w, h = img.size
    target = config.BLIP2_UPSCALE_MIN_EDGE
    if target > 0 and max(w, h) < target:
        scale = target / float(max(w, h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.25)
    return img


def _resolve_device() -> str:
    d = (config.HF_IMAGE_DEVICE or "auto").strip().lower()
    if d != "auto":
        return d
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _pipeline_device_arg(resolved: str) -> int | str:
    if resolved == "cpu":
        return -1
    if resolved.startswith("cuda"):
        return resolved if ":" in resolved else "cuda:0"
    if resolved == "mps":
        return "mps"
    return -1


def _accelerated_torch_dtype():
    """Use fp16 on CUDA/MPS for faster BLIP inference; CPU stays default (fp32)."""
    try:
        import torch

        dev = _resolve_device()
        if dev.startswith("cuda") or dev == "mps":
            return torch.float16
    except ImportError:
        pass
    return None


def extract_text_from_image(image_bytes: bytes) -> str:
    if not image_bytes:
        return ""
    backend = config.IMAGE_TEXT_BACKEND

    try:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.warning("Could not open image: %s", e)
        return ""

    try:
        if backend == "tesseract":
            return _clean(_tesseract_bytes(image_bytes))
        if backend == "blip2":
            return _clean(_run_blip2(_prepare_image_for_blip2(image)))
        if backend == "blip_vqa":
            return _clean(_blip_vqa(image))
        if backend == "blip_caption":
            return _clean(_blip_caption(image))
        if backend == "llava":
            return _clean(_run_llava(image))
        if backend == "clip_interrogator":
            return _clean(_clip_interrogator(image))
    except Exception as e:
        logger.warning("Image text extraction failed (%s): %s", backend, e)
        return ""

    logger.warning("Unknown IMAGE_TEXT_BACKEND=%r", backend)
    return ""


def _model_id(backend_key: str) -> str:
    return config.IMAGE_TEXT_MODEL or DEFAULT_MODELS.get(
        backend_key, DEFAULT_MODELS["blip2"]
    )


def _run_blip2(image) -> str:
    """BLIP-2 conditional generation — much better than BLIP-1 VQA for reading UI text."""
    global _blip2_bundle
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    mid = _model_id("blip2")
    device_s = _resolve_device()
    # CUDA: fp16 is faster. CPU/MPS: fp32 avoids BLIP-2 numerical issues and flaky MPS half kernels.
    torch_dtype = torch.float16 if device_s.startswith("cuda") else torch.float32

    with _lock:
        if _blip2_bundle is None or _blip2_bundle[2] != mid:
            processor = Blip2Processor.from_pretrained(mid)
            model = Blip2ForConditionalGeneration.from_pretrained(
                mid,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            if device_s == "cpu":
                model = model.to("cpu")
            elif device_s.startswith("cuda"):
                model = model.to(device_s if ":" in device_s else "cuda:0")
            elif device_s == "mps":
                model = model.to("mps")
            model.eval()
            _blip2_bundle = (processor, model, mid)

    processor, model, _ = _blip2_bundle
    prompt = os.environ.get("BLIP2_OCR_PROMPT", "").strip() or _BLIP2_OCR_PROMPT
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    if torch_dtype == torch.float16:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    input_len = inputs["input_ids"].shape[1]
    beams = config.BLIP2_NUM_BEAMS
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max(48, config.HF_IMAGE_MAX_NEW_TOKENS),
            do_sample=False,
            num_beams=beams,
            early_stopping=True,
        )
    new_tokens = out[0, input_len:]
    text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return text


def _blip_vqa(image) -> str:
    global _pipe_vqa
    mid = _model_id("blip_vqa")
    dev = _resolve_device()
    with _lock:
        if _pipe_vqa is None:
            from transformers import pipeline

            kw: dict[str, Any] = {
                "model": mid,
                "device": _pipeline_device_arg(dev),
            }
            dt = _accelerated_torch_dtype()
            if dt is not None:
                kw["torch_dtype"] = dt
            _pipe_vqa = pipeline("visual-question-answering", **kw)
    q = (
        "What exact words and numbers are shown as text in this image? Copy them verbatim "
        "in reading order as one line, including search box or field text. Do not describe the image."
    )
    out = _pipe_vqa(image=image, question=q, top_k=1)
    if isinstance(out, list) and out:
        return str(out[0].get("answer", ""))
    if isinstance(out, dict):
        return str(out.get("answer", ""))
    return str(out)


def _blip_caption(image) -> str:
    global _pipe_caption
    mid = _model_id("blip_caption")
    dev = _resolve_device()
    with _lock:
        if _pipe_caption is None:
            from transformers import pipeline

            kw: dict[str, Any] = {
                "model": mid,
                "device": _pipeline_device_arg(dev),
            }
            dt = _accelerated_torch_dtype()
            if dt is not None:
                kw["torch_dtype"] = dt
            _pipe_caption = pipeline("image-to-text", **kw)
    out = _pipe_caption(image)
    if isinstance(out, list) and out:
        return str(out[0].get("generated_text", ""))
    return str(out)


def _run_llava(image) -> str:
    global _llava_bundle
    import torch

    mid = _model_id("llava")
    device = _resolve_device()
    torch_dtype = (
        torch.float16
        if device.startswith("cuda") or device == "mps"
        else torch.float32
    )

    with _lock:
        if _llava_bundle is None or _llava_bundle[2] != mid:
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            processor = AutoProcessor.from_pretrained(mid)
            model = LlavaForConditionalGeneration.from_pretrained(
                mid,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            if device == "cpu":
                model = model.to("cpu")
            elif device.startswith("cuda"):
                model = model.to(device)
            elif device == "mps":
                model = model.to("mps")
            model.eval()
            _llava_bundle = (processor, model, mid)

    processor, model, _ = _llava_bundle
    instruction = (
        "Transcribe all visible text in the image exactly as written (search queries, "
        "labels, UI text). One line of plain text only. If there is no text, reply NONE."
    )
    prompt = (
        f"USER: <image>\n{instruction}\nASSISTANT:"
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=max(32, config.HF_IMAGE_MAX_NEW_TOKENS),
            do_sample=False,
        )
    # Decoding the full sequence repeats the prompt and breaks extraction; only decode new tokens.
    new_tokens = gen[0, input_len:]
    text = processor.decode(new_tokens, skip_special_tokens=True).strip()
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    return text


def _clip_interrogator(image) -> str:
    global _ci
    try:
        from clip_interrogator import Config, Interrogator
    except ImportError:
        logger.warning(
            "clip_interrogator backend requires: pip install clip-interrogator"
        )
        return ""

    import torch

    with _lock:
        if _ci is None:
            cfg = Config()
            r = _resolve_device()
            if r.startswith("cuda"):
                cfg.device = "cuda"
            elif r == "mps" and torch.backends.mps.is_available():
                cfg.device = "mps"
            else:
                cfg.device = "cpu"
            _ci = Interrogator(cfg)
    return str(_ci.interrogate(image))


def _tesseract_bytes(image_bytes: bytes) -> str:
    cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if cmd:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning("Tesseract backend needs: pip install pytesseract Pillow")
        return ""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    try:
        raw = pytesseract.image_to_string(img, lang="eng")
    except pytesseract.TesseractNotFoundError:
        logger.warning("Tesseract executable not found on PATH.")
        return ""
    return (raw or "").strip()
