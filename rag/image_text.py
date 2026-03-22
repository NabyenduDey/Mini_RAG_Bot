"""
Read text or a description from an image using a **local Hugging Face** model.

Backends (IMAGE_TEXT_BACKEND):
  blip_vqa (default) — BLIP VQA; asks the model to describe / transcribe visible text.
  blip_caption     — BLIP image captioning (short scene description).
  llava            — LLaVA 1.5 7B; strong if you have GPU RAM.
  clip_interrogator — CLIP Interrogator style caption (pip install clip-interrogator).
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
_ci: Any = None

DEFAULT_MODELS = {
    "blip_vqa": "Salesforce/blip-vqa-base",
    "blip_caption": "Salesforce/blip-image-captioning-base",
    "llava": "llava-hf/llava-1.5-7b-hf",
}


def _clean(s: str) -> str:
    return _WS.sub(" ", (s or "").strip()).strip()


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
        backend_key, DEFAULT_MODELS["blip_vqa"]
    )


def _blip_vqa(image) -> str:
    global _pipe_vqa
    mid = _model_id("blip_vqa")
    dev = _resolve_device()
    with _lock:
        if _pipe_vqa is None:
            from transformers import pipeline

            _pipe_vqa = pipeline(
                "visual-question-answering",
                model=mid,
                device=_pipeline_device_arg(dev),
            )
    q = (
        "What text appears in this image? Transcribe all readable words, numbers, "
        "and symbols as plain text."
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

            _pipe_caption = pipeline(
                "image-to-text",
                model=mid,
                device=_pipeline_device_arg(dev),
            )
    out = _pipe_caption(image)
    if isinstance(out, list) and out:
        return str(out[0].get("generated_text", ""))
    return str(out)


def _run_llava(image) -> str:
    global _llava_bundle
    import torch

    mid = _model_id("llava")
    device = _resolve_device()
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    with _lock:
        if _llava_bundle is None or _llava_bundle[2] != mid:
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            processor = AutoProcessor.from_pretrained(mid)
            model = LlavaForConditionalGeneration.from_pretrained(
                mid,
                torch_dtype=torch_dtype,
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
    prompt = (
        "USER: <image>\nRead all visible text in this image. If there is no text, "
        "reply exactly: NONE. Output plain transcript only, no commentary.\nASSISTANT:"
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    # Move tensors to the same device as the model (BatchFeature supports .to).
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=config.HF_IMAGE_MAX_NEW_TOKENS,
            do_sample=False,
        )
    text = processor.decode(gen[0], skip_special_tokens=True)
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1]
    return text.strip()


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
