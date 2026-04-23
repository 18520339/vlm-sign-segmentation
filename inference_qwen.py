"""Qwen3-VL local inference for sign language phrase segmentation.

Loads the model once, then processes each video to produce phrase-level
temporal segments.  Designed for Google Colab G4 (RTX PRO 6000, 96 GB VRAM).

CRITICAL: Qwen3-VL uses "Text-Timestamp Alignment" (NOT T-RoPE like Qwen2.5-VL).
This requires passing `video_metadata` to the processor so it can inject the
correct frame-to-time mapping.  Without it, the model outputs timestamps
clustered near 0 because it has no temporal reference.

The correct pipeline for Qwen3-VL is:
1. process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)
2. Unpack video_inputs as (tensor, metadata) tuples
3. Pass video_metadata to the processor alongside **video_kwargs
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from data_utils import Segment, save_segments_json, load_segments_json
from config import (
    QWEN_MODEL_ID, QWEN_VIDEO_FPS, QWEN_MAX_PIXELS,
    QWEN_MAX_NEW_TOKENS, PHRASE_SEGMENTATION_PROMPT, OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

# Module-level model cache (loaded once per session)
_model = None
_processor = None


# ── Model loading ──────────────────────────────────────────────────────────────

def load_qwen_model(model_id: str = QWEN_MODEL_ID):
    """Load Qwen3-VL model and processor, cached at module level."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    logger.info("Loading %s …", model_id)

    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",    # PyTorch-native; avoids slow flash-attn install
        device_map="auto",
    )
    _processor = AutoProcessor.from_pretrained(model_id)

    logger.info("Model loaded successfully.")
    return _model, _processor


# ── JSON parsing ───────────────────────────────────────────────────────────────

def _parse_segments_json(text: str) -> List[Segment]:
    """Parse a JSON array of {start, end} from model output."""
    text = text.strip()

    # Strip think blocks if present (Qwen3-VL-Thinking variants)
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):].strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _validate_segments([Segment.from_dict(d) for d in data])
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fallback: extract first JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return _validate_segments([Segment.from_dict(d) for d in data])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    logger.warning("Could not parse JSON segments from Qwen response:\n%s", text[:500])
    return []


def _validate_segments(segments: List[Segment]) -> List[Segment]:
    """Filter invalid segments and resolve overlaps."""
    valid = [s for s in segments if s.end_s > s.start_s >= 0]
    valid.sort(key=lambda s: s.start_s)

    cleaned = []
    for seg in valid:
        if cleaned and seg.start_s < cleaned[-1].end_s:
            seg = Segment(start_s=cleaned[-1].end_s, end_s=seg.end_s)
            if seg.end_s <= seg.start_s:
                continue
        cleaned.append(seg)
    return cleaned


# ── Sanitise video_kwargs ──────────────────────────────────────────────────────

def _sanitize_video_kwargs(video_kwargs: dict) -> dict:
    """Fix process_vision_info quirk: fps comes as a list, processor wants scalar."""
    if not video_kwargs:
        return video_kwargs

    sanitized = dict(video_kwargs)

    # fps: list[float] → float
    if "fps" in sanitized:
        fps_val = sanitized["fps"]
        if isinstance(fps_val, (list, tuple)):
            if len(fps_val) >= 1:
                sanitized["fps"] = float(fps_val[0])

    return sanitized


# ── Single-video inference ─────────────────────────────────────────────────────

def _infer_single(
    video_path: Path,
    model,
    processor,
    fps: float = QWEN_VIDEO_FPS,
    max_pixels: int = QWEN_MAX_PIXELS,
    max_new_tokens: int = QWEN_MAX_NEW_TOKENS,
) -> List[Segment]:
    """Run inference on a single video (≤ ~3 min).

    Qwen3-VL requires video_metadata for its Text-Timestamp Alignment
    mechanism.  Without it, the model outputs garbled near-zero timestamps.
    """
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path.resolve()),
                    "max_pixels": max_pixels,
                    "fps": float(fps),
                },
                {"type": "text", "text": PHRASE_SEGMENTATION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # ── Extract frames AND temporal metadata ──────────────────────────────
    # Qwen3-VL needs BOTH:
    #   - video_kwargs (fps info for temporal position IDs)
    #   - video_metadata (frame indices + timing for Text-Timestamp Alignment)
    #
    # Without return_video_metadata=True, the processor falls back to a
    # default FPS (often 24), causing "timestamp drift" where the model's
    # internal time doesn't match the actual video timeline.
    try:
        # Qwen3-VL path: return_video_metadata=True
        image_inputs, video_inputs_raw, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        # Unpack (tensor, metadata) tuples
        if video_inputs_raw is not None and len(video_inputs_raw) > 0:
            video_inputs = [item[0] for item in video_inputs_raw]
            video_metadatas = [item[1] for item in video_inputs_raw]
            logger.info("  Extracted %d video(s) with metadata (Qwen3-VL path)",
                        len(video_inputs))
            if video_metadatas:
                logger.info("  Video metadata: %s",
                            {k: v for k, v in video_metadatas[0].items()
                             if k != 'video'} if isinstance(video_metadatas[0], dict) else type(video_metadatas[0]))
        else:
            video_inputs = None
            video_metadatas = None

    except TypeError:
        # Fallback for older qwen-vl-utils without return_video_metadata
        logger.warning("  qwen-vl-utils does not support return_video_metadata; "
                       "temporal grounding may be degraded. Upgrade: "
                       "pip install --upgrade qwen-vl-utils")
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True,
        )
        video_metadatas = None

    # Sanitize fps from list to scalar
    video_kwargs = _sanitize_video_kwargs(video_kwargs)
    logger.info("  video_kwargs (sanitized): %s", video_kwargs)

    # Build processor inputs
    processor_kwargs = dict(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    # Add video_metadata if available (Qwen3-VL Text-Timestamp Alignment)
    if video_metadatas is not None:
        processor_kwargs["video_metadata"] = video_metadatas

    inputs = processor(**processor_kwargs).to(model.device)

    logger.info("  Input tokens: %d", inputs.input_ids.shape[-1])

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        do_sample=True,
    )

    # Trim input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    logger.info("  Raw Qwen output (first 500 chars): %s", output_text[:500])
    return _parse_segments_json(output_text)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_qwen_inference(
    video_path: Path,
    *,
    model_id: str = QWEN_MODEL_ID,
    cache_dir: Optional[Path] = None,
) -> List[Segment]:
    """Run Qwen3-VL phrase segmentation on *video_path*.

    Results are cached as JSON.  On subsequent calls the cached result is
    returned immediately.
    """
    cache_dir = cache_dir or OUTPUT_DIR
    cache_path = cache_dir / f"{video_path.stem}_qwen.json"

    if cache_path.exists():
        logger.info("Loading cached Qwen result: %s", cache_path)
        return load_segments_json(cache_path)

    model, processor = load_qwen_model(model_id)

    logger.info("Running Qwen3-VL on %s …", video_path.name)
    segments = _infer_single(video_path, model, processor)

    # Cache result
    save_segments_json(segments, cache_path)
    logger.info("Qwen3-VL returned %d segments for %s", len(segments), video_path.name)
    return segments
