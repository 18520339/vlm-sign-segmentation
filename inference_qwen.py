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
import re
import json
from pathlib import Path
from typing import List, Optional

from data_utils import Segment, save_segments_json, load_segments_json, get_video_duration
from config import (
    QWEN_MODEL_ID, QWEN_VIDEO_FPS, QWEN_MAX_PIXELS,
    QWEN_MAX_NEW_TOKENS, PHRASE_SEGMENTATION_PROMPT, OUTPUT_DIR,
)
_model, _processor = None, None # Module-level model cache (loaded once per session)


def load_qwen_model(model_id: str = QWEN_MODEL_ID): # Load Qwen3-VL model and processor, cached at module level
    global _model, _processor
    if _model is not None: return _model, _processor
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"[Qwen] Loading {model_id} …")
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto"
    )
    _processor = AutoProcessor.from_pretrained(model_id)
    print("[Qwen] Model loaded successfully.")
    return _model, _processor


def _parse_segments_json(text: str) -> List[Segment]: # Parse a JSON array of {start, end} from model output
    text = text.strip()

    # Strip think blocks if present (Qwen3-VL-Thinking variants)
    think_end = text.rfind("</think>")
    if think_end != -1: text = text[think_end + len("</think>"):].strip()
    
    try: # Try direct parse
        data = json.loads(text)
        if isinstance(data, list):
            return _validate_segments([Segment.from_dict(d) for d in data])
    except (json.JSONDecodeError, KeyError, TypeError): pass

    # Fallback: extract first JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list): return _validate_segments([Segment.from_dict(d) for d in data])
        except (json.JSONDecodeError, KeyError, TypeError): pass
        
    print(f"[Qwen] WARNING: Could not parse JSON segments from response:\n{text[:500]}")
    return []


def _validate_segments(segments: List[Segment]) -> List[Segment]: # Filter invalid segments and resolve overlaps
    valid = [s for s in segments if s.end_s > s.start_s >= 0]
    valid.sort(key=lambda s: s.start_s)

    cleaned = []
    for seg in valid:
        if cleaned and seg.start_s < cleaned[-1].end_s:
            seg = Segment(start_s=cleaned[-1].end_s, end_s=seg.end_s)
            if seg.end_s <= seg.start_s: continue
        cleaned.append(seg)
    return cleaned


def _sanitize_video_kwargs(video_kwargs: dict) -> dict: # fps comes as a list, processor wants scalar
    if not video_kwargs: return video_kwargs
    sanitized = dict(video_kwargs)
    
    if "fps" in sanitized: # fps: list[float] → float
        fps_val = sanitized["fps"]
        if isinstance(fps_val, (list, tuple)):
            if len(fps_val) >= 1:
                sanitized["fps"] = float(fps_val[0])
    return sanitized


def _infer_single(
    video_path: Path, model,
    processor, duration_s: float,
    fps: float = QWEN_VIDEO_FPS,
    max_pixels: int = QWEN_MAX_PIXELS,
    max_new_tokens: int = QWEN_MAX_NEW_TOKENS,
) -> List[Segment]:
    """Run inference on a single video (≤ ~3 min).

    Qwen3-VL requires video_metadata for its Text-Timestamp Alignment
    mechanism.  Without it, the model outputs garbled near-zero timestamps.

    Uses GREEDY decoding (do_sample=False) for deterministic, reproducible
    results on this structured JSON output task.
    """
    from qwen_vl_utils import process_vision_info

    # Format prompt with actual video duration (zero-cost calibration)
    prompt = PHRASE_SEGMENTATION_PROMPT.format(duration_s=duration_s)

    messages = [{"role": "user", "content": [{
        "type": "video", "video": str(video_path.resolve()),
        "max_pixels": max_pixels, "fps": float(fps),
    }, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ── Extract frames AND temporal metadata ──────────────────────────────
    # Qwen3-VL needs BOTH:
    #   - video_kwargs (fps info for temporal position IDs)
    #   - video_metadata (frame indices + timing for Text-Timestamp Alignment)
    #
    # Without return_video_metadata=True, the processor falls back to a
    # default FPS (often 24), causing "timestamp drift" where the model's
    # internal time doesn't match the actual video timeline.
    try:
        image_inputs, video_inputs_raw, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True)

        # Unpack (tensor, metadata) tuples
        if video_inputs_raw is not None and len(video_inputs_raw) > 0:
            video_inputs = [item[0] for item in video_inputs_raw]
            video_metadatas = [item[1] for item in video_inputs_raw]
            print(f"  Extracted {len(video_inputs)} video(s) with metadata (Qwen3-VL path)")
            if video_metadatas and isinstance(video_metadatas[0], dict):
                meta_summary = {k: v for k, v in video_metadatas[0].items() if k != 'video'}
                print(f"  Video metadata: {meta_summary}")
        else: video_inputs, video_metadatas = None, None

    except TypeError: # Fallback for older qwen-vl-utils without return_video_metadata
        print("  WARNING: qwen-vl-utils does not support return_video_metadata; "
              "temporal grounding may be degraded. Upgrade: pip install --upgrade qwen-vl-utils")
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        video_metadatas = None

    # Sanitize fps from list to scalar
    video_kwargs = _sanitize_video_kwargs(video_kwargs)
    print(f"  video_kwargs (sanitized): {video_kwargs}")

    # Build processor inputs
    processor_kwargs = dict(
        text=[text], images=image_inputs, videos=video_inputs, 
        padding=True, return_tensors="pt", **video_kwargs,
    )
    # Add video_metadata if available (Qwen3-VL Text-Timestamp Alignment)
    if video_metadatas is not None: processor_kwargs["video_metadata"] = video_metadatas
    inputs = processor(**processor_kwargs).to(model.device)
    print(f"  Input tokens: {inputs.input_ids.shape[-1]}")

    # GREEDY decoding: deterministic output for structured JSON tasks.
    # This fixes the "random results" issue where do_sample=True caused
    # wildly different segmentations across runs.
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Trim input tokens
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"  Raw Qwen output (first 500 chars): {output_text[:500]}")
    return _parse_segments_json(output_text)


def run_qwen_inference(
    video_path: Path, *,
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
        print(f"[Qwen] Loading cached result: {cache_path}")
        return load_segments_json(cache_path)

    model, processor = load_qwen_model(model_id)
    duration_s = get_video_duration(video_path)

    print(f"[Qwen] Running on {video_path.name} ({duration_s:.1f}s) …")
    segments = _infer_single(video_path, model, processor, duration_s=duration_s)

    # Cache result
    save_segments_json(segments, cache_path)
    print(f"[Qwen] Returned {len(segments)} segments for {video_path.name}")
    return segments