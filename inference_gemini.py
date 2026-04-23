"""Gemini API inference for sign language phrase segmentation.

Uploads a video to the Gemini File API, prompts the model to produce
phrase-level temporal segments, and returns parsed Segment objects.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

from data_utils import Segment, save_segments_json, load_segments_json
from config import (
    GEMINI_API_KEY, GEMINI_MODEL, PHRASE_SEGMENTATION_PROMPT, OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


# ── JSON response parsing ──────────────────────────────────────────────────────

def _parse_segments_json(text: str) -> List[Segment]:
    """Parse a JSON array of {start, end} objects from model output.

    Tries direct JSON parse first, then falls back to regex extraction.
    """
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _validate_segments([Segment.from_dict(d) for d in data])
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fallback: extract the first JSON array from the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return _validate_segments([Segment.from_dict(d) for d in data])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    logger.warning("Could not parse JSON segments from model response:\n%s", text)
    return []


def _validate_segments(segments: List[Segment]) -> List[Segment]:
    """Filter out invalid segments and ensure chronological order."""
    valid = [s for s in segments if s.end_s > s.start_s >= 0]
    valid.sort(key=lambda s: s.start_s)

    # Remove overlaps: if segment i overlaps with i-1, clip its start
    cleaned = []
    for seg in valid:
        if cleaned and seg.start_s < cleaned[-1].end_s:
            seg = Segment(start_s=cleaned[-1].end_s, end_s=seg.end_s)
            if seg.end_s <= seg.start_s:
                continue
        cleaned.append(seg)
    return cleaned


# ── Gemini inference ───────────────────────────────────────────────────────────

def run_gemini_inference(
    video_path: Path,
    *,
    api_key: Optional[str] = None,
    model: str = GEMINI_MODEL,
    cache_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> List[Segment]:
    """Upload *video_path* to Gemini, prompt for phrase segmentation.

    Results are cached as JSON in *cache_dir* (default: OUTPUT_DIR).
    On subsequent calls, the cached result is returned immediately.
    """
    cache_dir = cache_dir or OUTPUT_DIR
    cache_path = cache_dir / f"{video_path.stem}_gemini.json"

    if cache_path.exists():
        logger.info("Loading cached Gemini result: %s", cache_path)
        return load_segments_json(cache_path)

    # Lazy import so the module can be imported even without the SDK
    from google import genai

    api_key = api_key or GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "Gemini API key not provided. Set GEMINI_API_KEY environment "
            "variable or pass api_key= argument."
        )

    client = genai.Client(api_key=api_key)

    # Upload video
    logger.info("Uploading %s to Gemini File API …", video_path.name)
    video_file = client.files.upload(file=str(video_path))

    # Wait for processing
    while video_file.state.name == "PROCESSING":
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise RuntimeError(
            f"Gemini file processing failed for {video_path.name}: "
            f"{getattr(video_file, 'error', 'unknown error')}"
        )

    logger.info("File ready. Sending prompt to %s …", model)

    # Generate with retries
    segments: List[Segment] = []
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[video_file, PHRASE_SEGMENTATION_PROMPT],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                },
            )
            segments = _parse_segments_json(response.text)
            if segments:
                break
            logger.warning("Attempt %d: empty segment list, retrying …", attempt)
        except Exception as e:
            last_error = e
            logger.warning("Attempt %d failed: %s", attempt, e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    if not segments and last_error:
        raise RuntimeError(f"Gemini inference failed after {max_retries} attempts") from last_error

    # Clean up remote file
    try:
        client.files.delete(name=video_file.name)
    except Exception:
        pass

    # Cache result
    save_segments_json(segments, cache_path)
    logger.info("Gemini returned %d segments for %s", len(segments), video_path.name)
    return segments
