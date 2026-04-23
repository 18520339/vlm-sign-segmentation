"""Utilities for loading ground-truth subtitles, base-repo EAF outputs,
discovering video file groups, and converting between segment formats."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Segment dataclass ──────────────────────────────────────────────────────────

@dataclass
class Segment:
    """A time span in seconds, optionally carrying text."""
    start_s: float
    end_s: float
    text: Optional[str] = field(default=None, repr=False)

    def duration(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {"start": round(self.start_s, 3), "end": round(self.end_s, 3)}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(start_s=float(d["start"]), end_s=float(d["end"]),
                   text=d.get("text"))


# ── SRT parsing ────────────────────────────────────────────────────────────────

def parse_srt(path: Path) -> List[Segment]:
    """Parse an SRT subtitle file into a list of Segments (seconds)."""
    import srt

    with open(path, "r", encoding="utf-8-sig") as f:
        subs = list(srt.parse(f.read()))

    segments = []
    for sub in subs:
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        if end > start:
            segments.append(Segment(start_s=start, end_s=end, text=sub.content))
    return sorted(segments, key=lambda s: s.start_s)


# ── VTT parsing ────────────────────────────────────────────────────────────────

def parse_vtt(path: Path) -> List[Segment]:
    """Parse a WebVTT subtitle file into a list of Segments (seconds)."""
    import webvtt

    segments = []
    for caption in webvtt.read(str(path)):
        start = _vtt_ts_to_seconds(caption.start)
        end = _vtt_ts_to_seconds(caption.end)
        text = caption.text.strip()
        if end > start and text:
            segments.append(Segment(start_s=start, end_s=end, text=text))
    return sorted(segments, key=lambda s: s.start_s)


def _vtt_ts_to_seconds(ts: str) -> float:
    """Convert a VTT timestamp string (HH:MM:SS.mmm) to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


def parse_subtitles(path: Path) -> List[Segment]:
    """Parse subtitles from either SRT or VTT format (auto-detected)."""
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return parse_srt(path)
    elif suffix == ".vtt":
        return parse_vtt(path)
    else:
        raise ValueError(f"Unsupported subtitle format: {suffix} ({path})")


# ── EAF parsing ────────────────────────────────────────────────────────────────

def parse_eaf_sentences(path: Path, tier: str = "SENTENCE") -> List[Segment]:
    """Extract segments from an ELAN EAF file's SENTENCE (phrase) tier.

    Falls back to tier names containing 'sentence' (case-insensitive) if the
    exact tier name is not found.
    """
    import pympi

    eaf = pympi.Elan.Eaf(str(path))
    tier_names = eaf.get_tier_names()

    # Resolve tier name
    target = None
    if tier in tier_names:
        target = tier
    else:
        for t in tier_names:
            if "sentence" in t.lower():
                target = t
                break
    if target is None:
        raise ValueError(
            f"No SENTENCE tier found in {path}. Available tiers: {tier_names}"
        )

    segments = []
    for start_ms, end_ms, value in eaf.get_annotation_data_for_tier(target):
        segments.append(Segment(
            start_s=start_ms / 1000.0,
            end_s=end_ms / 1000.0,
            text=value if value else None,
        ))
    return sorted(segments, key=lambda s: s.start_s)


# ── Video metadata ─────────────────────────────────────────────────────────────

def get_video_duration(path: Path) -> float:
    """Return video duration in seconds using ffprobe (fast) or OpenCV (fallback)."""
    # Try ffprobe first (no decoding, instant)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass

    # Fallback: OpenCV
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0 and frame_count > 0:
            return frame_count / fps
    finally:
        cap.release()

    raise RuntimeError(f"Could not determine duration of {path}")


def get_video_fps(path: Path) -> float:
    """Return video FPS using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0
    finally:
        cap.release()


# ── Segment ↔ binary vector conversion ─────────────────────────────────────────

def segments_to_binary(segments: List[Segment], duration_s: float,
                       resolution_s: float = 0.04) -> np.ndarray:
    """Convert a list of segments to a binary vector at *resolution_s* steps."""
    n_bins = int(np.ceil(duration_s / resolution_s))
    vec = np.zeros(n_bins, dtype=np.int8)
    for seg in segments:
        i_start = max(0, int(seg.start_s / resolution_s))
        i_end = min(n_bins, int(np.ceil(seg.end_s / resolution_s)))
        vec[i_start:i_end] = 1
    return vec


# ── File-group discovery ───────────────────────────────────────────────────────

# Patterns for VTT files that belong to a video stem, ordered by preference.
# Given a stem "my_video", we search for files like:
#   my_video.vtt, my_video.en.vtt, my_video.en-en.vtt,
#   my_video.en-GB.vtt, my_video.en-en-GB.vtt, my_video.en-orig.vtt, etc.
_VTT_PATTERNS = [
    "{stem}.vtt",
    "{stem}.en.vtt",
    "{stem}.en-orig.vtt",
    "{stem}.en-en.vtt",
    "{stem}.en-GB.vtt",
    "{stem}.en-en-GB.vtt",
]


def _find_subtitle(data_dir: Path, stem: str) -> Optional[Path]:
    """Find the best matching subtitle file (SRT or VTT) for a video stem.

    Priority: SRT > VTT (exact stem) > VTT (with language tags).
    If multiple VTTs exist, picks the first match from _VTT_PATTERNS.
    Falls back to a glob search for any remaining {stem}.*.vtt pattern.
    """
    # 1. Direct SRT
    srt_path = data_dir / f"{stem}.srt"
    if srt_path.exists():
        return srt_path

    # 2. VTT — known patterns
    for pattern in _VTT_PATTERNS:
        vtt_path = data_dir / pattern.format(stem=stem)
        if vtt_path.exists():
            return vtt_path

    # 3. VTT — glob fallback for any {stem}.*.vtt
    vtt_matches = sorted(data_dir.glob(f"{stem}.*.vtt"))
    if vtt_matches:
        return vtt_matches[0]

    # 4. Plain VTT without any middle name
    plain_vtt = data_dir / f"{stem}.vtt"
    if plain_vtt.exists():
        return plain_vtt

    return None


def discover_videos(data_dir: Path) -> List[dict]:
    """Discover groups of related files (video + subtitle + eaf).

    Returns a list of dicts with keys:
        name  : str        — stem of the video file
        video : Path       — path to .mp4
        subs  : Path|None  — path to .srt or .vtt (required for evaluation)
        eaf   : Path|None  — path to .eaf  (base-repo output, optional)
    Only includes groups where *both* video and subtitle file exist.
    """
    data_dir = Path(data_dir)
    mp4_files = sorted(data_dir.glob("*.mp4"))

    groups = []
    for mp4 in mp4_files:
        stem = mp4.stem
        subs = _find_subtitle(data_dir, stem)
        eaf = data_dir / f"{stem}.eaf"

        if subs is None:
            continue  # skip videos without GT subtitles

        groups.append({
            "name": stem,
            "video": mp4,
            "subs": subs,
            "eaf": eaf if eaf.exists() else None,
        })
    return groups


# ── JSON cache helpers ─────────────────────────────────────────────────────────

def save_segments_json(segments: List[Segment], path: Path) -> None:
    """Persist segments as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2)


def load_segments_json(path: Path) -> List[Segment]:
    """Load segments from a cached JSON file."""
    with open(path) as f:
        return [Segment.from_dict(d) for d in json.load(f)]
