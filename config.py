"""Central configuration for VLM sign language phrase segmentation."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./results"))

# ── Gemini API ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-pro-latest"

# ── Qwen3-VL (Colab G4 — RTX PRO 6000 Blackwell, 96 GB VRAM) ─────────────────
# Default model.  To try a different variant, override via --qwen_model_id.
# Available Qwen3-VL models on HuggingFace (as of Apr 2026):
#   Dense:   2B, 4B, 8B, 32B  (Instruct or Thinking editions)
#   MoE:     30B-A3B (3B active), 235B-A22B (22B active)
#   FP8:     Qwen/Qwen3-VL-32B-Instruct-FP8  (~18 GB, fits easily)
#            Qwen/Qwen3-VL-235B-A22B-Instruct-FP8  (needs multi-GPU)
#   MoE fit: Qwen/Qwen3-VL-30B-A3B-Instruct  (~30 GB bf16, fits alongside 32B)
QWEN_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-VL-32B-Instruct")

# Frames sampled per second from the video.  Using the full 30 fps is
# impractical: a 2-min video at 30 fps = 3 600 frames, each producing
# hundreds of visual tokens — far exceeding the 256K context window.
# 4 fps gives 480 frames for 2 min, which is a good balance.
QWEN_VIDEO_FPS = 30.0
QWEN_MAX_PIXELS = 360 * 640   # per-frame resolution budget (~360p)
QWEN_MAX_NEW_TOKENS = 4096    # max tokens for the generated JSON response

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_RESOLUTION_S = 0.04      # 25-fps equivalent for IoU discretisation

# Segment F1 uses pairwise IoU matching (standard in temporal action detection).
# Report at multiple thresholds like ActivityNet: 0.3, 0.5, 0.7.
SEGMENT_IOU_THRESHOLDS = [0.3, 0.5, 0.7]

# ── VLM Prompt ─────────────────────────────────────────────────────────────────
# Shared across Gemini and Qwen so that the only variable is the model itself.
PHRASE_SEGMENTATION_PROMPT = """\
You are an expert in sign language video analysis, specifically Australian \
Sign Language (Auslan). Your task is to perform precise phrase-level temporal \
segmentation on this video.

CONTEXT: This is a YouTube video containing Auslan signing. The video may \
include non-signing segments such as title cards, transitions, captions, \
the signer adjusting the camera, or brief pauses between topics. Focus ONLY \
on the periods where the person is actively signing.

TASK: Identify every distinct phrase or sentence being signed. A phrase is a \
complete, coherent unit of signed communication — typically corresponding to \
one subtitle line or one sentence worth of meaning.

HOW TO DETECT PHRASE BOUNDARIES:
- Pauses: Brief holds or rest periods between phrases (hands may drop or \
  return to a neutral position)
- Rhythm changes: Each phrase has a natural rhythm; transitions between \
  phrases often have a brief deceleration then acceleration
- Non-manual markers: Head nods, slight body shifts, brow movements, or \
  changes in facial expression that signal sentence completion
- Topic shifts: Changes in the spatial area where signs are produced, or \
  shifts in eye gaze direction

RULES:
1. Provide start and end timestamps in SECONDS with decimal precision \
   (e.g., 3.2)
2. Include ALL signed phrases — do not skip any
3. EXCLUDE non-signing periods (idle time, title screens, camera adjustments)
4. Segments must not overlap and must be in chronological order
5. If two phrases have no visible gap between them (continuous signing), \
   place the boundary at the point where the transition occurs
6. Very short hesitations within a phrase should NOT create a new boundary

Return ONLY a JSON array with no additional text:
[{"start": <seconds>, "end": <seconds>}, ...]
"""
