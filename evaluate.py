"""Main evaluation script for VLM sign language phrase segmentation.

Discovers videos, loads GT (SRT/VTT) and base-repo (EAF) results, runs Gemini
and/or Qwen3-VL inference, computes metrics, generates visualisations + overlay
videos, and prints a summary table.

Usage:
    python evaluate.py --data_dir ./data --output_dir ./results
    python evaluate.py --methods gemini          # Gemini only
    python evaluate.py --skip_inference          # re-evaluate from cache
    python evaluate.py --no_video                # skip overlay video rendering
    python evaluate.py --qwen_model_id Qwen/Qwen3-VL-72B-Instruct-AWQ
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import (
    DATA_DIR, OUTPUT_DIR, GEMINI_API_KEY, QWEN_MODEL_ID,
    EVAL_RESOLUTION_S, SEGMENT_IOU_THRESHOLDS,
)
from data_utils import (
    Segment, discover_videos, parse_subtitles, parse_eaf_sentences,
    get_video_duration, get_video_fps, load_segments_json,
)
from metrics import compute_all_metrics, boundary_errors as compute_boundary_errors
from visualize import generate_all_plots, render_overlay_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate VLM phrase segmentation against GT subtitles",
    )
    p.add_argument("--data_dir", type=Path, default=DATA_DIR,
                   help="Directory containing *.mp4, *.srt/*.vtt, *.eaf files")
    p.add_argument("--output_dir", type=Path, default=OUTPUT_DIR,
                   help="Directory for cached results and plots")
    p.add_argument("--methods", type=str, default="gemini,qwen",
                   help="Comma-separated methods to run: gemini, qwen (default: both)")
    p.add_argument("--gemini_api_key", type=str, default=None,
                   help="Gemini API key (overrides GEMINI_API_KEY env var)")
    p.add_argument("--qwen_model_id", type=str, default=QWEN_MODEL_ID,
                   help="HuggingFace model ID for Qwen (default: %(default)s)")
    p.add_argument("--skip_inference", action="store_true",
                   help="Skip inference; only evaluate from cached JSONs")
    p.add_argument("--videos", type=str, default=None,
                   help="Comma-separated video stems to process (default: all)")
    p.add_argument("--no_video", action="store_true",
                   help="Skip overlay video rendering")
    return p.parse_args()


# ── Inference dispatch ─────────────────────────────────────────────────────────

def run_method(
    method: str,
    video_path: Path,
    cache_dir: Path,
    api_key: Optional[str] = None,
    qwen_model_id: str = QWEN_MODEL_ID,
    skip_inference: bool = False,
) -> List[Segment]:
    """Run a single method on a single video, with caching."""
    cache_path = cache_dir / f"{video_path.stem}_{method}.json"

    if cache_path.exists():
        return load_segments_json(cache_path)

    if skip_inference:
        logger.warning("No cached result for %s/%s and --skip_inference is set",
                       method, video_path.stem)
        return []

    if method == "gemini":
        from inference_gemini import run_gemini_inference
        return run_gemini_inference(video_path, api_key=api_key, cache_dir=cache_dir)
    elif method == "qwen":
        from inference_qwen import run_qwen_inference
        return run_qwen_inference(
            video_path, model_id=qwen_model_id, cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Pretty-print table ────────────────────────────────────────────────────────

def _fmt(val, fmt=".3f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "  —  "
    return f"{val:{fmt}}"


def print_summary_table(
    aggregated: Dict[str, Dict[str, float]],
    n_videos: int,
    iou_thresholds: List[float],
):
    """Print a formatted summary table to stdout."""
    methods = list(aggregated.keys())

    # Header row
    width = 86
    header = f"\n{'═' * width}\n"
    header += f"  Phrase Segmentation — {n_videos} video(s) (Auslan)\n"
    header += f"{'═' * width}\n"

    # Column headers
    cols = f"  {'Method':<10} {'tIoU':>7}"
    for thr in iou_thresholds:
        cols += f" {'F1@'+f'{thr:.1f}':>8}"
    cols += f" {'MAE(s)':>8} {'CntRat':>8} {'#Pred':>6} {'#GT':>6}"
    header += cols + f"\n{'─' * width}"
    print(header)

    for m in methods:
        d = aggregated[m]
        row = f"  {m:<10} {_fmt(d.get('temporal_iou')):>7}"
        for thr in iou_thresholds:
            row += f" {_fmt(d.get(f'seg_f1@{thr:.1f}')):>8}"
        row += f" {_fmt(d.get('boundary_mean_abs_error_s'), '.2f'):>8}"
        row += f" {_fmt(d.get('count_ratio'), '.2f'):>8}"
        row += f" {_fmt(d.get('pred_count'), '.0f'):>6}"
        row += f" {_fmt(d.get('gt_count'), '.0f'):>6}"
        print(row)

    print(f"{'═' * width}\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve methods
    enabled_methods = [m.strip().lower() for m in args.methods.split(",")]
    valid = {"gemini", "qwen"}
    for m in enabled_methods:
        if m not in valid:
            print(f"Error: unknown method '{m}'. Choose from: {valid}")
            sys.exit(1)

    # Discover videos
    groups = discover_videos(args.data_dir)
    if args.videos:
        allowed = set(v.strip() for v in args.videos.split(","))
        groups = [g for g in groups if g["name"] in allowed]

    if not groups:
        print(f"No videos found in {args.data_dir} (need *.mp4 + matching *.srt or *.vtt)")
        sys.exit(1)

    logger.info("Found %d video(s): %s", len(groups),
                ", ".join(g["name"] for g in groups))

    api_key = args.gemini_api_key or GEMINI_API_KEY

    # Process each video
    video_results: List[dict] = []

    for group in groups:
        vname = group["name"]
        logger.info("── Processing: %s ──", vname)

        # Load GT
        gt_segments = parse_subtitles(group["subs"])
        duration_s = get_video_duration(group["video"])
        logger.info("  GT: %d phrases (from %s), duration: %.1fs",
                    len(gt_segments), group["subs"].suffix, duration_s)

        # Load base repo (EAF)
        base_segments: List[Segment] = []
        if group["eaf"]:
            try:
                base_segments = parse_eaf_sentences(group["eaf"])
                logger.info("  Base: %d phrases from EAF", len(base_segments))
            except Exception as e:
                logger.warning("  Could not parse EAF: %s", e)

        # Collect all method segments
        all_segments: Dict[str, List[Segment]] = {"GT": gt_segments}
        if base_segments:
            all_segments["Base"] = base_segments

        for method in enabled_methods:
            try:
                segs = run_method(
                    method, group["video"], args.output_dir,
                    api_key=api_key,
                    qwen_model_id=args.qwen_model_id,
                    skip_inference=args.skip_inference,
                )
                method_label = "Gemini" if method == "gemini" else "Qwen"
                all_segments[method_label] = segs
                logger.info("  %s: %d phrases", method_label, len(segs))
            except Exception as e:
                logger.error("  %s failed: %s", method, e)

        # Compute metrics for each prediction method
        method_metrics: Dict[str, dict] = {}
        method_boundary_errors: Dict[str, List[float]] = {}

        for method_label, segs in all_segments.items():
            if method_label == "GT":
                continue
            m = compute_all_metrics(
                segs, gt_segments, duration_s,
                resolution_s=EVAL_RESOLUTION_S,
                iou_thresholds=SEGMENT_IOU_THRESHOLDS,
            )
            method_metrics[method_label] = m

            be = compute_boundary_errors(segs, gt_segments)
            method_boundary_errors[method_label] = be["errors"]

        video_results.append({
            "name": vname,
            "video_path": group["video"],
            "duration_s": duration_s,
            "segments": all_segments,
            "metrics": method_metrics,
            "boundary_errors": method_boundary_errors,
        })

    # ── Aggregate metrics ──────────────────────────────────────────────────
    all_methods = set()
    for vr in video_results:
        all_methods.update(vr["metrics"].keys())

    aggregated: Dict[str, Dict[str, float]] = {}
    for method in sorted(all_methods):
        per_video_metrics = [
            vr["metrics"][method]
            for vr in video_results
            if method in vr["metrics"]
        ]
        if not per_video_metrics:
            continue

        avg: Dict[str, float] = {}
        for key in per_video_metrics[0]:
            vals = [m[key] for m in per_video_metrics
                    if key in m and not np.isnan(m.get(key, 0))]
            avg[key] = float(np.mean(vals)) if vals else float("nan")
        aggregated[method] = avg

    # ── Print summary ──────────────────────────────────────────────────────
    print_summary_table(aggregated, len(video_results), SEGMENT_IOU_THRESHOLDS)

    # ── Save detailed results ──────────────────────────────────────────────
    summary_path = args.output_dir / "summary.json"
    serialisable = {}
    for vr in video_results:
        vdata = {
            "duration_s": vr["duration_s"],
            "segments": {
                m: [s.to_dict() for s in segs]
                for m, segs in vr["segments"].items()
            },
            "metrics": vr["metrics"],
        }
        serialisable[vr["name"]] = vdata
    serialisable["_aggregated"] = aggregated

    with open(summary_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    logger.info("Detailed results saved to %s", summary_path)

    # ── Visualisations ─────────────────────────────────────────────────────
    logger.info("Generating plot visualisations …")
    plots = generate_all_plots(video_results, args.output_dir)
    for p in plots:
        logger.info("  Saved: %s", p)

    # ── Overlay videos ─────────────────────────────────────────────────────
    if not args.no_video:
        logger.info("Rendering overlay videos …")
        for vr in video_results:
            try:
                out_path = render_overlay_video(
                    video_path=vr["video_path"],
                    method_segments=vr["segments"],
                    output_dir=args.output_dir,
                )
                logger.info("  Saved: %s", out_path)
            except Exception as e:
                logger.error("  Overlay video failed for %s: %s", vr["name"], e)

    logger.info("Done.")


if __name__ == "__main__":
    main()
