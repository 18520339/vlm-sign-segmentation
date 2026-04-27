"""Visualisation functions for comparing phrase segmentation methods.

All plots are saved as PNG to the output directory.  Each function is
self-contained and can be called independently.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from typing import Dict, List
from data_utils import Segment

METHOD_COLORS = {
    "GT":     "#2ecc71",   # green
    "Base":   "#3498db",   # blue
    "Gemini": "#e67e22",   # orange
    "Qwen":   "#9b59b6",   # purple
}

def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

_setup_style()

# ── 1. Timeline comparison (per video) ─────────────────────────────────────────

def plot_timeline(
    video_name: str, duration_s: float,
    method_segments: Dict[str, List[Segment]], output_dir: Path,
) -> Path:
    # Horizontal bar chart showing segments for each method on a shared time axis.
    # *method_segments* maps method name (e.g. "GT", "Base", "Gemini", "Qwen") to its list of Segments.
    methods = list(method_segments.keys())
    n = len(methods)
    fig, ax = plt.subplots(figsize=(14, max(2.5, 0.8 * n + 1.2)))
    bar_height = 0.6
    
    for i, method in enumerate(methods):
        y = n - 1 - i  # top to bottom
        color = METHOD_COLORS.get(method, "#95a5a6")
        segments = method_segments[method]

        for seg in segments:
            ax.barh(y, seg.end_s - seg.start_s, left=seg.start_s, height=bar_height, 
                    color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(methods)), fontweight="bold")
    ax.set_xlim(0, duration_s)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Phrase Segmentation Timeline — {video_name}", fontweight="bold")

    # Legend
    handles = [mpatches.Patch(color=METHOD_COLORS.get(m, "#95a5a6"), label=m) for m in methods]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    path = output_dir / f"timeline_{video_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── 2. Metrics summary bar chart ──────────────────────────────────────────────

def plot_metrics_summary(all_metrics: Dict[str, Dict[str, List[float]]], output_dir: Path) -> Path: 
    # Grouped bar chart of key metrics across methods. *all_metrics*: {method_name: {metric_name: [per-video values]}}
    display_metrics = [
        ("temporal_iou", "Temporal IoU"),
        ("seg_f1@0.3",   "F1 @0.3"),
        ("seg_f1@0.5",   "F1 @0.5"),
        ("seg_f1@0.7",   "F1 @0.7"),
    ]

    methods = list(all_metrics.keys())
    n_metrics = len(display_metrics)
    n_methods = len(methods)
    x = np.arange(n_metrics)
    width = 0.7 / n_methods
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        means, stds = [], []
        for key, _ in display_metrics:
            values = all_metrics[method].get(key, [0])
            means.append(np.mean(values))
            stds.append(np.std(values) if len(values) > 1 else 0)

        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds if any(s > 0 for s in stds) else None,
                      label=method, color=METHOD_COLORS.get(method, "#95a5a6"),
                      alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)

        # Value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in display_metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Phrase Segmentation — Metrics Comparison", fontweight="bold")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    path = output_dir / "metrics_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── 3. IoU distribution box plot ──────────────────────────────────────────────

def plot_iou_distribution(all_metrics: Dict[str, Dict[str, List[float]]], output_dir: Path) -> Path:
    # Box plot of per-video temporal IoU for each method
    methods = list(all_metrics.keys())
    data = [all_metrics[m].get("temporal_iou", [0]) for m in methods]
    colors = [METHOD_COLORS.get(m, "#95a5a6") for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=methods, patch_artist=True, widths=0.5)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = np.random.normal(0, 0.04, size=len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, color=color, edgecolors="white", s=50, zorder=3, alpha=0.9)

    ax.set_ylabel("Temporal IoU")
    ax.set_title("IoU Distribution Across Videos", fontweight="bold")
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    path = output_dir / "iou_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── 4. Segmentation tendency scatter ──────────────────────────────────────────

def plot_segmentation_tendency(all_metrics: Dict[str, Dict[str, List[float]]], output_dir: Path) -> Path:
    # Scatter: predicted vs GT segment count per method.
    # Points on the diagonal = perfect count.  Above = over-segmentation.
    methods = list(all_metrics.keys())
    fig, ax = plt.subplots(figsize=(6, 6))
    all_counts = []
    
    for method in methods:
        pred_counts = all_metrics[method].get("pred_count", [])
        gt_counts = all_metrics[method].get("gt_count", [])
        all_counts.extend(gt_counts + pred_counts)
        color = METHOD_COLORS.get(method, "#95a5a6")
        ax.scatter(gt_counts, pred_counts, label=method, color=color,
                   s=80, alpha=0.85, edgecolors="white", linewidth=1)

    if all_counts: max_val = max(all_counts) * 1.15
    else: max_val = 10
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Perfect")

    ax.set_xlabel("GT Segment Count")
    ax.set_ylabel("Predicted Segment Count")
    ax.set_title("Over/Under-Segmentation Tendency", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    plt.tight_layout()
    path = output_dir / "segmentation_tendency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── 5. Boundary error histogram ───────────────────────────────────────────────

def plot_boundary_errors(boundary_errors_by_method: Dict[str, List[float]], output_dir: Path) -> Path:
    # Histogram of boundary timing errors for each method.
    # Centered at 0 = perfect alignment.  Positive = prediction is late.
    methods = list(boundary_errors_by_method.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        errors = boundary_errors_by_method[method]
        if not errors:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(method, fontweight="bold")
            continue

        color = METHOD_COLORS.get(method, "#95a5a6")
        ax.hist(errors, bins=30, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", alpha=0.4)

        mean_err = np.mean(np.abs(errors))
        ax.axvline(np.mean(errors), color="red", linestyle="-", alpha=0.6,
                   label=f"Mean = {np.mean(errors):+.2f}s")

        ax.set_xlabel("Error (seconds)")
        ax.set_title(f"{method}  (MAE={mean_err:.2f}s)", fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)

    axes[0].set_ylabel("Count")
    fig.suptitle("Boundary Timing Error Distribution", fontweight="bold", y=1.02)
    plt.tight_layout()
    path = output_dir / "boundary_errors.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(video_results: List[dict], output_dir: Path) -> List[Path]:
    """Generate all visualisations from evaluation results.

    *video_results*: list of dicts, each with keys:
        name       : str
        duration_s : float
        segments   : {method_name: List[Segment]}
        metrics    : {method_name: dict of metric values}
        boundary_errors : {method_name: List[float]}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: List[Path] = []

    # Determine which methods are present (excluding GT)
    pred_methods = set()
    for vr in video_results: pred_methods.update(k for k in vr["segments"] if k != "GT")
    pred_methods = sorted(pred_methods)

    # 1. Per-video timelines
    for vr in video_results:
        p = plot_timeline(vr["name"], vr["duration_s"], vr["segments"], output_dir)
        plots.append(p)

    # Aggregate per-method metric lists
    agg_metrics: Dict[str, Dict[str, List[float]]] = {m: {} for m in pred_methods}
    agg_boundary_errors: Dict[str, List[float]] = {m: [] for m in pred_methods}

    for vr in video_results:
        for method in pred_methods:
            if method not in vr["metrics"]: continue
            for key, val in vr["metrics"][method].items(): agg_metrics[method].setdefault(key, []).append(val)
            if method in vr.get("boundary_errors", {}): agg_boundary_errors[method].extend(vr["boundary_errors"][method])

    # 2–5. Aggregate plots
    if pred_methods:
        plots.append(plot_metrics_summary(agg_metrics, output_dir))
        plots.append(plot_iou_distribution(agg_metrics, output_dir))
        plots.append(plot_segmentation_tendency(agg_metrics, output_dir))
        plots.append(plot_boundary_errors(agg_boundary_errors, output_dir))
    return plots


# ── 6. Overlay video rendering ─────────────────────────────────────────────────

def _get_active_text(t: float, segments: List[Segment]) -> str: # Return the text of the GT segment that is active at time *t*
    for s in segments:
        if s.start_s <= t <= s.end_s: return s.text or ""
        if s.start_s > t: break
    return ""


def _draw_text(frame, text: str, pos: tuple, font_scale: float = 0.6, color=(255, 255, 255), thickness: int = 1, bg_color=None):
    # Draw text with optional background rectangle using OpenCV
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    if bg_color is not None: cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2), bg_color, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _hex_to_bgr(hex_color: str) -> tuple: # Convert '#RRGGBB' to (B, G, R) for OpenCV
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _fmt_time(t: float) -> str: # Format seconds as M:SS.s for timeline tick labels
    m = int(t) // 60
    s = t - m * 60
    return f"{m}:{s:04.1f}"


def render_overlay_video(
    video_path: Path, method_segments: Dict[str, List[Segment]],
    output_dir: Path, output_fps: float = 30.0,
) -> Path: # Render an overlay comparison video with a clear segmentation timeline
    """ Layout (top → bottom):
    ┌─────────────────────────────┐
    │       Original frame        │
    ├─────────────────────────────┤
    │  GT subtitle text           │
    ├─────────────────────────────┤
    │  GT     ████  ████  ████    │  ← 20px per method row
    │  Qwen   ██████   ████      │    with 1px borders on each
    │  Base   ███  ██  ████      │    segment block so gaps
    │  ...                        │    are clearly visible
    ├── 0:00  0:15  0:30  0:45 ──┤  ← time axis with ticks
    │  0:12.34 / 1:04.50         │  ← current time readout
    └─────────────────────────────┘
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = frame_count / src_fps if src_fps > 0 else 0
    methods = list(method_segments.keys())
    n_methods = len(methods)

    # ── Layout dimensions ──────────────────────────────────────────────────
    pad = 14                       # horizontal padding
    label_w = 70                   # width reserved for method labels
    subtitle_h = 48                # subtitle text area
    tl_row_h = 20                  # height per timeline method row
    tl_row_gap = 4                 # gap between timeline rows
    tick_h = 22                    # time axis tick area
    time_readout_h = 22            # current time display
    top_gap = 8                    # gap between video and panel

    tl_total_h = n_methods * tl_row_h + (n_methods - 1) * tl_row_gap
    panel_h = (top_gap + subtitle_h + 6 + tl_total_h + 4 + tick_h + time_readout_h + 6)
    out_w, out_h = src_w, src_h + panel_h

    # Timeline pixel coordinates
    tl_x0 = pad + label_w          # left edge of timeline bars
    tl_x1 = out_w - pad            # right edge
    tl_w = tl_x1 - tl_x0
    tl_top = src_h + top_gap + subtitle_h + 6  # y of first timeline row

    # Pre-compute time axis ticks (every 5s, 10s, or 15s depending on duration)
    if duration_s <= 30: tick_interval = 5
    elif duration_s <= 90: tick_interval = 10
    elif duration_s <= 180: tick_interval = 15
    else: tick_interval = 30
    tick_times = np.arange(0, duration_s + 0.1, tick_interval)

    # Output path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"overlay_{video_path.stem}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video writer for {out_path}")

    # Pre-compute segment pixel ranges (avoids re-computing every frame)
    seg_px: Dict[str, list] = {}
    for method in methods:
        segs = method_segments[method]
        px_list = []
        for seg in segs:
            if duration_s <= 0: continue
            x1 = tl_x0 + int(seg.start_s / duration_s * tl_w)
            x2 = tl_x0 + int(seg.end_s / duration_s * tl_w)
            x2 = max(x2, x1 + 2)  # at least 2px wide so it's visible
            px_list.append((x1, x2))
        seg_px[method] = px_list

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t = frame_idx / src_fps

        # ── Compose output frame ──────────────────────────────────────────
        out_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        out_frame[:src_h, :, :] = frame
        out_frame[src_h:, :, :] = 25  # dark panel background
        panel_top = src_h

        # ── 1. GT subtitle text ───────────────────────────────────────────
        sub_y = panel_top + top_gap
        gt_segs = method_segments.get("GT", [])
        subtitle_text = _get_active_text(t, gt_segs)
        if subtitle_text:
            max_chars = max(50, out_w // 10)
            lines = [subtitle_text[i:i + max_chars] for i in range(0, len(subtitle_text), max_chars)]
            for i, line in enumerate(lines[:2]):
                _draw_text(out_frame, line, (pad, sub_y + 18 + i * 22),
                           font_scale=0.55, color=(255, 255, 255), thickness=1, bg_color=(50, 50, 50))
        else: _draw_text(out_frame, "(no subtitle)", (pad, sub_y + 18),
                         font_scale=0.45, color=(90, 90, 90), thickness=1)

        # ── 2. Timeline rows ──────────────────────────────────────────────
        for i, method in enumerate(methods):
            row_y = tl_top + i * (tl_row_h + tl_row_gap)
            color_hex = METHOD_COLORS.get(method, "#95a5a6")
            color_bgr = _hex_to_bgr(color_hex)
            dim_bgr = tuple(max(0, c // 6) for c in color_bgr)
            border_bgr = tuple(max(0, c // 3) for c in color_bgr)

            # Row background (very dark tint of the method color)
            cv2.rectangle(out_frame, (tl_x0, row_y), (tl_x1, row_y + tl_row_h - 1), dim_bgr, cv2.FILLED)

            # Method label (left of the bar)
            _draw_text(out_frame, method, (pad, row_y + tl_row_h - 5),
                       font_scale=0.45, color=color_bgr, thickness=1)

            # Segment blocks with 1px dark border (makes gaps clearly visible)
            for (x1, x2) in seg_px[method]:
                # Filled block
                cv2.rectangle(out_frame, (x1, row_y + 1), (x2, row_y + tl_row_h - 2), color_bgr, cv2.FILLED)
                # 1px dark border around each block
                cv2.rectangle(out_frame, (x1, row_y + 1), (x2, row_y + tl_row_h - 2), border_bgr, 1)

        # ── 3. Time axis ticks ────────────────────────────────────────────
        tick_y = tl_top + tl_total_h + 2
        cv2.line(out_frame, (tl_x0, tick_y), (tl_x1, tick_y), (80, 80, 80), 1) # Thin horizontal axis line
        for tt in tick_times:
            tx = tl_x0 + int(tt / duration_s * tl_w) if duration_s > 0 else tl_x0
            cv2.line(out_frame, (tx, tick_y), (tx, tick_y + 5), (120, 120, 120), 1) # Tick mark
            _draw_text(out_frame, _fmt_time(tt), (tx - 15, tick_y + 17),
                       font_scale=0.33, color=(150, 150, 150), thickness=1) # Tick label

        # ── 4. Playhead cursor (bright white line across all rows) ────────
        if duration_s > 0:
            cursor_x = tl_x0 + int(t / duration_s * tl_w)
            cv2.line(out_frame, (cursor_x, tl_top - 2), (cursor_x, tick_y), (255, 255, 255), 2)
            # Small triangle at top of playhead
            pts = np.array([[cursor_x - 4, tl_top - 6], [cursor_x + 4, tl_top - 6], [cursor_x, tl_top - 1]], dtype=np.int32)
            cv2.fillPoly(out_frame, [pts], (255, 255, 255))

        # ── 5. Time readout (bottom-right) ────────────────────────────────
        t_mins, t_secs = int(t) // 60, t % 60
        d_mins, d_secs = int(duration_s) // 60, duration_s % 60
        time_str = f"{t_mins}:{t_secs:05.2f} / {d_mins}:{d_secs:05.2f}"
        _draw_text(out_frame, time_str, (out_w - 180, tick_y + tick_h + 12), font_scale=0.45, color=(180, 180, 180), thickness=1)
        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return out_path