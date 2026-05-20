"""Evaluation metrics for temporal phrase segmentation.

All metrics operate in the **time domain** (seconds) so they are directly
comparable across methods that produce segments in different ways.

Metrics:
    temporal_iou       — Binary-vector IoU (coverage metric, same as base repo)
    segment_f1         — Pairwise IoU matching at configurable thresholds
                         (standard in temporal action detection: ActivityNet)
    boundary_errors    — Per-boundary timing error distribution
    count_ratio        — Predicted / GT segment count
"""
import numpy as np
from typing import Dict, List
from data_utils import Segment, segments_to_binary


def temporal_iou(pred: List[Segment], gt: List[Segment], duration_s: float, resolution_s: float = 0.04) -> float:
    """Frame-vector IoU at *resolution_s* time steps.

    This is the primary coverage metric — analogous to the base repo's Phrase
    IoU.  Both segment lists are discretised into binary "signing / not-signing"
    vectors and compared.  It measures how well predictions cover the correct
    time regions, regardless of how many segments are used.
    """
    pred_vec = segments_to_binary(pred, duration_s, resolution_s)
    gt_vec = segments_to_binary(gt, duration_s, resolution_s)

    # Ensure equal length (may differ by ±1 due to rounding)
    min_len = min(len(pred_vec), len(gt_vec))
    pred_vec, gt_vec = pred_vec[:min_len], gt_vec[:min_len]

    intersection = np.logical_and(pred_vec, gt_vec).sum()
    union = np.logical_or(pred_vec, gt_vec).sum()
    if union == 0: return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def _pairwise_iou(a: Segment, b: Segment) -> float:
    """Compute temporal IoU between two individual segments.

    IoU = intersection(a, b) / union(a, b)
    where union = duration(a) + duration(b) - intersection.
    """
    inter_start = max(a.start_s, b.start_s)
    inter_end = min(a.end_s, b.end_s)
    intersection = max(0.0, inter_end - inter_start)

    union = (a.duration() + b.duration()) - intersection
    if union <= 0: return 0.0
    return intersection / union


def _count_matches(query: List[Segment], pool: List[Segment], iou_threshold: float) -> int:
    """Count how many *query* segments match a *pool* segment at ≥ iou_threshold.

    Uses greedy best-IoU matching: for each query segment, find the pool
    segment with the highest pairwise IoU.  If IoU ≥ threshold, count as a
    match and remove the pool segment from further consideration.

    This is the standard protocol in temporal action detection
    (ActivityNet / THUMOS benchmarks). Examples showing correctness:
      - 1 giant pred covering the whole video vs. 30 small GTs:
        IoU per pair ≈ 4s/120s = 0.033 → no match at 0.3 threshold ✓
      - Well-aligned pred: IoU ≈ 0.85 → match ✓
    """
    if not query or not pool: return 0
    used_pool = set()
    hits = 0

    for q in query:
        best_iou, best_idx = 0.0, -1
        for idx, p in enumerate(pool):
            if idx in used_pool: continue
            iou = _pairwise_iou(q, p)
            if iou > best_iou: best_iou, best_idx = iou, idx
        if best_iou >= iou_threshold and best_idx >= 0:
            hits += 1
            used_pool.add(best_idx)
    return hits


def segment_f1(pred: List[Segment], gt: List[Segment], iou_threshold: float = 0.5) -> Dict[str, float]:
    """Segment-level precision, recall, and F1 using pairwise IoU matching.

    For each GT segment, find the predicted segment with the highest pairwise
    IoU. Count as a match only if IoU ≥ *iou_threshold*. This correctly penalises both:
    - Under-segmentation (1 giant pred → tiny pairwise IoU → no match)
    - Over-segmentation  (many tiny preds → tiny pairwise IoU → no match)
    """
    if not pred and not gt: return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred or not gt: return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp_recall = _count_matches(gt, pred, iou_threshold)
    tp_precision = _count_matches(pred, gt, iou_threshold)
    precision = tp_precision / len(pred)
    recall = tp_recall / len(gt)

    if precision + recall == 0: return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def boundary_errors(pred: List[Segment], gt: List[Segment]) -> Dict:
    """Per-boundary signed error (pred − nearest GT) in seconds.

    For each predicted boundary (start or end), find the nearest GT boundary
    and compute the signed error.  Positive = prediction is late.

    Returns a dict with:
        errors             : list of all signed errors (seconds)
        mean_abs_error_s   : mean |error|
        median_abs_error_s : median |error|
    """
    gt_boundaries = sorted({s.start_s for s in gt} | {s.end_s for s in gt})
    if not gt_boundaries or not pred:
        return {"errors": [], "mean_abs_error_s": float("nan"), "median_abs_error_s": float("nan")}

    gt_arr = np.array(gt_boundaries)
    errors: list[float] = []
    for seg in pred:
        for t in (seg.start_s, seg.end_s):
            idx = np.argmin(np.abs(gt_arr - t))
            errors.append(t - gt_arr[idx])

    abs_errors = np.abs(errors)
    return {
        "errors": errors,
        "mean_abs_error_s": float(np.mean(abs_errors)),
        "median_abs_error_s": float(np.median(abs_errors)),
    }


def count_ratio(pred: List[Segment], gt: List[Segment]) -> float:
    # Predicted / GT segment count.  1.0 = perfect, >1 = over-seg
    if not gt: return float("inf") if pred else 1.0
    return len(pred) / len(gt)


def compute_all_metrics(
    pred: List[Segment], gt: List[Segment],
    duration_s: float, resolution_s: float = 0.04,
    iou_thresholds: List[float] = None,
) -> Dict[str, float]:
    """Run every metric and return a flat dictionary.

    Segment F1 is computed at each IoU threshold in *iou_thresholds* (default: [0.3, 0.5, 0.7]). 
    This follows the ActivityNet convention for temporal action detection evaluation.
    """
    if iou_thresholds is None: iou_thresholds = [0.3, 0.5, 0.7]
    result: Dict[str, float] = {}
    result["temporal_iou"] = temporal_iou(pred, gt, duration_s, resolution_s) # Coverage metric
    
    for thr in iou_thresholds: # Segment F1 at multiple IoU thresholds
        sf1 = segment_f1(pred, gt, iou_threshold=thr)
        suffix = f"@{thr:.1f}"
        result[f"seg_precision{suffix}"] = sf1["precision"]
        result[f"seg_recall{suffix}"] = sf1["recall"]
        result[f"seg_f1{suffix}"] = sf1["f1"]

    be = boundary_errors(pred, gt)
    result["boundary_mean_abs_error_s"] = be["mean_abs_error_s"]
    result["boundary_median_abs_error_s"] = be["median_abs_error_s"]

    result["count_ratio"] = count_ratio(pred, gt)
    result["pred_count"] = len(pred)
    result["gt_count"] = len(gt)
    return result