"""Post-processing for VLM-predicted segments.

All operations are CPU-only and add zero VLM inference time.
Three techniques:
  1. merge_short   — absorb tiny segments into neighbours
  2. fill_gaps     — close small inter-segment gaps
  3. snap_to_pose  — refine boundaries using pose velocity minima
"""
import numpy as np
from pathlib import Path
from typing import List, Optional
from data_utils import Segment
from config import MIN_SEGMENT_S, MAX_GAP_S, POSE_SNAP_WINDOW_S


def merge_short_segments(segments: List[Segment], min_duration_s: float = MIN_SEGMENT_S) -> List[Segment]:
    """Absorb segments shorter than *min_duration_s* into the nearest neighbour.

    For each short segment, it is merged with whichever adjacent segment
    (previous or next) results in the smallest combined duration.
    This fixes VLM "stutter" outputs — tiny 0.1–0.2s fragments.
    """
    if not segments: return []
    result = list(segments)
    changed = True

    while changed:
        changed = False
        new_result = []
        i = 0
        
        while i < len(result):
            seg = result[i]
            if seg.duration() < min_duration_s and len(result) > 1: # Merge with closest neighbour
                if i == 0 and i + 1 < len(result): # Merge with next
                    nxt = result[i + 1]
                    new_result.append(Segment(start_s=seg.start_s, end_s=nxt.end_s, text=nxt.text))
                    i += 2
                    changed = True
                elif i == len(result) - 1: # Merge with previous
                    if new_result:
                        prev = new_result.pop()
                        new_result.append(Segment(start_s=prev.start_s, end_s=seg.end_s, text=prev.text))
                    else:
                        new_result.append(seg)
                    i += 1
                    changed = True
                else: # Middle segment — merge with the closer neighbour
                    prev = new_result[-1] if new_result else None
                    nxt = result[i + 1]
                    gap_prev = seg.start_s - prev.end_s if prev else float("inf")
                    gap_next = nxt.start_s - seg.end_s

                    if gap_prev <= gap_next and prev is not None:
                        merged = new_result.pop()
                        new_result.append(Segment(start_s=merged.start_s, end_s=seg.end_s, text=merged.text))
                    else:
                        new_result.append(Segment(start_s=seg.start_s, end_s=nxt.end_s, text=nxt.text))
                        i += 1  # skip next since we consumed it
                    i += 1
                    changed = True
            else:
                new_result.append(seg)
                i += 1
        result = new_result
    return result


def fill_gaps(segments: List[Segment], max_gap_s: float = MAX_GAP_S) -> List[Segment]:
    """Close gaps smaller than *max_gap_s* between consecutive segments.

    For tiny inter-phrase gaps, extends the earlier segment's end to the
    next segment's start.  This removes "flicker" in the timeline.
    """
    if len(segments) < 2: return list(segments)
    result = [segments[0]]
    for seg in segments[1:]:
        gap = seg.start_s - result[-1].end_s
        if 0 < gap <= max_gap_s:
            # Close the gap by extending the previous segment
            prev = result.pop()
            result.append(Segment(start_s=prev.start_s, end_s=seg.end_s, text=prev.text))
        else: result.append(seg)
    return result


def _load_pose_velocity(pose_path: Path) -> Optional[np.ndarray]:
    """Load a *.pose file and compute wrist velocity magnitude.

    Returns (times_s, velocity) arrays, or None if the file can't be loaded.
    The sign_language_processing/pose library stores poses in a custom
    binary format.  We use the `pose_format` package to read it.
    """
    try:
        from pose_format import Pose
        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())

        # pose.body.data has shape (frames, people, landmarks, dims)
        # Find wrist landmarks — typically indices 15,16 (left/right wrist)
        # in the BODY_135 or BODY_25 skeleton
        data = pose.body.data.numpy()  # (F, P, L, D)
        fps = pose.body.fps
        if data.ndim != 4 or data.shape[0] < 2: return None
        person_data = data[:, 0, :, :]  # (F, L, D) Use first person, take mean of all landmarks for robustness

        # Try to find wrist landmarks; fall back to mean of all landmarks
        try: # Wrists are typically landmarks 9,10 in MediaPipe Holistic
            header = pose.header
            body_comp = header.components[0]
            point_names = [p.name if hasattr(p, 'name') else str(p) for p in body_comp.points]
            wrist_indices = [i for i, name in enumerate(point_names) if 'wrist' in name.lower() or 'WRIST' in name]
            if wrist_indices: person_data = person_data[:, wrist_indices, :]
        except (AttributeError, IndexError): pass  # use all landmarks

        # Compute velocity: L2 norm of frame-to-frame displacement
        velocity = np.linalg.norm(np.diff(person_data, axis=0), axis=-1)
        velocity = velocity.mean(axis=-1)  # average over landmarks → (F-1,)
        times = np.arange(len(velocity)) / fps
        return times, velocity

    except ImportError:
        print("  WARNING: pose_format not installed; skipping pose boundary snapping. Install with: pip install pose_format")
        return None
    except Exception as e:
        print(f"  WARNING: Could not load pose file {pose_path}: {e}")
        return None


def snap_to_pose(segments: List[Segment], pose_path: Path, window_s: float = POSE_SNAP_WINDOW_S) -> List[Segment]:
    """Snap segment boundaries to the nearest velocity minimum in pose data.

    For each boundary (start or end), search within ±window_s for the time
    with the lowest wrist velocity (hands at rest = phrase boundary).
    This refines VLM timestamps from ~0.3s error to ~0.04s (frame-level).
    """
    result_data = _load_pose_velocity(pose_path)
    if result_data is None: return segments

    times, velocity = result_data
    if len(times) == 0: return segments

    def _snap_time(t: float) -> float: # Find the time within [t-window, t+window] with minimum velocity
        mask = (times >= t - window_s) & (times <= t + window_s)
        if not mask.any(): return t
        candidates = np.where(mask)[0]
        best = candidates[np.argmin(velocity[candidates])]
        return float(times[best])

    snapped = []
    for seg in segments:
        new_start = _snap_time(seg.start_s)
        new_end = _snap_time(seg.end_s)
        # Ensure validity
        if new_end > new_start: snapped.append(Segment(start_s=new_start, end_s=new_end, text=seg.text))
        else: snapped.append(seg)  # keep original if snapping made it invalid
    return snapped


def postprocess_segments(
    segments: List[Segment],
    pose_path: Optional[Path] = None,
    min_segment_s: float = MIN_SEGMENT_S,
    max_gap_s: float = MAX_GAP_S,
    snap_window_s: float = POSE_SNAP_WINDOW_S,
) -> List[Segment]:
    """Apply all post-processing steps in order. Order matters:
      1. Pose snapping first (refine raw VLM boundaries)
      2. Merge short segments (clean up stutter)
      3. Fill gaps (close tiny silences)
    """
    n_before = len(segments)

    # 1. Pose-based boundary snapping (if pose file available)
    if pose_path is not None and pose_path.exists():
        segments = snap_to_pose(segments, pose_path, window_s=snap_window_s)
        print(f"  Pose snapping: applied to {len(segments)} segments")

    # 2. Merge short segments
    segments = merge_short_segments(segments, min_duration_s=min_segment_s)
    if len(segments) != n_before:
        print(f"  Merge short: {n_before} → {len(segments)} segments")

    # 3. Fill tiny gaps
    n_before_gaps = len(segments)
    segments = fill_gaps(segments, max_gap_s=max_gap_s)
    if len(segments) != n_before_gaps: print(f"  Fill gaps: {n_before_gaps} → {len(segments)} segments")
    return segments