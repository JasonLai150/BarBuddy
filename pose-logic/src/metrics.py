"""
Metrics computation — joint angles, ROM, rep counting.

Computes lift-specific metrics from smoothed 3D keypoints.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from src.utils import COCO_KEYPOINTS


# ---------------------------------------------------------------------------
# Joint angle definitions per lift type
#
# Each entry: (parent_idx, joint_idx, child_idx) → angle at joint_idx
# Using COCO-17 indices.
# ---------------------------------------------------------------------------
LIFT_ANGLES: dict[str, dict[str, tuple[int, int, int]]] = {
    "squat": {
        "left_knee":   (11, 13, 15),   # hip → knee → ankle
        "right_knee":  (12, 14, 16),
        "left_hip":    (5, 11, 13),    # shoulder → hip → knee
        "right_hip":   (6, 12, 14),
    },
    "deadlift": {
        "left_hip":    (5, 11, 13),
        "right_hip":   (6, 12, 14),
        "left_knee":   (11, 13, 15),
        "right_knee":  (12, 14, 16),
        "torso":       (0, 5, 11),     # nose → shoulder → hip (trunk angle approx)
    },
    "bench": {
        "left_elbow":  (5, 7, 9),      # shoulder → elbow → wrist
        "right_elbow": (6, 8, 10),
        "left_shoulder": (7, 5, 11),   # elbow → shoulder → hip
        "right_shoulder": (8, 6, 12),
    },
    "overhead_press": {
        "left_elbow":  (5, 7, 9),
        "right_elbow": (6, 8, 10),
        "left_shoulder": (7, 5, 11),
        "right_shoulder": (8, 6, 12),
    },
}

# Default angles for unknown lift types
DEFAULT_ANGLES: dict[str, tuple[int, int, int]] = {
    "left_knee":   (11, 13, 15),
    "right_knee":  (12, 14, 16),
    "left_hip":    (5, 11, 13),
    "right_hip":   (6, 12, 14),
    "left_elbow":  (5, 7, 9),
    "right_elbow": (6, 8, 10),
}

# Primary angle for rep detection per lift type
PRIMARY_ANGLE: dict[str, str] = {
    "squat":          "left_knee",
    "deadlift":       "left_hip",
    "bench":          "left_elbow",
    "overhead_press": "left_elbow",
}


# ---------------------------------------------------------------------------
# Angle computation
# ---------------------------------------------------------------------------
def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle (in degrees) at point b formed by rays ba and bc.

    Args:
        a, b, c: 3D points as np.ndarray of shape (3,)

    Returns:
        Angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_frame_angles(
    landmarks: list[dict],
    angle_defs: dict[str, tuple[int, int, int]],
    conf_threshold: float = 0.3,
) -> dict[str, float | None]:
    """
    Compute joint angles for a single frame.

    Returns dict mapping angle_name → degrees (or None if insufficient confidence).
    """
    # Build lookup: idx → (x, y, z, conf)
    kpt = {}
    for lm in landmarks:
        kpt[lm["idx"]] = np.array([lm["x"], lm["y"], lm["z"]])
        kpt[(lm["idx"], "conf")] = lm["conf"]

    angles: dict[str, float | None] = {}
    for name, (pi, ji, ci) in angle_defs.items():
        if (pi not in kpt or ji not in kpt or ci not in kpt):
            angles[name] = None
            continue
        if (kpt.get((pi, "conf"), 0) < conf_threshold or
            kpt.get((ji, "conf"), 0) < conf_threshold or
            kpt.get((ci, "conf"), 0) < conf_threshold):
            angles[name] = None
            continue
        angles[name] = angle_3d(kpt[pi], kpt[ji], kpt[ci])

    return angles


# ---------------------------------------------------------------------------
# Rep detection (peak/valley analysis on joint angle time series)
# ---------------------------------------------------------------------------
def detect_reps(
    angle_series: list[float | None],
    sample_fps: int,
    min_rep_duration_sec: float = 0.8,
    max_rep_duration_sec: float = 8.0,
    min_rom_degrees: float = 30.0,
) -> list[dict]:
    """
    Detect repetitions from a joint angle time series.

    Uses peak/valley detection: a rep is a valley→peak→valley cycle.

    Args:
        angle_series: per-frame angle values (None for missing)
        sample_fps: sampling rate
        min_rep_duration_sec: minimum time for one rep
        max_rep_duration_sec: maximum time for one rep
        min_rom_degrees: minimum angle change to count as a rep

    Returns:
        List of rep dicts: {start_frame, end_frame, rom_degrees, duration_sec}
    """
    # Interpolate None values
    values = _interpolate_series(angle_series)
    if len(values) < 5:
        return []

    min_frames = int(min_rep_duration_sec * sample_fps)
    max_frames = int(max_rep_duration_sec * sample_fps)

    # Find local minima (valleys = bottom of movement)
    valleys = _find_valleys(values, min_prominence=min_rom_degrees * 0.5)

    reps = []
    for i in range(len(valleys) - 1):
        v1, v2 = valleys[i], valleys[i + 1]
        duration_frames = v2 - v1

        if duration_frames < min_frames or duration_frames > max_frames:
            continue

        # Find peak between valleys
        segment = values[v1:v2 + 1]
        peak_val = max(segment)
        valley_val = min(values[v1], values[v2])
        rom = peak_val - valley_val

        if rom < min_rom_degrees:
            continue

        reps.append({
            "start_frame": v1,
            "end_frame": v2,
            "rom_degrees": float(rom),
            "duration_sec": float(duration_frames / sample_fps),
            "peak_angle": float(peak_val),
            "valley_angle": float(valley_val),
        })

    return reps


def _find_valleys(values: list[float], min_prominence: float = 15.0) -> list[int]:
    """
    Find local minima in a 1D signal.

    Simple approach: a valley is a point lower than both neighbours
    by at least min_prominence.
    """
    n = len(values)
    if n < 3:
        return []

    # Smooth slightly to avoid noise-induced false valleys
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(values, sigma=2.0).tolist()

    valleys = []
    for i in range(1, n - 1):
        # Check if local minimum
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            # Check prominence: max of neighbours minus valley
            left_max = max(smoothed[:i])
            right_max = max(smoothed[i + 1:]) if i + 1 < n else smoothed[i]
            prominence = min(left_max, right_max) - smoothed[i]
            if prominence >= min_prominence:
                valleys.append(i)

    return valleys


def _interpolate_series(series: list[float | None]) -> list[float]:
    """Linearly interpolate None values in a series."""
    result = list(series)
    n = len(result)

    # Forward-fill then backward-fill edges
    valid_indices = [i for i, v in enumerate(result) if v is not None]
    if not valid_indices:
        return [0.0] * n

    for i in range(n):
        if result[i] is not None:
            continue

        before = [vi for vi in valid_indices if vi < i]
        after = [vi for vi in valid_indices if vi > i]

        if before and after:
            b, a = before[-1], after[0]
            alpha = (i - b) / (a - b)
            result[i] = (1 - alpha) * result[b] + alpha * result[a]
        elif before:
            result[i] = result[before[-1]]
        elif after:
            result[i] = result[after[0]]
        else:
            result[i] = 0.0

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_rep_metrics(
    frames_3d: list[dict],
    lift_type: str = "unknown",
    sample_fps: int = 12,
    conf_threshold: float = 0.3,
) -> dict:
    """
    Compute rep count, ROM, and joint angle time series from 3D frames.

    Args:
        frames_3d: list of smoothed frame dicts with "landmarks"
        lift_type: "squat", "deadlift", "bench", "overhead_press", "unknown"
        sample_fps: sampling rate
        conf_threshold: minimum confidence for angle computation

    Returns:
        Dict with keys:
            reps             – int, number of detected reps
            repDetails       – list of per-rep dicts
            avgRom           – float, average ROM in degrees
            peakRom          – float, max ROM across reps
            jointAngles      – dict of angle name → list of per-frame values
            primaryAngle     – str, name of the angle used for rep detection
            confidence       – float, average frame confidence
    """
    # Select angle definitions for this lift type
    angle_defs = LIFT_ANGLES.get(lift_type, DEFAULT_ANGLES)

    # Compute per-frame angles
    angle_series: dict[str, list[float | None]] = defaultdict(list)

    for frame in frames_3d:
        lms = frame.get("landmarks")
        if lms is None:
            for name in angle_defs:
                angle_series[name].append(None)
            continue

        frame_angles = compute_frame_angles(lms, angle_defs, conf_threshold)
        for name in angle_defs:
            angle_series[name].append(frame_angles.get(name))

    # Select primary angle for rep detection
    primary = PRIMARY_ANGLE.get(lift_type, list(angle_defs.keys())[0])
    primary_series = angle_series.get(primary, [])

    # Detect reps
    reps = detect_reps(primary_series, sample_fps)

    # Compute summary statistics
    avg_rom = float(np.mean([r["rom_degrees"] for r in reps])) if reps else 0.0
    peak_rom = float(max([r["rom_degrees"] for r in reps])) if reps else 0.0

    # Average confidence across all frames
    avg_conf = float(np.mean([
        f.get("confidence", 0.0) for f in frames_3d
    ])) if frames_3d else 0.0

    # Convert angle series to plain lists (replace None with null-safe values)
    joint_angles_out: dict[str, list[float | None]] = {}
    for name, series in angle_series.items():
        joint_angles_out[name] = [
            round(v, 2) if v is not None else None for v in series
        ]

    return {
        "reps": len(reps),
        "repDetails": reps,
        "avgRom": round(avg_rom, 2),
        "peakRom": round(peak_rom, 2),
        "jointAngles": joint_angles_out,
        "primaryAngle": primary,
        "confidence": round(avg_conf, 3),
    }
