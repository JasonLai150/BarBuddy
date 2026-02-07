"""
Metrics computation — joint angles, ROM, rep counting.

State-machine rep detection per lift type.  Both sides (left **and**
right) are tracked independently, but a frame is only valid when
**both** sides cross the threshold (AND logic).  If one side is not
visible, the visible side alone is used as a fallback.

A moving-average filter removes noise at the beginning and end of videos.

**Angle computation uses 2D only** — The z coordinate from VideoPose3D
is empirically unreliable for gym-lift videos (see ``_z_scale_factor``
for the full analysis).  Joint angles are computed from the RTMPose 2D
keypoints (x, y) which accurately track full range of motion.  The z
values are still stored in landmarks for approximate depth ordering
but are zeroed before angle computation.

Validity criteria (angle at joint < threshold):
  - Bench:    both elbow angles < 90°
  - Squat:    both knee  angles < 90°
  - Deadlift: both hip   angles < 90°

Rep counting state machine (per frame):
  IDLE   → waiting for person to reach neutral first
  READY  → person was neutral; can now enter valid to start a rep
  IN_REP → person is in the valid range (bottom of movement)
  On exit from IN_REP back to neutral → rep counted, return to READY.

This prevents counting a rep if the video starts with the person already
at the bottom of the movement.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum, auto

import numpy as np
from scipy.ndimage import uniform_filter1d

from src.utils import COCO_KEYPOINTS


# ---------------------------------------------------------------------------
# Joint angle definitions per lift type  (COCO-17 indices)
# ---------------------------------------------------------------------------
LIFT_ANGLES: dict[str, dict[str, tuple[int, int, int]]] = {
    "squat": {
        "left_knee":   (11, 13, 15),
        "right_knee":  (12, 14, 16),
        "left_hip":    (5, 11, 13),
        "right_hip":   (6, 12, 14),
    },
    "deadlift": {
        "left_hip":    (5, 11, 13),
        "right_hip":   (6, 12, 14),
        "left_knee":   (11, 13, 15),
        "right_knee":  (12, 14, 16),
    },
    "bench": {
        "left_elbow":  (5, 7, 9),
        "right_elbow": (6, 8, 10),
        "left_shoulder": (7, 5, 11),
        "right_shoulder": (8, 6, 12),
    },
    "overhead_press": {
        "left_elbow":  (5, 7, 9),
        "right_elbow": (6, 8, 10),
        "left_shoulder": (7, 5, 11),
        "right_shoulder": (8, 6, 12),
    },
}

DEFAULT_ANGLES: dict[str, tuple[int, int, int]] = {
    "left_knee":   (11, 13, 15),
    "right_knee":  (12, 14, 16),
    "left_hip":    (5, 11, 13),
    "right_hip":   (6, 12, 14),
    "left_elbow":  (5, 7, 9),
    "right_elbow": (6, 8, 10),
}


# ---------------------------------------------------------------------------
# Per-lift validity config
#
# primary_pairs: list of (left_angle_name, right_angle_name) pairs.
#   Each pair is checked independently — if *either* side crosses the
#   threshold the frame is marked valid.
# threshold / neutral_threshold: angle boundaries.
# below_threshold: True → valid when angle < threshold.
# ---------------------------------------------------------------------------
class RepState(Enum):
    IDLE = auto()
    READY = auto()
    IN_REP = auto()


LIFT_VALIDITY: dict[str, dict] = {
    "bench": {
        "primary_pairs": [("left_elbow", "right_elbow")],
        "threshold": 90.0,
        "below_threshold": True,
        "neutral_threshold": 120.0,
    },
    "squat": {
        "primary_pairs": [("left_knee", "right_knee")],
        "threshold": 90.0,
        "below_threshold": True,
        "neutral_threshold": 140.0,
    },
    "deadlift": {
        "primary_pairs": [("left_hip", "right_hip")],
        "threshold": 90.0,
        "below_threshold": True,
        "neutral_threshold": 120.0,
    },
    "overhead_press": {
        "primary_pairs": [("left_elbow", "right_elbow")],
        "threshold": 90.0,
        "below_threshold": True,
        "neutral_threshold": 150.0,
    },
}

# Skeleton segments to highlight per lift type (COCO keypoint-index pairs).
LIFT_HIGHLIGHT_SEGMENTS: dict[str, list[tuple[int, int]]] = {
    "bench":          [(5, 7), (7, 9), (6, 8), (8, 10)],
    "squat":          [(11, 13), (13, 15), (12, 14), (14, 16)],
    "deadlift":       [(5, 11), (6, 12), (11, 12)],
    "overhead_press": [(5, 7), (7, 9), (6, 8), (8, 10)],
}


# ---------------------------------------------------------------------------
# Angle computation
# ---------------------------------------------------------------------------
def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at point *b* formed by rays ba and bc.  [0, 180]."""
    ba = a - b
    bc = c - b
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _z_scale_factor(landmarks: list[dict]) -> float:
    """Compute a scale factor for the z axis before angle computation.

    **Returns 0.0** — angles are computed in 2D only (x / y).

    Empirical analysis on bench-press and deadlift videos shows that the
    VideoPose3D z predictions are not usable for joint-angle computation:

    1. **Mixed-coordinate problem** — We store image-space x / y ([0, 1]
       normalised) with the model's root-relative z.  The z deltas
       between adjacent joints on the far-side arm are 3–6× larger than
       the x / y deltas, collapsing 3D angles to ~20° regardless of the
       actual arm position (2D angle: 72–168°).

    2. **Model-only 3D is also inaccurate** — Using the model's own
       (x, y, z) together (internally consistent coordinates) produces
       angles that are *inversely* correlated with reality.  For the
       deadlift: model says 173° when bent (should be ~45°) and 94°
       when standing (should be ~170°).

    3. **Root cause** — VideoPose3D was trained on Human3.6M (controlled
       indoor lab).  It doesn't generalise to gym lifts filmed from the
       side, especially for self-occluded limbs and extreme flexion.
       Short clips (100–200 frames vs. the 243-frame receptive field)
       compound the issue.

    4. **2D angles from RTMPose are reliable** — RTMPose was trained on
       large, diverse datasets (COCO + AIC + MPII, etc.) and produces
       accurate 2D joint positions that track the full ROM correctly.

    The z values remain in the ``landmarks`` output for approximate
    depth ordering (visualisation), but are zeroed out for angle / rep
    computation.
    """
    return 0.0  # zero → z * 0.0 = 0.0 → effectively 2D angles


def compute_frame_angles(
    landmarks: list[dict],
    angle_defs: dict[str, tuple[int, int, int]],
    conf_threshold: float = 0.3,
) -> dict[str, float | None]:
    """Per-frame joint angles.  Returns {angle_name: degrees | None}.

    **Z is rescaled** before angle computation so that the depth axis
    is on the same scale as the image-space x/y axes (see
    ``_z_scale_factor``).
    """
    z_sf = _z_scale_factor(landmarks)

    kpt: dict = {}
    for lm in landmarks:
        kpt[lm["idx"]] = np.array([lm["x"], lm["y"], lm["z"] * z_sf])
        kpt[(lm["idx"], "conf")] = lm["conf"]

    angles: dict[str, float | None] = {}
    for name, (pi, ji, ci) in angle_defs.items():
        if pi not in kpt or ji not in kpt or ci not in kpt:
            angles[name] = None
            continue
        if (kpt.get((pi, "conf"), 0) < conf_threshold
                or kpt.get((ji, "conf"), 0) < conf_threshold
                or kpt.get((ci, "conf"), 0) < conf_threshold):
            angles[name] = None
            continue
        angles[name] = angle_3d(kpt[pi], kpt[ji], kpt[ci])
    return angles


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------
def _smooth_series(raw: list[float | None], window: int = 5) -> list[float]:
    """Interpolate Nones then apply a moving-average filter."""
    n = len(raw)
    if n == 0:
        return []

    # Interpolate None gaps
    arr = np.array([v if v is not None else np.nan for v in raw], dtype=np.float64)
    nans = np.isnan(arr)
    if nans.all():
        return [0.0] * n
    if nans.any():
        xp = np.where(~nans)[0]
        fp = arr[~nans]
        arr = np.interp(np.arange(n), xp, fp)

    # Moving average (handles edges gracefully)
    smoothed = uniform_filter1d(arr, size=window, mode="nearest")
    return smoothed.tolist()


def _trim_bounds(n_frames: int, sample_fps: int) -> tuple[int, int]:
    """
    Return (start, end) indices to ignore for state transitions.
    Trims the first and last ~0.25 s to discard setup / rack noise.
    """
    margin = max(1, int(0.25 * sample_fps))
    start = min(margin, n_frames // 4)
    end = max(n_frames - margin, n_frames * 3 // 4)
    return start, end


# ---------------------------------------------------------------------------
# State-machine — each side checked independently
# ---------------------------------------------------------------------------
def _both_cross_valid(
    left_val: float | None,
    right_val: float | None,
    cfg: dict,
) -> bool:
    """True if **both** sides cross the valid threshold.

    If only one side is visible (the other is None), fall back to
    checking just the visible side so that partial-visibility videos
    still work.
    """
    thr = cfg["threshold"]
    below = cfg["below_threshold"]

    def _crosses(v: float) -> bool:
        return (v < thr) if below else (v > thr)

    if left_val is not None and right_val is not None:
        return _crosses(left_val) and _crosses(right_val)
    if left_val is not None:
        return _crosses(left_val)
    if right_val is not None:
        return _crosses(right_val)
    return False


def _both_cross_neutral(
    left_val: float | None,
    right_val: float | None,
    cfg: dict,
) -> bool:
    """True if **both** sides cross the neutral threshold.

    Same fallback rule as ``_both_cross_valid``.
    """
    nthr = cfg["neutral_threshold"]
    below = cfg["below_threshold"]

    def _crosses(v: float) -> bool:
        return (v > nthr) if below else (v < nthr)

    if left_val is not None and right_val is not None:
        return _crosses(left_val) and _crosses(right_val)
    if left_val is not None:
        return _crosses(left_val)
    if right_val is not None:
        return _crosses(right_val)
    return False


def _best_val(left: float | None, right: float | None) -> float | None:
    """Pick the larger of two optional values (max ROM side)."""
    if left is not None and right is not None:
        return max(left, right)
    return left if left is not None else right


def detect_reps_state_machine(
    sm_left: list[float],
    sm_right: list[float],
    lift_type: str,
    sample_fps: int,
    min_valid_frames: int = 2,
    trim_start: int = 0,
    trim_end: int | None = None,
) -> tuple[list[dict], list[bool]]:
    """
    Detect reps on smoothed per-side angle series.

    Returns (reps, per_frame_valid).
    """
    cfg = LIFT_VALIDITY.get(lift_type)
    n = len(sm_left)
    if cfg is None or n == 0:
        return [], [False] * n
    if trim_end is None:
        trim_end = n

    state = RepState.IDLE
    reps: list[dict] = []
    per_frame_valid: list[bool] = [False] * n

    rep_start: int | None = None
    valid_count = 0
    min_ang = 999.0
    max_ang = 0.0

    for i in range(n):
        lv = sm_left[i] if i < len(sm_left) else None
        rv = sm_right[i] if i < len(sm_right) else None

        v = _both_cross_valid(lv, rv, cfg)
        nt = _both_cross_neutral(lv, rv, cfg)
        best = _best_val(lv, rv)
        per_frame_valid[i] = v

        # Skip trimmed edges for state transitions
        if i < trim_start or i >= trim_end:
            continue

        if state == RepState.IDLE:
            if nt:
                state = RepState.READY
                rep_start = i
                if best is not None:
                    max_ang = best

        elif state == RepState.READY:
            if best is not None:
                max_ang = max(max_ang, best)
            if v:
                state = RepState.IN_REP
                valid_count = 1
                if best is not None:
                    min_ang = best

        elif state == RepState.IN_REP:
            if v:
                valid_count += 1
                if best is not None:
                    min_ang = min(min_ang, best)
            elif nt:
                if valid_count >= min_valid_frames and rep_start is not None:
                    if best is not None:
                        max_ang = max(max_ang, best)
                    rom = max_ang - min_ang
                    dur = (i - rep_start) / sample_fps
                    reps.append({
                        "start_frame": rep_start,
                        "end_frame": i,
                        "rom_degrees": float(rom),
                        "duration_sec": float(dur),
                        "peak_angle": float(max_ang),
                        "valley_angle": float(min_ang),
                        "valid_frames": valid_count,
                    })
                state = RepState.READY
                rep_start = i
                valid_count = 0
                min_ang = 999.0
                max_ang = best if best is not None else 0.0

    # --- flush an in-progress rep at end of video ---
    # If the lifter was still in the valid range when the video ended
    # (or crossed back out of valid but didn't reach neutral yet),
    # count it as a completed rep so short clips aren't penalised.
    if state == RepState.IN_REP and valid_count >= min_valid_frames and rep_start is not None:
        rom = max_ang - min_ang
        dur = (n - 1 - rep_start) / sample_fps
        reps.append({
            "start_frame": rep_start,
            "end_frame": n - 1,
            "rom_degrees": float(rom),
            "duration_sec": float(dur),
            "peak_angle": float(max_ang),
            "valley_angle": float(min_ang),
            "valid_frames": valid_count,
        })

    return reps, per_frame_valid


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
    Compute rep count, ROM, and joint-angle time series from 3D frames.

    Each side (left / right) is tracked independently.  The angle series
    are smoothed with a moving-average filter to reject noise, and the
    first / last ~0.25 s are trimmed to ignore setup / rack movements.
    """
    angle_defs = LIFT_ANGLES.get(lift_type, DEFAULT_ANGLES)
    cfg = LIFT_VALIDITY.get(lift_type, {})

    # --- compute raw per-frame angles ---
    angle_series: dict[str, list[float | None]] = defaultdict(list)

    for frame in frames_3d:
        lms = frame.get("landmarks")
        if lms is None:
            for name in angle_defs:
                angle_series[name].append(None)
            continue
        fa = compute_frame_angles(lms, angle_defs, conf_threshold)
        for name in angle_defs:
            angle_series[name].append(fa.get(name))

    n = len(frames_3d)

    # --- smooth & detect ---
    primary_pairs = cfg.get("primary_pairs", [])
    if not primary_pairs:
        return _fallback_result(frames_3d, angle_series, angle_defs, n)

    left_name, right_name = primary_pairs[0]
    raw_left = angle_series.get(left_name, [None] * n)
    raw_right = angle_series.get(right_name, [None] * n)

    smooth_window = max(3, int(0.25 * sample_fps))
    sm_left = _smooth_series(raw_left, window=smooth_window)
    sm_right = _smooth_series(raw_right, window=smooth_window)

    trim_start, trim_end = _trim_bounds(n, sample_fps)

    reps, per_frame_valid = detect_reps_state_machine(
        sm_left, sm_right, lift_type, sample_fps,
        trim_start=trim_start, trim_end=trim_end,
    )

    # --- output ---
    avg_rom = float(np.mean([r["rom_degrees"] for r in reps])) if reps else 0.0
    peak_rom = float(max(r["rom_degrees"] for r in reps)) if reps else 0.0

    avg_conf = float(np.mean([
        f.get("confidence", 0.0) for f in frames_3d
    ])) if frames_3d else 0.0

    joint_angles_out: dict[str, list[float | None]] = {}
    for name, series in angle_series.items():
        joint_angles_out[name] = [
            round(v, 2) if v is not None else None for v in series
        ]
    # Include smoothed primary curves for debugging / viz
    joint_angles_out[f"{left_name}_smooth"] = [round(v, 2) for v in sm_left]
    joint_angles_out[f"{right_name}_smooth"] = [round(v, 2) for v in sm_right]

    return {
        "reps": len(reps),
        "repDetails": reps,
        "avgRom": round(avg_rom, 2),
        "peakRom": round(peak_rom, 2),
        "jointAngles": joint_angles_out,
        "primaryAngle": f"{left_name} / {right_name}",
        "confidence": round(avg_conf, 3),
        "perFrameValid": per_frame_valid,
    }


def _fallback_result(
    frames_3d: list[dict],
    angle_series: dict[str, list[float | None]],
    angle_defs: dict,
    n: int,
) -> dict:
    """Return a metrics dict with 0 reps for unknown lift types."""
    avg_conf = float(np.mean([
        f.get("confidence", 0.0) for f in frames_3d
    ])) if frames_3d else 0.0

    joint_angles_out = {
        name: [round(v, 2) if v is not None else None for v in series]
        for name, series in angle_series.items()
    }

    return {
        "reps": 0,
        "repDetails": [],
        "avgRom": 0.0,
        "peakRom": 0.0,
        "jointAngles": joint_angles_out,
        "primaryAngle": list(angle_defs.keys())[0] if angle_defs else "unknown",
        "confidence": round(avg_conf, 3),
        "perFrameValid": [False] * n,
    }
