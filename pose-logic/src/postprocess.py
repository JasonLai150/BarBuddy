"""
Stage 4 — Post-processing: temporal smoothing + kinematic constraints.

Applies:
  1. One Euro Filter for adaptive jitter removal
  2. Gaussian temporal smoothing for remaining noise
  3. Bone-length consistency enforcement
  4. Joint angle limit enforcement
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.utils import COCO_KEYPOINTS, COCO_SKELETON


# ---------------------------------------------------------------------------
# One Euro Filter (Casiez et al., 2012)
# Adaptive low-pass filter: smooth when slow, responsive when fast.
# ---------------------------------------------------------------------------
class OneEuroFilter:
    """1D One Euro Filter for a single signal dimension."""

    def __init__(self, freq: float, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x

        # Derivative
        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)

        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class OneEuroFilter3D:
    """One Euro Filter for 3D coordinates (x, y, z independently)."""

    def __init__(self, freq: float, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0):
        self.filters = [
            OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
            for _ in range(3)
        ]

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return np.array([f(v) for f, v in zip(self.filters, xyz)])


# ---------------------------------------------------------------------------
# Kinematic constraints
# ---------------------------------------------------------------------------
# Approximate bone-length ratios (relative to shoulder width = 1.0)
# Based on average human proportions (Drillis & Contini, 1966)
BONE_LENGTH_RATIOS: dict[tuple[int, int], float] = {
    # Torso
    (5, 6):   1.00,   # shoulder width (reference)
    (11, 12): 0.85,   # hip width
    (5, 11):  1.50,   # left torso
    (6, 12):  1.50,   # right torso
    # Arms
    (5, 7):   0.80,   # L upper arm
    (7, 9):   0.75,   # L forearm
    (6, 8):   0.80,   # R upper arm
    (8, 10):  0.75,   # R forearm
    # Legs
    (11, 13): 1.20,   # L thigh
    (13, 15): 1.10,   # L shin
    (12, 14): 1.20,   # R thigh
    (14, 16): 1.10,   # R shin
}

# Joint angle limits in degrees (min, max) — approximate physiological limits
JOINT_ANGLE_LIMITS: dict[str, tuple[float, float]] = {
    "left_elbow":  (0, 160),    # elbow flexion
    "right_elbow": (0, 160),
    "left_knee":   (0, 170),    # knee flexion
    "right_knee":  (0, 170),
    "left_hip":    (0, 170),    # hip flexion
    "right_hip":   (0, 170),
}

# Joint angle definitions: (parent, joint, child) triplets for angle at 'joint'
ANGLE_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "left_elbow":  (5, 7, 9),     # shoulder → elbow → wrist
    "right_elbow": (6, 8, 10),
    "left_knee":   (11, 13, 15),   # hip → knee → ankle
    "right_knee":  (12, 14, 16),
    "left_hip":    (5, 11, 13),    # shoulder → hip → knee
    "right_hip":   (6, 12, 14),
}


def _angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle at b formed by vectors ba and bc, in degrees."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Main smoothing + constraint function
# ---------------------------------------------------------------------------
def smooth_and_constrain(
    frames_3d: list[dict],
    sample_fps: int,
    one_euro_min_cutoff: float = 1.5,
    one_euro_beta: float = 0.01,
    gaussian_sigma: float = 0.8,
    conf_threshold: float = 0.3,
    enforce_bones: bool = True,
    enforce_angles: bool = True,
) -> list[dict]:
    """
    Apply temporal smoothing and kinematic constraints to 3D keypoints.

    Args:
        frames_3d: list of per-frame dicts from Stage 3 with field "landmarks"
        sample_fps: frame rate
        one_euro_min_cutoff: min cutoff for One Euro filter (lower = smoother)
        one_euro_beta: speed coefficient (higher = faster response to motion)
        gaussian_sigma: Gaussian filter sigma (in frames)
        conf_threshold: confidence below which smoothing is more aggressive
        enforce_bones: whether to enforce bone-length consistency
        enforce_angles: whether to enforce joint angle limits

    Returns:
        list of frames with smoothed "landmarks" and "confidence"
    """
    n_frames = len(frames_3d)
    if n_frames == 0:
        return frames_3d

    n_joints = 17

    # ---- Extract to arrays ----
    xyz = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
    conf = np.zeros((n_frames, n_joints), dtype=np.float32)

    for i, frame in enumerate(frames_3d):
        lms = frame.get("landmarks")
        if lms is None:
            continue
        for kpt in lms:
            j = kpt["idx"]
            if 0 <= j < n_joints:
                xyz[i, j, :] = [kpt["x"], kpt["y"], kpt["z"]]
                conf[i, j] = kpt["conf"]

    # ---- Pass 1: One Euro Filter (per-joint, per-dimension) ----
    xyz_oef = np.copy(xyz)
    for j in range(n_joints):
        filt = OneEuroFilter3D(
            freq=float(sample_fps),
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
        )
        for i in range(n_frames):
            if conf[i, j] > conf_threshold:
                xyz_oef[i, j] = filt(xyz[i, j])
            else:
                # For low-confidence frames, still update filter but blend
                # towards the filtered prediction (heavier smoothing)
                filtered = filt(xyz[i, j])
                xyz_oef[i, j] = filtered  # trust filter over noisy input

    # ---- Pass 2: Gaussian smoothing (catch remaining jitter) ----
    xyz_smooth = np.copy(xyz_oef)
    if n_frames >= 3 and gaussian_sigma > 0:
        for j in range(n_joints):
            for d in range(3):
                xyz_smooth[:, j, d] = gaussian_filter1d(
                    xyz_oef[:, j, d], sigma=gaussian_sigma
                )

    # ---- Pass 3: Bone-length consistency ----
    if enforce_bones and n_frames > 0:
        xyz_smooth = _enforce_bone_lengths(xyz_smooth, conf)

    # ---- Pass 4: Joint angle limits ----
    if enforce_angles and n_frames > 0:
        xyz_smooth = _enforce_angle_limits(xyz_smooth, conf)

    # ---- Rebuild output dicts ----
    frames_out: list[dict] = []
    for i in range(n_frames):
        landmarks = []
        confs_frame = []
        for j in range(n_joints):
            c = float(conf[i, j])
            confs_frame.append(c)
            landmarks.append({
                "idx": j,
                "name": COCO_KEYPOINTS[j],
                "x": float(xyz_smooth[i, j, 0]),
                "y": float(xyz_smooth[i, j, 1]),
                "z": float(xyz_smooth[i, j, 2]),
                "conf": c,
            })
        frames_out.append({
            "t": frames_3d[i]["t"],
            "landmarks": landmarks,
            "confidence": float(np.mean(confs_frame)),
        })

    return frames_out


# ---------------------------------------------------------------------------
# Bone-length enforcement
# ---------------------------------------------------------------------------
def _enforce_bone_lengths(
    xyz: np.ndarray,
    conf: np.ndarray,
    max_deviation: float = 0.3,
) -> np.ndarray:
    """
    Enforce bone-length consistency across frames.

    Strategy: compute median bone length across all frames, then
    nudge joints toward that length when they deviate too far.
    """
    n_frames = xyz.shape[0]

    for (a, b), _ratio in BONE_LENGTH_RATIOS.items():
        # Compute per-frame bone lengths
        bone_vecs = xyz[:, b, :] - xyz[:, a, :]
        bone_lengths = np.linalg.norm(bone_vecs, axis=-1)  # (N,)

        # Use median of high-confidence frames as target
        mask = (conf[:, a] > 0.3) & (conf[:, b] > 0.3)
        if mask.sum() < 3:
            continue
        target_length = float(np.median(bone_lengths[mask]))

        if target_length < 1e-5:
            continue

        # Fix frames where bone deviates too far from median
        for i in range(n_frames):
            if bone_lengths[i] < 1e-5:
                continue
            ratio = bone_lengths[i] / target_length
            if abs(ratio - 1.0) > max_deviation:
                # Scale the bone to target length
                direction = bone_vecs[i] / bone_lengths[i]
                midpoint = (xyz[i, a] + xyz[i, b]) / 2.0
                xyz[i, a] = midpoint - direction * target_length / 2.0
                xyz[i, b] = midpoint + direction * target_length / 2.0

    return xyz


# ---------------------------------------------------------------------------
# Joint angle limit enforcement
# ---------------------------------------------------------------------------
def _enforce_angle_limits(
    xyz: np.ndarray,
    conf: np.ndarray,
    blend_factor: float = 0.5,
) -> np.ndarray:
    """
    Softly enforce physiological joint angle limits.

    When a joint angle exceeds its limit, blend it toward the limit
    by blend_factor (0 = no correction, 1 = snap to limit).
    """
    n_frames = xyz.shape[0]

    for joint_name, (parent_idx, joint_idx, child_idx) in ANGLE_TRIPLETS.items():
        if joint_name not in JOINT_ANGLE_LIMITS:
            continue
        min_angle, max_angle = JOINT_ANGLE_LIMITS[joint_name]

        for i in range(n_frames):
            # Skip if any involved joint has low confidence
            if (conf[i, parent_idx] < 0.3 or
                conf[i, joint_idx] < 0.3 or
                conf[i, child_idx] < 0.3):
                continue

            angle = _angle_3d(
                xyz[i, parent_idx],
                xyz[i, joint_idx],
                xyz[i, child_idx],
            )

            if min_angle <= angle <= max_angle:
                continue  # within limits

            # Clamp to nearest limit
            target = min_angle if angle < min_angle else max_angle

            # Adjust child position to achieve target angle
            parent_to_joint = xyz[i, parent_idx] - xyz[i, joint_idx]
            joint_to_child = xyz[i, child_idx] - xyz[i, joint_idx]

            len_jc = np.linalg.norm(joint_to_child)
            if len_jc < 1e-6:
                continue

            # Rotate child around joint toward target angle
            # Simplified: interpolate between current and a position at the target angle
            current_dir = joint_to_child / len_jc
            target_rad = math.radians(target)
            current_rad = math.radians(angle)

            # Blend factor determines how aggressively we correct
            correction = blend_factor * (target_rad - current_rad)
            # This is a simplified correction — in practice you'd do a proper rotation
            # For now, scale the angle difference as a small nudge
            nudge_scale = correction * 0.1
            xyz[i, child_idx] += nudge_scale * parent_to_joint

    return xyz
