"""
Shared utilities for the BarBuddy 3-stage 3D pose pipeline.

Keypoint definitions, video probing, frame sampling, model download helpers.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# COCO 17 keypoints (output of Stage 2 — RTMPose / YOLOv8-Pose)
# ---------------------------------------------------------------------------
COCO_KEYPOINTS: list[str] = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

COCO_SKELETON: list[tuple[int, int]] = [
    # torso
    (5, 6),    # left_shoulder – right_shoulder
    (5, 11),   # left_shoulder – left_hip
    (6, 12),   # right_shoulder – right_hip
    (11, 12),  # left_hip – right_hip
    # arms
    (5, 7),  (7, 9),    # left arm
    (6, 8),  (8, 10),   # right arm
    # legs
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    # face
    (0, 1), (0, 2),     # nose – eyes
    (1, 3), (2, 4),     # eyes – ears
]

# ---------------------------------------------------------------------------
# Human3.6M 17 joints (used by VideoPose3D / temporal 3D lifter)
# ---------------------------------------------------------------------------
H36M_JOINTS: list[str] = [
    "hip",             # 0  (root / pelvis)
    "right_hip",       # 1
    "right_knee",      # 2
    "right_ankle",     # 3
    "left_hip",        # 4
    "left_knee",       # 5
    "left_ankle",      # 6
    "spine",           # 7
    "neck",            # 8
    "nose",            # 9
    "head",            # 10
    "left_shoulder",   # 11
    "left_elbow",      # 12
    "left_wrist",      # 13
    "right_shoulder",  # 14
    "right_elbow",     # 15
    "right_wrist",     # 16
]

H36M_SKELETON: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),      # right leg
    (0, 4), (4, 5), (5, 6),      # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine → head
    (8, 11), (11, 12), (12, 13), # left arm
    (8, 14), (14, 15), (15, 16), # right arm
]


def coco_to_h36m(kpts_coco: np.ndarray) -> np.ndarray:
    """
    Convert COCO 17-keypoint array to H36M 17-joint array.

    Args:
        kpts_coco: (..., 17, C) where C is 2 or 3 (xy or xyz + optional conf).

    Returns:
        kpts_h36m: (..., 17, C)

    Mapping:
        H36M hip (0)       = midpoint( COCO left_hip(11), right_hip(12) )
        H36M right_hip (1) = COCO right_hip (12)
        H36M right_knee(2) = COCO right_knee(14)
        H36M right_ankle(3)= COCO right_ankle(16)
        H36M left_hip (4)  = COCO left_hip (11)
        H36M left_knee (5) = COCO left_knee (13)
        H36M left_ankle(6) = COCO left_ankle(15)
        H36M spine (7)     = midpoint(shoulders + hips)
        H36M neck (8)      = midpoint( COCO left_shoulder(5), right_shoulder(6) )
        H36M nose (9)      = COCO nose (0)
        H36M head (10)     = midpoint( COCO left_ear(3), right_ear(4) )
        H36M left_shoulder(11) = COCO left_shoulder (5)
        H36M left_elbow(12)    = COCO left_elbow (7)
        H36M left_wrist(13)    = COCO left_wrist (9)
        H36M right_shoulder(14)= COCO right_shoulder(6)
        H36M right_elbow(15)   = COCO right_elbow (8)
        H36M right_wrist(16)   = COCO right_wrist (10)
    """
    shape = list(kpts_coco.shape)
    shape[-2] = 17  # same count, different order
    out = np.zeros(shape, dtype=kpts_coco.dtype)

    out[..., 0, :] = (kpts_coco[..., 11, :] + kpts_coco[..., 12, :]) / 2.0  # hip
    out[..., 1, :] = kpts_coco[..., 12, :]   # right hip
    out[..., 2, :] = kpts_coco[..., 14, :]   # right knee
    out[..., 3, :] = kpts_coco[..., 16, :]   # right ankle
    out[..., 4, :] = kpts_coco[..., 11, :]   # left hip
    out[..., 5, :] = kpts_coco[..., 13, :]   # left knee
    out[..., 6, :] = kpts_coco[..., 15, :]   # left ankle
    out[..., 7, :] = (
        kpts_coco[..., 5, :] + kpts_coco[..., 6, :]
        + kpts_coco[..., 11, :] + kpts_coco[..., 12, :]
    ) / 4.0                                    # spine
    out[..., 8, :] = (kpts_coco[..., 5, :] + kpts_coco[..., 6, :]) / 2.0  # neck
    out[..., 9, :] = kpts_coco[..., 0, :]     # nose
    out[..., 10, :] = (kpts_coco[..., 3, :] + kpts_coco[..., 4, :]) / 2.0 # head
    out[..., 11, :] = kpts_coco[..., 5, :]    # left shoulder
    out[..., 12, :] = kpts_coco[..., 7, :]    # left elbow
    out[..., 13, :] = kpts_coco[..., 9, :]    # left wrist
    out[..., 14, :] = kpts_coco[..., 6, :]    # right shoulder
    out[..., 15, :] = kpts_coco[..., 8, :]    # right elbow
    out[..., 16, :] = kpts_coco[..., 10, :]   # right wrist

    return out


def h36m_to_coco(kpts_h36m: np.ndarray) -> np.ndarray:
    """
    Convert H36M 17-joint array back to COCO 17-keypoint array.

    Inverse of coco_to_h36m (best-effort; some COCO joints are derived
    from midpoints so the inverse is approximate).

    Args:
        kpts_h36m: (..., 17, C)

    Returns:
        kpts_coco: (..., 17, C)
    """
    shape = list(kpts_h36m.shape)
    shape[-2] = 17
    out = np.zeros(shape, dtype=kpts_h36m.dtype)

    out[..., 0, :]  = kpts_h36m[..., 9, :]   # nose
    # eyes/ears are not in H36M; approximate from head & nose
    head = kpts_h36m[..., 10, :]
    nose = kpts_h36m[..., 9, :]
    out[..., 1, :]  = (nose + head) / 2.0     # left_eye  (approx)
    out[..., 2, :]  = (nose + head) / 2.0     # right_eye (approx)
    out[..., 3, :]  = head                     # left_ear  (approx)
    out[..., 4, :]  = head                     # right_ear (approx)
    out[..., 5, :]  = kpts_h36m[..., 11, :]   # left_shoulder
    out[..., 6, :]  = kpts_h36m[..., 14, :]   # right_shoulder
    out[..., 7, :]  = kpts_h36m[..., 12, :]   # left_elbow
    out[..., 8, :]  = kpts_h36m[..., 15, :]   # right_elbow
    out[..., 9, :]  = kpts_h36m[..., 13, :]   # left_wrist
    out[..., 10, :] = kpts_h36m[..., 16, :]   # right_wrist
    out[..., 11, :] = kpts_h36m[..., 4, :]    # left_hip
    out[..., 12, :] = kpts_h36m[..., 1, :]    # right_hip
    out[..., 13, :] = kpts_h36m[..., 5, :]    # left_knee
    out[..., 14, :] = kpts_h36m[..., 2, :]    # right_knee
    out[..., 15, :] = kpts_h36m[..., 6, :]    # left_ankle
    out[..., 16, :] = kpts_h36m[..., 3, :]    # right_ankle

    return out


# ---------------------------------------------------------------------------
# Video probing & frame sampling (reused from original pipeline)
# ---------------------------------------------------------------------------
@dataclass
class VideoMeta:
    duration_sec: float
    src_fps: float
    width: int
    height: int


def probe_video(path: str) -> VideoMeta:
    """Use ffprobe to extract video metadata."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=0",
        path,
    ]
    out = subprocess.check_output(cmd, text=True)
    kv: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    width = int(kv.get("width", "0") or 0)
    height = int(kv.get("height", "0") or 0)
    dur = float(kv.get("duration", "0") or 0.0)

    r = kv.get("r_frame_rate", "0/1")
    num, den = r.split("/")
    fps = (float(num) / float(den)) if float(den) != 0 else 0.0
    return VideoMeta(duration_sec=dur, src_fps=fps, width=width, height=height)


def sample_frames(input_path: str, frames_dir: str, sample_fps: int, max_dim: int) -> None:
    """Extract frames from video at given FPS, resizing longest side to max_dim."""
    os.makedirs(frames_dir, exist_ok=True)
    vf = (
        f"fps={sample_fps},"
        f"scale='if(gt(iw,ih),{max_dim},-2)':'if(gt(iw,ih),-2,{max_dim})'"
    )
    out_pattern = os.path.join(frames_dir, "frame_%06d.png")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", vf, "-vsync", "0", out_pattern]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------
MODEL_CACHE_DIR = os.environ.get(
    "BARBUDDY_MODEL_CACHE",
    os.path.join(Path.home(), ".cache", "barbuddy", "models"),
)


def download_model(url: str, filename: str | None = None, expected_sha256: str | None = None) -> str:
    """
    Download a model file to the local cache directory.

    Returns the local file path.  Skips download if file already exists
    (and checksum matches, when provided).
    """
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1].split("?")[0]

    local_path = os.path.join(MODEL_CACHE_DIR, filename)

    if os.path.isfile(local_path):
        if expected_sha256 is None:
            return local_path
        h = hashlib.sha256(open(local_path, "rb").read()).hexdigest()
        if h == expected_sha256:
            return local_path
        print(f"[download] Checksum mismatch for {filename}, re-downloading …")

    print(f"[download] {url} → {local_path}")
    urllib.request.urlretrieve(url, local_path)

    if expected_sha256:
        h = hashlib.sha256(open(local_path, "rb").read()).hexdigest()
        if h != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: expected {expected_sha256}, got {h}"
            )

    return local_path


# ---------------------------------------------------------------------------
# Visualization helpers (3D-aware skeleton drawing)
# ---------------------------------------------------------------------------
# Colour palette: high-confidence = green, low = red
_CONF_COLORS = [
    (0, 0, 255),    # 0.0 – red
    (0, 128, 255),  # 0.3 – orange
    (0, 255, 255),  # 0.5 – yellow
    (0, 255, 128),  # 0.7 – lime
    (0, 255, 0),    # 1.0 – green
]


def _conf_to_color(conf: float) -> tuple[int, int, int]:
    """Map confidence [0, 1] to a BGR colour."""
    idx = min(int(conf * (len(_CONF_COLORS) - 1)), len(_CONF_COLORS) - 1)
    return _CONF_COLORS[max(0, idx)]


def draw_skeleton_on_frame(
    img: np.ndarray,
    keypoints: list[dict] | None,
    skeleton: list[tuple[int, int]] = COCO_SKELETON,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """
    Draw 2D/3D skeleton on image.

    Args:
        img: BGR image (H, W, 3)
        keypoints: list of dicts with keys {idx, x, y, conf} where x, y are normalised [0,1]
        skeleton: pairs of keypoint indices to connect
        conf_threshold: minimum confidence to draw

    Returns:
        img with skeleton drawn (modified in-place)
    """
    if keypoints is None:
        return img

    h, w = img.shape[:2]

    # Build a lookup: idx → (px, py, conf)
    kpt_map: dict[int, tuple[int, int, float]] = {}
    for kpt in keypoints:
        if kpt["conf"] < conf_threshold:
            continue
        px = int(kpt["x"] * w)
        py = int(kpt["y"] * h)
        kpt_map[kpt["idx"]] = (px, py, kpt["conf"])

    # Draw connections
    for a, b in skeleton:
        if a not in kpt_map or b not in kpt_map:
            continue
        ax, ay, ac = kpt_map[a]
        bx, by, bc = kpt_map[b]
        colour = _conf_to_color(min(ac, bc))
        cv2.line(img, (ax, ay), (bx, by), colour, 2)

    # Draw keypoints
    for idx, (px, py, conf) in kpt_map.items():
        colour = _conf_to_color(conf)
        cv2.circle(img, (px, py), 4, colour, -1)

    return img
