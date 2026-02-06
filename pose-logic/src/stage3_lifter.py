"""
Stage 3 — Temporal 3D Lifting (VideoPose3D).

Lifts 2D COCO keypoints to 3D using a temporal convolutional model
trained on Human3.6M.  Falls back to a simple heuristic when no
pretrained weights are available.

Architecture: dilated temporal convolutions with residual connections
(Pavllo et al., CVPR 2019 — "3D Human Pose Estimation in Video with
Temporal Convolutions and Semi-supervised Training").
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import (
    COCO_KEYPOINTS,
    H36M_JOINTS,
    coco_to_h36m,
    h36m_to_coco,
    download_model,
)


# ---------------------------------------------------------------------------
# VideoPose3D model architecture
# ---------------------------------------------------------------------------
class _TemporalBlock(nn.Module):
    """Single temporal convolution block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.25, causal: bool = False):
        super().__init__()
        self.causal = causal
        padding = (kernel_size - 1) * dilation // 2
        self.pad = (kernel_size - 1) * dilation if causal else padding

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.1)
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = self.residual(x)

        if self.causal:
            x = F.pad(x, (self.pad, 0))  # left-pad for causal
        else:
            x = F.pad(x, (self.pad, self.pad))

        out = self.dropout(F.relu(self.bn(self.conv(x))))

        # Trim to match residual length
        if out.shape[-1] != res.shape[-1]:
            out = out[..., :res.shape[-1]]

        return out + res


class VideoPose3DModel(nn.Module):
    """
    VideoPose3D temporal convolution model.

    For a receptive field of 243 frames: filter_widths = [3, 3, 3, 3, 3]
    For a receptive field of 81 frames:  filter_widths = [3, 3, 3, 3]
    For a receptive field of 27 frames:  filter_widths = [3, 3, 3]

    Input:  (B, T, 17*2)  — 2D keypoints in H36M ordering, normalised
    Output: (B, T', 17, 3) — 3D keypoints, root-relative
    """

    def __init__(
        self,
        num_joints: int = 17,
        in_features: int = 2,
        out_features: int = 3,
        filter_widths: list[int] | None = None,
        causal: bool = False,
        dropout: float = 0.25,
        channels: int = 1024,
    ):
        super().__init__()
        if filter_widths is None:
            filter_widths = [3, 3, 3, 3, 3]  # 243-frame receptive field

        self.num_joints = num_joints
        self.in_features = in_features
        self.out_features = out_features
        self.filter_widths = filter_widths

        # Compute receptive field
        self.receptive_field = self._compute_receptive_field(filter_widths)

        # Expand to channel dimension
        self.expand_conv = nn.Conv1d(num_joints * in_features, channels,
                                     filter_widths[0], bias=False)
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.expand_dropout = nn.Dropout(dropout)

        # Temporal blocks with increasing dilation
        layers = []
        dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            layers.append(_TemporalBlock(
                channels, channels, filter_widths[i],
                dilation=dilation, dropout=dropout, causal=causal,
            ))
            dilation *= filter_widths[i]
        self.layers = nn.ModuleList(layers)

        # Project to output
        self.shrink = nn.Conv1d(channels, num_joints * out_features, 1)

    @staticmethod
    def _compute_receptive_field(filter_widths: list[int]) -> int:
        rf = filter_widths[0]
        dilation = filter_widths[0]
        for fw in filter_widths[1:]:
            rf += (fw - 1) * dilation
            dilation *= fw
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J*2) — 2D keypoints over T frames

        Returns:
            (B, T', J, 3) — 3D predictions (T' ≤ T due to valid convolutions)
        """
        B, T, _ = x.shape
        x = x.permute(0, 2, 1)  # (B, J*2, T)

        # Expand
        pad = (self.filter_widths[0] - 1) // 2
        x = F.pad(x, (pad, pad))
        x = self.expand_dropout(F.relu(self.expand_bn(self.expand_conv(x))))

        # Temporal blocks
        for layer in self.layers:
            x = layer(x)

        # Shrink
        x = self.shrink(x)  # (B, J*3, T')
        x = x.permute(0, 2, 1)  # (B, T', J*3)
        x = x.reshape(x.shape[0], x.shape[1], self.num_joints, self.out_features)
        return x


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------
# Facebook Research pretrained checkpoint (243-frame receptive field)
_PRETRAINED_URL = (
    "https://dl.fbaipublicfiles.com/video-pose-3d/"
    "pretrained_h36m_detectron_coco.bin"
)


def _load_pretrained_weights(model: VideoPose3DModel, checkpoint_path: str) -> bool:
    """
    Load pretrained VideoPose3D weights.

    The official checkpoint has a slightly different key structure
    (prefixed with "model_pos."), so we strip the prefix.

    Returns True if weights were loaded successfully.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "model_pos" in checkpoint:
            state_dict = checkpoint["model_pos"]
        elif isinstance(checkpoint, dict) and any(k.startswith("model_pos.") for k in checkpoint):
            state_dict = {
                k.replace("model_pos.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("model_pos.")
            }
        else:
            state_dict = checkpoint

        # Try to load; non-strict to handle minor architecture differences
        model.load_state_dict(state_dict, strict=False)
        print(f"[stage3] Loaded pretrained weights from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"[stage3] Warning: could not load pretrained weights: {e}")
        return False


# ---------------------------------------------------------------------------
# Public Lifter3D class
# ---------------------------------------------------------------------------
class Lifter3D:
    """
    Temporal 3D pose lifter.

    Converts per-frame 2D COCO keypoints into 3D using a temporal
    convolutional model (VideoPose3D architecture).

    Usage:
        lifter = Lifter3D(device="cuda")
        frames_3d = lifter.lift(frames_2d, sample_fps=12)
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        filter_widths: list[int] | None = None,
        channels: int = 1024,
    ):
        self.device = device
        self.filter_widths = filter_widths or [3, 3, 3, 3, 3]

        self.model = VideoPose3DModel(
            num_joints=17,
            in_features=2,
            out_features=3,
            filter_widths=self.filter_widths,
            channels=channels,
        )
        self.receptive_field = self.model.receptive_field
        self.has_weights = False

        # Try to load pretrained weights
        if model_path and os.path.isfile(model_path):
            self.has_weights = _load_pretrained_weights(self.model, model_path)
        else:
            # Try downloading the pretrained checkpoint
            try:
                ckpt_path = download_model(_PRETRAINED_URL, filename="videopose3d_h36m_cpn.bin")
                self.has_weights = _load_pretrained_weights(self.model, ckpt_path)
            except Exception as e:
                print(f"[stage3] Could not download pretrained model: {e}")
                print("[stage3] Falling back to heuristic 3D estimation.")

        self.model.to(device)
        self.model.eval()

    def lift(self, frames_2d: list[dict], sample_fps: int) -> list[dict]:
        """
        Lift 2D keypoints to 3D.

        Args:
            frames_2d: list of per-frame dicts:
                {
                    "t": float,          # timestamp in seconds
                    "keypoints": [...] | None,  # list of 17 COCO kpt dicts
                }
            sample_fps: sampling rate of the frames

        Returns:
            list of per-frame dicts:
                {
                    "t": float,
                    "landmarks": [...],   # list of 17 dicts {idx, name, x, y, z, conf}
                    "confidence": float,  # frame-level average confidence
                }
        """
        n_frames = len(frames_2d)
        if n_frames == 0:
            return []

        # ---- Build (N, 17, 3) array: [x, y, conf] ----
        kpts_2d = np.zeros((n_frames, 17, 3), dtype=np.float32)
        valid_mask = np.zeros(n_frames, dtype=bool)

        for i, frame in enumerate(frames_2d):
            kp = frame.get("keypoints")
            if kp is None:
                continue
            valid_mask[i] = True
            for pt in kp:
                j = pt["idx"]
                if 0 <= j < 17:
                    kpts_2d[i, j, 0] = pt["x"]
                    kpts_2d[i, j, 1] = pt["y"]
                    kpts_2d[i, j, 2] = pt["conf"]

        # Interpolate missing frames (linear on xy, zero conf)
        kpts_2d = self._interpolate_missing(kpts_2d, valid_mask)

        if self.has_weights:
            kpts_3d = self._lift_with_model(kpts_2d)
        else:
            kpts_3d = self._lift_heuristic(kpts_2d)

        # ---- Build output ----
        # Use original 2D x/y (normalised image coords) for visualization,
        # and only the model's z for depth.  The 3D model output is
        # root-relative and NOT in image space.
        frames_3d: list[dict] = []
        for i in range(n_frames):
            landmarks = []
            confs = []
            for j in range(17):
                c = float(kpts_2d[i, j, 2]) if valid_mask[i] else 0.0
                confs.append(c)
                landmarks.append({
                    "idx": j,
                    "name": COCO_KEYPOINTS[j],
                    "x": float(kpts_2d[i, j, 0]),   # original 2D (image-space)
                    "y": float(kpts_2d[i, j, 1]),   # original 2D (image-space)
                    "z": float(kpts_3d[i, j, 2]),   # depth from 3D model
                    "conf": c,
                })
            frames_3d.append({
                "t": frames_2d[i]["t"],
                "landmarks": landmarks,
                "confidence": float(np.mean(confs)) if confs else 0.0,
            })

        return frames_3d

    # ------------------------------------------------------------------
    # Model-based 3D lifting
    # ------------------------------------------------------------------
    def _lift_with_model(self, kpts_2d: np.ndarray) -> np.ndarray:
        """
        Use the VideoPose3D model to lift 2D → 3D.

        Args:
            kpts_2d: (N, 17, 3) COCO keypoints [x, y, conf]

        Returns:
            kpts_3d: (N, 17, 3) COCO keypoints [x, y, z]
        """
        N = kpts_2d.shape[0]
        xy = kpts_2d[:, :, :2].copy()  # (N, 17, 2) — just x, y

        # Convert COCO → H36M ordering for the model
        xy_h36m = coco_to_h36m(xy)  # (N, 17, 2)

        # Normalise: centre on hip, scale by image coords
        hip = xy_h36m[:, 0:1, :]  # (N, 1, 2) — root joint
        xy_h36m_centred = xy_h36m - hip

        # Reshape for model: (1, N, 17*2)
        model_input = xy_h36m_centred.reshape(1, N, 17 * 2)

        # Pad to at least receptive_field frames
        rf = self.receptive_field
        if N < rf:
            pad_left = (rf - N) // 2
            pad_right = rf - N - pad_left
            model_input = np.pad(model_input, ((0, 0), (pad_left, pad_right), (0, 0)), mode="reflect")
        else:
            pad_left = 0

        with torch.no_grad():
            x = torch.from_numpy(model_input).float().to(self.device)
            pred = self.model(x)  # (1, T', 17, 3)
            pred = pred.cpu().numpy()[0]  # (T', 17, 3)

        # Extract the frames corresponding to our input
        # The model outputs fewer frames than input (valid convolution)
        T_out = pred.shape[0]
        T_in = model_input.shape[1]
        offset = (T_in - T_out) // 2

        # Map back to original frame indices
        kpts_3d_h36m = np.zeros((N, 17, 3), dtype=np.float32)
        for i in range(N):
            src_idx = i + pad_left - offset
            src_idx = max(0, min(T_out - 1, src_idx))
            kpts_3d_h36m[i] = pred[src_idx]

        # De-normalise: add back hip position (xy), keep z as-is
        kpts_3d_h36m[:, :, :2] += hip

        # Convert H36M → COCO ordering
        kpts_3d_coco = h36m_to_coco(kpts_3d_h36m)

        return kpts_3d_coco

    # ------------------------------------------------------------------
    # Heuristic fallback (no model weights)
    # ------------------------------------------------------------------
    @staticmethod
    def _lift_heuristic(kpts_2d: np.ndarray) -> np.ndarray:
        """
        Simple heuristic 3D: use 2D x/y, estimate z from joint hierarchy.

        This is a rough approximation that still provides *some* depth
        ordering for angle computation.  Upgrade to real weights ASAP.
        """
        N, J, _ = kpts_2d.shape
        kpts_3d = np.zeros((N, J, 3), dtype=np.float32)
        kpts_3d[:, :, 0] = kpts_2d[:, :, 0]  # x
        kpts_3d[:, :, 1] = kpts_2d[:, :, 1]  # y

        # Heuristic: estimate relative depth from torso proportions
        # Shoulder width → rough body scale → approximate z offsets
        for i in range(N):
            ls = kpts_2d[i, 5, :2]  # left shoulder
            rs = kpts_2d[i, 6, :2]  # right shoulder
            shoulder_w = np.linalg.norm(ls - rs)

            if shoulder_w < 1e-4:
                continue

            # Arms and legs get slight z offsets based on x deviation from torso centre
            torso_x = (ls[0] + rs[0]) / 2.0
            for j in range(J):
                dx = kpts_2d[i, j, 0] - torso_x
                kpts_3d[i, j, 2] = -abs(dx) * 0.3  # further from midline → slightly forward

        return kpts_3d

    # ------------------------------------------------------------------
    # Temporal interpolation for missing frames
    # ------------------------------------------------------------------
    @staticmethod
    def _interpolate_missing(kpts: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Linearly interpolate missing frames in the keypoint sequence.

        Args:
            kpts:  (N, 17, 3) — [x, y, conf]
            valid: (N,) bool mask

        Returns:
            kpts with missing frames filled via interpolation.
        """
        N = kpts.shape[0]
        if valid.all():
            return kpts

        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            return kpts  # nothing to interpolate from

        for i in range(N):
            if valid[i]:
                continue

            # Find nearest valid frames before and after
            before = valid_indices[valid_indices < i]
            after = valid_indices[valid_indices > i]

            if len(before) > 0 and len(after) > 0:
                b_idx, a_idx = before[-1], after[0]
                alpha = (i - b_idx) / (a_idx - b_idx)
                kpts[i] = (1 - alpha) * kpts[b_idx] + alpha * kpts[a_idx]
                kpts[i, :, 2] = 0.0  # zero confidence for interpolated
            elif len(before) > 0:
                kpts[i] = kpts[before[-1]]
                kpts[i, :, 2] = 0.0
            elif len(after) > 0:
                kpts[i] = kpts[after[0]]
                kpts[i, :, 2] = 0.0

        return kpts
