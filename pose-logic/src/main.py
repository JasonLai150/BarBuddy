"""
BarBuddy 3-stage temporal 3D pose analysis pipeline.

Stages:
  0. Probe video + sample frames (ffmpeg)
  1. Person detection (YOLOv8)
  2. 2D pose estimation (RTMPose / YOLOv8-Pose)
  3. Temporal 3D lifting (VideoPose3D)
  4. Post-processing (smoothing + kinematic constraints)
  5. Metrics (joint angles, ROM, rep counting)

Output:  landmarks.json, summary.json, viz.mp4 (optional)
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import orjson

from src.utils import (
    COCO_SKELETON,
    VideoMeta,
    draw_skeleton_on_frame,
    probe_video,
    sample_frames,
)
from src.stage1_detector import PersonDetector
from src.stage2_pose import PoseEstimator2D
from src.stage3_lifter import Lifter3D
from src.postprocess import smooth_and_constrain
from src.metrics import compute_rep_metrics, LIFT_HIGHLIGHT_SEGMENTS


# ---------------------------------------------------------------------------
# Visualization (3D-aware skeleton video)
# ---------------------------------------------------------------------------
def draw_viz(
    frames_dir: str,
    frames_3d: list[dict],
    out_mp4: str,
    viz_fps: int,
    lift_type: str = "unknown",
    per_frame_valid: list[bool] | None = None,
    rep_count: int = 0,
) -> None:
    """Draw 3D-aware skeleton overlay with colour-coded lift highlights."""
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(out_mp4) or ".", "_viz_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    n = min(len(frame_paths), len(frames_3d))

    highlight_segs = LIFT_HIGHLIGHT_SEGMENTS.get(lift_type)
    if per_frame_valid is None:
        per_frame_valid = [False] * n

    running_reps = 0  # count reps completed so far for the HUD
    prev_valid = False
    was_in_rep = False

    for i in range(n):
        img = cv2.imread(frame_paths[i])
        if img is None:
            continue

        is_valid = per_frame_valid[i] if i < len(per_frame_valid) else False

        # Track running rep count for HUD
        if prev_valid and not is_valid and was_in_rep:
            running_reps += 1
        if is_valid:
            was_in_rep = True
        if not is_valid and not prev_valid:
            was_in_rep = False
        prev_valid = is_valid

        lms = frames_3d[i].get("landmarks")
        draw_skeleton_on_frame(
            img, lms, COCO_SKELETON,
            conf_threshold=0.3,
            highlight_segments=highlight_segs,
            is_valid=is_valid,
        )

        # HUD: timestamp, confidence, rep count, valid indicator
        t = frames_3d[i].get("t", 0.0)
        conf = frames_3d[i].get("confidence", 0.0)
        h_img, w_img = img.shape[:2]

        # Top-left: time + confidence
        cv2.putText(
            img, f"t={t:.2f}s  conf={conf:.2f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        # Top-right: rep counter
        rep_text = f"Reps: {running_reps}/{rep_count}"
        (tw, _), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            img, rep_text,
            (w_img - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2,
        )

        # Valid state indicator
        if is_valid:
            cv2.putText(
                img, "VALID",
                (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2,
            )

        out_png = os.path.join(tmp_dir, f"viz_{i:06d}.png")
        cv2.imwrite(out_png, img)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(viz_fps),
        "-i", os.path.join(tmp_dir, "viz_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="BarBuddy 3-stage temporal 3D pose pipeline"
    )
    # I/O
    ap.add_argument("--input", required=True, help="Path to input mp4")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--job-id", default="local-job")
    ap.add_argument("--user-id", default="")
    ap.add_argument("--lift-type", default="unknown")

    # Frame sampling
    ap.add_argument("--sample-fps", type=int, default=12)
    ap.add_argument("--max-dim", type=int, default=720)
    ap.add_argument("--max-frames", type=int, default=360)

    # Detection / pose config
    ap.add_argument("--min-det", type=float, default=0.5,
                    help="Minimum detection confidence (Stage 1)")
    ap.add_argument("--min-trk", type=float, default=0.3,
                    help="Minimum keypoint confidence (Stage 2)")
    ap.add_argument("--pose-backend", default="rtmpose",
                    choices=["rtmpose", "yolov8"],
                    help="2D pose estimation backend")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="Inference device")

    # Visualization
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--viz-fps", type=int, default=24)

    args = ap.parse_args()

    # ---- Auto-detect device ----
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[MAIN] CUDA not available, falling back to CPU")
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"

    print("=" * 60)
    print("  BarBuddy 3D Pose Pipeline v2")
    print("=" * 60)
    print(f"  Input:   {args.input}")
    print(f"  Device:  {args.device}")
    print(f"  Backend: {args.pose_backend}")
    print(f"  Lift:    {args.lift_type}")
    print()

    workdir = "/tmp/work"
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)

    input_local = os.path.join(workdir, "input.mp4")
    shutil.copyfile(args.input, input_local)

    # ===================================================================
    # Stage 0 — Probe video + sample frames
    # ===================================================================
    print("[STAGE 0] Probing video + sampling frames …")
    t0 = time.time()

    meta = probe_video(input_local)
    print(f"  Video: {meta.width}×{meta.height} @ {meta.src_fps:.1f} FPS, "
          f"{meta.duration_sec:.1f}s")

    # Auto-adjust FPS to stay under max_frames
    if meta.duration_sec > 0 and (meta.duration_sec * args.sample_fps) > args.max_frames:
        args.sample_fps = max(1, int(args.max_frames / meta.duration_sec))
        print(f"  Auto-adjusted sample_fps → {args.sample_fps}")

    frames_dir = os.path.join(workdir, "frames")
    sample_frames(input_local, frames_dir, args.sample_fps, args.max_dim)
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    n_frames = len(frame_paths)
    t_stage0 = time.time() - t0
    print(f"  Sampled {n_frames} frames in {t_stage0:.1f}s")

    # ===================================================================
    # Stage 1 — Person Detection (YOLOv8)
    # ===================================================================
    print(f"\n[STAGE 1] Person detection (YOLOv8) …")
    t1 = time.time()

    detector = PersonDetector(device=args.device, confidence=args.min_det)
    all_bboxes: list[list[dict]] = []

    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        if img is None:
            all_bboxes.append([])
            continue
        bboxes = detector.detect(img)
        all_bboxes.append(bboxes)

    t_stage1 = time.time() - t1
    frames_with_person = sum(1 for b in all_bboxes if b)
    print(f"  Detected person in {frames_with_person}/{n_frames} frames "
          f"({t_stage1:.1f}s)")

    # ===================================================================
    # Stage 2 — 2D Pose Estimation
    # ===================================================================
    backend_name = args.pose_backend.upper()
    print(f"\n[STAGE 2] 2D pose estimation ({backend_name}) …")
    t2 = time.time()

    pose_2d = PoseEstimator2D(backend=args.pose_backend, device=args.device)
    frames_2d: list[dict] = []

    for i, frame_path in enumerate(frame_paths):
        img = cv2.imread(frame_path)
        t_sec = i / args.sample_fps

        if img is None or not all_bboxes[i]:
            frames_2d.append({"t": t_sec, "keypoints": None})
            continue

        # Use the primary person bbox
        primary_bbox = detector.select_primary(all_bboxes[i])
        bboxes_for_pose = [primary_bbox] if primary_bbox else all_bboxes[i][:1]

        keypoints = pose_2d.estimate(img, bboxes_for_pose)
        frames_2d.append({"t": t_sec, "keypoints": keypoints})

    t_stage2 = time.time() - t2
    frames_with_pose = sum(1 for f in frames_2d if f["keypoints"] is not None)
    print(f"  2D pose in {frames_with_pose}/{n_frames} frames ({t_stage2:.1f}s)")

    # ===================================================================
    # Stage 3 — Temporal 3D Lifting (VideoPose3D)
    # ===================================================================
    print(f"\n[STAGE 3] Temporal 3D lifting (VideoPose3D) …")
    t3 = time.time()

    lifter = Lifter3D(device=args.device)
    frames_3d = lifter.lift(frames_2d, args.sample_fps)

    t_stage3 = time.time() - t3
    print(f"  3D lifting complete ({t_stage3:.1f}s)")
    if lifter.has_weights:
        print(f"  Using pretrained VideoPose3D model (receptive field: "
              f"{lifter.receptive_field} frames)")
    else:
        print(f"  ⚠ Using heuristic fallback — download weights for best results")

    # ===================================================================
    # Stage 4 — Post-processing (smoothing + constraints)
    # ===================================================================
    print(f"\n[STAGE 4] Post-processing (smoothing + constraints) …")
    t4 = time.time()

    frames_3d_smooth = smooth_and_constrain(frames_3d, args.sample_fps)

    t_stage4 = time.time() - t4
    print(f"  Post-processing complete ({t_stage4:.1f}s)")

    # ===================================================================
    # Metrics — joint angles, ROM, rep counting
    # ===================================================================
    print(f"\n[METRICS] Computing rep metrics for '{args.lift_type}' …")
    t5 = time.time()

    rep_metrics = compute_rep_metrics(
        frames_3d_smooth,
        lift_type=args.lift_type,
        sample_fps=args.sample_fps,
    )
    t_metrics = time.time() - t5
    print(f"  Reps detected: {rep_metrics['reps']}")
    print(f"  Avg ROM: {rep_metrics['avgRom']:.1f}°")
    print(f"  Confidence: {rep_metrics['confidence']:.3f}")

    # ===================================================================
    # Write outputs
    # ===================================================================
    os.makedirs(args.outdir, exist_ok=True)
    landmarks_path = os.path.join(args.outdir, "landmarks.json")
    summary_path = os.path.join(args.outdir, "summary.json")
    viz_path = os.path.join(args.outdir, "viz.mp4")

    timing = {
        "stage0SampleSec": round(t_stage0, 2),
        "stage1DetSec": round(t_stage1, 2),
        "stage2PoseSec": round(t_stage2, 2),
        "stage3LifterSec": round(t_stage3, 2),
        "stage4PostprocSec": round(t_stage4, 2),
        "metricsSec": round(t_metrics, 2),
        "totalSec": round(time.time() - t0, 2),
    }

    pose_stats = {
        "framesTotal": n_frames,
        "framesWithPerson": frames_with_person,
        "framesWithPose": frames_with_pose,
        "poseBackend": args.pose_backend,
        "lifterHasWeights": lifter.has_weights,
        "receptiveField": lifter.receptive_field,
    }

    metrics_block = {
        "video": {
            "durationSec": meta.duration_sec,
            "srcFps": meta.src_fps,
            "width": meta.width,
            "height": meta.height,
        },
        "timing": timing,
        "pose": pose_stats,
        "config": {
            "sampleFps": args.sample_fps,
            "maxDim": args.max_dim,
            "maxFrames": args.max_frames,
            "device": args.device,
            "poseBackend": args.pose_backend,
            "minDet": args.min_det,
            "minTrk": args.min_trk,
        },
    }

    # landmarks.json — full per-frame 3D keypoints
    landmarks_result = {
        "version": 2,   # v2 = 3D COCO-17 format (v1 = 2D MediaPipe-33)
        "jobId": args.job_id,
        "userId": args.user_id or None,
        "liftType": args.lift_type or None,
        "sampleFps": float(args.sample_fps),
        "frames": frames_3d_smooth,
        "metrics": metrics_block,
    }

    with open(landmarks_path, "wb") as f:
        f.write(orjson.dumps(landmarks_result))

    # viz.mp4 (optional)
    if args.viz:
        print(f"\n[VIZ] Drawing skeleton overlay …")
        draw_viz(
            frames_dir, frames_3d_smooth, viz_path, args.viz_fps,
            lift_type=args.lift_type,
            per_frame_valid=rep_metrics.get("perFrameValid"),
            rep_count=rep_metrics["reps"],
        )
        print(f"  Wrote: {viz_path}")

    # summary.json — compact results for frontend
    summary = {
        "version": 2,
        "jobId": args.job_id,
        "status": "DONE",
        "liftType": args.lift_type,
        "reps": rep_metrics["reps"],
        "repDetails": rep_metrics["repDetails"],
        "avgRom": rep_metrics["avgRom"],
        "peakRom": rep_metrics["peakRom"],
        "primaryAngle": rep_metrics["primaryAngle"],
        "confidence": rep_metrics["confidence"],
        "output": {
            "landmarks": os.path.abspath(landmarks_path),
            "viz": os.path.abspath(viz_path) if args.viz else None,
        },
        "metrics": metrics_block,
    }

    with open(summary_path, "wb") as f:
        f.write(orjson.dumps(summary))

    # Cleanup
    shutil.rmtree(workdir, ignore_errors=True)

    print()
    print("=" * 60)
    print("  ✅ Pipeline complete!")
    print(f"  Landmarks: {landmarks_path}")
    print(f"  Summary:   {summary_path}")
    if args.viz:
        print(f"  Viz:       {viz_path}")
    print(f"  Total:     {timing['totalSec']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
