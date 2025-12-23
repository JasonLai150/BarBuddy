from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import orjson


@dataclass
class VideoMeta:
    duration_sec: float
    src_fps: float
    width: int
    height: int


def probe_video(path: str) -> VideoMeta:
    # ffprobe prints key=value lines
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
    os.makedirs(frames_dir, exist_ok=True)
    # limit longest side to max_dim, keep aspect
    vf = (
        f"fps={sample_fps},"
        f"scale='if(gt(iw,ih),{max_dim},-2)':'if(gt(iw,ih),-2,{max_dim})'"
    )
    out_pattern = os.path.join(frames_dir, "frame_%06d.png")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", vf, "-vsync", "0", out_pattern]
    subprocess.check_call(cmd)


def run_blazepose(frames_dir: str, sample_fps: int, model_complexity: int,
                  min_det: float, min_trk: float, time_budget_sec: int = 0):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))

    PoseLandmark = mp.solutions.pose.PoseLandmark
    landmark_names = [lm.name for lm in PoseLandmark]

    t0 = time.time()
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_trk,
    )

    frames_out = []
    with_pose = 0

    for i, p in enumerate(frame_paths):
        if time_budget_sec and (time.time() - t0) > time_budget_sec:
            break

        img = cv2.imread(p)
        if img is None:
            frames_out.append({"t": i / sample_fps, "landmarks": None})
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        t = i / sample_fps
        if res.pose_landmarks is None:
            frames_out.append({"t": t, "landmarks": None})
        else:
            lms = []
            for j, lm in enumerate(res.pose_landmarks.landmark):
                lms.append({
                    "name": landmark_names[j],
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility),
                })
            frames_out.append({"t": t, "landmarks": lms})
            with_pose += 1

    pose.close()
    dt = time.time() - t0
    return frames_out, {"framesTotal": len(frames_out), "framesWithPose": with_pose, "seconds": dt}


def draw_viz(frames_dir: str, frames_out: list[dict], out_mp4: str, viz_fps: int):
    """
    Simple custom skeleton drawing (no mediapipe protobuf dependency).
    Draws points + some connections.
    """
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(out_mp4), "_viz_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    # subset of connections (enough for MVP)
    # using PoseLandmark indices
    P = mp.solutions.pose.PoseLandmark
    connections = [
        (P.LEFT_SHOULDER.value, P.RIGHT_SHOULDER.value),
        (P.LEFT_HIP.value, P.RIGHT_HIP.value),

        (P.LEFT_SHOULDER.value, P.LEFT_ELBOW.value),
        (P.LEFT_ELBOW.value, P.LEFT_WRIST.value),
        (P.RIGHT_SHOULDER.value, P.RIGHT_ELBOW.value),
        (P.RIGHT_ELBOW.value, P.RIGHT_WRIST.value),

        (P.LEFT_HIP.value, P.LEFT_KNEE.value),
        (P.LEFT_KNEE.value, P.LEFT_ANKLE.value),
        (P.RIGHT_HIP.value, P.RIGHT_KNEE.value),
        (P.RIGHT_KNEE.value, P.RIGHT_ANKLE.value),

        (P.LEFT_SHOULDER.value, P.LEFT_HIP.value),
        (P.RIGHT_SHOULDER.value, P.RIGHT_HIP.value),
    ]

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    n = min(len(frame_paths), len(frames_out))

    for i in range(n):
        img = cv2.imread(frame_paths[i])
        if img is None:
            continue
        h, w = img.shape[:2]
        item = frames_out[i]
        lms = item.get("landmarks")

        if lms:
            # draw connections
            for a, b in connections:
                la, lb = lms[a], lms[b]
                if la["visibility"] < 0.4 or lb["visibility"] < 0.4:
                    continue
                ax, ay = int(la["x"] * w), int(la["y"] * h)
                bx, by = int(lb["x"] * w), int(lb["y"] * h)
                cv2.line(img, (ax, ay), (bx, by), (0, 255, 0), 2)

            # draw points
            for lm in lms:
                if lm["visibility"] < 0.4:
                    continue
                x, y = int(lm["x"] * w), int(lm["y"] * h)
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        out_png = os.path.join(tmp_dir, f"viz_{i:06d}.png")
        cv2.imwrite(out_png, img)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(viz_fps),
        "-i", os.path.join(tmp_dir, "viz_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    subprocess.check_call(cmd)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input mp4")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--job-id", default="local-job")
    ap.add_argument("--user-id", default="")
    ap.add_argument("--lift-type", default="unknown")

    ap.add_argument("--sample-fps", type=int, default=12)
    ap.add_argument("--max-dim", type=int, default=720)
    ap.add_argument("--max-frames", type=int, default=360)
    ap.add_argument("--model-complexity", type=int, default=1)
    ap.add_argument("--min-det", type=float, default=0.5)
    ap.add_argument("--min-trk", type=float, default=0.5)
    ap.add_argument("--time-budget-sec", type=int, default=0)

    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--viz-fps", type=int, default=24)

    args = ap.parse_args()

    workdir = "/tmp/work"
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)

    input_local = os.path.join(workdir, "input.mp4")
    shutil.copyfile(args.input, input_local)

    meta = probe_video(input_local)

    # auto-adjust fps to fit max_frames
    if meta.duration_sec > 0 and (meta.duration_sec * args.sample_fps) > args.max_frames:
        args.sample_fps = max(1, int(args.max_frames / meta.duration_sec))

    frames_dir = os.path.join(workdir, "frames")
    t0 = time.time()
    sample_frames(input_local, frames_dir, args.sample_fps, args.max_dim)
    sample_sec = time.time() - t0

    t1 = time.time()
    frames_out, pose_stats = run_blazepose(
        frames_dir, args.sample_fps, args.model_complexity,
        args.min_det, args.min_trk, args.time_budget_sec
    )
    pose_sec = time.time() - t1

    os.makedirs(args.outdir, exist_ok=True)
    landmarks_path = os.path.join(args.outdir, "landmarks.json")
    summary_path = os.path.join(args.outdir, "summary.json")
    viz_path = os.path.join(args.outdir, "viz.mp4")

    metrics = {
        "video": {
            "durationSec": meta.duration_sec,
            "srcFps": meta.src_fps,
            "width": meta.width,
            "height": meta.height,
        },
        "timing": {
            "sampleSec": sample_sec,
            "poseSec": pose_sec,
        },
        "pose": pose_stats,
        "config": {
            "sampleFps": args.sample_fps,
            "maxDim": args.max_dim,
            "modelComplexity": args.model_complexity,
            "maxFrames": args.max_frames,
        }
    }

    result = {
        "jobId": args.job_id,
        "userId": args.user_id or None,
        "liftType": args.lift_type or None,
        "sampleFps": float(args.sample_fps),
        "frames": frames_out,
        "metrics": metrics,
    }

    with open(landmarks_path, "wb") as f:
        f.write(orjson.dumps(result))

    if args.viz:
        draw_viz(frames_dir, frames_out, viz_path, args.viz_fps)

    summary = {
        "jobId": args.job_id,
        "status": "DONE",
        "output": {
            "landmarks": os.path.abspath(landmarks_path),
            "viz": os.path.abspath(viz_path) if args.viz else None,
        },
        "metrics": metrics,
    }

    with open(summary_path, "wb") as f:
        f.write(orjson.dumps(summary))

    shutil.rmtree(workdir, ignore_errors=True)
    print(f"Wrote: {landmarks_path}")
    print(f"Wrote: {summary_path}")
    if args.viz:
        print(f"Wrote: {viz_path}")


if __name__ == "__main__":
    main()
