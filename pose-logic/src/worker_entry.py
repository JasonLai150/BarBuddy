"""
BarBuddy Worker — SQS polling mode.

Long-running process that polls an SQS queue for pose-analysis jobs.
Designed to run inside a Docker container on a GPU EC2 spot instance.

Supports two modes:
  - WORKER_MODE=sqs  → long-running SQS poller (default)
  - WORKER_MODE=once → single job from env vars (legacy / local testing)
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import boto3

import cv2


# ── Globals ────────────────────────────────────
_shutdown_requested = False

def _handle_signal(signum, frame):
    """Graceful shutdown on SIGTERM (spot interruption) or SIGINT."""
    global _shutdown_requested
    print(f"\n⚠️  Received signal {signum}, shutting down gracefully …")
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def must_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v.strip()


# ── CUDA warmup ────────────────────────────────
def warmup_cuda():
    """Pre-initialize CUDA to avoid first-inference latency."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🔥 CUDA warmup: {torch.cuda.get_device_name(0)}")
            torch.zeros(1, device="cuda")  # trigger CUDA context creation
            return "cuda"
        else:
            print("⚡ CUDA not available, using CPU")
            return "cpu"
    except ImportError:
        print("⚡ PyTorch not installed, using CPU")
        return "cpu"


# ── Single job processor ──────────────────────
def process_job(msg: dict, device: str):
    """
    Process a single pose-analysis job.

    Args:
        msg: Parsed SQS message body with jobId, rawS3Key, liftType, userId, results.
        device: 'cuda' or 'cpu'.
    """
    job_id = msg["jobId"]
    raw_key = msg["rawS3Key"]
    user_id = msg.get("userId", "anon")
    lift_type = msg.get("liftType", "unknown")
    results = msg.get("results", {})

    landmarks_key = results["landmarksKey"]
    summary_key = results["summaryKey"]
    viz_key = results["vizKey"]
    meta_key = results.get("metaKey")
    thumbnail_key = results.get("thumbnailKey")

    bucket = must_env("BUCKET_NAME")
    table = must_env("JOBS_TABLE_NAME")

    # Config knobs (from container env)
    sample_fps = os.environ.get("SAMPLE_FPS", "12")
    max_dim = os.environ.get("MAX_DIM", "720")
    max_frames = os.environ.get("MAX_FRAMES", "360")
    min_det = os.environ.get("MIN_DET", "0.5")
    min_trk = os.environ.get("MIN_TRK", "0.3")
    pose_backend = os.environ.get("POSE_BACKEND", "rtmpose")
    viz = os.environ.get("VIZ", "0") in ("1", "true", "True", "yes")
    viz_fps = os.environ.get("VIZ_FPS", "24")

    s3 = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    # Set status to PROCESSING
    ddb.update_item(
        TableName=table,
        Key={"jobId": {"S": job_id}},
        UpdateExpression="SET #s=:s, updatedAt=:u",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": {"S": "PROCESSING"},
            ":u": {"S": iso_now()},
        },
    )

    download_dir = "/tmp/download"
    outdir = "/tmp/out"

    # Clean up from previous job
    for d in [download_dir, outdir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    in_path = os.path.join(download_dir, "input.mp4")
    landmarks_local = os.path.join(outdir, "landmarks.json")
    summary_local = os.path.join(outdir, "summary.json")
    viz_local = os.path.join(outdir, "viz.mp4")
    thumbnail_local = os.path.join(outdir, "thumbnail.jpg")

    print(f"📥 Downloading s3://{bucket}/{raw_key}")
    s3.download_file(bucket, raw_key, in_path)

    # Phase 1: Extract frame 0 as early thumbnail and upload immediately
    if thumbnail_key:
        try:
            cap = cv2.VideoCapture(in_path)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                cv2.imwrite(thumbnail_local, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                s3.upload_file(thumbnail_local, bucket, thumbnail_key,
                               ExtraArgs={"ContentType": "image/jpeg"})
                print(f"🖼️  Early thumbnail uploaded → s3://{bucket}/{thumbnail_key}")
            else:
                print("⚠️  Could not read frame 0 for early thumbnail")
        except Exception as e:
            print(f"⚠️  Early thumbnail failed (non-fatal): {e}")

    # Run the 3-stage pipeline
    cmd = [
        "python", "-m", "src.main",
        "--input", in_path,
        "--outdir", outdir,
        "--job-id", job_id,
        "--user-id", user_id,
        "--lift-type", lift_type,
        "--sample-fps", str(sample_fps),
        "--max-dim", str(max_dim),
        "--max-frames", str(max_frames),
        "--min-det", str(min_det),
        "--min-trk", str(min_trk),
        "--pose-backend", pose_backend,
        "--device", device,
    ]
    if viz:
        cmd += ["--viz", "--viz-fps", str(viz_fps)]
    if thumbnail_key:
        cmd += ["--thumbnail", thumbnail_local]

    subprocess.check_call(cmd)

    # Validate outputs
    if not os.path.exists(landmarks_local):
        raise RuntimeError(f"Missing expected output file: {landmarks_local}")
    if not os.path.exists(summary_local):
        raise RuntimeError(f"Missing expected output file: {summary_local}")

    # Upload results to S3
    print(f"📤 Uploading results …")
    s3.upload_file(landmarks_local, bucket, landmarks_key,
                   ExtraArgs={"ContentType": "application/json"})
    s3.upload_file(summary_local, bucket, summary_key,
                   ExtraArgs={"ContentType": "application/json"})

    viz_uploaded = None
    if viz and os.path.exists(viz_local):
        s3.upload_file(viz_local, bucket, viz_key,
                       ExtraArgs={"ContentType": "video/mp4"})
        viz_uploaded = viz_key

    # Optional meta.json
    if meta_key:
        meta = {
            "jobId": job_id,
            "rawS3Key": raw_key,
            "userId": user_id,
            "liftType": lift_type,
            "generatedAt": iso_now(),
        }
        s3.put_object(
            Bucket=bucket,
            Key=meta_key,
            Body=(json.dumps(meta, indent=2) + "\n").encode("utf-8"),
            ContentType="application/json",
        )

    # Phase 2: Upload annotated thumbnail (overwrites early thumbnail)
    if thumbnail_key and os.path.exists(thumbnail_local):
        s3.upload_file(thumbnail_local, bucket, thumbnail_key,
                       ExtraArgs={"ContentType": "image/jpeg"})
        print(f"🖼️  Annotated thumbnail uploaded → s3://{bucket}/{thumbnail_key}")

    # Update DynamoDB → DONE
    update_expr = "SET #s=:s, updatedAt=:u, resultLandmarksKey=:lk, resultSummaryKey=:sk"
    expr_vals = {
        ":s": {"S": "DONE"},
        ":u": {"S": iso_now()},
        ":lk": {"S": landmarks_key},
        ":sk": {"S": summary_key},
    }

    if viz_uploaded:
        update_expr += ", resultVizKey=:vk"
        expr_vals[":vk"] = {"S": viz_uploaded}

    if meta_key:
        update_expr += ", resultMetaKey=:mk"
        expr_vals[":mk"] = {"S": meta_key}

    if thumbnail_key:
        update_expr += ", resultThumbnailKey=:tk"
        expr_vals[":tk"] = {"S": thumbnail_key}

    ddb.update_item(
        TableName=table,
        Key={"jobId": {"S": job_id}},
        UpdateExpression=update_expr,
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues=expr_vals,
    )

    print(f"✅ Job {job_id} complete")


# ── SQS polling loop ─────────────────────────
def poll_sqs(device: str):
    """
    Long-running SQS poller. Receives messages, processes jobs, deletes messages.
    Handles SIGTERM for graceful shutdown on spot interruption.
    """
    queue_url = must_env("QUEUE_URL")
    table = must_env("JOBS_TABLE_NAME")
    sqs = boto3.client("sqs")
    ddb = boto3.client("dynamodb")

    idle_since = time.time()
    idle_timeout = int(os.environ.get("IDLE_TIMEOUT_SEC", "300"))  # 5 min default

    print(f"🔄 Polling SQS: {queue_url}")
    print(f"   Idle timeout: {idle_timeout}s")

    while not _shutdown_requested:
        try:
            resp = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,          # long poll
                VisibilityTimeout=600,       # 10 min per job
                MessageAttributeNames=["All"],
            )
        except Exception as e:
            print(f"❌ SQS receive error: {e}")
            time.sleep(5)
            continue

        messages = resp.get("Messages", [])

        if not messages:
            elapsed = time.time() - idle_since
            if elapsed > idle_timeout:
                print(f"💤 No jobs for {idle_timeout}s, exiting (ASG will scale down)")
                break
            continue

        # Reset idle timer
        idle_since = time.time()

        for message in messages:
            receipt_handle = message["ReceiptHandle"]
            body = json.loads(message["Body"])
            job_id = body.get("jobId", "unknown")

            print(f"\n{'='*60}")
            print(f"📋 Processing job: {job_id}")
            print(f"{'='*60}")

            try:
                process_job(body, device)

                # Delete message on success
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle,
                )
                print(f"🗑️  Message deleted for job {job_id}")

            except Exception as e:
                print(f"❌ Job {job_id} failed: {e}")

                # Set DynamoDB status to ERROR
                try:
                    ddb.update_item(
                        TableName=table,
                        Key={"jobId": {"S": job_id}},
                        UpdateExpression="SET #s=:s, updatedAt=:u, errorMessage=:e",
                        ExpressionAttributeNames={"#s": "status"},
                        ExpressionAttributeValues={
                            ":s": {"S": "ERROR"},
                            ":u": {"S": iso_now()},
                            ":e": {"S": str(e)[:1000]},
                        },
                    )
                except Exception as ddb_err:
                    print(f"❌ Failed to update DynamoDB error status: {ddb_err}")

                # Message will become visible again after visibility timeout
                # and will be retried (up to maxReceiveCount before going to DLQ)
                # Optionally make it visible sooner:
                try:
                    sqs.change_message_visibility(
                        QueueUrl=queue_url,
                        ReceiptHandle=receipt_handle,
                        VisibilityTimeout=30,  # retry in 30s
                    )
                except Exception:
                    pass  # message will reappear after original visibility timeout

    print("👋 Worker shutting down")


# ── Legacy single-job mode ────────────────────
def run_once(device: str):
    """Run a single job from environment variables (legacy / local testing)."""
    msg = {
        "jobId": must_env("JOB_ID"),
        "rawS3Key": must_env("RAW_S3_KEY"),
        "userId": os.environ.get("USER_ID", "anon").strip(),
        "liftType": os.environ.get("LIFT_TYPE", "unknown").strip(),
        "results": {
            "landmarksKey": must_env("RESULT_LANDMARKS_KEY"),
            "summaryKey": must_env("RESULT_SUMMARY_KEY"),
            "vizKey": must_env("RESULT_VIZ_KEY"),
            "metaKey": os.environ.get("RESULT_META_KEY", "").strip() or None,
            "thumbnailKey": os.environ.get("RESULT_THUMBNAIL_KEY", "").strip() or None,
        },
    }
    process_job(msg, device)


# ── Main ──────────────────────────────────────
def main():
    mode = os.environ.get("WORKER_MODE", "once").lower()

    print(f"BarBuddy Worker v2")
    print(f"  Mode: {mode}")

    device = warmup_cuda()
    print(f"  Device: {device}")
    print()

    if mode == "sqs":
        poll_sqs(device)
    else:
        run_once(device)


if __name__ == "__main__":
    main()
