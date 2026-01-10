import os
import json
import subprocess
from datetime import datetime, timezone

import boto3


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def must_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v.strip()


def main():
    # Required env
    job_id = must_env("JOB_ID")
    raw_key = must_env("RAW_S3_KEY")
    bucket = must_env("BUCKET_NAME")
    table = must_env("JOBS_TABLE_NAME")

    # Explicit output keys (required in the new design)
    landmarks_key = must_env("RESULT_LANDMARKS_KEY")
    summary_key = must_env("RESULT_SUMMARY_KEY")
    viz_key = must_env("RESULT_VIZ_KEY")
    meta_key = os.environ.get("RESULT_META_KEY")
    meta_key = meta_key.strip() if meta_key else None

    # Optional env
    user_id = os.environ.get("USER_ID", "anon").strip()
    lift_type = os.environ.get("LIFT_TYPE", "unknown").strip()

    # Config knobs (optional)
    sample_fps = os.environ.get("SAMPLE_FPS", "12")
    max_dim = os.environ.get("MAX_DIM", "720")
    max_frames = os.environ.get("MAX_FRAMES", "360")
    model_complexity = os.environ.get("MODEL_COMPLEXITY", "1")
    min_det = os.environ.get("MIN_DET", "0.5")
    min_trk = os.environ.get("MIN_TRK", "0.5")
    viz = os.environ.get("VIZ", "0") in ("1", "true", "True", "yes")
    viz_fps = os.environ.get("VIZ_FPS", "24")

    s3 = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    download_dir = "/tmp/download"
    os.makedirs(download_dir, exist_ok=True)
    in_path = os.path.join(download_dir, "input.mp4")

    outdir = "/tmp/out"
    os.makedirs(outdir, exist_ok=True)

    # Local outputs produced by src.main
    landmarks_local = os.path.join(outdir, "landmarks.json")
    summary_local = os.path.join(outdir, "summary.json")
    viz_local = os.path.join(outdir, "viz.mp4")

    print("Starting worker")
    print(" job_id:", job_id)
    print(" raw_key:", raw_key)
    print(" outputs:")
    print("  landmarks_key:", landmarks_key)
    print("  summary_key:", summary_key)
    print("  viz_key:", viz_key)
    print("  meta_key:", meta_key)

    # Download raw video
    s3.download_file(bucket, raw_key, in_path)

    # Run your teammate's module
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
        "--model-complexity", str(model_complexity),
        "--min-det", str(min_det),
        "--min-trk", str(min_trk),
    ]
    if viz:
        cmd += ["--viz", "--viz-fps", str(viz_fps)]

    subprocess.check_call(cmd)

    # Upload required JSON outputs
    if not os.path.exists(landmarks_local):
        raise RuntimeError(f"Missing expected output file: {landmarks_local}")
    if not os.path.exists(summary_local):
        raise RuntimeError(f"Missing expected output file: {summary_local}")

    s3.upload_file(
        landmarks_local,
        bucket,
        landmarks_key,
        ExtraArgs={"ContentType": "application/json"},
    )
    s3.upload_file(
        summary_local,
        bucket,
        summary_key,
        ExtraArgs={"ContentType": "application/json"},
    )

    # Upload viz if requested and produced
    viz_uploaded = None
    if viz and os.path.exists(viz_local):
        s3.upload_file(
            viz_local,
            bucket,
            viz_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
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

    # Update DynamoDB pointers (no status change; Step Functions owns status)
    update_expr = "SET updatedAt=:u, resultLandmarksKey=:lk, resultSummaryKey=:sk"
    expr_vals = {
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

    ddb.update_item(
        TableName=table,
        Key={"jobId": {"S": job_id}},
        UpdateExpression=update_expr,
        ExpressionAttributeValues=expr_vals,
    )

    print("Uploaded:", landmarks_key, summary_key, viz_uploaded, meta_key)


if __name__ == "__main__":
    main()
