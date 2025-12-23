import os
import json
import subprocess
from datetime import datetime, timezone

import boto3

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def main():
    # Required env
    job_id = os.environ["JOB_ID"].strip()
    raw_key = os.environ["RAW_S3_KEY"].strip()
    bucket = os.environ["BUCKET_NAME"].strip()
    table = os.environ["JOBS_TABLE_NAME"].strip()

    # Optional env
    user_id = os.environ.get("USER_ID", "anon")
    lift_type = os.environ.get("LIFT_TYPE", "unknown")

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

    # Mark processing
    ddb.update_item(
        TableName=table,
        Key={"jobId": {"S": job_id}},
        UpdateExpression="SET #s=:s, updatedAt=:u",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": {"S": "POSE_RUNNING"},
            ":u": {"S": iso_now()},
        },
    )

    # Download
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

    # Upload outputs to S3
    landmarks_local = os.path.join(outdir, "landmarks.json")
    summary_local = os.path.join(outdir, "summary.json")
    viz_local = os.path.join(outdir, "viz.mp4")

    landmarks_key = f"results/{job_id}/landmarks.json"
    summary_key = f"results/{job_id}/summary.json"
    viz_key = f"results/{job_id}/viz.mp4"

    s3.upload_file(landmarks_local, bucket, landmarks_key, ExtraArgs={"ContentType": "application/json"})
    s3.upload_file(summary_local, bucket, summary_key, ExtraArgs={"ContentType": "application/json"})
    viz_uploaded = None
    if viz and os.path.exists(viz_local):
        s3.upload_file(viz_local, bucket, viz_key, ExtraArgs={"ContentType": "video/mp4"})
        viz_uploaded = viz_key

    # Update DDB pointers
    ddb.update_item(
        TableName=table,
        Key={"jobId": {"S": job_id}},
        UpdateExpression="SET #s=:s, updatedAt=:u, resultLandmarksKey=:lk, resultSummaryKey=:sk, resultVizKey=:vk",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": {"S": "POSE_DONE"},
            ":u": {"S": iso_now()},
            ":lk": {"S": landmarks_key},
            ":sk": {"S": summary_key},
            ":vk": {"S": viz_uploaded} if viz_uploaded else {"NULL": True},
        },
    )

    print("Uploaded:", landmarks_key, summary_key, viz_uploaded)

if __name__ == "__main__":
    main()
