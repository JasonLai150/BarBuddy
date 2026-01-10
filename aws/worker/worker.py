import json
import os
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def must_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def main():
    job_id = must_env("JOB_ID")
    bucket = must_env("BUCKET_NAME")
    raw_key = must_env("RAW_S3_KEY")
    table = must_env("JOBS_TABLE_NAME")

    # ✅ Explicit output keys (no prefix construction)
    meta_key = must_env("RESULT_META_KEY")
    landmarks_key = must_env("RESULT_LANDMARKS_KEY")
    summary_key = must_env("RESULT_SUMMARY_KEY")
    viz_key = must_env("RESULT_VIZ_KEY")

    s3 = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    # 1) Validate raw upload exists
    try:
        head = s3.head_object(Bucket=bucket, Key=raw_key)
    except ClientError as e:
        msg = f"head_object failed: {e}"
        print(msg)
        ddb.update_item(
            TableName=table,
            Key={"jobId": {"S": job_id}},
            UpdateExpression="SET updatedAt=:u, errorMessage=:e",
            ExpressionAttributeValues={
                ":u": {"S": iso_now()},
                ":e": {"S": msg},
            },
        )
        raise

    meta = {
        "jobId": job_id,
        "rawS3Key": raw_key,
        "contentLength": int(head.get("ContentLength", 0)),
        "contentType": head.get("ContentType"),
        "etag": head.get("ETag"),
        "lastModified": head.get("LastModified").isoformat() if head.get("LastModified") else None,
        "generatedAt": iso_now(),
        "note": "Skeleton worker meta.json (no pose yet).",
    }

    # 2) Write meta.json to explicit key
    s3.put_object(
        Bucket=bucket,
        Key=meta_key,
        Body=(json.dumps(meta, indent=2) + "\n").encode("utf-8"),
        ContentType="application/json",
    )

    # 3) Placeholder writes showing explicit-key usage
    # Replace with your real outputs later.
    s3.put_object(
        Bucket=bucket,
        Key=landmarks_key,
        Body=b"{}",
        ContentType="application/json",
    )

    s3.put_object(
        Bucket=bucket,
        Key=summary_key,
        Body=b"{}",
        ContentType="application/json",
    )

    # If you don't produce a viz yet, you can skip this write
    # or only write it when VIZ=1 and you actually have bytes.
    if os.environ.get("VIZ") == "1":
        s3.put_object(
            Bucket=bucket,
            Key=viz_key,
            Body=b"",  # replace with actual mp4 bytes
            ContentType="video/mp4",
        )

    # 4) Update DynamoDB pointers (no status mutation)
    ddb.update_item(
        TableName=table,
        Key={"jobId": {"S": job_id}},
        UpdateExpression="SET updatedAt=:u, resultMetaKey=:mk, resultLandmarksKey=:lk, resultSummaryKey=:sk, resultVizKey=:vk",
        ExpressionAttributeValues={
            ":u": {"S": iso_now()},
            ":mk": {"S": meta_key},
            ":lk": {"S": landmarks_key},
            ":sk": {"S": summary_key},
            ":vk": {"S": viz_key},
        },
    )

    print("Worker complete")
    print(f"  meta: s3://{bucket}/{meta_key}")
    print(f"  landmarks: s3://{bucket}/{landmarks_key}")
    print(f"  summary: s3://{bucket}/{summary_key}")
    print(f"  viz: s3://{bucket}/{viz_key}")


if __name__ == "__main__":
    main()
