import json
import os
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def main():
    job_id = os.environ["JOB_ID"]
    bucket = os.environ["BUCKET_NAME"]
    raw_key = os.environ["RAW_S3_KEY"]
    table = os.environ["JOBS_TABLE_NAME"]

    s3 = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    # 1) Validate raw video exists + fetch metadata
    try:
        head = s3.head_object(Bucket=bucket, Key=raw_key)
    except ClientError as e:
        msg = f"head_object failed: {e}"
        print(msg)
        # Mark ERROR in DynamoDB
        ddb.update_item(
            TableName=table,
            Key={"jobId": {"S": job_id}},
            UpdateExpression="SET #s=:s, updatedAt=:u, errorMessage=:e",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": {"S": "ERROR"},
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
        "note": "Skeleton worker meta.json (no pose yet)."
    }

    # 2) Write results/{jobId}/meta.json
    result_key = f"results/{job_id}/meta.json"
    s3.put_object(
        Bucket=bucket,
        Key=result_key,
        Body=(json.dumps(meta, indent=2) + "\n").encode("utf-8"),
        ContentType="application/json",
    )

    # 3) Update DynamoDB with result pointer
    ddb.update_item(
        TableName=table,
        Key={"jobId": {"S": job_id}},
        UpdateExpression="SET #s=:s, updatedAt=:u, resultMetaKey=:rk",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": {"S": "WORKER_DONE"},
            ":u": {"S": iso_now()},
            ":rk": {"S": result_key},
        },
    )

    print(f"Done. Wrote s3://{bucket}/{result_key}")

if __name__ == "__main__":
    main()
