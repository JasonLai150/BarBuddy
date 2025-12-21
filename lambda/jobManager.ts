import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand, UpdateCommand, GetCommand } from "@aws-sdk/lib-dynamodb";
import { S3Client, HeadObjectCommand, PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { randomUUID } from "crypto";
import { SFNClient, StartExecutionCommand } from "@aws-sdk/client-sfn";

const BUCKET_NAME = process.env.BUCKET_NAME!;
const JOBS_TABLE_NAME = process.env.JOBS_TABLE_NAME!;

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const s3 = new S3Client({});

const sfn = new SFNClient({});
const STATE_MACHINE_ARN = process.env.STATE_MACHINE_ARN!;

function resp(statusCode: number, body: any) {
  return {
    statusCode,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  };
}

export async function handler(event: any) {
  const method = event.httpMethod;
  const resource = event.resource; // "/upload-url" | "/jobs" | "/jobs/{jobId}"

  try {
    if (method === "POST" && resource === "/upload-url") return await uploadUrl(event);
    if (method === "POST" && resource === "/jobs") return await createJob(event);
    if (method === "GET" && resource === "/jobs/{jobId}") return await getJob(event);

    return resp(404, { message: "Not found", method, resource });
  } catch (e: any) {
    console.error(e);
    return resp(500, { message: "Internal server error", error: String(e?.message ?? e) });
  }
}

async function uploadUrl(event: any) {
  const body = event.body ? JSON.parse(event.body) : {};
  const contentType = body.contentType ?? "video/mp4";
  const liftType = body.liftType ?? null;

  // TODO: replace with Cognito user sub later
  const userId = "anon";

  const jobId = randomUUID();
  const s3Key = `raw/${userId}/${jobId}.mp4`;

  // Create job record
  const now = new Date().toISOString();
  await ddb.send(new PutCommand({
    TableName: JOBS_TABLE_NAME,
    Item: {
      jobId,
      userId,
      rawS3Key: s3Key,
      liftType,
      status: "CREATED",
      createdAt: now,
      updatedAt: now,
    },
    ConditionExpression: "attribute_not_exists(jobId)",
  }));

  // Presigned PUT URL to S3
  const putCmd = new PutObjectCommand({
    Bucket: BUCKET_NAME,
    Key: s3Key,
    ContentType: contentType,
  });
  const uploadUrl = await getSignedUrl(s3, putCmd, { expiresIn: 600 });

  return resp(200, { jobId, s3Key, uploadUrl });
}

async function createJob(event: any) {
  const body = event.body ? JSON.parse(event.body) : {};
  const jobId = body.jobId;
  const liftType = body.liftType;

  if (!jobId || !liftType) return resp(400, { message: "jobId and liftType are required" });

  const res = await ddb.send(new GetCommand({ TableName: JOBS_TABLE_NAME, Key: { jobId } }));
  if (!res.Item) return resp(404, { message: "Job not found" });

  const s3Key = res.Item.rawS3Key;

  // Ensure upload finished
  try {
    await s3.send(new HeadObjectCommand({ Bucket: BUCKET_NAME, Key: s3Key }));
  } catch {
    return resp(409, { message: "Upload not found in S3 yet. Finish upload before starting job.", s3Key });
  }

  const now = new Date().toISOString();
  await ddb.send(new UpdateCommand({
    TableName: JOBS_TABLE_NAME,
    Key: { jobId },
    UpdateExpression: "SET #s=:s, liftType=:lt, updatedAt=:u",
    ExpressionAttributeNames: { "#s": "status" },
    ExpressionAttributeValues: { ":s": "UPLOADED", ":lt": liftType, ":u": now },
  }));

  const exec = await sfn.send(new StartExecutionCommand({
    stateMachineArn: STATE_MACHINE_ARN,
    input: JSON.stringify({
        jobId,
        rawS3Key: s3Key,
        liftType,
        userId: res.Item.userId ?? "anon",
    }),
  }));

  await ddb.send(new UpdateCommand({
    TableName: JOBS_TABLE_NAME,
    Key: { jobId },
    UpdateExpression: "SET executionArn=:e, updatedAt=:u",
    ExpressionAttributeValues: { ":e": exec.executionArn, ":u": new Date().toISOString() },
  }));
  return resp(200, { jobId, status: "UPLOADED" });
}

async function getJob(event: any) {
  let jobId = event.pathParameters?.jobId;
  if (!jobId) return resp(400, { message: "jobId required" });

  // Defensively decode + trim (handles %0A, %0D, spaces, etc.)
  try {
    jobId = decodeURIComponent(jobId);
  } catch {
    // if it isn't valid URI encoding, ignore
  }
  jobId = jobId.trim();

  const res = await ddb.send(new GetCommand({ TableName: JOBS_TABLE_NAME, Key: { jobId } }));
  if (!res.Item) return resp(404, { message: "Job not found", debug: { jobId } });

  return resp(200, res.Item);
}
