import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand, UpdateCommand, GetCommand, QueryCommand } from "@aws-sdk/lib-dynamodb";
import { S3Client, HeadObjectCommand, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { randomUUID } from "crypto";
import { SFNClient, StartExecutionCommand } from "@aws-sdk/client-sfn";

const BUCKET_NAME = process.env.BUCKET_NAME!;
const JOBS_TABLE_NAME = process.env.JOBS_TABLE_NAME!;
const STATE_MACHINE_ARN = process.env.STATE_MACHINE_ARN!;
const PRESIGN_GET_TTL_SECONDS = Number(process.env.PRESIGN_GET_TTL_SECONDS ?? "900"); // 15 min
const PRESIGN_PUT_TTL_SECONDS = Number(process.env.PRESIGN_PUT_TTL_SECONDS ?? "600"); // 10 min

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const s3 = new S3Client({});
const sfn = new SFNClient({});

function resp(statusCode: number, body: any) {
  return {
    statusCode,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  };
}

/**
 * REST API + Cognito User Pool Authorizer puts claims here:
 *   event.requestContext.authorizer.claims
 */
function requireUserSub(event: any): string {
  const claims = event?.requestContext?.authorizer?.claims;
  const sub = claims?.sub;
  if (!sub) throw new Error("Unauthorized: missing Cognito claims.sub");
  return String(sub);
}

const ALLOW_LEGACY_ANON = (process.env.ALLOW_LEGACY_ANON ?? "false").toLowerCase() === "true";

async function getJobOwned(jobId: string, userSub: string) {
  const res = await ddb.send(new GetCommand({ TableName: JOBS_TABLE_NAME, Key: { jobId } }));
  const job = res.Item;
  if (!job) return { job: null as any, err: resp(404, { message: "Job not found" }) };

  const owner = job.userId;
  const isOwner = owner === userSub || (ALLOW_LEGACY_ANON && owner === "anon");
  if (!isOwner) return { job: null as any, err: resp(403, { message: "Forbidden" }) };

  return { job, err: null as any };
}

export async function handler(event: any) {
  const method = event.httpMethod;
  const resource = event.resource; // "/upload-url" | "/jobs" | "/jobs/{jobId}" | "/jobs/{jobId}/results"

  try {
    if (method === "POST" && resource === "/upload-url") return await uploadUrl(event);
    if (method === "POST" && resource === "/jobs") return await createJob(event);
    if (method === "GET" && resource === "/jobs") return await getJobsList(event);
    if (method === "GET" && resource === "/jobs/{jobId}") return await getJob(event);
    if (method === "GET" && resource === "/jobs/{jobId}/results") return await getJobResults(event);

    return resp(404, { message: "Not found", method, resource });
  } catch (e: any) {
    console.error(e);
    const msg = String(e?.message ?? e);
    if (msg.toLowerCase().includes("unauthorized")) return resp(401, { message: "Unauthorized" });
    return resp(500, { message: "Internal server error", error: msg });
  }
}

async function uploadUrl(event: any) {
  const userSub = requireUserSub(event);

  const body = event.body ? JSON.parse(event.body) : {};
  const contentType = body.contentType ?? "video/mp4";

  const jobId = randomUUID();
  const rawS3Key = `raw/${userSub}/${jobId}.mp4`;

  // ✅ New results layout
  const resultsPrefix = `results/${userSub}/${jobId}/`;

  const resultMetaKey = `${resultsPrefix}meta.json`;
  const resultLandmarksKey = `${resultsPrefix}landmarks.json`;
  const resultSummaryKey = `${resultsPrefix}summary.json`;
  const resultVizKey = `${resultsPrefix}viz.mp4`;


  const now = new Date().toISOString();

  await ddb.send(
    new PutCommand({
      TableName: JOBS_TABLE_NAME,
      Item: {
        jobId,
        userId: userSub,
        rawS3Key,
        resultsPrefix,
        resultMetaKey,
        resultLandmarksKey,
        resultSummaryKey,
        resultVizKey,
        status: "CREATED",
        createdAt: now,
        updatedAt: now,
      },
      ConditionExpression: "attribute_not_exists(jobId)",
    })
  );

  const putCmd = new PutObjectCommand({
    Bucket: BUCKET_NAME,
    Key: rawS3Key,
    ContentType: contentType,
  });

  const uploadUrl = await getSignedUrl(s3, putCmd, { expiresIn: PRESIGN_PUT_TTL_SECONDS });

  return resp(200, { jobId, s3Key: rawS3Key, uploadUrl });
}

async function createJob(event: any) {
  const userSub = requireUserSub(event);

  const body = event.body ? JSON.parse(event.body) : {};
  const jobId = String(body.jobId ?? "").trim();
  const liftType = body.liftType;

  if (!jobId || !liftType) return resp(400, { message: "jobId and liftType are required" });

  const { job, err } = await getJobOwned(jobId, userSub);
  if (err) return err;

  const rawS3Key = job.rawS3Key;

  // Ensure upload finished
  try {
    await s3.send(new HeadObjectCommand({ Bucket: BUCKET_NAME, Key: rawS3Key }));
  } catch {
    return resp(409, { message: "Upload not found in S3 yet. Finish upload before starting job.", s3Key: rawS3Key });
  }

  const now = new Date().toISOString();

  // Mark uploaded
  await ddb.send(
    new UpdateCommand({
      TableName: JOBS_TABLE_NAME,
      Key: { jobId },
      UpdateExpression: "SET #s=:s, liftType=:lt, updatedAt=:u",
      ExpressionAttributeNames: { "#s": "status" },
      ExpressionAttributeValues: { ":s": "UPLOADED", ":lt": liftType, ":u": now },
    })
  );

  // ✅ Pass results prefix + keys into Step Functions
  const resultsPrefix = job.resultsPrefix ?? `results/${job.userId}/${jobId}/`;
  const exec = await sfn.send(
    new StartExecutionCommand({
      stateMachineArn: STATE_MACHINE_ARN,
      input: JSON.stringify({
        jobId,
        rawS3Key,
        liftType,
        userId: job.userId, // sub
        results: {
          metaKey: job.resultMetaKey,
          landmarksKey: job.resultLandmarksKey,
          summaryKey: job.resultSummaryKey,
          vizKey: job.resultVizKey,
        },
      }),
    })
  );

  await ddb.send(
    new UpdateCommand({
      TableName: JOBS_TABLE_NAME,
      Key: { jobId },
      UpdateExpression: "SET executionArn=:e, updatedAt=:u",
      ExpressionAttributeValues: { ":e": exec.executionArn, ":u": new Date().toISOString() },
    })
  );

  return resp(200, { jobId, status: "UPLOADED" });
}

async function getJob(event: any) {
  const userSub = requireUserSub(event);

  let jobId = event.pathParameters?.jobId;
  if (!jobId) return resp(400, { message: "jobId required" });

  try {
    jobId = decodeURIComponent(jobId);
  } catch {
    // ignore
  }
  jobId = jobId.trim();

  const { job, err } = await getJobOwned(jobId, userSub);
  if (err) return err;

  return resp(200, job);
}

async function getJobsList(event: any) {
  const userSub = requireUserSub(event);

  const queryParams = event.queryStringParameters ?? {};
  let nextToken = queryParams.nextToken;
  const pageSize = 20;

  let exclusiveStartKey: any = undefined;
  if (nextToken) {
    try {
      const decoded = Buffer.from(nextToken, "base64").toString("utf-8");
      exclusiveStartKey = JSON.parse(decoded);
    } catch {
      return resp(400, { message: "Invalid nextToken" });
    }
  }

  try {
    const queryResult = await ddb.send(
      new QueryCommand({
        TableName: JOBS_TABLE_NAME,
        IndexName: "ByUser",
        KeyConditionExpression: "userId = :uid",
        ExpressionAttributeValues: { ":uid": userSub },
        ScanIndexForward: false, // newest first
        Limit: pageSize,
        ExclusiveStartKey: exclusiveStartKey,
        ProjectionExpression: "jobId, userId, liftType, #s, createdAt, updatedAt, resultMetaKey, resultLandmarksKey, resultSummaryKey, resultVizKey",
        ExpressionAttributeNames: { "#s": "status" },
      })
    );

    const jobs = (queryResult.Items ?? []).map((item: any) => ({
      jobId: item.jobId,
      userId: item.userId,
      liftType: item.liftType ?? null,
      status: item.status ?? null,
      createdAt: item.createdAt ?? null,
      updatedAt: item.updatedAt ?? null,
      resultMetaKey: item.resultMetaKey ?? null,
      resultLandmarksKey: item.resultLandmarksKey ?? null,
      resultSummaryKey: item.resultSummaryKey ?? null,
      resultVizKey: item.resultVizKey ?? null,
    }));

    let newNextToken: string | undefined;
    if (queryResult.LastEvaluatedKey) {
      const encoded = Buffer.from(JSON.stringify(queryResult.LastEvaluatedKey)).toString("base64");
      newNextToken = encoded;
    }

    return resp(200, {
      jobs,
      nextToken: newNextToken ?? null,
    });
  } catch (e: any) {
    console.error("Error querying jobs", e);
    throw e;
  }
}

async function getJobResults(event: any) {
  const userSub = requireUserSub(event);

  let jobId = event.pathParameters?.jobId;
  if (!jobId) return resp(400, { message: "jobId required" });

  try {
    jobId = decodeURIComponent(jobId);
  } catch {
    // ignore
  }
  jobId = jobId.trim();

  const { job, err } = await getJobOwned(jobId, userSub);
  if (err) return err;

  if (job.status !== "DONE" && job.status !== "WORKER_DONE") {
    return resp(200, { jobId, status: job.status, message: "Results not ready yet", urls: [] });
  }

  // Include meta.json too if you want (since your current worker writes meta.json)
  const metaKey = job.resultMetaKey ?? (job.resultsPrefix ? `${job.resultsPrefix}meta.json` : undefined);

  const keys: Array<{ name: string; key?: string }> = [
    { name: "meta", key: metaKey },
    { name: "landmarks", key: job.resultLandmarksKey },
    { name: "summary", key: job.resultSummaryKey },
    { name: "viz", key: job.resultVizKey },
  ];

  const urls: Array<{ name: string; key: string; url: string }> = [];

  for (const item of keys) {
    if (!item.key) continue;

    const getCmd = new GetObjectCommand({
      Bucket: BUCKET_NAME,
      Key: item.key,
    });

    const url = await getSignedUrl(s3, getCmd, { expiresIn: PRESIGN_GET_TTL_SECONDS });
    urls.push({ name: item.name, key: item.key, url });
  }

  return resp(200, { jobId, status: job.status, urls });
}
