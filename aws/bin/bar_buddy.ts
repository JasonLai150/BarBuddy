#!/usr/bin/env node
import * as cdk from "aws-cdk-lib/core";
import { BarBuddyStack } from "../lib/bar_buddy-core-stack";
import { ApiStack } from "../lib/bar_buddy-api-stack";
import { ComputeStack } from "../lib/bar_buddy-compute-stack";

const app = new cdk.App();

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION,
};

// ── Core: S3, DynamoDB, ECR, SQS, IAM ───────
const core = new BarBuddyStack(app, "BarBuddyStack", { env });

// ── Compute: VPC, ASG (g4dn.xlarge spot), auto-scaling ──
const compute = new ComputeStack(app, "BarBuddyComputeStack", {
  env,
  bucket: core.bucket,
  jobsTable: core.jobsTable,
  workerRepo: core.workerRepo,
  jobQueue: core.jobQueue,
  ec2InstanceRole: core.ec2InstanceRole,
  scaleTriggerLambdaRole: core.scaleTriggerLambdaRole,
  workerLogGroup: core.workerLogGroup,
});

// ── API: API Gateway + Lambda ────────────────
const api = new ApiStack(app, "BarBuddyApiStack", {
  env,
  bucket: core.bucket,
  jobsTable: core.jobsTable,
  jobQueue: core.jobQueue,
  apiLambdaRole: core.apiLambdaRole,
  userPoolId: "us-east-1_0WHxHJ2Lf",
});
