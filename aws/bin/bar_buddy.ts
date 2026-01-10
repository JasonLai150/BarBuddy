#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib/core';
import * as cognito from "aws-cdk-lib/aws-cognito";
import { BarBuddyStack } from '../lib/bar_buddy-core-stack';
import { ApiStack } from '../lib/bar_buddy-api-stack';
import { ComputeStack } from "../lib/bar_buddy-compute-stack";
import { OrchestrationStack } from '../lib/bar_buddy-orchestration-stack';

const app = new cdk.App();
const core = new BarBuddyStack(app, 'BarBuddyStack', {
  /* If you don't specify 'env', this stack will be environment-agnostic.
   * Account/Region-dependent features and context lookups will not work,
   * but a single synthesized template can be deployed anywhere. */

  /* Uncomment the next line to specialize this stack for the AWS Account
   * and Region that are implied by the current CLI configuration. */
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },

  /* Uncomment the next line if you know exactly what Account and Region you
   * want to deploy the stack to. */
  // env: { account: '123456789012', region: 'us-east-1' },

  /* For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html */
});

const compute = new ComputeStack(app, "BarBuddyComputeStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  bucket: core.bucket,
  jobsTable: core.jobsTable,
  workerRepo: core.workerRepo,
  ecsTaskRole: core.ecsTaskRole,
  ecsExecutionRole: core.ecsExecutionRole,
  workerLogGroup: core.workerLogGroup,
});

const orchestration = new OrchestrationStack(app, "BarBuddyOrchestrationStack", {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
  cluster: compute.cluster,
  taskDef: compute.taskDef,
  vpc: compute.vpc,
  taskSecurityGroup: compute.taskSecurityGroup,
  jobsTable: core.jobsTable,
});

const api = new ApiStack(app, "BarBuddyApiStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  bucket: core.bucket,
  jobsTable: core.jobsTable,
  apiLambdaRole: core.apiLambdaRole,
  stateMachineArn: orchestration.stateMachine.stateMachineArn,
  userPoolId: "us-east-1_0WHxHJ2Lf",
});
