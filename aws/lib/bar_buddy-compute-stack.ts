import { Stack, StackProps, Duration, RemovalPolicy } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as logs from "aws-cdk-lib/aws-logs";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";

type ComputeStackProps = StackProps & {
  bucket: s3.Bucket;
  jobsTable: dynamodb.Table;
  workerRepo: ecr.Repository;
  ecsTaskRole: iam.Role;
  ecsExecutionRole: iam.Role;
  workerLogGroup: logs.ILogGroup;
};

export class ComputeStack extends Stack {
  public readonly cluster: ecs.Cluster;
  public readonly taskDef: ecs.FargateTaskDefinition;
  public readonly vpc: ec2.Vpc;
  public readonly taskSecurityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: ComputeStackProps) {
    super(scope, id, props);

    // VPC for Fargate (CDK creates a simple one; good for beta)
    this.vpc = new ec2.Vpc(this, "WorkerVpc", {
      maxAzs: 2,
      //TODO: look into whether nat is neeeded
      natGateways: 1, // costs $, but simplifies outbound access to AWS APIs
    });

    this.taskSecurityGroup = new ec2.SecurityGroup(this, "WorkerTaskSg", {
      vpc: this.vpc,
      allowAllOutbound: true,
      description: "Security group for one-off pose worker tasks",
    });

    this.cluster = new ecs.Cluster(this, "WorkerCluster", {
      vpc: this.vpc,
    });

    this.taskDef = new ecs.FargateTaskDefinition(this, "PoseWorkerTaskDef", {
      cpu: 2048, // 2 vCPU
      memoryLimitMiB: 8192, // 8 GB
      taskRole: props.ecsTaskRole,
      executionRole: props.ecsExecutionRole,
    });

    const container = this.taskDef.addContainer("worker", {
      image: ecs.ContainerImage.fromEcrRepository(props.workerRepo, "latest"),
      logging: ecs.LogDrivers.awsLogs({
        logGroup: props.workerLogGroup,
        streamPrefix: "worker",
      }),
      environment: {
        BUCKET_NAME: props.bucket.bucketName,
        JOBS_TABLE_NAME: props.jobsTable.tableName,
      },
    });

    // no ports required
  }
}
