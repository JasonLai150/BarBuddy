import { Stack, StackProps, RemovalPolicy, Duration, CfnOutput } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as sqs from "aws-cdk-lib/aws-sqs";

export class BarBuddyStack extends Stack {
  public readonly bucket: s3.Bucket;
  public readonly jobsTable: dynamodb.Table;
  public readonly workerRepo: ecr.Repository;
  public readonly jobQueue: sqs.Queue;

  public readonly apiLambdaRole: iam.Role;
  public readonly ec2InstanceRole: iam.Role;
  public readonly scaleTriggerLambdaRole: iam.Role;

  public readonly workerLogGroup: logs.LogGroup;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // ── S3 bucket ────────────────────────────
    this.bucket = new s3.Bucket(this, "liftVideos", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: false,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      cors: [
        {
          allowedMethods: [
            s3.HttpMethods.PUT,
            s3.HttpMethods.GET,
            s3.HttpMethods.HEAD,
          ],
          allowedOrigins: ["*"],
          allowedHeaders: ["*"],
          exposedHeaders: ["ETag"],
          maxAge: 3000,
        },
      ],
    });

    // ── DynamoDB ──────────────────────────────
    this.jobsTable = new dynamodb.Table(this, "LiftJobsTable", {
      tableName: "LiftJobs",
      partitionKey: { name: "jobId", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY,
      pointInTimeRecovery: true,
      timeToLiveAttribute: "ttl",
    });

    this.jobsTable.addGlobalSecondaryIndex({
      indexName: "ByUser",
      partitionKey: { name: "userId", type: dynamodb.AttributeType.STRING },
      sortKey: { name: "createdAt", type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // ── ECR ──────────────────────────────────
    this.workerRepo = new ecr.Repository(this, "PoseWorkerRepo", {
      repositoryName: "pose-worker",
      removalPolicy: RemovalPolicy.DESTROY,
      emptyOnDelete: true,
      imageScanOnPush: true,
    });

    new CfnOutput(this, "PoseWorkerRepoUri", {
      value: this.workerRepo.repositoryUri,
    });

    this.workerLogGroup = new logs.LogGroup(this, "PoseWorkerLogGroup", {
      logGroupName: "/ecs/pose-worker",
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // ── SQS job queue ────────────────────────
    const deadLetterQueue = new sqs.Queue(this, "JobDLQ", {
      queueName: "barbuddy-jobs-dlq",
      retentionPeriod: Duration.days(14),
    });

    this.jobQueue = new sqs.Queue(this, "JobQueue", {
      queueName: "barbuddy-jobs",
      visibilityTimeout: Duration.minutes(10),
      receiveMessageWaitTime: Duration.seconds(20),
      retentionPeriod: Duration.days(4),
      deadLetterQueue: {
        queue: deadLetterQueue,
        maxReceiveCount: 3,
      },
    });

    // ── IAM: API Lambda ──────────────────────
    this.apiLambdaRole = new iam.Role(this, "ApiLambdaRole", {
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
    });
    this.apiLambdaRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSLambdaBasicExecutionRole")
    );
    this.jobsTable.grantReadWriteData(this.apiLambdaRole);
    this.bucket.grantReadWrite(this.apiLambdaRole);
    this.jobQueue.grantSendMessages(this.apiLambdaRole);

    // ── IAM: EC2 instance role (worker) ──────
    this.ec2InstanceRole = new iam.Role(this, "Ec2InstanceRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
    });
    this.bucket.grantReadWrite(this.ec2InstanceRole);
    this.jobsTable.grantReadWriteData(this.ec2InstanceRole);
    this.jobQueue.grantConsumeMessages(this.ec2InstanceRole);
    this.workerRepo.grantPull(this.ec2InstanceRole);
    this.ec2InstanceRole.addToPolicy(new iam.PolicyStatement({
      actions: ["ecr:GetAuthorizationToken"],
      resources: ["*"],
    }));
    this.ec2InstanceRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams",
      ],
      resources: [this.workerLogGroup.logGroupArn + ":*"],
    }));
    this.ec2InstanceRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore")
    );

    // ── IAM: Scale-trigger Lambda ────────────
    this.scaleTriggerLambdaRole = new iam.Role(this, "ScaleTriggerLambdaRole", {
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
    });
    this.scaleTriggerLambdaRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSLambdaBasicExecutionRole")
    );
    this.scaleTriggerLambdaRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        "autoscaling:SetDesiredCapacity",
        "autoscaling:DescribeAutoScalingGroups",
      ],
      resources: ["*"],
    }));
  }
}
