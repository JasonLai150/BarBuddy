import { Stack, StackProps, RemovalPolicy, Duration } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";

export class BarBuddyStack extends Stack {
  public readonly bucket: s3.Bucket;
  public readonly jobsTable: dynamodb.Table;
  public readonly workerRepo: ecr.Repository;

  public readonly apiLambdaRole: iam.Role;
  public readonly ecsTaskRole: iam.Role;
  public readonly ecsExecutionRole: iam.Role;
  public readonly stepFunctionsRole: iam.Role;

  public readonly workerLogGroup: logs.LogGroup;

  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    // S3 bucket for raw videos + intermediate work + results
    this.bucket = new s3.Bucket(this, "liftVideos", {
      // You can replace this with an explicit name once you decide env naming
      // bucketName: `pl-lifts-${this.account}-${this.region}`,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: false,
      removalPolicy: RemovalPolicy.DESTROY, // change to RETAIN for prod
      autoDeleteObjects: true,              // remove for prod
      cors: [
        {
          allowedMethods: [
            s3.HttpMethods.PUT,
            s3.HttpMethods.GET,
            s3.HttpMethods.HEAD,
          ],
          allowedOrigins: ["*"], // tighten later to your app domains
          allowedHeaders: ["*"],
          exposedHeaders: ["ETag"],
          maxAge: 3000,
        },
      ],
    });

    // DynamoDB table for job status + pointers to S3 artifacts
    this.jobsTable = new dynamodb.Table(this, "LiftJobsTable", {
      tableName: "LiftJobs",
      partitionKey: { name: "jobId", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY, // change to RETAIN for prod
      pointInTimeRecovery: true,
      timeToLiveAttribute: "ttl", // optional: set TTL later for cleanup
    });

    // Optional but recommended: query by userId for “my jobs”
    this.jobsTable.addGlobalSecondaryIndex({
      indexName: "ByUser",
      partitionKey: { name: "userId", type: dynamodb.AttributeType.STRING },
      sortKey: { name: "createdAt", type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // ECR repo for your worker container image (ffmpeg + mediapipe etc.)
    this.workerRepo = new ecr.Repository(this, "PoseWorkerRepo", {
      repositoryName: "pose-worker",
      removalPolicy: RemovalPolicy.DESTROY, // change to RETAIN for prod
      emptyOnDelete: true,
      imageScanOnPush: true,
    });

    this.workerLogGroup = new logs.LogGroup(this, "PoseWorkerLogGroup", {
      logGroupName: "/ecs/pose-worker",
      removalPolicy: RemovalPolicy.DESTROY, // RETAIN later for prod
    });

    this.apiLambdaRole = new iam.Role(this, "ApiLambdaRole", {
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
    });
    // Basic Lambda logging
    this.apiLambdaRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSLambdaBasicExecutionRole")
    );
    // DynamoDB access (jobs table)
    this.jobsTable.grantReadWriteData(this.apiLambdaRole);
    // S3 access needed for presign + head checks
    // (Presigning itself doesn't require permission, but validating existence does)
    this.bucket.grantReadWrite(this.apiLambdaRole); // for HeadObject/GetObject if needed
    // Start Step Functions executions (we’ll attach the actual ARN later)
    // For now allow starting any state machine in this account/region.
    // We'll tighten this once we create the state machine.
    this.apiLambdaRole.addToPolicy(new iam.PolicyStatement({
      actions: ["states:StartExecution"],
      resources: ["*"],
    }));

    this.ecsTaskRole = new iam.Role(this, "EcsTaskRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    });
    // Worker reads raw videos and writes results
    this.bucket.grantReadWrite(this.ecsTaskRole);
    // Worker updates job status + writes result pointers
    this.jobsTable.grantReadWriteData(this.ecsTaskRole);


    this.ecsExecutionRole = new iam.Role(this, "EcsExecutionRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    });
    this.ecsExecutionRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AmazonECSTaskExecutionRolePolicy")
    );
    // Allow pulling from your specific ECR repo
    this.workerRepo.grantPull(this.ecsExecutionRole);


    this.stepFunctionsRole = new iam.Role(this, "StepFunctionsRole", {
      assumedBy: new iam.ServicePrincipal("states.amazonaws.com"),
    });
    // Step Functions can run ECS tasks
    this.stepFunctionsRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        "ecs:RunTask",
        "ecs:StopTask",
        "ecs:DescribeTasks",
      ],
      resources: ["*"], // tighten once we have cluster + task definition ARNs
    }));
    // Needed so Step Functions can attach the ECS task roles when running tasks
    this.stepFunctionsRole.addToPolicy(new iam.PolicyStatement({
      actions: ["iam:PassRole"],
      resources: [
        this.ecsTaskRole.roleArn,
        this.ecsExecutionRole.roleArn,
      ],
    }));

  }
}
