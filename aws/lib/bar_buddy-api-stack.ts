import { Stack, StackProps, Duration } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as apigw from "aws-cdk-lib/aws-apigateway";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as nodeLambda from "aws-cdk-lib/aws-lambda-nodejs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as sqs from "aws-cdk-lib/aws-sqs";
import * as cognito from "aws-cdk-lib/aws-cognito";

type ApiStackProps = StackProps & {
  bucket: s3.Bucket;
  jobsTable: dynamodb.Table;
  jobQueue: sqs.Queue;
  apiLambdaRole: iam.Role;
  userPoolId: string;
};

export class ApiStack extends Stack {
  public readonly api: apigw.RestApi;

  constructor(scope: Construct, id: string, props: ApiStackProps) {
    super(scope, id, props);

    const jobManagerFn = new nodeLambda.NodejsFunction(this, "JobManagerFn", {
      runtime: lambda.Runtime.NODEJS_20_X,
      entry: "lambda/jobManager.ts",
      handler: "handler",
      timeout: Duration.seconds(15),
      memorySize: 512,
      role: props.apiLambdaRole,
      environment: {
        BUCKET_NAME: props.bucket.bucketName,
        JOBS_TABLE_NAME: props.jobsTable.tableName,
        JOB_QUEUE_URL: props.jobQueue.queueUrl,
        PRESIGN_GET_TTL_SECONDS: "900",
      },
      bundling: {
        minify: true,
        sourceMap: true,
      },
    });

    this.api = new apigw.RestApi(this, "BarBuddyApi", {
      restApiName: "barbuddy-job-api",
      defaultCorsPreflightOptions: {
        allowOrigins: apigw.Cors.ALL_ORIGINS,
        allowMethods: ["GET", "POST", "OPTIONS"],
        allowHeaders: ["Content-Type", "Authorization"],
      },
    });

    const userPool = cognito.UserPool.fromUserPoolId(
      this,
      "ImportedUserPool",
      props.userPoolId
    );

    const authorizer = new apigw.CognitoUserPoolsAuthorizer(this, "CognitoAuthorizer", {
      cognitoUserPools: [userPool],
    });

    const auth = {
      authorizer,
      authorizationType: apigw.AuthorizationType.COGNITO,
    };

    const jobs = this.api.root.addResource("jobs");

    // POST /upload-url (protected)
    const uploadUrl = this.api.root.addResource("upload-url");
    uploadUrl.addMethod("POST", new apigw.LambdaIntegration(jobManagerFn), auth);

    // POST /jobs (protected)
    jobs.addMethod("POST", new apigw.LambdaIntegration(jobManagerFn), auth);

    // GET /jobs (protected)
    jobs.addMethod("GET", new apigw.LambdaIntegration(jobManagerFn), auth);

    // GET /jobs/{jobId} (protected)
    const jobById = jobs.addResource("{jobId}");
    jobById.addMethod("GET", new apigw.LambdaIntegration(jobManagerFn), auth);

    // GET /jobs/{jobId}/results (protected)
    const jobResults = jobById.addResource("results");
    jobResults.addMethod("GET", new apigw.LambdaIntegration(jobManagerFn), auth);
  }
}
