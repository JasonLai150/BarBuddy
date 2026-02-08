import { Stack, StackProps, Duration } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as sfn from "aws-cdk-lib/aws-stepfunctions";
import * as tasks from "aws-cdk-lib/aws-stepfunctions-tasks";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";

type OrchestrationStackProps = StackProps & {
  cluster: ecs.Cluster;
  taskDef: ecs.FargateTaskDefinition;
  vpc: ec2.Vpc;
  taskSecurityGroup: ec2.SecurityGroup;
  jobsTable: dynamodb.Table;
};

export class OrchestrationStack extends Stack {
  public readonly stateMachine: sfn.StateMachine;

  constructor(scope: Construct, id: string, props: OrchestrationStackProps) {
    super(scope, id, props);

    // 1) status -> PROCESSING
    const setProcessing = new tasks.DynamoUpdateItem(this, "SetStatusProcessing", {
        table: props.jobsTable,
        key: {
            jobId: tasks.DynamoAttributeValue.fromString(sfn.JsonPath.stringAt("$.jobId")),
        },
        updateExpression: "SET #s = :s, updatedAt = :u",
        expressionAttributeNames: { "#s": "status" },
        expressionAttributeValues: {
            ":s": tasks.DynamoAttributeValue.fromString("PROCESSING"),
            ":u": tasks.DynamoAttributeValue.fromString(sfn.JsonPath.stringAt("$.now")),
        },
        resultPath: sfn.JsonPath.DISCARD,
    });

    // 2) run ECS task
    const runWorker = new tasks.EcsRunTask(this, "RunPoseWorkerTask", {
      integrationPattern: sfn.IntegrationPattern.RUN_JOB, // wait for task to finish
      cluster: props.cluster,
      taskDefinition: props.taskDef,
      launchTarget: new tasks.EcsFargateLaunchTarget({
        platformVersion: ecs.FargatePlatformVersion.LATEST,
      }),
      assignPublicIp: true, // simplify for now
      securityGroups: [props.taskSecurityGroup],
      subnets: props.vpc.selectSubnets({ subnetType: ec2.SubnetType.PUBLIC }),
      containerOverrides: [
        {
          containerDefinition: props.taskDef.defaultContainer!,
          environment: [
            { name: "JOB_ID", value: sfn.JsonPath.stringAt("$.jobId") },
            { name: "RAW_S3_KEY", value: sfn.JsonPath.stringAt("$.rawS3Key") },
            { name: "LIFT_TYPE", value: sfn.JsonPath.stringAt("$.liftType") },
            { name: "USER_ID", value: sfn.JsonPath.stringAt("$.userId") },

            { name: "RESULT_META_KEY", value: sfn.JsonPath.stringAt("$.results.metaKey") },
            { name: "RESULT_LANDMARKS_KEY", value: sfn.JsonPath.stringAt("$.results.landmarksKey") },
            { name: "RESULT_SUMMARY_KEY", value: sfn.JsonPath.stringAt("$.results.summaryKey") },
            { name: "RESULT_VIZ_KEY", value: sfn.JsonPath.stringAt("$.results.vizKey") },

            { name: "VIZ", value: "1" },
          ],
        },
      ],
      resultPath: sfn.JsonPath.DISCARD,
    });

    // 3) status -> DONE
    const setDone = new tasks.DynamoUpdateItem(this, "SetStatusDone", {
      table: props.jobsTable,
      key: {
        jobId: tasks.DynamoAttributeValue.fromString(sfn.JsonPath.stringAt("$.jobId")),
      },
      updateExpression: "SET #s = :s, updatedAt = :u",
      expressionAttributeNames: { "#s": "status" },
      expressionAttributeValues: {
        ":s": tasks.DynamoAttributeValue.fromString("DONE"),
        ":u": tasks.DynamoAttributeValue.fromString(sfn.JsonPath.stringAt("$.now")),
      },
      resultPath: sfn.JsonPath.DISCARD,
    });

    // error handler: status -> ERROR
    const setError = new tasks.DynamoUpdateItem(this, "SetStatusError", {
      table: props.jobsTable,
      key: {
        jobId: tasks.DynamoAttributeValue.fromString(sfn.JsonPath.stringAt("$.jobId")),
      },
      updateExpression: "SET #s = :s, updatedAt = :u, errorMessage = :e",
      expressionAttributeNames: { "#s": "status" },
      expressionAttributeValues: {
        ":s": tasks.DynamoAttributeValue.fromString("ERROR"),
        ":u": tasks.DynamoAttributeValue.fromString(sfn.JsonPath.stringAt("$.now")),
        ":e": tasks.DynamoAttributeValue.fromString(sfn.JsonPath.stringAt("$.error.Cause")),
      },
      resultPath: sfn.JsonPath.DISCARD,
    });

    // Add a Pass state to inject "now" timestamp once (cheap + consistent)
    const addNow = new sfn.Pass(this, "AddNowTimestamp", {
      parameters: {
        "jobId.$": "$.jobId",
        "rawS3Key.$": "$.rawS3Key",
        "liftType.$": "$.liftType",
        "userId.$": "$.userId",
        "results.$": "$.results",          // ✅ keep results object
        "now.$": "$$.State.EnteredTime",
      },
    });

    // ---- TRY chain ----
    const processingChain = sfn.Chain
        .start(setProcessing)
        .next(runWorker)
        .next(setDone);

    // Wrap the try chain in a Parallel so we can attach ONE catch handler.
    // (Parallel is a State; Chain is not.)
    const tryProcessing = new sfn.Parallel(this, "TryProcessing");
    tryProcessing.branch(processingChain);

    // If anything in the branch fails, attach error info at $.error and jump to setError
    tryProcessing.addCatch(setError, { resultPath: "$.error" });

    // Final state machine definition
    const definition = addNow.next(tryProcessing);

    this.stateMachine = new sfn.StateMachine(this, "LiftProcessingStateMachine", {
        stateMachineName: "lift-processing",
        definitionBody: sfn.DefinitionBody.fromChainable(definition),
        timeout: Duration.minutes(15),
    });
  }
}
