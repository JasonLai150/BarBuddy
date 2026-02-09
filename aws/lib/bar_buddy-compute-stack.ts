import { Stack, StackProps, Duration, CfnOutput } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as autoscaling from "aws-cdk-lib/aws-autoscaling";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as sqs from "aws-cdk-lib/aws-sqs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as logs from "aws-cdk-lib/aws-logs";

type ComputeStackProps = StackProps & {
  bucket: s3.Bucket;
  jobsTable: dynamodb.Table;
  workerRepo: ecr.Repository;
  jobQueue: sqs.Queue;
  ec2InstanceRole: iam.Role;
  scaleTriggerLambdaRole: iam.Role;
  workerLogGroup: logs.ILogGroup;
};

export class ComputeStack extends Stack {
  public readonly vpc: ec2.Vpc;
  public readonly asg: autoscaling.AutoScalingGroup;

  constructor(scope: Construct, id: string, props: ComputeStackProps) {
    super(scope, id, props);

    // ── VPC (public subnets only, no NAT) ────
    this.vpc = new ec2.Vpc(this, "WorkerVpc", {
      maxAzs: 2,
      natGateways: 0,
      subnetConfiguration: [
        {
          name: "Public",
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
      ],
    });

    // ── Security group ──────────────────────
    const workerSg = new ec2.SecurityGroup(this, "WorkerSg", {
      vpc: this.vpc,
      allowAllOutbound: true,
      description: "Security group for GPU worker instances",
    });

    // ── Instance profile from core-stack role ─
    const instanceProfile = new iam.CfnInstanceProfile(this, "WorkerInstanceProfile", {
      roles: [props.ec2InstanceRole.roleName],
    });

    // ── User data script ─────────────────────
    // Runs on instance boot: pull Docker image, start worker container
    const userData = ec2.UserData.forLinux();
    const region = Stack.of(this).region;
    const account = Stack.of(this).account;
    const repoUri = props.workerRepo.repositoryUri;

    userData.addCommands(
      "#!/bin/bash",
      "set -euo pipefail",
      "exec > >(tee /var/log/user-data.log) 2>&1",
      "echo '=============================='",
      "echo 'BarBuddy Worker Bootstrap'",
      "echo \"Started: $(date -u)\"",
      "echo '=============================='",

      // Install Docker
      "echo '[1/7] Installing Docker...'",
      "yum update -y",
      "amazon-linux-extras install docker -y || yum install docker -y",
      "systemctl enable docker && systemctl start docker",
      "echo '[1/7] ✅ Docker installed: '$(docker --version)",

      // Install NVIDIA driver (g4dn = Tesla T4, needs kernel modules)
      "echo '[2/7] Installing NVIDIA driver...'",
      "yum install -y gcc10 kernel-devel-$(uname -r)",
      "export CC=/usr/bin/gcc10-cc",
      // Download NVIDIA Tesla driver via HTTPS (the S3 bucket requires EULA acceptance)
      "echo '       Downloading driver...'",
      `curl -fSL -o /tmp/NVIDIA-Linux.run https://us.download.nvidia.com/tesla/535.183.01/NVIDIA-Linux-x86_64-535.183.01.run`,
      "echo '       Installing driver (compiling kernel module)...'",
      "chmod +x /tmp/NVIDIA-Linux.run",
      "/tmp/NVIDIA-Linux.run --silent --tmpdir /tmp",
      "echo '[2/7] ✅ NVIDIA driver installed:'",
      "nvidia-smi",

      // Install NVIDIA Container Toolkit
      "echo '[3/7] Installing NVIDIA Container Toolkit...'",
      `curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \\
  tee /etc/yum.repos.d/nvidia-container-toolkit.repo`,
      "yum install -y nvidia-container-toolkit",
      "nvidia-ctk runtime configure --runtime=docker",
      "systemctl restart docker",
      "echo '[3/7] ✅ NVIDIA Container Toolkit installed'",

      // Authenticate with ECR
      "echo '[4/7] Authenticating with ECR...'",
      `aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com`,
      "echo '[4/7] ✅ ECR login succeeded'",

      // Pull worker image
      "echo '[5/7] Pulling worker image...'",
      `docker pull ${repoUri}:latest`,
      "echo '[5/7] ✅ Image pulled'",

      // Run worker container with GPU support, passing env vars
      "echo '[6/7] Starting worker container...'",
      `docker run -d --restart=on-failure \\
  --gpus all \\
  --name barbuddy-worker \\
  -e QUEUE_URL="${props.jobQueue.queueUrl}" \\
  -e BUCKET_NAME="${props.bucket.bucketName}" \\
  -e JOBS_TABLE_NAME="${props.jobsTable.tableName}" \\
  -e AWS_DEFAULT_REGION="${region}" \\
  -e WORKER_MODE=sqs \\
  -e VIZ=1 \\
  ${repoUri}:latest`,

      // Verify container started
      "echo '[7/7] Verifying...'",
      "sleep 5",
      "docker ps",
      "docker logs barbuddy-worker 2>&1 | head -20",
      "echo '=============================='",
      "echo \"Bootstrap complete: $(date -u)\"",
      "echo '=============================='",
    );

    // ── Launch Template ──────────────────────
    const launchTemplate = new ec2.LaunchTemplate(this, "WorkerLaunchTemplate", {
      instanceType: new ec2.InstanceType("g4dn.xlarge"),
      machineImage: ec2.MachineImage.latestAmazonLinux2(),
      userData,
      securityGroup: workerSg,
      role: props.ec2InstanceRole,
      blockDevices: [
        {
          deviceName: "/dev/xvda",
          volume: ec2.BlockDeviceVolume.ebs(50, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
          }),
        },
      ],
      spotOptions: {
        requestType: ec2.SpotRequestType.ONE_TIME,
      },
    });

    // ── Auto Scaling Group (0 → 1) ───────────
    this.asg = new autoscaling.AutoScalingGroup(this, "WorkerAsg", {
      vpc: this.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      launchTemplate,
      minCapacity: 0,
      maxCapacity: 1,
      desiredCapacity: 0,
      newInstancesProtectedFromScaleIn: false,
      // Health check grace period — give time for Docker pull + CUDA init
      healthCheck: autoscaling.HealthCheck.ec2({
        grace: Duration.minutes(10),
      }),
    });

    new CfnOutput(this, "AsgName", { value: this.asg.autoScalingGroupName });

    // ── Auto-scaling based on SQS queue depth ─
    // Scale up to 1 when messages arrive, scale down to 0 when queue is empty
    this.asg.scaleOnMetric("ScaleOnQueueDepth", {
      metric: props.jobQueue.metricApproximateNumberOfMessagesVisible({
        period: Duration.minutes(1),
        statistic: "Maximum",
      }),
      adjustmentType: autoscaling.AdjustmentType.EXACT_CAPACITY,
      scalingSteps: [
        { upper: 0, change: 0 },   // 0 messages → 0 instances
        { lower: 1, change: 1 },   // 1+ messages → 1 instance
      ],
    });
  }
}
