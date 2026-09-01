"""CDK stack for the stateful serverless ML inference pipeline (S3 Files pattern).

Provisions:
  * A VPC with private (egress) subnets in two Availability Zones.
  * A versioned S3 bucket that stores the ML models under the ``models/`` prefix.
  * An Amazon S3 Files file system backed by the bucket, mount targets in each
    private subnet, and an access point rooted at ``/models`` (UID/GID 1000:1000).
  * Security groups that permit NFS (TCP 2049) only between the Lambda ENIs and
    the S3 Files mount targets.
  * Three ONNX inference Lambda functions (``predict``, ``model_a``, ``model_b``)
    that mount the S3 Files access point at ``/mnt/models`` and read the model via
    memory-mapped I/O, plus a lightweight ensemble ``combiner`` function.
  * An AWS Step Functions Express state machine that runs the two ensemble models
    in parallel and combines their predictions.
  * A REST API with ``POST /predict`` (single model) and ``POST /ensemble``
    (synchronous Step Functions execution).

Amazon S3 Files currently exposes only L1 (CloudFormation) constructs in CDK
(``aws_cdk.aws_s3files``), so the file system, mount targets, and access point
are created with ``Cfn*`` constructs and wired together manually.
"""
from __future__ import annotations

import os

from aws_cdk import (
    Aws,
    BundlingOptions,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigateway as apigateway,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3files as s3files,
    aws_stepfunctions as sfn,
)
from constructs import Construct

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAMBDA_DIR = os.path.join(PROJECT_ROOT, "lambdas")
ASL_FILE = os.path.join(PROJECT_ROOT, "stepfunctions", "ensemble_workflow.asl.json")

# Where the S3 Files file system is mounted inside every inference function.
MODEL_MOUNT_PATH = "/mnt/models"
# Access point root directory (maps to the bucket's ``models/`` prefix).
ACCESS_POINT_ROOT = "/models"
POSIX_UID = "1000"
POSIX_GID = "1000"
NFS_PORT = 2049


class MlInferenceStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- Tunable context values -------------------------------------
        provisioned_concurrency = int(
            self.node.try_get_context("provisionedConcurrency") or 0
        )
        lambda_memory = int(self.node.try_get_context("lambdaMemoryMb") or 2048)
        cache_enabled = str(self.node.try_get_context("cacheEnabled") or "true")
        cache_ttl = str(self.node.try_get_context("cacheTtlSeconds") or 3600)
        ensemble_strategy = str(self.node.try_get_context("ensembleStrategy") or "average")
        use_interface_endpoints = bool(
            self.node.try_get_context("useInterfaceEndpoints") or False
        )
        # Default to IAM so an omitted context key cannot silently expose an
        # unauthenticated public API; set apiAuth=NONE explicitly to opt out.
        api_auth = str(self.node.try_get_context("apiAuth") or "IAM").upper()

        # Match the Lambda architecture to the Docker bundling platform so the
        # installed wheels match the runtime. Defaults to arm64 (Graviton),
        # which also builds natively on Apple Silicon machines (no emulation).
        lambda_arch = str(
            self.node.try_get_context("lambdaArchitecture") or "arm64"
        ).lower()
        if lambda_arch in ("x86_64", "amd64", "x86-64"):
            self._lambda_architecture = lambda_.Architecture.X86_64
            self._bundling_platform = "linux/amd64"
        else:
            self._lambda_architecture = lambda_.Architecture.ARM_64
            self._bundling_platform = "linux/arm64"

        # ---- Networking --------------------------------------------------
        vpc = ec2.Vpc(
            self,
            "InferenceVpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # A free S3 gateway endpoint keeps model traffic off the NAT Gateway.
        vpc.add_gateway_endpoint(
            "S3GatewayEndpoint", service=ec2.GatewayVpcEndpointAwsService.S3
        )

        # Optional interface endpoints for a NAT-free deployment (extra cost).
        if use_interface_endpoints:
            for name, svc in (
                ("StepFunctionsEndpoint", ec2.InterfaceVpcEndpointAwsService.STEP_FUNCTIONS),
                ("CloudWatchLogsEndpoint", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS),
                ("MonitoringEndpoint", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_MONITORING),
                ("XRayEndpoint", ec2.InterfaceVpcEndpointAwsService.XRAY),
            ):
                vpc.add_interface_endpoint(name, service=svc)

        # ---- Model storage bucket (versioning is required by S3 Files) ---
        model_bucket = s3.Bucket(
            self,
            "ModelBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            # RETAIN protects model artifacts from accidental deletion on
            # `cdk destroy`. Empty and delete the bucket manually to fully clean up.
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ---- Security groups --------------------------------------------
        lambda_sg = ec2.SecurityGroup(
            self,
            "LambdaSecurityGroup",
            vpc=vpc,
            description="Lambda inference functions (S3 Files NFS client)",
            allow_all_outbound=True,
        )
        mount_target_sg = ec2.SecurityGroup(
            self,
            "MountTargetSecurityGroup",
            vpc=vpc,
            description="S3 Files mount targets",
            allow_all_outbound=False,
        )
        # Only allow NFS from the Lambda security group to the mount targets.
        mount_target_sg.add_ingress_rule(
            peer=lambda_sg,
            connection=ec2.Port.tcp(NFS_PORT),
            description="NFS from inference Lambda functions",
        )

        # ---- S3 Files file system, mount targets, access point ----------
        file_system, access_point, mount_targets = self._create_s3_files(
            model_bucket, vpc, mount_target_sg
        )

        # ---- Inference Lambda functions ---------------------------------
        common_env = {
            "MODEL_MOUNT_PATH": MODEL_MOUNT_PATH,
            "CACHE_ENABLED": cache_enabled,
            "CACHE_DIR": f"{MODEL_MOUNT_PATH}/cache",
            "CACHE_TTL_SECONDS": cache_ttl,
            "POWERTOOLS_METRICS_NAMESPACE": "ServerlessMLInference",
            "POWERTOOLS_LOG_LEVEL": "INFO",
        }

        predict_alias = self._create_inference_function(
            "Predict",
            model_file="model.onnx",
            model_name="predict",
            vpc=vpc,
            lambda_sg=lambda_sg,
            access_point=access_point,
            mount_targets=mount_targets,
            model_bucket=model_bucket,
            memory=lambda_memory,
            env=common_env,
            provisioned_concurrency=provisioned_concurrency,
        )
        model_a_alias = self._create_inference_function(
            "ModelA",
            model_file="model_a.onnx",
            model_name="model_a",
            vpc=vpc,
            lambda_sg=lambda_sg,
            access_point=access_point,
            mount_targets=mount_targets,
            model_bucket=model_bucket,
            memory=lambda_memory,
            env=common_env,
            provisioned_concurrency=provisioned_concurrency,
        )
        model_b_alias = self._create_inference_function(
            "ModelB",
            model_file="model_b.onnx",
            model_name="model_b",
            vpc=vpc,
            lambda_sg=lambda_sg,
            access_point=access_point,
            mount_targets=mount_targets,
            model_bucket=model_bucket,
            memory=lambda_memory,
            env=common_env,
            provisioned_concurrency=provisioned_concurrency,
        )

        # ---- Ensemble combiner (no mount / no VPC needed) ---------------
        combiner_fn = lambda_.Function(
            self,
            "CombinerFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=self._lambda_architecture,
            handler="handler.handler",
            code=self._bundle(os.path.join(LAMBDA_DIR, "combiner")),
            memory_size=512,
            timeout=Duration.seconds(30),
            environment={
                "ENSEMBLE_STRATEGY": ensemble_strategy,
                "POWERTOOLS_METRICS_NAMESPACE": "ServerlessMLInference",
                "POWERTOOLS_LOG_LEVEL": "INFO",
                "POWERTOOLS_SERVICE_NAME": "ensemble-combiner",
            },
            tracing=lambda_.Tracing.ACTIVE,
        )

        # ---- Step Functions Express workflow ----------------------------
        state_machine = self._create_state_machine(
            model_a_alias, model_b_alias, combiner_fn, ensemble_strategy
        )

        # ---- REST API ----------------------------------------------------
        self._create_api(predict_alias, state_machine, api_auth)

        # ---- Outputs -----------------------------------------------------
        CfnOutput(self, "ModelBucketName", value=model_bucket.bucket_name)
        CfnOutput(self, "ModelBucketPrefix", value="models/")
        CfnOutput(self, "S3FilesFileSystemId", value=file_system.attr_file_system_id)
        CfnOutput(self, "S3FilesAccessPointArn", value=access_point.attr_access_point_arn)
        CfnOutput(self, "EnsembleStateMachineArn", value=state_machine.state_machine_arn)

    # ------------------------------------------------------------------ #
    # S3 Files resources
    # ------------------------------------------------------------------ #
    def _create_s3_files(self, model_bucket, vpc, mount_target_sg):
        """Create the S3 Files file system, mount targets, and access point."""
        # Role that S3 Files assumes to synchronize data between S3 and the
        # NFS file system. It needs S3 data access plus permission to manage the
        # EventBridge rules (prefixed DO-NOT-DELETE-S3-Files) that detect object
        # changes.
        sync_role = iam.Role(
            self,
            "S3FilesSyncRole",
            assumed_by=iam.ServicePrincipal("elasticfilesystem.amazonaws.com"),
            description="Role assumed by Amazon S3 Files to sync the model bucket",
        )
        sync_policy = iam.Policy(
            self,
            "S3FilesSyncPolicy",
            roles=[sync_role],
            statements=[
                iam.PolicyStatement(
                    sid="S3BucketList",
                    actions=["s3:ListBucket*", "s3:GetBucketLocation"],
                    resources=[model_bucket.bucket_arn],
                ),
                iam.PolicyStatement(
                    sid="S3ObjectAccess",
                    actions=[
                        "s3:AbortMultipartUpload",
                        "s3:DeleteObject",
                        "s3:GetObject*",
                        "s3:List*",
                        "s3:PutObject*",
                    ],
                    resources=[model_bucket.arn_for_objects("*")],
                ),
                iam.PolicyStatement(
                    sid="EventBridgeManageSyncRules",
                    actions=[
                        "events:DeleteRule",
                        "events:DisableRule",
                        "events:EnableRule",
                        "events:PutRule",
                        "events:PutTargets",
                        "events:RemoveTargets",
                    ],
                    resources=[
                        f"arn:{Aws.PARTITION}:events:*:*:rule/DO-NOT-DELETE-S3-Files*"
                    ],
                    conditions={
                        "StringEquals": {
                            "events:ManagedBy": "elasticfilesystem.amazonaws.com"
                        }
                    },
                ),
                iam.PolicyStatement(
                    sid="EventBridgeReadRules",
                    actions=[
                        "events:DescribeRule",
                        "events:ListRuleNamesByTarget",
                        "events:ListRules",
                        "events:ListTargetsByRule",
                    ],
                    resources=[f"arn:{Aws.PARTITION}:events:*:*:rule/*"],
                ),
            ],
        )

        file_system = s3files.CfnFileSystem(
            self,
            "S3FilesFileSystem",
            bucket=model_bucket.bucket_arn,
            role_arn=sync_role.role_arn,
        )
        # The sync role (and its policy) must exist before the file system.
        file_system.node.add_dependency(sync_policy)

        # One mount target per private subnet so every AZ can reach the
        # file system over NFS.
        mount_targets = []
        private_subnets = vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
        ).subnets
        for i, subnet in enumerate(private_subnets):
            mount_target = s3files.CfnMountTarget(
                self,
                f"MountTarget{i}",
                file_system_id=file_system.attr_file_system_id,
                subnet_id=subnet.subnet_id,
                security_groups=[mount_target_sg.security_group_id],
            )
            mount_targets.append(mount_target)

        access_point = s3files.CfnAccessPoint(
            self,
            "ModelsAccessPoint",
            file_system_id=file_system.attr_file_system_id,
            root_directory=s3files.CfnAccessPoint.RootDirectoryProperty(
                path=ACCESS_POINT_ROOT,
                creation_permissions=s3files.CfnAccessPoint.CreationPermissionsProperty(
                    owner_uid=POSIX_UID,
                    owner_gid=POSIX_GID,
                    permissions="750",
                ),
            ),
            posix_user=s3files.CfnAccessPoint.PosixUserProperty(
                uid=POSIX_UID, gid=POSIX_GID
            ),
        )
        return file_system, access_point, mount_targets

    # ------------------------------------------------------------------ #
    # Lambda helpers
    # ------------------------------------------------------------------ #
    def _bundle(self, asset_path: str) -> lambda_.Code:
        """Bundle a Python Lambda by pip-installing its requirements (needs Docker).

        The bundling container runs for the target Lambda architecture
        (``self._bundling_platform``), so the installed wheels match the runtime
        and, on Apple Silicon, the arm64 image runs natively instead of under
        emulation.
        """
        return lambda_.Code.from_asset(
            asset_path,
            bundling=BundlingOptions(
                image=lambda_.Runtime.PYTHON_3_12.bundling_image,
                platform=self._bundling_platform,
                command=[
                    "bash",
                    "-c",
                    "pip install --no-cache-dir -r requirements.txt -t /asset-output "
                    "&& cp -au . /asset-output",
                ],
            ),
        )

    def _create_inference_function(
        self,
        construct_id: str,
        *,
        model_file: str,
        model_name: str,
        vpc,
        lambda_sg,
        access_point,
        mount_targets,
        model_bucket,
        memory: int,
        env: dict,
        provisioned_concurrency: int,
    ) -> lambda_.Alias:
        """Create a VPC inference function that mounts the S3 Files access point."""
        fn = lambda_.Function(
            self,
            f"{construct_id}Function",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=self._lambda_architecture,
            handler="handler.handler",
            code=self._bundle(os.path.join(LAMBDA_DIR, "inference")),
            memory_size=memory,
            # >= 2x expected inference duration to absorb occasional NFS latency.
            timeout=Duration.minutes(2),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            security_groups=[lambda_sg],
            # Build the mount config directly rather than via
            # FileSystem.from_s3_files_access_point(): that helper creates a
            # child security group scoped to the access point, which collides
            # when several functions mount the same access point. We manage the
            # security groups and mount targets ourselves, so we only need the
            # access point ARN, the mount path, and a dependency on the mount
            # targets being created first.
            filesystem=lambda_.FileSystem(
                arn=access_point.attr_access_point_arn,
                local_mount_path=MODEL_MOUNT_PATH,
                dependency=mount_targets,
                # Supply the mount/write permission through the FileSystem config
                # (the same mechanism CDK uses for EFS): CDK attaches it to the
                # execution role and orders it ahead of the mount, with no extra
                # policy resource that could form a circular dependency.
                # s3files:ClientMount is evaluated against the file system, so
                # these actions use "*" (matching the AWS Lambda docs example);
                # scoping only to the access-point ARN yields a 403 at mount.
                policies=[
                    iam.PolicyStatement(
                        actions=["s3files:ClientMount", "s3files:ClientWrite"],
                        resources=["*"],
                    ),
                ],
            ),
            environment={
                **env,
                "MODEL_FILE": model_file,
                "MODEL_NAME": model_name,
                "POWERTOOLS_SERVICE_NAME": f"inference-{model_name}",
            },
            tracing=lambda_.Tracing.ACTIVE,
        )

        # Direct S3 reads used by the S3 Files direct-read optimization
        # (functions with >= 512 MB memory read model bytes straight from S3).
        # This is a runtime permission, so it can live in the role's default
        # policy without any create-time ordering concern.
        fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="S3DirectRead",
                actions=["s3:GetObject", "s3:GetObjectVersion"],
                resources=[model_bucket.arn_for_objects("*")],
            )
        )

        # Alias enables (optional) provisioned concurrency to remove the
        # cold-start NFS mount penalty for latency-sensitive endpoints.
        alias = lambda_.Alias(
            self,
            f"{construct_id}Alias",
            alias_name="live",
            version=fn.current_version,
            provisioned_concurrent_executions=(provisioned_concurrency or None),
        )
        return alias

    # ------------------------------------------------------------------ #
    # Step Functions
    # ------------------------------------------------------------------ #
    def _create_state_machine(
        self, model_a_alias, model_b_alias, combiner_fn, ensemble_strategy: str
    ) -> sfn.StateMachine:
        with open(ASL_FILE, "r", encoding="utf-8") as fh:
            asl_definition = fh.read()

        log_group = logs.LogGroup(
            self,
            "EnsembleWorkflowLogs",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        state_machine = sfn.StateMachine(
            self,
            "EnsembleStateMachine",
            state_machine_type=sfn.StateMachineType.EXPRESS,
            definition_body=sfn.DefinitionBody.from_string(asl_definition),
            definition_substitutions={
                "ModelAFunctionArn": model_a_alias.function_arn,
                "ModelBFunctionArn": model_b_alias.function_arn,
                "CombinerFunctionArn": combiner_fn.function_arn,
                "EnsembleStrategy": ensemble_strategy,
            },
            logs=sfn.LogOptions(
                destination=log_group,
                level=sfn.LogLevel.ALL,
                include_execution_data=True,
            ),
            tracing_enabled=True,
            timeout=Duration.minutes(5),
        )

        # Allow the workflow to invoke the model and combiner functions.
        model_a_alias.grant_invoke(state_machine)
        model_b_alias.grant_invoke(state_machine)
        combiner_fn.grant_invoke(state_machine)
        return state_machine

    # ------------------------------------------------------------------ #
    # API Gateway
    # ------------------------------------------------------------------ #
    def _create_api(self, predict_alias, state_machine, api_auth: str) -> None:
        authorization_type = (
            apigateway.AuthorizationType.IAM
            if api_auth == "IAM"
            else apigateway.AuthorizationType.NONE
        )

        api = apigateway.RestApi(
            self,
            "InferenceApi",
            rest_api_name="serverless-ml-inference",
            description="Inference API for the S3 Files serverless ML pattern",
            deploy_options=apigateway.StageOptions(
                stage_name="prod",
                tracing_enabled=True,
                metrics_enabled=True,
                logging_level=apigateway.MethodLoggingLevel.INFO,
            ),
        )

        # POST /predict -> single-model inference (Lambda proxy).
        predict = api.root.add_resource("predict")
        predict.add_method(
            "POST",
            apigateway.LambdaIntegration(predict_alias, proxy=True),
            authorization_type=authorization_type,
        )

        # POST /ensemble -> synchronous Step Functions Express execution.
        ensemble = api.root.add_resource("ensemble")
        ensemble.add_method(
            "POST",
            apigateway.StepFunctionsIntegration.start_execution(state_machine),
            authorization_type=authorization_type,
        )

        CfnOutput(self, "ApiUrl", value=api.url)
        CfnOutput(self, "PredictEndpoint", value=f"{api.url}predict")
        CfnOutput(self, "EnsembleEndpoint", value=f"{api.url}ensemble")
        if authorization_type == apigateway.AuthorizationType.NONE:
            CfnOutput(
                self,
                "ApiAuthWarning",
                value=(
                    "API methods are UNAUTHENTICATED (apiAuth=NONE). Set context "
                    "apiAuth=IAM (or add an authorizer) before exposing publicly."
                ),
            )
