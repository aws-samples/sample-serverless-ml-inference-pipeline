#!/usr/bin/env python3
"""CDK application entry point.

Deploys the stateful serverless ML inference pipeline described in the APG
pattern "Build stateful serverless ML inference pipelines using Lambda and
S3 Files".
"""
import os

import aws_cdk as cdk

from infrastructure.ml_inference_stack import MlInferenceStack

app = cdk.App()

MlInferenceStack(
    app,
    "ServerlessMlInferenceS3FilesStack",
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION"),
    ),
    description=(
        "Stateful serverless ML inference pipeline using AWS Lambda and "
        "Amazon S3 Files (APG cloud-native pattern)."
    ),
)

app.synth()
