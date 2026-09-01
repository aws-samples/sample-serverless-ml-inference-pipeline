#!/usr/bin/env bash
# Upload the generated sample models to the S3 Files backing bucket.
#
# Usage:
#   ./scripts/upload_models.sh <ModelBucketName> [local_dir]
#
# The bucket name is printed as the `ModelBucketName` CloudFormation output
# after `cdk deploy`.
set -euo pipefail

BUCKET="${1:?Usage: upload_models.sh <ModelBucketName> [local_dir]}"
LOCAL_DIR="${2:-models_local}"

if [ ! -d "${LOCAL_DIR}" ]; then
  echo "Model directory '${LOCAL_DIR}' not found. Generate models first:"
  echo "  python scripts/generate_model.py --out-dir ${LOCAL_DIR}"
  exit 1
fi

echo "Uploading ${LOCAL_DIR}/*.onnx to s3://${BUCKET}/models/ ..."
aws s3 cp "${LOCAL_DIR}/" "s3://${BUCKET}/models/" --recursive --exclude "*" --include "*.onnx"
echo "Done. Objects under s3://${BUCKET}/models/:"
aws s3 ls "s3://${BUCKET}/models/"
