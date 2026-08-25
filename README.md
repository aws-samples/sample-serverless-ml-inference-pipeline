# Stateful serverless ML inference with AWS Lambda and Amazon S3 Files

A complete, deployable reference implementation of the AWS Prescriptive Guidance
(APG) pattern **"Build stateful serverless ML inference pipelines using Lambda
and S3 Files."** It runs CPU-based ONNX model inference on AWS Lambda, reading
the models directly from an **Amazon S3 Files** NFS mount (so you pay S3 prices
for model storage, never copy the model to `/tmp`, and skip per-invocation
downloads), with cross-invocation result caching on the shared mount and a
multi-model ensemble orchestrated by AWS Step Functions.

This repository is fully working and has been deployed and tested end to end.

---

## 1. What this is (in plain terms)

You send a small JSON payload of numbers to a REST API. That triggers a Lambda
function which loads a pre-trained ML model and returns a prediction. The novel
part of the pattern is *how the model reaches the function*:

- The model file lives in an **S3 bucket** (cheap storage: ~$0.023/GB).
- **Amazon S3 Files** mounts that bucket onto the Lambda function as a network
  (NFS) file system at `/mnt/models`.
- The function reads the model **straight from the mount** using memory-mapped
  I/O — no download, no `/tmp` copy — and keeps the loaded model in memory
  across warm invocations.
- Results can be **cached** back onto the same shared mount, so a prediction
  computed by one invocation can be reused by any other.
- A **Step Functions** workflow can run several models in parallel and combine
  their outputs (an ensemble).

`cdk deploy` builds all the infrastructure; the model files are uploaded
separately (they are data, not code), which means you can swap models later
without redeploying.

---

## 2. Architecture

```
Client
  |  POST /predict                              POST /ensemble
  v  (SigV4 / IAM auth)                          v  (SigV4 / IAM auth)
API Gateway ──────► Predict Lambda        API Gateway ──────► Step Functions (Express)
                       │                                          │  Parallel
                       │                                +---------+---------+
                       ▼                                ▼                   ▼
                 S3 Files mount                   ModelA Lambda       ModelB Lambda
                 /mnt/models/model.onnx                │                   │
                       │                               └──── S3 Files ─────┘
                       ▼                                 (memory-mapped .onnx reads)
             S3 bucket (models/ prefix)                        │
             + /mnt/models/cache/                       Combiner Lambda (average /
                                                        majority / inverse-latency)
                                                               │
                                                               ▼  ensemble prediction
```

Target technology stack:

| Layer | Service | Role |
|-------|---------|------|
| Compute | AWS Lambda (arm64/Graviton) | Serverless inference execution |
| Model storage | Amazon S3 + S3 Files | NFS-mounted model access at S3 pricing |
| Orchestration | AWS Step Functions (Express) | Parallel multi-model ensemble |
| API | Amazon API Gateway (REST) | `/predict` and `/ensemble` endpoints |
| Networking | Amazon VPC | Required for S3 Files NFS (port 2049) |
| Observability | Amazon CloudWatch | Structured logs + EMF custom metrics |
| Inference | ONNX Runtime + NumPy | Cross-framework model execution |
| IaC | AWS CDK v2 (Python) | Repeatable, version-controlled deploys |

---

## 3. How it works

**Request path (single model):** Client → API Gateway (`POST /predict`, IAM
auth) → Predict Lambda (Lambda proxy integration). The function memory-maps
`/mnt/models/model.onnx`, runs inference, optionally checks/writes the result
cache, and returns JSON with the prediction and latency.

**Request path (ensemble):** Client → API Gateway (`POST /ensemble`, IAM auth) →
Step Functions Express state machine. A `Parallel` state invokes `model_a` and
`model_b` Lambdas concurrently (each reads its own `.onnx` from the mount), then
a combiner Lambda merges the outputs (default strategy: average) and the result
is returned synchronously.

**S3 Files mount:** The stack creates an S3 Files file system backed by a
versioned bucket, one mount target per private subnet, and an access point
rooted at `/models` (POSIX UID/GID `1000:1000`), mounted on every function at
`/mnt/models`. `s3://<bucket>/models/model.onnx` therefore appears to the
function as `/mnt/models/model.onnx`.

**Warm-start optimization:** The ONNX `InferenceSession` is stored in a
module-level variable, so the model is loaded at most once per execution
environment; subsequent warm invocations skip loading.

**Caching:** Inputs are hashed with SHA-256; a matching, non-expired result file
under `/mnt/models/cache/` is returned directly (`cache_hit: true`). Cache writes
are best-effort (a read-only mount or missing `s3files:ClientWrite` simply skips
caching rather than failing).

**Observability:** Each function emits CloudWatch **EMF** metrics — inference
latency, cache hit/miss, model load time, cold starts — plus structured logs and
X-Ray tracing via AWS Lambda Powertools.

---

## 4. Repository layout

```
app.py                              CDK app entry point
cdk.json                            CDK config + tunable context
pytest.ini                          Test config (only collects tests/, ignores cdk.out)
infrastructure/
  ml_inference_stack.py             VPC, S3, S3 Files, Lambdas, Step Functions, API GW
lambdas/
  inference/handler.py              ONNX inference: module-level session, caching, EMF,
                                    handles API-proxy / direct / Step Functions payloads
  inference/requirements.txt        onnxruntime, numpy, aws-lambda-powertools[tracer]
  combiner/handler.py               ensemble combine (average / majority / inverse_latency)
  combiner/requirements.txt         numpy, aws-lambda-powertools
stepfunctions/
  ensemble_workflow.asl.json        Parallel(model_a, model_b) -> combine
scripts/
  generate_model.py                 build sample ONNX models (model, model_a, model_b)
  upload_models.sh                  upload models to the bucket's models/ prefix
events/                             sample invocation payloads
tests/                              unit tests for both handlers
```

---

## 5. Prerequisites

- An AWS account, AWS CLI v2 configured (or credentials exported as env vars).
- A Region that offers Amazon S3 Files (this project was validated in `us-east-1`).
- **AWS CDK v2 CLI** — a Node.js tool: `npm install -g aws-cdk` (needs Node.js).
  This is separate from the `aws-cdk-lib` Python library.
- **Python 3.11 or newer** for the local tooling. (Validated on 3.14 with
  `onnx`/`onnxruntime`/`numpy` wheels available and the unit tests passing; the
  Lambda runtime itself is pinned to 3.12 independently of your local version.)
- **A container engine** for CDK Lambda bundling — Docker Desktop **or**
  [Colima](https://github.com/abiosoft/colima) (`brew install colima docker && colima start`).

---

## 6. Deploy (end to end)

```bash
# From the project root: /path/to/serverless-APG

# 6.1 Credentials + region (either `aws configure`, or export env vars)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...          # only for temporary/STS credentials
export AWS_DEFAULT_REGION=us-east-1

# 6.2 Python environment for CDK + tooling (use python3.11/3.12 if available)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# 6.3 Make sure the container engine is running (Docker Desktop or Colima)
docker run --rm hello-world           # should succeed

# 6.4 Bootstrap the account/Region once (creates the CDK asset bucket + roles)
cdk bootstrap

# 6.5 Build the three sample models locally
python scripts/generate_model.py --out-dir models_local

# 6.6 Deploy the stack (bundles Lambda deps in Docker; approve the IAM prompt)
cdk deploy

# 6.7 Upload the models into the bucket (name is in the stack Outputs)
./scripts/upload_models.sh <ModelBucketName> models_local

# ...equivalent to, if you prefer the raw command:
# aws s3 cp models_local/ s3://<ModelBucketName>/models/ \
#   --recursive --exclude "*" --include "*.onnx" --region us-east-1
```

Get the deployed values any time:
```bash
aws cloudformation describe-stacks \
  --stack-name ServerlessMlInferenceS3FilesStack \
  --query "Stacks[0].Outputs" --output table
```
Key outputs: `ModelBucketName`, `PredictEndpoint`, `EnsembleEndpoint`.

Testing the endpoints is documented separately in **[TESTING.md](./TESTING.md)**.

---

## 7. Configuration (CDK context)

Set in `cdk.json` or with `--context key=value`:

| Key | Default | Purpose |
|-----|---------|---------|
| `provisionedConcurrency` | `0` | Pre-warm instances per function alias to remove cold-start mount latency. |
| `lambdaArchitecture` | `"arm64"` | `arm64` (Graviton) or `x86_64`. The Docker bundling platform is matched to it, so arm64 builds natively on Apple Silicon (no emulation). |
| `lambdaMemoryMb` | `2048` | Inference memory (>= 512 MB required for S3 Files direct reads). |
| `cacheEnabled` | `"true"` | Toggle inference-result caching on the shared mount. |
| `cacheTtlSeconds` | `3600` | Cache entry time-to-live. |
| `ensembleStrategy` | `"average"` | `average`, `majority`, or `inverse_latency`. |
| `useInterfaceEndpoints` | `false` | Add interface VPC endpoints for a NAT-light setup. |
| `apiAuth` | `"IAM"` | API methods require SigV4-signed (`AWS_IAM`) requests. Set to `NONE` to disable auth (not recommended). |

**Updating a model without redeploying:** upload a new `.onnx` to
`s3://<bucket>/models/` (same file name). Functions read the new version on the
next cold start.

---

## 8. Security

- **API auth is `AWS_IAM` by default** — requests must be SigV4-signed and the
  caller needs `execute-api:Invoke`. Overriding to `apiAuth=NONE` makes the
  endpoints public and emits an `ApiAuthWarning` output.
- **Least-privilege IAM:** execution roles get `s3files:ClientMount` /
  `ClientWrite` (mount/cache) and `s3:GetObject*` (direct reads). The mount
  actions use `Resource:"*"` per the AWS Lambda docs example, since
  `s3files:ClientMount` is evaluated against the file system.
- **NFS isolation:** the mount-target security group accepts TCP 2049 only from
  the Lambda security group.
- **Encryption:** the bucket uses SSE-S3; switch to SSE-KMS for sensitive model
  IP and scope the access point root per model family.
- **Networking:** Lambdas run in private subnets only; a NAT gateway (plus a free
  S3 gateway endpoint) provides egress.
- The model bucket uses `RemovalPolicy.RETAIN` to protect model artifacts.

---

## 9. Cost notes

- **S3 Files** stores models at S3 pricing (~$0.023/GB) vs. ~$0.30/GB for EFS.
- The stack runs a **NAT gateway** (hourly + per-GB) — the main idle cost — plus
  API Gateway, Lambda, Step Functions, S3, and CloudWatch.
- `provisionedConcurrency` defaults to `0` (no always-on Lambda charge). Enable
  it only for latency-sensitive endpoints.
- Result caching avoids recomputing inference for repeated inputs.

---

## 10. Cleanup

```bash
cdk destroy
# then remove the retained model bucket:
aws s3 rm s3://<ModelBucketName> --recursive
aws s3 rb s3://<ModelBucketName>
```
Failed create-rollbacks can leave orphaned empty buckets (RETAIN policy); delete
those from the S3 console if any accumulated.

---

## 11. When NOT to use this pattern

- **GPU inference** → Amazon SageMaker endpoints (Lambda has no GPU).
- **Sub-50 ms p99 latency** → SageMaker Serverless/Provisioned (NFS mount adds
  ~200–500 ms cold start).
- **Strongly consistent cross-invocation state** → DynamoDB or ElastiCache (the
  S3 Files NFS layer can briefly serve stale data).
- **Very large models (>50 GB)** → EFS-backed Lambda or SageMaker.
- **>15 min inference** → not possible within Lambda's max execution time.

---

## Product versions

| Product | Version |
|---------|---------|
| AWS CDK | v2 (aws-cdk-lib >= 2.266 for `aws_s3files`) |
| Python (Lambda runtime) | 3.12 |
| ONNX Runtime | >= 1.17 |
| NumPy | >= 1.26, < 2.0 |
| AWS Lambda Powertools | >= 2.30 (`[tracer]` extra) |
| Amazon S3 Files | GA (April 2026) |
