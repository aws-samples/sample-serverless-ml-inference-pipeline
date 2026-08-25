# Testing guide

How to test the stateful serverless ML inference stack — every command, what it
does, and what to expect. Commands that call AWS run in a terminal where your
AWS credentials are configured/exported and the Region is `us-east-1`.

The commands below use placeholders — substitute the values from your own
stack's Outputs:

- **`<ModelBucketName>`** — the model bucket name (Output: `ModelBucketName`)
- **`<ApiId>`** — the API Gateway ID in the endpoint URLs (Outputs: `PredictEndpoint`, `EnsembleEndpoint`)
- **`<EnsembleStateMachineArn>`** — the Step Functions ARN (Output: `EnsembleStateMachineArn`)

Get your own values with:
```bash
aws cloudformation describe-stacks \
  --stack-name ServerlessMlInferenceS3FilesStack \
  --query "Stacks[0].Outputs" --output table
```
**What it does:** reads the deployed stack's outputs (bucket name, endpoint URLs,
file system id, state machine ARN) so you can plug them into the commands below.

---

## 0. Prerequisites for testing

```bash
pip install awscurl
```
**What it does:** installs `awscurl`, a curl-like tool that automatically
SigV4-signs requests with your AWS credentials. The API uses `AWS_IAM`
authorization, so requests must be signed; `awscurl` handles that for you. (Plain
`curl` would return 403 unless you sign manually.)

If you use temporary credentials, make sure they're exported and current:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
export AWS_DEFAULT_REGION=us-east-1
```
**What it does:** provides credentials/region to the AWS CLI and `awscurl`.
Temporary (STS) credentials expire — if calls start returning 403, refresh these.

---

## 1. Local unit tests (no AWS needed)

```bash
./.venv/bin/python -m pytest -q
```
**What it does:** runs the handler unit tests in `tests/`. It uses a fake ONNX
session, so it validates the inference handler's logic (input parsing for API
proxy / direct / Step Functions payloads, caching, validation errors) and the
combiner's ensemble strategies without any AWS calls or `onnxruntime`. `pytest.ini`
restricts collection to `tests/` so it never imports the Linux-built packages in
`cdk.out/`. Expect `12 passed`.

---

## 2. Confirm the models are in the bucket

```bash
aws s3 ls s3://<ModelBucketName>/models/ --region us-east-1
```
**What it does:** lists the objects under the bucket's `models/` prefix — the
location the S3 Files access point exposes to the functions as `/mnt/models`.
You should see `model.onnx`, `model_a.onnx`, `model_b.onnx`. If empty, upload
them:
```bash
aws s3 cp models_local/ s3://<ModelBucketName>/models/ \
  --recursive --exclude "*" --include "*.onnx" --region us-east-1
```
**What it does:** copies the three locally generated `.onnx` files into the
bucket. `--recursive` walks the folder; `--exclude "*" --include "*.onnx"` limits
it to model files. The functions read these at inference time, so this must be
done before the endpoints will return predictions.

---

## 3. Single-model inference — `POST /predict`

```bash
awscurl --service execute-api --region us-east-1 -X POST \
  -d '{"input_data":[5.1,3.5,1.4,0.2]}' \
  "https://<ApiId>.execute-api.us-east-1.amazonaws.com/prod/predict"
```
**What it does:** sends a SigV4-signed POST to the `/predict` endpoint.
`--service execute-api` tells `awscurl` which service to sign for;
`-d '{"input_data":[...]}'` is the request body (a 4-feature input vector). This
invokes the Predict Lambda, which memory-maps `model.onnx` from the S3 Files
mount and runs inference.

**Expect:**
```json
{"predictions": [[...]], "raw_outputs": [[[...]]], "input_shape": [1,4],
 "model": "predict", "cache_hit": false, "latency_ms": 537.3, "request_id": "..."}
```
The first call is slower (cold start: NFS mount + model load).

---

## 4. Prove the cross-invocation cache

```bash
awscurl --service execute-api --region us-east-1 -X POST \
  -d '{"input_data":[5.1,3.5,1.4,0.2]}' \
  "https://<ApiId>.execute-api.us-east-1.amazonaws.com/prod/predict"
```
**What it does:** repeats the **exact same** input. The first call wrote the
result (keyed by a SHA-256 of the input) to `/mnt/models/cache/` on the shared S3
Files mount; this call finds it and returns it directly.

**Expect:** `"cache_hit": true` and a noticeably lower `latency_ms`.

---

## 5. Multi-model ensemble — `POST /ensemble`

```bash
awscurl --service execute-api --region us-east-1 -X POST \
  -d '{"input_data":[6.2,2.9,4.3,1.3]}' \
  "https://<ApiId>.execute-api.us-east-1.amazonaws.com/prod/ensemble"
```
**What it does:** sends a signed POST to `/ensemble`, which starts a synchronous
Step Functions Express execution. The workflow runs `model_a` and `model_b`
Lambdas in parallel (each reading its own model from the mount), then a combiner
Lambda merges the results (default: average).

**Expect:**
```json
{"ensemble_strategy":"average","model_count":2,
 "prediction":{"label":1,"probabilities":[...]},
 "members":[{"model":"model_a","label":1,"latency_ms":...,"cache_hit":false},
            {"model":"model_b","label":1,"latency_ms":...,"cache_hit":false}]}
```

> Note: the bundled models have **random weights** (from `generate_model.py`), so
> the specific `label` is not meaningful — the calls demonstrate the mechanics.
> Upload your own `.onnx` files to `models/` to run real predictions (no redeploy).

---

## 6. Try different inputs (optional)

```bash
awscurl --service execute-api --region us-east-1 -X POST \
  -d '{"input_data":[4.9,3.0,1.4,0.2]}' \
  "https://<ApiId>.execute-api.us-east-1.amazonaws.com/prod/predict"
```
**What it does:** same as the predict test with a different feature vector; a new
(uncached) input, so `cache_hit` is `false` again.

**Validation check** — send a bad payload to see the 400 path:
```bash
awscurl --service execute-api --region us-east-1 -X POST \
  -d '{"foo":"bar"}' \
  "https://<ApiId>.execute-api.us-east-1.amazonaws.com/prod/predict"
```
**What it does:** omits `input_data`; the handler returns HTTP 400 with
`{"error": "Missing required field 'input_data'."}`.

---

## 7. Inspect behind the scenes (optional)

**Recent Step Functions executions:**
```bash
aws stepfunctions list-executions \
  --state-machine-arn <EnsembleStateMachineArn> \
  --max-results 5 --region us-east-1
```
**What it does:** lists recent ensemble workflow runs (Express executions appear
if logging is enabled; use the state machine ARN from your Outputs).

**Tail a function's logs:**
```bash
aws logs tail /aws/lambda/<PredictFunctionName> --since 10m --region us-east-1
```
**What it does:** streams the last 10 minutes of a function's CloudWatch logs
(structured JSON from Powertools, including EMF metrics and any errors). Find the
exact function name in the Lambda console or via `aws lambda list-functions`.

**Direct Lambda invoke (bypasses API Gateway/SigV4):**
```bash
aws lambda invoke --function-name <PredictFunctionName> \
  --payload '{"input_data":[5.1,3.5,1.4,0.2]}' \
  --cli-binary-format raw-in-base64-out /tmp/out.json --region us-east-1 && cat /tmp/out.json
```
**What it does:** calls the function directly with a raw payload (no API layer).
`--cli-binary-format raw-in-base64-out` lets you pass the JSON payload as-is; the
response is written to `/tmp/out.json`. Useful to isolate whether an issue is in
the function or the API/auth layer.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `403` / auth error | Temporary creds expired, or caller lacks `execute-api:Invoke` | Refresh credentials; ensure the principal can invoke the API |
| `500` + "Model not found" in logs | Models not uploaded to `models/` | Re-run the upload in step 2 |
| `Missing required field 'input_data'` on `/ensemble` | Old function code (pre body-unwrap fix) | Redeploy: `cdk deploy` |
| `Could not connect to the endpoint URL` | Transient network / DNS / proxy | Retry; add `--region us-east-1`; check `HTTPS_PROXY` |
| First `/predict` is slow (~0.5–2 s) | Cold start (NFS mount + model load) | Expected; set `provisionedConcurrency>=1` for low latency |
| `cache_hit` always false | `cacheEnabled=false`, or write perms/TTL | Check `cacheEnabled`, `cacheTtlSeconds`, and `s3files:ClientWrite` |

---

## 9. Clean up when done

```bash
cdk destroy
aws s3 rm s3://<ModelBucketName> --recursive
aws s3 rb s3://<ModelBucketName>
```
**What it does:** `cdk destroy` deletes the stack (VPC, NAT gateway, Lambdas,
Step Functions, API, S3 Files resources). The model bucket is `RETAIN`, so the
two `aws s3` commands empty and delete it to stop all charges. Also rotate any
AWS credentials you may have shared during setup.
