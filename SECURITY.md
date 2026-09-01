# Security

## Reporting a vulnerability

If you discover a potential security issue in this project, please notify
AWS/Amazon Security via the
[vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/)
or email aws-security@amazon.com. **Please do not create a public GitHub issue.**

## Important: this is sample (non-production) code

This repository is **sample/reference code** published to demonstrate the AWS
Prescriptive Guidance pattern "Build stateful serverless ML inference pipelines
using Lambda and S3 Files." It provisions real AWS infrastructure (VPC, S3, S3
Files, Lambda, Step Functions, API Gateway, IAM) but is **not intended for
production use without additional security review and hardening.**

It ships with strong baseline defaults — IAM-authenticated API by default,
S3 encryption + `BlockPublicAccess` + TLS enforcement + versioning, network
isolation for the NFS mount, and least-privilege IAM in most statements. The
items below are intentional trade-offs to keep the sample focused; review and
address them before any production deployment.

AWS services used: AWS Lambda, Amazon S3, Amazon S3 Files, AWS Step Functions,
Amazon API Gateway, Amazon VPC, Amazon CloudWatch, AWS IAM, Amazon EventBridge.

## Known security considerations (accepted trade-offs)

These are documented, accepted trade-offs for a sample release. Address them
within your own hardening process before production use.

1. **Model bucket uses SSE-S3, not a customer-managed KMS key (CMK).**
   Encryption *is* enabled, and the sample models are random-weight with no
   sensitive IP. *If unaddressed:* storing proprietary/regulated models under
   SSE-S3 loses key-level access control and independent key-usage auditing.

2. **S3 Files sync role uses service-scoped action wildcards.**
   The policy is already scoped to this bucket's objects; this is a
   documentation-hygiene note, not over-privilege. *If unaddressed:* copying the
   policy grants broader object mutation (Delete/Put) than a read-only serving
   use case needs.

3. **Lambda environment variables are not encrypted with a CMK.**
   The environment holds only non-sensitive configuration today. *If
   unaddressed:* any secret later added to the environment would be protected
   only by the default key, with no dedicated key policy.

4. **CloudWatch Logs are not encrypted with a CMK.**
   Logs contain operational metadata only — no payloads or PII. *If
   unaddressed:* if handlers are adapted to log request/response bodies,
   sensitive inference data would sit under default encryption.

5. **No S3 access logging on the model bucket.**
   Omitted to keep the sample focused. *If unaddressed:* no object-level access
   audit trail for the model bucket in production, hampering incident
   investigation.

6. **Runtime Lambda dependencies use open (lower-bound) version ranges.**
   Intentional "track latest" strategy for a sample. `onnxruntime` and `numpy`
   are capped below their next major; `aws-lambda-powertools` is capped below
   `3.0.0`. *If unaddressed:* deployed functions are not byte-reproducible, and a
   compromised patch/minor within an allowed range could be pulled in on a build.

## Production hardening recommendations

- **Encryption with customer-managed keys.** Create a CMK (with rotation) and set
  `encryption=s3.BucketEncryption.KMS` with `encryption_key=<cmk>` on the model
  bucket; scope the key policy to the Lambda execution roles and the S3 Files
  sync role. Use a CMK for Lambda environment encryption and on the CloudWatch
  log groups if they may contain sensitive data. Keep true secrets in AWS
  Secrets Manager or SSM Parameter Store — never in environment variables.
- **Narrow IAM for read-only serving.** The S3 Files sync role's object wildcard
  can be reduced to `s3:GetObject`, `s3:GetObjectVersion`, `s3:ListBucket` for a
  read-only model-serving use case. Add an inline comment noting the wildcards
  are broad by design and should be narrowed.
- **Audit logging.** Enable S3 server access logging (a dedicated access-log
  bucket) or CloudTrail S3 data events on the model bucket, with a retention
  lifecycle.
- **Deterministic dependencies.** Pin runtime and build/test dependencies to
  exact tested versions with `==` (commit a `pip-compile` lock file), refresh via
  reviewed dependency-update PRs (Dependabot/Renovate), and run `pip-audit`
  against all requirements files to assess CVEs.
- **Network.** Consider VPC interface endpoints (`useInterfaceEndpoints=true`) to
  remove the NAT gateway dependency for AWS service calls.
- **API protection.** Keep `apiAuth=IAM` (or add a Cognito/Lambda authorizer),
  and consider AWS WAF plus API Gateway usage plans/throttling for public-facing
  deployments.
- **Observability.** Add CloudWatch alarms on error rates and latency, and keep
  raw model inputs/outputs out of logs.

## Tested versions

| Component | Version |
|-----------|---------|
| AWS CDK (`aws-cdk-lib`) | >= 2.266 |
| Python (Lambda runtime) | 3.12 |
| ONNX Runtime | >= 1.17, < 2.0 |
| NumPy | >= 1.26, < 2.0 |
| AWS Lambda Powertools | >= 2.30, < 3.0 (`[tracer]` extra) |
