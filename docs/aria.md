# Aria governance

Gail supports Aria over a versioned HTTP contract without importing the Aria project. Configure `governance` in `gail.yaml`:

```yaml
governance:
  mode: enforce
  aria_url: http://aria.aria.svc.cluster.local:8091
  evaluation_token: "${ARIA_EVALUATION_TOKEN}"
  assessment_token: "${ARIA_GAIL_TOKEN}"
  timeout_ms: 20000
  assessment_timeout_ms: 12000
  max_body_bytes: 262144
  max_in_flight: 32
  fail_open: false
```

The default mode is `disabled`. `monitor` evaluates traffic and reports would-be blocks; `enforce` withholds blocked requests and responses. Assessment and evaluation credentials require at least 32 characters and must differ. Never add the assessment credential to ordinary `security.api_tokens`.

The gateway checks both sides of supported AI HTTP routes, including compatible chat and response APIs, and its in-process completion APIs. Content inspection runs before response bytes are released. Streamed bodies are buffered with a fixed cap; unsupported binary content is marked uninspectable and handled by Aria policy. Size and concurrency limits still apply in monitor mode. Source attribution comes from the authenticated client; caller-supplied bypass headers are ignored.

AER dense decoding also has an independent 16 MiB allocation limit, including automatically inferred lengths from sparse addresses. This applies in every mode; larger workloads can retain sparse event representations. Numeric resource budgets and execution flags remain visible in the text sent for assessment.

Aria calls `/v1/internal/aria/assess` using its separate credential. Gail fixes the classification instructions, uses configured providers, restricts output to a validated risk schema and bounds execution. Only an internal task-local scope skips governance for this assessment; ordinary clients cannot set it. Classifier material is excluded from the content ledger and learning mirrors. Other Gail audit and mirroring behaviour remains unchanged.

`GET /v1/status/governance` requires `status` scope and exposes mode and counters. Governed HTTP responses provide `X-Aria-Request-Id` for correlation and `X-Aria-Status`. Aria blocks produce HTTP 403 with `governance_error` and a decision ID. Fail-closed evaluation outages return 503. In-process callers receive a Gail upstream error attributed to `aria`.

Use the Aria dashboard for incident review, per-source blocking, thresholds and pause controls. A pause takes effect at evaluation boundaries and does not reverse completed operations. Keep service-level execution controls in place. The companion `continuum_governance_site.yml` deploys Aria, supplies matching Gail credentials and configures authenticated Prometheus monitoring.
