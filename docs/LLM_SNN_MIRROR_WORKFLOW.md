# Gail LLM/SNN Mirror Workflow

This document describes how Gail transforms LLM request/response text into SNN stimulation payloads for AARNN, how responses are fed back into orchestration, and how the same path is replayed asynchronously from the ledger worker.

## Scope

The mirror pipeline is implemented across:

- `src/orchestration.rs`
- `src/aarnn_bridge.rs`
- `src/llm_ledger.rs`
- `src/mirror_worker.rs`
- `src/aer.rs`
- `src/models.rs`

The runtime AARNN endpoint is `POST {aarnn_bridge.endpoint}/api/llm/mirror`.

## End-to-end flow

### 1) Inline input mirror (request path)

1. Orchestration receives an LLM request and flattens request text (`flatten_prompt_text`).
2. `build_aarnn_exchange(...)` creates an `AarnnMirrorExchange` with:
   - request metadata (`request_id`, `conversation_id`, `workflow`, `role`)
   - provider/model context
   - prompt/system text
   - flattened text and chat message roles
3. `spawn_aarnn_mirror(...)` enqueues the exchange on the bridge queue when input mirroring is enabled.
4. The queue worker calls `AarnnMirrorClient::mirror(...)`.
5. `build_request(...)` performs the LLM->SNN transform:
   - compact and truncate text fields
   - project text to a binary sensory spike vector (`text_to_spikes`)
   - encode active spikes into `AER1` bytes (`encode_spikes`)
   - serialize bytes as hex (`payload_hex`)
6. Gail posts `AarnnMirrorRequest` to `/api/llm/mirror` with retry/backoff for transient failures.
7. Returned invocation trace is attached to completion trace (`aarnn_mirroring.input`) if it arrives inside `candidate_wait_timeout_ms`.

### 2) Inline output mirror (response path)

1. After LLM completion, orchestration builds another `AarnnMirrorExchange` in `output` direction.
2. The same transform pipeline runs in `build_request(...)`, now on selected LLM response text.
3. If configured, Gail requests candidate reply data from AARNN (`request_candidate_reply=true` for output direction).
4. Response trace is attached to `aarnn_mirroring.output`.
5. If `response_preference=prefer_aarnn_when_confident`, Gail may replace the LLM text with the AARNN candidate when all gates pass:
   - candidate exists and is marked usable
   - candidate reply is non-empty and above `candidate_min_reply_chars`
   - candidate confidence >= `candidate_confidence_threshold`
   - candidate reply is not equivalent to the original LLM text

### 3) Deferred replay mirror (worker path)

1. Every completion is persisted into the LLM ledger (`llm_ledger`).
2. `mirror_worker` polls Postgres rows where mirroring is pending.
3. For each row:
   - replays input mirroring from `prompt_text`
   - replays output mirroring from `response_text` (if present)
4. Row state is updated:
   - `mark_mirror_success(...)` on full success
   - `mark_mirror_retry(...)` on errors, with bounded retry attempts and backoff

This gives both immediate mirroring for active requests and eventual replay for durability/recovery.

## Transformation details (LLM text -> SNN payload)

The transformation in `AarnnMirrorClient::build_request(...)` is deterministic:

1. **Normalize text**
   - collapse whitespace (`compact_text`)
   - apply `max_text_chars` truncation
2. **Generate sensory spikes**
   - `text_to_spikes(text, sensory_size)` maps bytes into deterministic spike indices
   - output is `Vec<u8>` with `0/1` values of length `sensory_size`
3. **Encode to AER**
   - `encode_spikes(ts_us, aer_sensory_base, sensory_spikes)` writes `AER1` event payload
   - each active spike becomes an event at `aer_sensory_base + index`
4. **Serialize transport payload**
   - raw bytes are hex-encoded into `aer_payload_hex`
   - both `sensory_spikes` and `aer_payload_hex` are sent in `AarnnMirrorRequest`

## Request/response contract

`AarnnMirrorRequest` carries:

- source metadata (`request_id`, `conversation_id`, `workflow`, `role`, `direction`)
- LLM context (`provider`, `model`, `request_category`, `system`, `prompt_text`, `text`)
- role context (`message_roles`)
- SNN addressing (`aer_base`, `output_base`)
- transformed payload (`sensory_spikes`, `aer_payload_hex`)
- optional targeting (`network_id`, `node_id`)
- candidate reply request flag (`request_candidate_reply`)

`AarnnMirrorResponse` may carry:

- accept/reject status (`accepted`)
- response-level counts (`text_chars`, `spike_count`)
- response payload (`aer_payload_hex`)
- optional candidate reply (`candidate.reply_text`, confidence, output spikes/payload)
- stimulation execution details (`stimulation`)

## Non-blocking and failure behaviour

- Mirroring runs through a bounded async queue (`queue_capacity`, `worker_count`).
- Enqueue uses `enqueue_timeout_ms`; on saturation, exchanges are dropped quickly and request flow continues.
- HTTP failures use bounded retries (`request_max_attempts`) with exponential backoff capped by `request_backoff_max`.
- Retryable mirror HTTP statuses: `408, 425, 429, 500, 502, 503, 504`.
- Candidate-waiting is bounded by `candidate_wait_timeout_ms`; if timeout is reached, orchestration returns the LLM result without waiting further.

## Audit and observability

When `audit_logging.enabled=true`, the mirror path emits:

- `GAIL_AUDIT_LLM_INTERACTION` (LLM request/response ledger record)
- `GAIL_AUDIT_AARNN_MIRROR_REQUEST` (mirrored payload and metadata)
- `GAIL_AUDIT_AARNN_MIRROR_RESPONSE` (AARNN response/candidate metadata)
- `GAIL_AUDIT_AARNN_MIRROR_ERROR` (mirror error details)

When `audit_logging.log_aer_payloads=true`, AER hex payloads and active spike indices are included (truncated by `audit_logging.max_chars`).

## Operational checks

- Confirm bridge status: `GET /v1/status/orchestration` (`aarnn_bridge` section).
- Confirm live traces on completions: `aarnn_mirroring.input` and `aarnn_mirroring.output`.
- Confirm worker replay health via logs:
  - `kubectl ... logs deployment/gail`
  - `kubectl ... logs deployment/gail-mirror-worker`
  - `kubectl ... logs deployment/gail-trainer-worker`
