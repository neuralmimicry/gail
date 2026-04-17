# Gail

Gail is the shared AI middleware for NeuralMimicry services. It consolidates the LLM routing, provider orchestration, neuromorphic specialist access, AER translation, transcription, and orchestration-status surfaces that were previously embedded inside Refiner.

The service is designed so Refiner can delegate immediately, while Tracey and Continuum/NMC can consume the same HTTP contract without re-implementing provider selection or neuromorphic transport glue.

## Goals

- Remove duplicated LLM and neuromorphic service-interface code from product repositories.
- Keep high-level workflow logic in each product and move cross-cutting provider/service routing into one Rust service.
- Preserve Refiner features such as concurrent provider selection, persisted metrics, Ollama inventory visibility, transcription, and neuromorphic specialist routing.
- Improve latency and efficiency centrally so optimisations benefit every Gail client.

## Runtime Surface

Gail exposes the following endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Service health |
| `POST /v1/llm/complete` | Workflow-aware multi-provider orchestration |
| `POST /v1/llm/direct-complete` | Direct provider invocation without orchestration |
| `POST /v1/llm/transcribe` | Speech-to-text proxy |
| `POST /v1/neuromorphic/analyze` | Specialist-engine task analysis |
| `POST /v1/neuromorphic/predict` | Neuromorphic score/prediction |
| `POST /v1/aer/encode` | Encode spikes/events into `AER1` payloads |
| `POST /v1/aer/decode` | Decode `AER1` payloads into spikes/events |
| `GET /v1/status/orchestration` | Provider, engine, and metrics status |

## What Moved From Refiner

| Capability | Gail | Refiner |
| --- | --- | --- |
| Direct provider HTTP adapters | Owns | Delegates |
| Concurrent provider orchestration | Owns | Delegates |
| Provider scoring and early-success selection | Owns | Delegates |
| Provider metrics persistence | Owns | Delegates |
| Ollama inventory and local-model visibility | Owns | Delegates |
| Transcription proxying | Owns | Delegates |
| Neuromorphic specialist routing | Owns | Delegates |
| AER encode/decode | Owns | Delegates |
| Workflow prompting and business logic | Shared input only | Owns |
| Jobs, UI, RAG, MCP, Jira, Confluence workflows | Not moved | Owns |

## Security Model

- Bearer-token authentication with per-client IDs.
- Route scopes: `health`, `llm`, `neuromorphic`, `aer`, `status`.
- `/healthz` can be configured to allow or deny unauthenticated probes.
- The supplied Ansible role publishes Gail behind TLS ingress and injects per-client bearer tokens for Refiner, Tracey, and Continuum/NMC.

## Local Development

Run Gail directly:

```bash
cargo run -- --config gail.yaml
```

Run the Rust tests:

```bash
cargo test
```

The bundled [`gail.yaml`](./gail.yaml) is an example configuration. It supports `${ENV_VAR}` interpolation so secrets can stay outside the file.

## Container Image

Build the container image locally:

```bash
podman build -t ghcr.io/neuralmimicry/gail:latest .
```

The image expects a config file at `/app/config/gail.yaml` unless `GAIL_CONFIG` is overridden.

## Configuration Notes

- `providers`: shared LLM backends Gail can orchestrate.
- `specialists`: explicit neuromorphic engines. Use this when you have named SNN/AARNN backends to register.
- `config/ai-routing-profiles.json`: shared workflow/keyword/provider routing contract used by Gail and mirrored in Refiner for offline fallback.
- `GAIL_ROUTING_PROFILES_PATH`: optional override for the routing contract path.
- `GAIL_AARNN_*` env vars: optional legacy auto-attach path for an AARNN backend, mirroring Refiner's previous automatic fallback behavior.
- `storage.metrics_path`: persisted provider quality/latency metrics.
- `storage.ollama_model_store_path`: cached Ollama model inventory summary.

## Product Integration

### Refiner

Refiner now keeps its `LLMProvider` interface stable while routing direct and workflow-mode requests through Gail when `REFINER_GAIL_ENABLED=1` is set.

### Tracey

Tracey can consume Gail's neuromorphic and AER endpoints as a stable HTTP contract instead of embedding its own cross-service adapters. The service boundary is documented and tokenised so Tracey can attach later without copying Refiner internals.

### Continuum / NMC

NMC already owns AARNN and cloud-control orchestration concerns. Gail sits alongside that stack as the shared AI middleware layer for LLM, neuromorphic scoring, and AER translation so Continuum-facing tooling can call one service contract instead of product-specific glue.

## Deployment

The Ansible role in `swarmhpc/swarmhpc/ansible/roles/continuum_tenant_gail`:

- builds or syncs the Gail source tree,
- builds and pushes the container image,
- renders the Gail config into a Kubernetes Secret,
- ships the shared AI-routing contract with the Gail runtime image,
- persists Gail metrics on shared storage,
- exposes Gail via ingress/TLS, and
- injects a matching bearer-token configuration into Refiner.
