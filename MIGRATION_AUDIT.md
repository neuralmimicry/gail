# Gail Migration Audit

This audit compares the Refiner service-integration surface against the new Gail middleware and records where each capability now lives.

## Migration Summary

Gail replaces the duplicated Refiner-side integration layer for:

- direct LLM provider HTTP adapters,
- concurrent provider orchestration and scoring,
- early-success selection,
- persisted provider metrics,
- transcription proxying,
- neuromorphic specialist analysis and prediction,
- AER encode/decode transport helpers, and
- orchestration / model-inventory status reporting.

Refiner retains ownership of workflow logic, prompt design, RAG, MCP, web/API jobs, Jira/Confluence logic, and product-specific orchestration.

## Feature Crosswalk

| Previous Refiner capability | Previous location | Gail replacement | Status |
| --- | --- | --- | --- |
| Direct OpenAI/Gemini/Ollama completion | `rag_demo/llm_providers.py` | `gail/src/providers/*`, `POST /v1/llm/direct-complete` | Migrated |
| Workflow-aware multi-provider orchestration | `rag_demo/refiner_ai_orchestration.py` | `gail/src/orchestration.rs`, `POST /v1/llm/complete` | Migrated |
| Provider specialty scoring and routing profiles | `rag_demo/refiner_ai_orchestration.py` | `gail/src/orchestration.rs` routing profiles and task tags | Migrated |
| Best vs fastest selection modes | `rag_demo/refiner_ai_orchestration.py` | `gail/src/orchestration.rs` selection mode handling | Migrated |
| Early-success return with settle window | `rag_demo/refiner_ai_orchestration.py` | `gail/src/orchestration.rs` early-success settings | Migrated |
| Persistent provider metrics registry | `rag_demo/refiner_ai_orchestration.py` | `gail/src/metrics.rs`, persisted via `storage.metrics_path` | Migrated |
| Ollama local-model visibility / inventory status | `rag_demo/refiner_ai_model_inventory.py` | `gail/src/providers/ollama.rs`, `GET /v1/status/orchestration` | Migrated |
| STT transcription proxy | `rag_demo/llm_providers.py` | `gail/src/providers/*`, `POST /v1/llm/transcribe` | Migrated |
| Specialist-engine task analysis | `rag_demo/refiner_ai_specialists.py` | `gail/src/specialists.rs`, `POST /v1/neuromorphic/analyze` | Migrated |
| Neuromorphic prediction / spike scoring | `rag_demo/refiner_ai_aarnn.py` | `gail/src/specialists.rs`, `POST /v1/neuromorphic/predict` | Migrated |
| AER1 encode/decode helpers | `rag_demo/refiner_ai_aer.py` | `gail/src/aer.rs`, `POST /v1/aer/encode`, `POST /v1/aer/decode` | Migrated |
| Orchestration status surface | `rag_demo/refiner_ai_orchestration.py` | `gail/src/orchestration.rs`, `GET /v1/status/orchestration` | Migrated |
| Refiner provider interface compatibility | `rag_demo/llm_providers.py` | `rag_demo/refiner_ai_gail.py` bridge keeps `LLMProvider` stable | Migrated |

## Refiner Integration Result

Refiner now uses Gail in two paths:

- `rag_demo/llm_providers.py`: direct `get_provider(...)` calls return Gail-backed providers when Gail is enabled.
- `rag_demo/refiner_ai_orchestration.py`: workflow provider construction, candidate orchestration, and status reporting delegate to Gail lazily when enabled.

This preserves Refiner's call sites and minimises product-level code churn.

## Configuration Preconditions

The following capabilities require real upstream backends to be configured, exactly as they did before the migration:

- OpenAI/Gemini direct or orchestrated calls require valid provider credentials.
- Ollama routing requires a reachable Ollama base URL.
- Live AARNN/SNN prediction requires either:
  - a configured specialist profile in Gail, or
  - `GAIL_AARNN_ENDPOINT`, `GAIL_AARNN_SOCKET_PATH`, or `GAIL_AARNN_REPO_ROOT`.

If a live neuromorphic backend is unavailable, Gail still preserves the offline heuristic fallback path for specialist analysis and prediction.

## Tracey And Continuum / NMC

No invasive code changes were required in Tracey or NMC for this migration because the duplicated LLM/SNN interface layer existed in Refiner, not in those repositories.

What Gail now provides for them is:

- a stable HTTP middleware contract,
- bearer-token auth for per-client access,
- TLS-capable ingress deployment via Ansible, and
- shared optimisation points for LLM, neuromorphic, and AER service calls.

This means Tracey and Continuum/NMC can adopt Gail without copying Refiner internals.

## Outcome

No Refiner service-facing feature was removed. The duplicated integration layer has been centralised in Gail, while Refiner's higher-level workflows remain intact.
