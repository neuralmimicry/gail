# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build (default features include training-libtorch, requires libtorch)
cargo build

# Build without libtorch (CI / environments without libtorch)
cargo build --no-default-features

# Run all tests (requires libtorch)
cargo test

# Run tests without libtorch (CI-safe)
cargo test --no-default-features --features ci-trading-tests

# Run a single test or module
cargo test --no-default-features --features ci-trading-tests trading::tests::

# Run trading tests specifically (CI-safe wrapper script)
./scripts/test-trading-ci-safe.sh

# Run with live config
RUST_LOG=debug cargo run -- --config gail.yaml

# Run as mirror worker role
GAIL_ROLE=mirror_worker cargo run -- --config gail.yaml

# Run as trainer worker role
GAIL_ROLE=trainer_worker cargo run -- --config gail.yaml
```

The `training-libtorch` feature is on by default and gates the `gail-qlora-sft` binary and tch/PyTorch bindings. Most tests and development work can be done with `--no-default-features`.

## Architecture

Gail is an axum HTTP server that acts as a shared AI middleware layer. The binary supports three runtime roles selected via `--role` / `GAIL_ROLE`: `serve` (default), `mirror_worker`, and `trainer_worker`.

### Request flow (`serve` role)

1. **`src/app.rs`** — axum router; all HTTP handler functions live here. Handlers extract auth via `GailService::authorize(&headers, scope)` and delegate to the service.
2. **`src/orchestration.rs`** — `GailService` (cheaply `Clone`able `Arc` wrapper). Owns the reqwest HTTP client, provider adapters, specialist engines, metrics, the LLM ledger, and the optional trading bridge. The `complete()` method orchestrates multi-provider fan-out with workload pools (`interactive_pool`, `solver_pool`), early-success settling, and automatic fallback.
3. **`src/providers/`** — one adapter per LLM backend: `openai.rs`, `gemini.rs`, `ollama.rs`. The `ProviderAdapter` enum wraps them. `normalize_provider_type()` canonicalizes provider strings. Nvidia uses the OpenAI adapter pointed at a different base URL.
4. **`src/routing.rs`** — `RoutingProfiles` loaded from `config/ai-routing-profiles.json`; maps workflow+role → capability tags used to score candidates.
5. **`src/config.rs`** — `GailConfig` loaded from `gail.yaml`; supports `${ENV_VAR}` interpolation. Providers are `Vec<ProviderProfile>`; each profile carries name, provider type, model, API key, roles, specialties, weight, and optional resource budgets.

### OpenAI-compatible routing (`/v1/chat/completions`, `/v1/responses`)

The model field controls routing in `resolve_openai_route()`:
- `gail-auto` / `auto` → `GailService::complete()` (orchestrated, best-of-N)
- `provider/model` (e.g. `openai/gpt-4o`, `ollama/llama3`) → direct to that provider
- `aarnn/…`, `snn/…`, `specialist/…` → orchestrated with neuromorphic context injected
- Bare model names (e.g. `gpt-4o`, `gemini-2.0-flash`) → inferred from model name prefix

### Neuromorphic layer

- **`src/specialists.rs`** — `SpecialistEngine` wraps AARNN endpoints (HTTP or Unix socket). Used for `/v1/neuromorphic/analyze` and `/v1/neuromorphic/predict`.
- **`src/aarnn_bridge.rs`** — `AarnnMirrorClient` mirrors every LLM request/response as AER spike trains to the AARNN runtime (fire-and-forget, non-blocking queue).
- **`src/aer.rs`** — AER1 binary codec (varint-encoded spike events with `AER1` magic header).

### Background workers

- **`src/mirror_worker.rs`** — polls Postgres for LLM ledger records and mirrors them to the AARNN bridge. Requires `storage.postgres_dsn`.
- **`src/trainer_worker.rs`** — pulls training data from the LLM ledger and runs fine-tuning jobs. Requires `trainer.enabled=true`.
- **`src/llm_ledger.rs`** — append-only JSONL file + async Postgres writer. All LLM completions are recorded here.

### Trading bridge (`src/trading/`)

An optional background tokio task (enabled via `trading.enabled` in config). Per-cycle pipeline: OctoBot market data → Refiner RAG context → parallel AI advisory (`JoinSet`) → Type-2 fuzzy inference → signal blend + risk gates → OctoBot execution. State is shared via `SharedTradingState` (`Arc<Mutex<TradingState>>`); HTTP handlers in `app.rs` read/write it directly.

### Other modules

- **`src/adaptive_schema.rs`** — global `Lazy<Mutex<AdaptiveApiRegistry>>` that records live API response shapes and health signals; used by providers to adapt requests when upstream APIs change.
- **`src/api_issues.rs`** — in-memory + disk registry of upstream API errors, surfaced at `/v1/status/api-issues` and `/metrics`.
- **`src/metrics.rs`** — per-provider Prometheus-style counters, used in `provider_prometheus_metrics()`.
- **`src/nmc_telemetry.rs`** — pushes NMC agent signals (latency, model selection, health) to the NMC telemetry service.

## Configuration

`gail.yaml` is the canonical config file, loaded at startup. All values support `${ENV_VAR}` interpolation (missing vars become empty strings). Key sections:

| Section | Purpose |
|---|---|
| `server` | `bind_addr`, `public_base_url` |
| `security.api_tokens` | Client IDs, bearer tokens, and scopes (`llm`, `neuromorphic`, `aer`, `status`, `health`, `trading`, `trading_admin`) |
| `orchestration` | Selection mode (`best`/`fastest`/`random`), pool sizes, timeout caps, quality thresholds |
| `providers` | List of `ProviderProfile`; supports `openai`, `gemini`, `nvidia`, `ollama` |
| `specialists` | AARNN/SNN specialist engine endpoints |
| `aarnn_bridge` | Mirror client config |
| `trading` | OctoBot URL/credentials, Refiner URL, eval interval, risk parameters |
| `storage` | File paths for metrics, ledger, adaptive schema; optional `postgres_dsn` |

## Features

| Feature | Default | Purpose |
|---|---|---|
| `training-libtorch` | yes | Enables tch (PyTorch) bindings and `gail-qlora-sft` binary |
| `ci-trading-tests` | no | Enables `trading::tests` module without libtorch |
