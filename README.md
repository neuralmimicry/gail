# Gail

Gail is the shared AI middleware for NeuralMimicry services. It consolidates LLM routing, provider orchestration, neuromorphic specialist access, AER translation, transcription, orchestration-status surfaces, and a live crypto-trading bridge — all in one non-blocking Rust service.

The service is designed so Refiner can delegate immediately, while Tracey and Continuum/NMC can consume the same HTTP contract without re-implementing provider selection or neuromorphic transport glue.

## Goals

- Remove duplicated LLM and neuromorphic service-interface code from product repositories.
- Keep high-level workflow logic in each product and move cross-cutting provider/service routing into one Rust service.
- Preserve Refiner features such as concurrent provider selection, persisted metrics, Ollama inventory visibility, transcription, and neuromorphic specialist routing.
- Improve latency and efficiency centrally so optimisations benefit every Gail client.
- Allow additional OpenAI-compatible providers, including NVIDIA NIM-hosted model families, without changing client integrations.
- Provide an intelligent, autonomous, observable crypto-trading bridge using multi-AI consensus and Type-2 fuzzy logic.

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
| `GET /v1/status/api-schema` | Global adaptive API registry for remote integrations |
| `GET /v1/status/api-issues` | Active provider/API issues and Gail's current mitigations |
| `GET /metrics` | Prometheus text metrics for Gail API issue and provider health |
| `GET /v1/trading/status` | Trading bridge status snapshot |
| `GET /v1/trading/portfolio` | OctoBot portfolio holdings |
| `GET /v1/trading/positions` | Open OctoBot orders |
| `GET /v1/trading/history` | Recent executed trades |
| `GET /v1/trading/logs` | Activity log ring buffer |
| `GET /v1/trading/api-schema` | Adaptive OctoBot API schema and feedback reference |
| `GET /v1/trading/exchanges` | Available exchanges from OctoBot |
| `GET /v1/trading/currencies` | Available trading pairs |
| `GET /v1/trading/config` | Current trading configuration |
| `POST /v1/trading/config` | Update runtime trading configuration |
| `POST /v1/trading/pause` | Pause the evaluation loop |
| `POST /v1/trading/resume` | Resume the evaluation loop |
| `POST /v1/trading/override` | Inject an operator trade override |
| `POST /v1/trading/evaluate` | Trigger an immediate evaluation cycle |

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
- Route scopes: `health`, `llm`, `neuromorphic`, `aer`, `status`, `trading`, `trading_admin`.
- `/healthz` can be configured to allow or deny unauthenticated probes.
- All `/v1/trading/*` read endpoints require the `trading` scope.
- All `/v1/trading/*` write endpoints (pause, resume, override, config POST, evaluate) additionally require either the `trading_admin` scope or a `client_id` listed in `trading.admin_client_ids` (default: `["pbisaacs"]`).
- When `aarnn_bridge` is enabled, Gail should call AARNN with its own Customers-issued service-account bearer token rather than a browser-style session.
- The supplied Ansible role publishes Gail behind TLS ingress and injects per-client bearer tokens for Refiner, Tracey, and Continuum/NMC.
- OctoBot credentials (password, API keys) are stored in Kubernetes Secrets and never logged.

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
podman build \
  -t ghcr.io/neuralmimicry/gail:local .
```

By default, the Containerfile builds a local architecture-correct Gail `.deb` from the checked-out source and installs that package into the runtime image. This keeps source-based Ansible rollouts independent of GitHub Release availability while still exercising the same Debian package layout as published releases.

To build an image from a published GitHub Release package instead, pass a release selector:

```bash
podman build \
  --build-arg GAIL_VERSION=0.2.0 \
  -t ghcr.io/neuralmimicry/gail:0.2.0 .
```

`TARGETARCH` is mapped to Debian's `amd64` or `arm64` package names, with `dpkg --print-architecture` as a fallback for local Podman builds. Set `GAIL_VERSION=latest` to resolve the newest release asset, or set `GAIL_DEB_URL` to install a specific package URL directly.

The image keeps the existing container contract: it expects a config file at `/app/config/gail.yaml` unless `GAIL_CONFIG` is overridden, and it stores runtime data under `/app/data`.

## Debian Releases

The release workflow builds native Debian packages on the repo's self-hosted Linux runners for both X64 and ARM64:

- `gail_<version>_amd64.deb`
- `gail_<version>_arm64.deb`
- `SHA256SUMS`

Create a release by pushing a SemVer tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

The same workflow can be run manually from GitHub Actions with a `version` input. On each release run, the workflow derives the Gail version from the tag or manual input, updates `Cargo.toml` and `Cargo.lock` inside the runner workspace before building, and publishes the `.deb` files to a GitHub Release. This keeps the binary's `CARGO_PKG_VERSION`, the Debian package version, and the release tag aligned without requiring generated release edits to be committed back to the branch.

The package installs:

- `/usr/bin/gail`
- `/etc/gail/gail.yaml`
- `/etc/gail/gail.env`
- `/etc/gail/ai-routing-profiles.json`
- `/lib/systemd/system/gail.service`
- `/var/lib/gail/data`

Install and start a release package with:

```bash
sudo apt install ./gail_0.2.0_amd64.deb
sudo systemctl enable --now gail
```

Provider credentials, Gail bearer tokens, Ollama endpoints, and trading defaults belong in `/etc/gail/gail.env`. Persistent runtime state is written under `/var/lib/gail` because the systemd unit starts Gail with that working directory.

## Configuration Notes

- `providers`: shared LLM backends Gail can orchestrate.
- `providers` can include `openai`, `gemini`, `ollama`, and OpenAI-compatible `nvidia` profiles backed by custom `base_url` values.
- Gail keeps an Ollama local fallback candidate available when configured provider lists omit it, using `GAIL_OLLAMA_BASE_URL`/`GAIL_OLLAMA_MODEL` or the Continuum Ollama defaults. Ollama health checks verify both `/api/tags` and a tiny bounded `/api/generate` probe by default, so a reachable inventory endpoint does not route production work to a stalled generation service. Use `GAIL_OLLAMA_HEALTH_GENERATE_PROBE=false` only when inventory-only health is required.
- Ollama endpoint selection is adaptive. Gail tries the request/profile endpoint first, derives an HTTPS variant for public HTTP endpoints, and then tries `GAIL_OLLAMA_FALLBACK_BASE_URLS` plus the cluster-local Ollama service when the configured endpoint is not local. These attempts share one request budget instead of multiplying the timeout by the number of endpoints.
- Local Ollama generation is deliberately conservative: `GAIL_OLLAMA_MAX_CONCURRENT_REQUESTS` defaults to `1`, `GAIL_OLLAMA_QUEUE_TIMEOUT_SECONDS` defaults to `2`, `GAIL_OLLAMA_TIMEOUT_SECONDS` defaults to `30`, `GAIL_OLLAMA_MAX_RETRIES` defaults to `0`, and `GAIL_OLLAMA_MAX_PREDICT` defaults to `512`.
- `gail-auto` dispatches providers in ranked waves. If a candidate reports quota, rate-limit, upstream HTTP 429, or transient upstream failures such as 502/503/504, Gail marks that provider family throttled for the request, records health and an API issue mitigation, and tries the next suitable provider family instead of surfacing the first failure.
- When every orchestrated non-interactive candidate fails or is in adaptive backoff, Gail records the issue and returns a degraded safety response instead of an upstream 502. Explicit JSON prompts and routing-tagged `json`/`structured_data` requests use `GAIL_AUTOMATION_CANDIDATE_TIMEOUT_SECONDS`/`automation_candidate_timeout_cap_seconds` to reach that fallback quickly, returning a valid hold/no-trade JSON payload so OctoBot and Refiner-style automation can continue safely while provider health recovers. Explicit single-provider requests still surface their real provider error.
- When OctoBot requests an `ExecutionPlan` JSON schema and every provider is unavailable, Gail returns a schema-valid empty plan (`{"steps":[]}`) instead of a trading-decision object, so OctoBot can fall back without Pydantic validation failures.
- Provider retirements, missing account functions, and authentication failures are classified separately from transient upstream failures, so Gail can stop repeating dead model/account paths while still trying healthy cloud or local alternatives.
- Gail records active API/provider issues without calling back into Refiner during Refiner-originated Gail failures. This prevents cyclic retry loops while still exposing the active mitigation and next retry window.
- Gail Trading's OctoBot and Refiner clients feed remote API failures and recoveries into the same issue registry, alongside the adaptive schema registry, so dashboard/Prometheus status covers trading dependencies as well as LLM providers.
- `specialists`: explicit neuromorphic engines. Use this when you have named SNN/AARNN backends to register.
- `aarnn_bridge`: mirrored Gail-to-AARNN LLM I/O bridge. Gail mirrors prompt-side and response-side text plus translated AER payloads to `POST /api/llm/mirror` and can optionally promote a future AARNN reply.
- `config/ai-routing-profiles.json`: shared workflow/keyword/provider routing contract used by Gail and mirrored in Refiner for offline fallback.
- `GAIL_ROUTING_PROFILES_PATH`: optional override for the routing contract path.
- `GAIL_AARNN_*` env vars: optional legacy auto-attach path for an AARNN backend, mirroring Refiner's previous automatic fallback behaviour.
- `storage.metrics_path`: persisted provider quality/latency metrics.
- `storage.adaptive_schema_path`: persisted adaptive API registry for provider, Refiner, AARNN, specialist, OctoBot, and trading feedback observations.
- `storage.api_issues_path`: persisted issue registry for provider/API failures, mitigations, recoveries, and Prometheus/dashboard visibility.
- `storage.postgres_dsn` or `GAIL_POSTGRES_DSN`: optional Postgres persistence for `gail_api_issues` and `gail_api_issue_snapshots`.
- `security.allow_unauthenticated_metrics`: allows Prometheus to scrape `/metrics` without sharing a bearer token. Keep it enabled only on trusted network paths.
- `orchestration.health_ttl_seconds`: cached provider-health TTL. Runtime quota health remains in backoff until this TTL expires, so later requests skip rate-limited candidates before probing them again.
- `storage.ollama_model_store_path`: cached Ollama model inventory summary.

## AARNN Bridge

Use `aarnn_bridge` when Gail should mirror every LLM input and output into an AARNN instance.

- `endpoint` points at the AARNN web UI base URL; Gail appends `POST /api/llm/mirror`.
- `access_token` should be the Gail Customers-issued service-account bearer token.
- `response_preference` defaults to `llm_preferred`.
- `prefer_aarnn_when_confident` is available, but the current AARNN candidate reply is a deliberately low-confidence bootstrap echo until network-output decoding is mature.

## Product Integration

### Refiner

Refiner now keeps its `LLMProvider` interface stable while routing direct and workflow-mode requests through Gail when `REFINER_GAIL_ENABLED=1` is set.

### Tracey

Tracey can consume Gail's neuromorphic and AER endpoints as a stable HTTP contract instead of embedding its own cross-service adapters. The service boundary is documented and tokenised so Tracey can attach later without copying Refiner internals.

### Continuum / NMC

NMC already owns AARNN and cloud-control orchestration concerns. Gail sits alongside that stack as the shared AI middleware layer for LLM, neuromorphic scoring, and AER translation so Continuum-facing tooling can call one service contract instead of product-specific glue.

### AARNN

The AARNN bridge lets Gail mirror both prompt-side and response-side LLM traffic into AARNN so the attached network can be stimulated over time and, later, provide a candidate reply back into Gail's selection logic.

## Trading Bridge

The trading bridge (`src/trading/`) is a self-contained module that runs a background tokio evaluation loop alongside all existing Gail capabilities. It is disabled by default (`trading.enabled: false`) and can be enabled without restarting or touching any other Gail functionality.

### Architecture

```
Gail HTTP Server
  /v1/trading/* ←→ TradingBridge (Arc-shared state)
                         │
              ┌──────────┼───────────────────┐
              ▼          ▼                   ▼
        OctobotClient  RefinerClient   GailService
        (market data,  (RAG research)  (all AI providers)
         order mgmt)
              │                              │
              └─────── Background loop ──────┘
                  Every N seconds (default 60):
                  1. Fetch market snapshots + portfolio
                  2. Query Refiner for market research
                  3. Consult all AIs in parallel (JoinSet)
                  4. Run Type-2 fuzzy inference
                  5. Blend fuzzy + AI signals → decision
                  6. Apply risk gates → execute or hold
                  7. Log to ring buffer; persist state
```

### Evaluation Pipeline (one cycle)

Each evaluation runs the following pipeline steps in sequence:

**Step 1 — Market data** (`octobot.rs`)
OctoBot is queried for market snapshots (price, 24 h change %, 24 h volume), portfolio totals where available, and open orders. The client probes `/api/ping` at startup; Continuum deployments normally keep OctoBot native web auth disabled behind shared ingress auth, so Gail does not attempt the old non-existent JSON password-login endpoint. Up to 20 snapshots are fetched across all configured exchanges and currency filters. Gail records each OctoBot endpoint's observed shape, status, failures, and log-derived semantic hints in an adaptive API schema; optional routes that prove missing or temporarily bad are skipped for a short TTL while fallback routes are used.

**Step 2 — Research** (`refiner.rs`)
The highest-signal market snapshot (scored by `|Δ%| × ln(volume+1)`) is used to build a Refiner RAG query from the `research_query_template`. Refiner's `/api/rag/query` endpoint returns ranked context passages (default top 5). The research context is passed verbatim to the AI advisors and contributes a sentiment signal to the fuzzy engine.

**Step 3 — Multi-AI advisory** (`advisor.rs`)
`TradingAdvisor::consult_all()` fires all configured Gail provider profiles in parallel using a `tokio::task::JoinSet`. Each provider receives a structured prompt containing market data, portfolio state, and research context, and is asked to respond with:
```json
{
  "action": "buy|sell|hold|strong_buy|strong_sell",
  "confidence": 0.0–1.0,
  "reasoning": "...",
  "suggested_amount_usd": null,
  "risk_score": 0.0–1.0,
  "risk_flags": [],
  "target_symbol": "BTC/USDT"
}
```
Providers are selected by quality weight and provider-family diversity, so a single rate-limited cloud family does not crowd out OpenAI/Gemini/Ollama alternatives. Responses are parsed defensively, schema-echo responses are rejected, and `AiConsensus` uses weighted voting with agreement, response coverage, and risk penalties before producing a signal in `[−1, +1]`.

**Step 4 — Type-2 fuzzy inference** (`fuzzy.rs`)
Five linguistic input variables are encoded from the gathered data:

| Variable | Range | Derivation |
| --- | --- | --- |
| `price_trend` | −1 to +1 | `clamp(Δ% / 5, -1, 1)` |
| `volume_ratio` | 0 to 2 | `clamp(vol24h / 1M, 0, 2)` |
| `ai_consensus` | −1 to +1 | Aggregated AI signal |
| `research_sentiment` | −1 to +1 | `(avg_rag_score − 0.5) × 0.4` |
| `portfolio_exposure` | 0 to 1 | Non-stablecoin fraction of portfolio |

Each variable has three linguistic terms with **interval Type-2 Gaussian membership functions** (lower/upper bounds encoding epistemic uncertainty). A rule base of 25 Mamdani rules fires against the five inputs and activates output terms (`strong_sell, sell, hold, buy, strong_buy`). Type reduction uses a simplified **Karnik-Mendel centroid** over the interval-weighted output terms to produce a crisp signal in `[−1, +1]` plus a confidence score.

**Step 5 — Decision blending** (`decision.rs`)
The fuzzy signal and AI consensus signal are blended with configurable weights (default: fuzzy 40%, AI 60%):
```
blended_signal     = fuzzy.signal × fuzzy_weight + ai.signal × ai_weight
blended_confidence = fuzzy.confidence × fuzzy_weight + ai.confidence × ai_weight
```
Three sequential risk gates are applied before a trade is placed:
1. **Confidence gate**: `blended_confidence < fuzzy_confidence_threshold` → hold
2. **Position gate**: open positions ≥ `max_open_positions` and signal is buy → hold
3. **Cooldown gate**: time since last trade < `min_trade_interval_seconds` → hold

Trade size scales with `signal_strength × confidence` between `micro_trade_min_usd` and `micro_trade_max_usd`.

Action thresholds on the blended signal:
- `signal ≥ 0.65` → `strong_buy`
- `signal ≥ 0.20` → `buy`
- `signal ≤ −0.65` → `strong_sell`
- `signal ≤ −0.20` → `sell`
- otherwise → `hold`

**Step 6 — Execution** (`mod.rs`)
Gail evaluates decisions by default but does not send live orders unless `trading.live_execution_enabled` is explicitly enabled. Direct `place_buy_order` / `place_sell_order` calls return an explicit unsupported error because OctoBot's current web API exposes order cancellation and trading-mode/user-command surfaces, not direct market-order placement. Live execution should be routed through a supported OctoBot trading mode or command bridge before operator overrides or autonomous trades are enabled.

**Override mechanism**: if `TradingState.pending_override` is set via `POST /v1/trading/override`, the decision pipeline is bypassed and the override decision is prepared with `confidence = 1.0`. The override still requires `trading.live_execution_enabled: true` before Gail submits anything to OctoBot. The override is cleared after the attempt.

### State and Persistence

`SharedTradingState` (`state.rs`) is an `Arc<Mutex<TradingState>>` shared between the background loop and all HTTP handlers. It holds:

- `paused` / `enabled` flags
- `evaluation_count` / `trade_count` counters
- `last_evaluation_at` / `last_trade_at` Unix timestamps
- `current_portfolio` — latest OctoBot portfolio snapshot
- `open_positions` — latest open orders
- `recent_trades` — `VecDeque` ring buffer (default 200 entries)
- `activity_log` — `VecDeque` ring buffer (default 1000 entries) with level, category, message, and JSON context
- `api_schema` — adaptive OctoBot endpoint/reference schema, semantic hints, and recent automatic adjustments
- `available_exchanges` — populated from OctoBot on each cycle
- `pending_override` — operator-injected trade override
- `config_overrides` — runtime-mutable subset of config
- `last_error` — most recent error string

State is persisted to `data_path` (default `./data/trading_state.json`) every 5 evaluation cycles and on shutdown, and restored at startup.

### Module Layout

| File | Responsibility |
| --- | --- |
| `src/adaptive_schema.rs` | Generic adaptive API registry shared by Gail providers, Refiner, AARNN, specialists, and trading |
| `src/api_issues.rs` | Persistent provider/API issue registry, mitigation log, Postgres sync, and Prometheus metric rendering |
| `src/trading/mod.rs` | `TradingBridge`, background loop, evaluation pipeline, execution |
| `src/trading/config.rs` | `TradingConfig`, `TradingConfigOverride` |
| `src/trading/state.rs` | `TradingState`, `SharedTradingState`, ring buffers, persistence |
| `src/trading/octobot.rs` | `OctobotClient` — OctoBot web API probe, market data, portfolio totals, orders |
| `src/trading/refiner.rs` | `RefinerClient` — RAG research queries |
| `src/trading/fuzzy.rs` | `FuzzyEngine` — Type-2 interval fuzzy logic, 25 rules, Karnik-Mendel |
| `src/trading/advisor.rs` | `TradingAdvisor` — parallel multi-AI advisory, consensus aggregation |
| `src/trading/decision.rs` | `DecisionEngine` — signal blending, risk gates, trade sizing |

### Configuration Reference

```yaml
storage:
  metrics_path: "./data/provider_metrics.json"
  adaptive_schema_path: "./data/adaptive_api_schema.json"
  api_issues_path: "./data/api_issues.json"
  postgres_dsn: "${GAIL_POSTGRES_DSN}"
  ollama_model_store_path: "./data/ollama_model_inventory.json"

trading:
  enabled: false                          # master switch
  octobot_base_url: "${GAIL_TRADING_OCTOBOT_URL}"
  octobot_password: null                   # only for native OctoBot web auth
  refiner_base_url: "${GAIL_TRADING_REFINER_URL}"
  refiner_api_token: "${GAIL_TRADING_REFINER_TOKEN}"
  admin_client_ids: ["pbisaacs"]          # write-access list
  evaluation_interval_seconds: 60         # minimum 10
  max_parallel_advisors: 5               # 1–20
  micro_trade_max_usd: 25.0              # per-trade ceiling
  micro_trade_min_usd: 1.0               # per-trade floor
  max_open_positions: 5                  # 1–50
  min_trade_interval_seconds: 120        # cooldown between trades
  target_exchanges: []                   # empty = all available
  target_currencies: []                  # empty = all available
  fuzzy_confidence_threshold: 0.65       # minimum blended confidence to trade
  fuzzy_weight: 0.4                      # fuzzy vs AI blend weight
  live_execution_enabled: false          # keep false until an order bridge exists
  research_query_template: "cryptocurrency market sentiment {currency} {exchange} {date}"
  research_top_k: 5
  log_ring_size: 1000
  trade_ring_size: 200
  data_path: "./data/trading_state.json"
  octobot_timeout_seconds: 10.0
  refiner_timeout_seconds: 15.0
  advisor_timeout_seconds: 30.0
  backtesting_enabled: false             # requires OctoBot .data files
  backtest_data_files: []                # explicit .data files, or auto-discovered when enabled
```

Runtime-mutable fields (via `POST /v1/trading/config`, no restart required):
`evaluation_interval_seconds`, `micro_trade_max_usd`, `micro_trade_min_usd`, `max_open_positions`, `fuzzy_confidence_threshold`, `target_exchanges`, `target_currencies`.

### NMC / Continuum Integration

The NMC server proxies all `/v1/trading/*` Gail endpoints at `/gail/trading/*`. The NMC dashboard shows a live **Trading Bot** card (status, trade count, last trade time) and a detail modal with portfolio holdings, recent trades table, activity log, and admin controls (pause, resume, force-evaluate). All card data auto-refreshes on the 15-second polling cycle. The NMC CLI exposes the full surface via `nmc gail trading <subcommand>`:

```
nmc gail trading status
nmc gail trading portfolio
nmc gail trading positions
nmc gail trading history [--limit N]
nmc gail trading logs [--limit N]
nmc gail trading api-schema
nmc gail trading exchanges
nmc gail trading currencies
nmc gail trading config
nmc gail trading config-set --json '{"micro_trade_max_usd":15.0}'
nmc gail trading pause
nmc gail trading resume
nmc gail trading override --action buy --symbol BTC/USDT --amount 10.0 --exchange binance
nmc gail trading evaluate
```

### Ansible Variables

The Ansible role (`roles/continuum_tenant_gail`) exposes the following defaults for the trading bridge (all off by default):

| Variable | Default | Description |
| --- | --- | --- |
| `continuum_tenant_gail_trading_enabled` | `false` | Enable the bridge |
| `continuum_tenant_gail_trading_octobot_url` | cluster-local OctoBot | OctoBot base URL |
| `continuum_tenant_gail_trading_octobot_api_key` | secret file lookup | OctoBot API key |
| `continuum_tenant_gail_trading_octobot_password` | env / secret / API key fallback | Native OctoBot web-auth password, only passed when native web auth is enabled |
| `continuum_tenant_gail_trading_octobot_native_web_auth_enable` | OctoBot role auth flag | Whether to pass an OctoBot native web-auth password to Gail |
| `continuum_tenant_gail_trading_refiner_url` | cluster-local Refiner | Refiner base URL |
| `continuum_tenant_gail_trading_refiner_token` | secret file lookup | Refiner API token |
| `continuum_tenant_gail_trading_admin_token` | env / secret file | `pbisaacs` bearer token with `trading` + `trading_admin` scopes |
| `continuum_tenant_gail_trading_eval_interval_seconds` | `60` | Evaluation interval |
| `continuum_tenant_gail_trading_max_parallel_advisors` | `5` | Max parallel AI advisors |
| `continuum_tenant_gail_trading_micro_trade_max_usd` | `25.0` | Per-trade ceiling (USD) |
| `continuum_tenant_gail_trading_micro_trade_min_usd` | `1.0` | Per-trade floor (USD) |
| `continuum_tenant_gail_trading_max_open_positions` | `5` | Max simultaneous positions |
| `continuum_tenant_gail_trading_fuzzy_confidence_threshold` | `0.65` | Minimum trade confidence |
| `continuum_tenant_gail_trading_admin_client_ids` | `["pbisaacs"]` | Admin client list |

Secrets are looked up from `continuum_tenant_gail_secret_store_dir` files (`trading_octobot_api_key`, `trading_refiner_token`, `trading_admin_token`) or from environment variables.

## Deployment

The Ansible role in `swarmhpc/swarmhpc/ansible/roles/continuum_tenant_gail`:

- builds or syncs the Gail source tree,
- builds and pushes the container image,
- renders the Gail config (including trading section) into a Kubernetes Secret,
- ships the shared AI-routing contract with the Gail runtime image,
- persists Gail metrics, adaptive API schema, API issue registry, Ollama inventory, and trading state on shared storage,
- optionally mirrors API issue state into the shared Postgres service and exposes Gail issue/provider metrics to Prometheus,
- exposes Gail via ingress/TLS, and
- injects a matching bearer-token configuration into Refiner.
