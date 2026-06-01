# Gail Project Workflows

This document maps the code-level workflows implemented in Gail as of June 2026, including how the trading bridge now feeds historical ROI outcomes back into live decision-making.

## 1. Runtime Roles

`src/main.rs` defines three runtime roles:

1. `serve`: starts `GailService`, builds the Axum router (`src/app.rs`), serves HTTP APIs.
2. `mirror-worker`: polls Postgres-backed LLM ledger rows and mirrors prompt/response exchanges to AARNN (`src/mirror_worker.rs`).
3. `trainer-worker`: polls ledger rows for training candidates, writes datasets, executes configured training command pipelines, optionally registers snapshots in Ollama (`src/trainer_worker.rs`).

## 2. API Service Workflow (`serve`)

### 2.1 Request ingress

`src/app.rs` routes requests to:

- LLM: `/v1/llm/*`, OpenAI-compatible `/v1/chat/completions`, `/v1/responses`
- Neuromorphic and AER: `/v1/neuromorphic/*`, `/v1/aer/*`
- Status/metrics: `/v1/status/*`, `/metrics`
- Trading bridge: `/v1/trading/*`

Auth/scoping is enforced per endpoint category (`src/orchestration.rs`, `GailService::authorize`).

### 2.2 LLM orchestration path

`src/orchestration.rs` handles:

1. Candidate selection from configured providers (`src/config.rs` profiles + `src/routing.rs` tags/specialties).
2. Per-candidate scoring and health gating (`src/metrics.rs`, `src/api_issues.rs`, `src/adaptive_schema.rs`).
3. Parallel candidate execution and early-success arbitration.
4. Result normalization, tracing, and response shaping back to API handlers.

Supporting flows:

- Provider adapters in `src/providers/*` (OpenAI/NVIDIA/Gemini/Ollama).
- Adaptive API observations (`src/adaptive_schema.rs`) for endpoint backoff/shape memory.
- Issue registry (`src/api_issues.rs`) for active failure + mitigation tracking.
- Optional NMC host-pressure signal (`src/nmc_telemetry.rs`) influencing candidate ranking (default-enabled; activates when `base_url` is configured).
- Optional mirrored prompt/response stimulation to AARNN (`src/aarnn_bridge.rs`) (default-enabled; activates when an endpoint is discoverable).
- Optional durable interaction logging (`src/llm_ledger.rs`) (default-enabled).

### 2.3 Neuromorphic path

`src/specialists.rs` resolves specialist engines and supports:

- health checks over HTTP/UDS/heuristic paths,
- `analyze` routing hints,
- `predict` inference output, including AER-native transforms via `src/aer.rs`.

## 3. Durable Async Worker Workflows

### 3.1 Mirror worker

`src/mirror_worker.rs` loop:

1. Fetch pending mirror rows from `gail_llm_interactions`.
2. Replay input/output mirror exchanges through `AarnnMirrorClient`.
3. Mark success or schedule retry with backoff.

### 3.2 Trainer worker

`src/trainer_worker.rs` loop:

1. Fetch pending training rows from `gail_llm_interactions`.
2. Build JSONL dataset snapshot.
3. Resolve training invocation template.
4. Execute command with hardware-aware env configuration.
5. Persist pipeline report and update training status/retry state.
6. Optional Ollama registration/rotation (default-enabled).

## 4. Trading Bridge Workflow

Core module: `src/trading/mod.rs`.

Per evaluation cycle:

1. Fetch market snapshots and portfolio/order context from OctoBot (`src/trading/octobot.rs`).
2. Build research query from highest-scoring market candidate and gather RAG context from Refiner (`src/trading/refiner.rs`).
3. Query AI advisors in parallel and aggregate weighted consensus (`src/trading/advisor.rs`).
4. Produce fuzzy signal/confidence from Type-2 fuzzy system (`src/trading/fuzzy.rs`).
5. Blend fuzzy + AI outputs and run decision risk gates (`src/trading/decision.rs`).
6. Execute order when warranted (buy/sell), with OctoBot mode fallbacks and safety checks.
7. Persist logs/state ring buffers (`src/trading/state.rs`) and periodic disk snapshots.
8. Optional periodic backtest viability checks (`src/trading/backtest.rs`) (default-enabled when trading is enabled).

## 5. New Trading ROI Feedback Workflow

### 5.1 Purpose

The decision engine now consumes prior buy/sell outcome quality to reduce repeated poor decisions and reinforce consistently profitable directional behavior.

### 5.2 Inputs

From `TradingState.recent_trades`:

- action (`buy/strong_buy/sell/strong_sell`)
- symbol
- filled price (when available)
- recency order

Config controls (`src/trading/config.rs`):

- `decision_roi_feedback_enabled`
- `decision_roi_feedback_lookback_trades`
- `decision_roi_feedback_min_samples`
- `decision_roi_feedback_target_roi_pct`
- `decision_roi_feedback_max_signal_adjustment`
- `decision_roi_feedback_max_confidence_penalty`
- `decision_roi_feedback_max_confidence_boost`

### 5.3 Computation

In `src/trading/decision.rs`:

1. Compute base blended signal/confidence from fuzzy + AI.
2. Identify target direction (`buy`/`sell`) from the current blended-signal sign.
3. Compute directional ROI samples by comparing each historical trade to the next priced trade for the same symbol:
   - buy directional ROI = `(next_price - entry_price) / entry_price`
   - sell directional ROI = `-((next_price - entry_price) / entry_price)`
4. Prefer symbol-scoped stats when enough samples; fallback to global directional stats.
5. Normalize average ROI against configured target and combine with directional win rate.
6. Apply bounded signal adjustment and confidence multiplier.
7. Run existing confidence/position/cooldown gates on adjusted values.

### 5.4 Decision observability

`TradeDecision` now includes ROI-feedback trace fields:

- whether applied,
- signal adjustment,
- confidence multiplier,
- sample count,
- average directional ROI,
- directional win rate.

These are logged in the trading decision log context for auditability.

## 6. Test Coverage

Trading tests (`src/trading/tests.rs`) now include ROI-feedback behavioral assertions:

- negative historical buy ROI dampens marginal buy to hold,
- positive historical buy ROI boosts marginal hold to buy,
- negative historical sell ROI dampens marginal sell to hold,
- config defaults and normalization bounds include new ROI fields.

Existing tests continue to cover:

- config normalization,
- state persistence and ring buffers,
- fuzzy engine invariants,
- AI consensus aggregation,
- decision risk gates and overrides,
- backtesting and OctoBot integration behavior.
