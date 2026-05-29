# Gail <-> OctoBot Connectivity

This document defines how Gail uses OctoBot HTTP APIs/controllers for trading,
which fallback paths are used, and how to validate behavior in production.

## Goals

1. Keep trading execution reliable when individual OctoBot endpoints are
   unavailable, disabled, or shape-shifted across OctoBot versions.
2. Prefer explicit JSON API endpoints where possible.
3. Keep fallback logic deterministic and observable through Gail adaptive schema,
   logs, and tests.
4. Avoid duplicate side effects on mutating routes (order placement/cancel).

## Endpoint Coverage Matrix

### Health and metadata

- `GET /api/ping`: startup/session probe.
- `GET /api/version`: optional version enrichment for status endpoint.

### Portfolio and balances

- Primary: `GET /api/portfolio`
- Fallback: `GET /portfolio` (HTML table parse)
- Enrichment: `GET /api/historical_portfolio_value?currency=USDT`
- Refresh trigger: `POST /api/refresh_portfolio`

### Exchanges and symbol discovery

- Primary exchange identity: `GET /api/first_exchange_details`
- Config pairs: `GET /api/get_config_currency`
- Exchange symbol universe fallback:
  - `GET /api/get_all_symbols/<exchange>`
  - `GET /api/currency_list` (pair-like symbols only)
  - `GET /dashboard/first_symbol`
- Legacy fallback: `GET /api/exchanges`

### Market snapshots

- Preferred (current OctoBot controller paths):
  - `GET /dashboard/watched_symbol/<symbol>`
  - `GET /dashboard/currency_price_graph_update/<exchange_id>/<symbol>/<time_frame>/live?display_orders=false`
- Legacy fallback:
  - `GET /api/market/ticker?exchange=<exchange>&symbol=<symbol>`

### Orders and trades

- Open orders: `GET /api/orders`
- Trade history: `GET /api/trades`
- Order placement modes (discovered automatically, cached when successful):
  - `POST /api/orders?action=create_order`
  - `POST /api/orders?action=create_orders`
  - `POST /api/orders` with action body
  - `POST /api/orders` with canonical order body
  - `POST /api/user_command` variants (`trading`, `gail_trading`, `trading_bridge`)
- Order cancellation:
  - `POST /api/orders?action=cancel_order`

### Log feedback ingestion

- Preferred:
  - `GET /logs?format=json&limit=<n>`
  - `GET /logs`
- Legacy fallback:
  - `GET /api/logs?limit=<n>`

## Reliability Rules

### Adaptive endpoint degradation

Gail tracks endpoint failures and automatically de-prioritizes optional routes
that repeatedly fail (404/5xx/parse errors), then retries them later.
Fallback endpoints remain active to keep service continuity.

### Sell precheck with forced portfolio refresh

Before a sell is submitted:

1. Gail checks cached portfolio balance for the symbol base asset.
2. If balance is missing or non-positive, Gail calls
   `POST /api/refresh_portfolio`.
3. Gail immediately re-fetches portfolio via `get_portfolio()` fallback chain.
4. If balance is still missing/non-positive, sell is skipped with explicit
   log context.

This prevents stale portfolio state from causing avoidable skipped sells.

### Order submission safety

Order creation endpoints are attempted in a known order. On HTTP success
without order acknowledgement, Gail verifies side effects (`/api/orders`,
`/api/trades`) before accepting the submission. If acknowledgement remains
ambiguous, Gail aborts further mutating attempts to avoid duplicate orders.

## Testing Coverage

Implemented in `src/trading/tests.rs`:

- exchange/symbol fallback behavior:
  - config symbols
  - `/api/get_all_symbols/<exchange>` fallback
  - `/api/currency_list` fallback
- portfolio fallback from missing `/api/portfolio` to `/portfolio` HTML parse
- refresh endpoint behavior:
  - success path
  - non-2xx error propagation
- sell precheck refresh flow:
  - missing cached balance
  - refresh + refetch updates state and restores sellability

## Operational Verification

### Validate endpoint health from Gail behavior

1. Check Gail logs for fallback and recovery messages.
2. Inspect Gail adaptive schema:
   - `GET /v1/trading/api-schema`
   - verify degraded endpoints and active numeric/semantic hints.
3. Confirm API issue registry:
   - `GET /v1/status/api-issues`

### Validate OctoBot bridge effectiveness

1. Ensure symbols populate in `GET /v1/trading/exchanges` and
   `GET /v1/trading/currencies`.
2. Ensure snapshots populate from dashboard routes even if legacy ticker route
   is absent.
3. Trigger a sell candidate with stale/empty portfolio cache and confirm Gail
   performs refresh+refetch before deciding to skip.

## Notes

- OctoBot deployments differ significantly by enabled modules and auth mode.
  The connectivity strategy is intentionally layered rather than bound to a
  single route set.
- HTML controller parsing (`/portfolio`, `/logs`) is fallback-only and used
  when JSON APIs are unavailable.
