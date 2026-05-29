# Gail Operations Runbook

## Trading Bridge

Detailed OctoBot integration behavior, endpoint fallback order, and reliability
rules are documented in:

- [OCTOBOT_CONNECTIVITY.md](./OCTOBOT_CONNECTIVITY.md)

## Key Runtime Checks

### Gail service health

- `GET /api/health`
- `GET /v1/status/api-issues`
- `GET /v1/status/api-schema`
- `GET /metrics`

### Trading bridge health

- `GET /v1/trading/status`
- `GET /v1/trading/portfolio`
- `GET /v1/trading/positions`
- `GET /v1/trading/exchanges`
- `GET /v1/trading/currencies`
- `GET /v1/trading/api-schema`

### Cluster logs (Kubernetes)

- `kubectl --kubeconfig /home/pbisaacs/.kube/config-continuum-gail -n gail logs deployment/gail`
- `kubectl --kubeconfig /home/pbisaacs/.kube/config-continuum-gail -n gail logs deployment/gail-mirror-worker`
- `kubectl --kubeconfig /home/pbisaacs/.kube/config-continuum-gail -n gail logs deployment/gail-trainer-worker`

## Common Failure Modes

### Sell skipped due to missing balance

Expected behavior now:

1. Gail triggers `POST /api/refresh_portfolio`.
2. Gail re-fetches portfolio using API/controller fallback chain.
3. Sell is skipped only if the balance remains missing/non-positive.

### Missing symbol/universe data

Expected Gail fallback order:

1. `/api/get_config_currency`
2. `/api/get_all_symbols/<exchange>`
3. `/api/currency_list` (pair-like symbols)
4. `/dashboard/first_symbol`

## Post-change validation checklist

1. Confirm `cargo fmt` passes.
2. Run trading tests in `src/trading/tests.rs` (or targeted wiremock tests).
3. Confirm `GET /v1/trading/exchanges` returns non-empty symbols.
4. Confirm `GET /v1/trading/api-schema` shows endpoint observations.
5. Confirm no persistent `sell skipped` events caused only by stale cache.
