# Trading history, policy and execution validation

Gail consumes OctoBot's public venue candle API at
`GET /api/market/candles?exchange=binance&symbol=BTC%2FUSDT&time_frame=1h&limit=500`.
It is independent of the web dashboard's watched-pair cache. A valid response
identifies the exact venue, symbol and timeframe, declares seconds and
`candle_open` timestamp semantics, excludes unfinished candles and identifies
whether the account is simulated. Gail rejects simulated history for this live
bootstrap. A missing route alone permits the older dashboard history fallback;
a failed venue read remains an error.

Historical closes become observable at candle end, not at the opening timestamp.
Gail preserves array indices across invalid rows, accepts numeric seconds,
milliseconds and supported dashboard date strings, and excludes missing,
unordered, future or unfinished observations. It never fabricates timestamps.
The datalake independently rejects absent or future observation times.

Datalake schema 3 records `observation_time_contract` for newly validated
observations. Older append-only file records remain available for audit but are
excluded from replay and features. The optional PostgreSQL mirror adds a column
with legacy default zero, writes the current version explicitly and reads only
the current contract. The schema change requests a fresh canonical history
bootstrap. It does not alter exchange ledgers or promote a model. Price history
does not reconstruct historical spread, book depth or derivatives telemetry;
replay continues to use its documented conservative fallbacks.

A partial bootstrap persists successful rows but remains failed/retryable until
every requested venue/symbol/timeframe has returned nonempty history. This
prevents one healthy venue from hiding another venue's missing data behind a
successful completion marker.
Bootstrap identity includes the compiled source revision even when the Cargo
version or deployment label is reused, so a repaired executable re-fetches data.

`GET /v1/trading/status` requires the normal trading authorization and includes
`decision_policy`: declared and effective limits, override presence, open
position count, available slots, buy-capacity exhaustion, live permission,
paper requirements and the shared profitability predicate used during actual
execution. No credentials or individual holdings are included. Runtime
overrides take precedence over deployment YAML. Operators should review and
back up the complete overrides before selectively clearing a drifted field
through the administrator-only configuration API.

The 29 October audit found declared capacity 40 versus persisted capacity 4,
with 11 open positions; declared confidence 0.65 versus persisted 0.77; and
declared maximum notional 25 versus persisted 28.5. These are concrete policy
differences, not evidence that markets offered no opportunities. The latest
retained native validation selected cash and did not qualify. Fixing data or
policy does not make that result profitable or replace clean paper evidence.

## Verification

Run `cargo test --no-default-features --features ci-trading-tests,elm --lib`.
The suite includes candle contract/fallback failures, aligned parsing and time
causality, invalid-history quarantine, effective-policy counterfactuals and an
isolated execution replay. The replay calls the real decision and execution
code against a loopback mock HTTP venue, checks paper and profitability gates,
persists an acknowledged fixture order and proves that process-restart replay
does not submit it again. Its evidence lives only in a temporary directory.
It never calls production or qualifies production paper/model state.

The ignored test `postgres_observation_contract_migrates_and_filters_legacy_rows`
exercises real schema migration, idempotent initialization, upsert and filtered
reload against an isolated PostgreSQL database. It requires
`GAIL_TEST_POSTGRES_DSN`, refuses non-loopback hosts or a database other than
`gail_test_history`, and must never target the platform database. Invoke it
explicitly with `-- --ignored`; the ordinary suite requires no live database.

Production transport and data acceptance is maintained in SwarmHPC's
`swarmhpc/ansible/octobot_trading_api_validation_site.yml`. It validates
authentication, current evaluation heartbeats, enabled venue identities,
account read refreshes, tickers, actual closed candle history, order API method
availability, retained execution ledgers, bounded logs and backtest catalogs.
It submits no order. The report explicitly leaves trading effectiveness
unqualified; a healthy endpoint or a hold decision is not a successful trade.

After deployment, inspect data-bootstrap completion, native validation rejection
reasons and paper-intent progression. Real fills must have venue ledger evidence
and a Gail order identity. Do not reduce risk thresholds, manufacture successful
backtests or submit a real diagnostic order to make an acceptance report green.
