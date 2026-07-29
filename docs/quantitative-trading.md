# Quantitative trading architecture

This document describes Gail's second-generation quantitative stack. Its
purpose is to improve fee-adjusted return and risk measurement without
weakening the existing shadow-to-primary migration guard. No component assumes
that more trading means better performance, and no configuration can guarantee
positive ROI.

## Architecture and concurrency

The live evaluation path performs network I/O with Tokio and delegates
CPU-bound research to blocking workers. Market, portfolio, order, exchange and
research requests remain concurrent. Within the quant stack:

1. the market datalake creates point-in-time replay frames in parallel by
   symbol;
2. native backtesting evaluates parameter arms in parallel with Rayon;
3. cross-sectional factors are calculated in parallel across the eligible
   universe;
4. pairs and carry sleeves run concurrently on a blocking worker; and
5. an LLM advisory round, when required in shadow mode, runs concurrently with
   sleeve research.

No Rayon or replay calculation runs on a Tokio I/O worker. Persistent state is
updated only after a complete worker result is available, and all ledgers have
configured bounds.

## 1. Gail-native replay

The scheduled native backtester calls the same Rust quant evaluator used by
the live bridge. This corrects the principal limitation of an OctoBot-only
backtest, which tests OctoBot's configured strategy rather than Gail's policy.

Replay frames contain only observations available at their timestamp. The
datalake calculates each historical feature from the prefix ending at that
sample, avoiding future-data leakage. Candidate arms use chronological
walk-forward training and validation partitions separated by an embargo.
Promotion requires:

- enough actionable out-of-sample trades;
- a positive one-sided lower confidence bound after costs;
- an acceptable probability-of-backtest-overfitting estimate; and
- a recent native validation of the parameter selected by the live controller.

If native validation fails, a shadow controller remains shadow. An already
primary controller is paused only when `backtest_pause_on_failure` is enabled.
OctoBot backtesting remains available as a compatibility fallback.

## 2. Multi-horizon executable-edge calibration

Raw factor strength is not treated as expected profit. Gail records
non-overlapping forward observations at 15 minutes, one hour, four hours and
24 hours by default. Each horizon maintains an independent label and selects
the most specific statistically usable estimate in this order:

1. symbol, market regime and horizon;
2. symbol and horizon; then
3. global horizon.

The execution gate subtracts a conservative round-trip transaction-cost
estimate from the one-sided gross-edge lower bound. During cold start, raw
intent continues to be observed but live action fails closed when
`require_calibrated_edge` is true. The selected horizon also becomes the trade
markout horizon, ensuring that prediction and evaluation refer to the same
holding period.

## 3. Strategy selection and cash

`cash-v1` is an explicit parameter arm with a zero return. It is therefore
possible for Gail to select no trade when every active strategy loses after
costs. Holds do not dilute per-trade losses: Gail reports both actionable-trade
expectancy and return per market opportunity.

A trading arm must have a positive confidence-adjusted absolute edge before it
can beat cash. Cash may become the active research choice but is never promoted
as an execution strategy. Existing saved state receives the cash arm during
normalisation.

## 4. Cross-sectional allocation

The allocator first excludes markets with inadequate quote volume, excessive
spread, insufficient executable depth or an inadequate listing history. It
then:

- winsorises factor extremes;
- volatility-normalises directional strength;
- removes the common market component;
- applies a bounded order/trade-flow contribution;
- penalises weak rather than merely absent volume confirmation;
- targets the strongest `top_k` markets;
- enforces asset and correlation-cluster caps; and
- leaves every unallocated fraction explicitly in cash.

The decision size is based on target weight minus current portfolio weight.
This makes the result a rebalance instruction rather than repeatedly buying an
asset already at its target.

## 5. Market and execution telemetry

Every live `MarketSnapshot` can carry best bid and ask, bid and ask executable
depth, order and trade-flow imbalance, funding, open interest, futures basis,
listing age, quote volume and source latency. OctoBot dashboard and ticker
requests run concurrently and are merged without discarding richer dashboard
fields.

One-way projected market cost is the half-spread plus bounded depth
participation. Missing fields use the configured slippage fallback; missing
data never implies zero cost. Once a venue/symbol/side has enough fills, Gail
compares the live projection with median observed adverse slippage and uses the
more conservative value.

Immediate repricing subtracts both adverse price movement and any increase in
round-trip cost since the decision. A priced successful order records its
reference price, fill price, notional, side and fee allowance in the persistent
telemetry ledger. JSONL and Postgres datalakes retain the complete
microstructure object. Native replay charges two fees plus the independently
projected entry and exit market costs.

The Postgres migration is additive: schema initialisation adds the
`microstructure JSONB NOT NULL DEFAULT '{}'` column when upgrading an existing
table. JSONL rows from schema version 1 remain readable through Serde defaults.

## 6. Diversifying sleeves and LLM risk overlay

### Pairs

Pairs are explicitly configured symbol pairs. Gail retains a rolling log-price
ratio and requires the configured minimum sample count and return correlation.
An entry requires a ratio z-score beyond the threshold and an expected
convergence move that covers both legs' round-trip costs and the safety margin.

### Carry

Carry combines annualised funding and basis convergence, then subtracts the
annualised two-leg execution cost. Funding size, open interest, leverage and an
approximate liquidation buffer are mandatory gates. Reverse carry is disabled
by default because borrow availability cannot be inferred from ticker data.

Both sleeves implement the same `AlphaSleeve` interface and produce a common
recommendation structure. They default to `shadow_only: true` and
`atomic_hedge_execution_supported: false`. Changing only one of those flags is
insufficient for execution. Gail does not presently submit sleeve orders; the
flags document the safety contract for a future atomic execution adapter.

Actionable sleeve opportunities create non-overlapping persistent shadow
markouts. Pairs resolve the return of both opposing legs after costs. Carry
resolves accrued entry funding and futures-basis convergence after costs. These
records make future promotion depend on measured net ROI rather than projected
annualised yield.

### LLM risk overlay

In LLM-primary shadow mode, a round with adequate provider coverage and
agreement updates a bounded-TTL risk overlay. A moderate risk score dampens
quant signal, confidence and calibrated edge; a high score vetoes action. The
overlay cannot create a trade, reverse direction or increase any quantity.

Once quant is primary, Gail makes no synchronous LLM or Refiner call. It uses
only the most recent non-expired overlay and ignores it after expiry. This
preserves deterministic latency while retaining a short-lived qualitative
risk brake.

## Configuration and safe rollout

All new settings are under `trading.quantitative`. Values are normalised to
bounded ranges at startup. The production sequence should remain:

1. deploy with quant shadow mode and both alternative sleeves shadow-only;
2. gather fill telemetry and multi-horizon observations;
3. inspect native walk-forward validation and overfitting rejection reasons;
4. compare quant and LLM on the same non-overlapping market periods;
5. permit automatic momentum promotion only through the existing guard; and
6. keep pairs and carry non-executable until an atomic multi-leg adapter,
   venue-specific margin data and dedicated promotion criteria are available.

Important activity markers are:

| Marker | Meaning |
| --- | --- |
| `GAIL_NATIVE_QUANT_BACKTEST_COMPLETE` | Native replay completed, including promotion evidence. |
| `QUANT_MULTI_HORIZON_OBSERVATIONS_RECORDED` | Raw intent was retained for future edge calibration. |
| `QUANT_MULTI_HORIZON_OBSERVATIONS_RESOLVED` | One or more horizon labels became available. |
| `QUANT_ALPHA_SLEEVES_EVALUATED` | Pairs and carry completed their concurrent shadow evaluation. |
| `QUANT_LLM_RISK_OVERLAY_APPLIED` | The bounded risk view was refreshed or used. |

## ROI interpretation

The changes improve ROI conditions through fewer false-positive trades,
lower transaction-cost surprise, better diversification and stronger
out-of-sample selection. They can also lower gross return by holding cash more
often. Evaluation should therefore report, over the same dates and market
regimes:

- net basis points per actionable trade and per opportunity;
- confidence lower bound and downside mean;
- maximum drawdown and realised exchange P&L;
- turnover, spread, projected slippage and observed slippage;
- cash allocation and rejected-liquidity counts;
- quant-minus-LLM paired net return; and
- pairs/carry shadow net return separately from projected edge.

Do not infer improvement from signal count, annualised projection or an
in-sample Sharpe ratio alone.

## Verification

Run the following before deployment:

```bash
cargo fmt --all -- --check
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test --no-default-features
git diff --check
```

Default-feature tests additionally require the configured libtorch runtime.
The test suite covers look-ahead-safe replay, walk-forward validation, cost
fallbacks, empirical fill estimates, portfolio caps, cash selection, pairs and
carry constraints, two-leg shadow markouts, overlay expiry and the prohibition
on overlay-created trades.
