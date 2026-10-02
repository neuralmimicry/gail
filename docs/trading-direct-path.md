# Gail trading path and Jev Trader analysis

Reviewed Jev Trader at `b587759e459ea049590102e54a0b07800864cdc3` and Gail's
OctoBot deployment path on 2026-10-02.

## Existing Gail workflow

`run_evaluation_loop` in `src/trading/mod.rs` schedules one evaluation at a
time. It reads a bounded OctoBot market universe, updates the market lake and
quant observations, fetches portfolio and order state, then selects a market.
In shadow mode, the LLM consensus supplies the execution signal while Rust
quant runs for evidence. Promotion to quant primary is guarded by paired
outcomes and native backtest qualification. The decision engine still applies
fuzzy, portfolio, cost, freshness and confidence gates.

`execute_if_warranted` checks balances and the exact venue's current price,
then rechecks drift and net edge. It requires paper and backtest qualification,
atomically persists an intent lease, verifies that the pair is active in
OctoBot and sends an order. OctoBot remains the sole order and exchange
authority. An acknowledged order updates Gail history and markouts. An
ambiguous response keeps the lease so another request cannot duplicate it.

The production inventory at
`swarmhpc/ansible/continuum_tenant_gail_site.yml` currently configures a
900-second evaluation interval and a 900-second advisory round. At review,
the deployed controller reported `QUANT_SHADOW_RESTORED` and its latest native
backtest had not qualified. A short network path cannot turn this configuration
into Jev Trader's 300 ms block cadence while shadow mode remains primary.

## Useful Jev Trader patterns

Jev Trader's `src/chain.ts`, `src/market.ts` and `src/trader.ts` keep one
decision in flight, read the book once, submit a transaction immediately, and
collect receipts, fills and fee updates later. The market-specific transaction
uses Kuru's `batchUpdate`, preselected gas and a local nonce. Its 300 ms
budget, post-only quote pricing, Monad RPC calls and event log decoder are
specific to Kuru and cannot be sent through OctoBot's spot-order API.

The reusable principle is to keep optional advisory and observation work off
the active decision path, and to send through an endpoint already proven to
acknowledge the order directly. These parts are implemented in native Rust:

1. Once quant is primary, Gail starts one bounded background LLM advisory
   round. A completed result refreshes the persistent risk overlay for future
   evaluations. The current Rust quant decision applies the last unexpired
   overlay and does not wait for that network round. On demotion or shutdown,
   the pending background advisory is cancelled. Shadow mode retains its
   synchronous LLM consensus and migration evidence.
2. The first OctoBot order captures open-order and trade baselines concurrently.
   A mode is marked direct only after a response containing an actual order
   acknowledgement. Later orders on that mode take one POST, with no baseline
   reads or side-effect polling. If the direct endpoint rejects an order, Gail
   captures a baseline before trying another endpoint. If its HTTP success is
   ambiguous, or the server returns an uncertain 5xx error, Gail does not
   retry a mutating endpoint; the durable lease is retained. OctoBot's
   Ansible compatibility patch already returns a created
   order id from `/api/orders?action=create_order`.
3. The selected order mode logs acknowledgement type, baseline use and elapsed
   path time at debug level so latency can be measured after rollout.

Gail continues to use OctoBot's market snapshots, balance checks, pair
activation and execution history. Post-only replacement quoting, Monad gas
logic and wallet signing are deliberately excluded because they would change
the trading strategy and bypass OctoBot's order authority.

## Validation and deployment

The `Trading CI-Safe Tests` workflow runs the whole Rust `trading::` test
module without libtorch. The main `Build, Version, and Release` workflow runs
`scripts/preflight.sh --ci`, which checks the default-feature build, clippy
and tests before packaging. The direct-order test exercises repeated explicit
acknowledgements, an ambiguous response, and a server error, including the
absence of extra baseline reads and unsafe fallback submissions.

The local Ansible deployment builds the Gail image from this working tree,
imports it into k3s and rolls out `deployment/gail`. OctoBot is built by its
own Ansible role from pinned source with a compatibility patch; its local
source checkout is read-only for this change. Production validation should
check `QUANT_LLM_RISK_OVERLAY_REFRESHED`, order-path debug timings when orders
qualify, and the existing intent-lease and quant promotion markers. The
currently shadowed production controller will not exercise the new primary
advisory path until its evidence gates promote it.
