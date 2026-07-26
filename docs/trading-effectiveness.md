# Trading effectiveness and execution safety

Gail is the sole trading decision authority. OctoBot is a market-data and order-execution venue; its autonomous `AIIndexTradingMode` must be disabled when Gail live execution is enabled. The SwarmHPC OctoBot site playbook enforces that ownership by activating `BlankTradingMode` while leaving the trader and external order API available.

The objective is positive fee-adjusted ROI, not order count. A timeout, stale signal, missing venue balance, or uneconomic recommendation therefore produces a documented hold rather than a speculative fallback.

## Decision and execution flow

1. Gail fetches market, portfolio, exchange, order, and research inputs concurrently.
2. Market rows are validated, de-duplicated, deterministically ranked, and reduced to `advisory_candidate_limit` rows.
3. Up to `max_parallel_advisors` providers race inside one `advisor_round_timeout_seconds` budget. Completed responses are retained; valid early quorum cancels stragglers. A deadline retains partial results and marks unfinished providers as failures.
4. Fuzzy and AI signals are blended. Lower provider coverage increases the confidence requirement.
5. Resolved fixed-horizon outcomes provide a bounded provider/symbol/regime calibration multiplier.
6. The economics gate estimates gross edge and subtracts round-trip fee and slippage allowances. A decision below `minimum_net_edge_bps` becomes a hold. Position size also scales with net edge.
7. Immediately before submission Gail validates advisory and snapshot age, fetches the exact exchange/symbol price, and rejects excessive adverse drift or an edge that became uneconomic.
8. Gail refreshes exchange-scoped balances. With `strict_exchange_selection: true`, OctoBot may execute only on the selected venue; it cannot silently try another exchange.
9. Gail persists an authority-tagged intent lease before invoking a mutating endpoint. A restart observes the lease and cannot repeat an ambiguous in-flight order.
10. A filled order creates a markout due at `markout_horizon_seconds`. The resolved directional return subtracts the same round-trip cost model and feeds later calibration.

## Core configuration

| Setting | Meaning | Production guidance |
|---|---|---|
| `advisor_timeout_seconds` | Per-provider request timeout | At least the round deadline |
| `advisor_round_timeout_seconds` | Maximum total advisor round duration | Below the point where market input loses relevance |
| `advisor_early_quorum` | Valid responses needed to return early | `1` for fastest-valid response; `2+` for stronger consensus |
| `market_snapshot_ttl_seconds` | Maximum source observation age | Match strategy horizon, not HTTP timeout |
| `advisory_ttl_seconds` | Maximum post-computation decision age | Keep short; execution repricing is immediate |
| `max_reprice_drift_bps` | Maximum adverse move before submit | Less than plausible expected net edge |
| `estimated_fee_bps` | One-way taker fee allowance | Highest applicable venue tier |
| `estimated_slippage_bps` | One-way price-impact allowance | Raise for thin symbols or larger orders |
| `minimum_net_edge_bps` | Profit margin after modeled costs | Positive in live trading |
| `execution_authority` | Stable owner written to durable leases | `gail` in this deployment |
| `markout_horizon_seconds` | Fixed outcome measurement horizon | Match intended holding horizon |

The pre-trade round-trip cost is `2 × (estimated_fee_bps + estimated_slippage_bps)`. `expected_move_bps` is the gross move represented by a perfect signal at perfect confidence. It is deliberately conservative and should be recalibrated from resolved markouts and realistic backtests, not increased merely to make more trades pass.

## Outcome interpretation

Markouts measure whether a recommendation direction was effective at a fixed horizon. For buys, rising price is positive; for sells, falling price is positive. Round-trip modeled costs are subtracted from both. Calibration prefers enough exact symbol/provider/regime observations, falls back to symbol observations, and finally to global resolved observations. The multiplier is bounded to prevent a small sample from dominating risk controls.

No implementation can guarantee improved ROI. These controls improve the conditions for ROI by eliminating fee-negative trades, stale decisions, duplicate execution, exchange ambiguity, and misleading next-trade feedback. Production evaluation should compare resolved net markout mean, median, win rate, skipped-gate reasons, fill rate, and realized exchange P&L over the same market regime.

## QA and deployment

```bash
./scripts/test-trading-ci-safe.sh
cargo fmt --all --check
cargo check --locked --all-targets
cargo clippy --locked --all-targets
```

The script runs every test under `trading::`, including colocated economics and outcome tests. GitHub workflows build and test both amd64 and arm64. After those checks pass, deploy the pinned Gail image and authority configuration with:

```bash
cd /home/pbisaacs/Developer/swarmhpc/swarmhpc/ansible
ansible-playbook continuum_tenant_octobot_site.yml
ansible-playbook continuum_tenant_gail_site.yml
```

After rollout, verify that OctoBot has `AIIndexTradingMode=false`, Gail reports `execution_authority=gail`, advisor rounds complete inside their deadline, and new orders include an immediately persisted intent lease and a pending markout.
