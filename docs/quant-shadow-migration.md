# Quant shadow evaluation and automatic migration

Gail starts with the LLM advisory path as the live decision source and runs a
deterministic quant controller in shadow. It automatically replaces the LLM
only when paired, fee-adjusted USDT markouts prove that quant is better under
the configured promotion guards. The active mode, evidence, selected
parameters, pending observations, and completed observations are stored inside
the same atomically persisted `trading_state.json` as execution leases and
trade markouts.

## Comparison integrity

Each shadow observation has one shared timestamp and horizon. The LLM and each
quant parameter arm retain their own selected venue, USDT symbol, entry price,
direction, and actionability, then resolve against their own exact future
quote. This measures complete policies without incorrectly scoring an LLM BTC
decision against an ETH interval selected by quant. An actionable direction
pays the configured round-trip fee and slippage allowance; a hold earns zero.
Samples are created no more frequently than `quant_shadow_horizon_seconds`,
preventing heavily overlapping returns from being counted as independent
migration evidence.

The quant signal uses only bounded numeric inputs:

- short, medium, long, and live momentum;
- short/long volume confirmation;
- realized volatility and drawdown attenuation;
- data completeness;
- exact USDT-quoted markets and held-inventory validation for sell signals.

Five explainable parameter arms independently rank the same market universe at
every sampled timestamp. After
`quant_tuning_min_samples`, Gail may select another arm when it has enough
actionable observations and improves the paired risk-adjusted score by at
least `quant_tuning_min_outperformance_bps`. Parameter changes are immediately
part of persisted state.

## Migration guard

Migration requires all of the following over the bounded rolling window:

1. At least `quant_migration_min_samples` paired non-overlapping outcomes.
2. At least `quant_migration_min_actionable_samples` actionable quant signals.
3. Positive mean quant return after modeled costs.
4. Mean quant outperformance of at least
   `quant_migration_min_outperformance_bps` versus the LLM.
5. Mean quant downside no more than
   `quant_migration_max_downside_regression_bps` worse than the LLM.
6. All conditions remain true for `quant_migration_required_streak`
   consecutive resolution checks.

The migration record is atomically persisted before the new mode is used. If
that write fails, Gail restores shadow mode and logs
`QUANT_MIGRATION_ABORTED_PERSISTENCE`. There is no automatic promotion based
only on latency, provider failure, trade count, or unadjusted price movement.

Once promoted, the synchronous LLM and Refiner calls are removed from the
normal, discovery, and pruning decision paths. Gail continues quant-only
markouts so parameter selection can adapt from later outcomes. Existing
confidence, position, economics, freshness, venue, balance, lease, and
execution gates remain in force.

## Stable log markers

These exact messages appear in the activity log and, where operationally
important, the process log:

| Marker | Meaning |
| --- | --- |
| `QUANT_SHADOW_INITIATED` | A new persistent shadow controller was created. |
| `QUANT_SHADOW_RESTORED` | Shadow evidence and parameters survived restart. |
| `QUANT_PRIMARY_RESTORED` | A prior migration survived restart; quant remains primary. |
| `QUANT_SHADOW_EVALUATION_RECORDED` | A paired future markout was durably queued. |
| `QUANT_PRIMARY_EVALUATION_RECORDED` | A post-migration quant-only tuning markout was durably queued. |
| `QUANT_SHADOW_MARKOUTS_RESOLVED` | One or more comparisons were resolved. |
| `QUANT_PARAMETERS_ADJUSTED` | A paired challenger replaced the active parameter arm. |
| `QUANT_REPLACED_LLM` | Every migration guard passed and quant became primary. |
| `QUANT_MIGRATION_ABORTED_PERSISTENCE` | The cutover was cancelled because durability failed. |
| `QUANT_SHADOW_PERSISTENCE_FAILED` | A controller update was rolled back because durability failed. |

`GET /v1/trading/status` exposes `quant_mode`,
`quant_active_parameter_id`, `quant_pending_evaluations`,
`quant_resolved_evaluations`, and `quant_promotion_streak` without exposing
trade credentials or prompt content.

## Defaults

The production defaults deliberately require more evidence for migration than
for parameter tuning: 24 observations tune arms, while 96 paired observations,
16 actionable quant signals, a 10 bps mean net advantage, at most 25 bps of
downside regression, and three confirmations are required for cutover. These
values are configuration, not claims that a particular strategy will produce
positive ROI.
