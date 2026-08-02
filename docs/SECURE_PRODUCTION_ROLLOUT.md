# Secure production rollout

This release changes trainer, trading, AARNN, and image-provenance safety
contracts. Deploy the Gail and AARNN revisions together; do not enable live
orders during the first rollout.

## Before building

1. Commit the exact source tree. The Ansible role rejects dirty production
   builds and gives local builds a commit-unique image tag.
2. For a prebuilt image, set `continuum_tenant_gail_image` to an immutable
   `repository@sha256:...` reference. Mutable release tags are rejected.
3. Confirm `/healthz` reports the expected `build.git_commit`, `source_tree`,
   and `source_dirty=false` values after deployment.

## Credential remediation

`GET /v1/trading/config` now removes credential-shaped fields recursively.
After the corrected endpoint is deployed, rotate `GAIL_TRADING_REFINER_TOKEN`
and update its Kubernetes/Ansible secret. Also rotate the OctoBot credential if
one was ever configured. Do not put replacement values in source control.

## Trainer recovery

The trainer startup reconciler converts legacy `trained_registration_pending`
rows back to due retries and clears invalid terminal timestamps. Retry rows are
selected before new interactions. A LoRA/QLoRA job becomes `trained` only after
Ollama accepts its adapter and the serving alias is updated; unsupported
adapters remain retryable and never fall back to the unchanged base model.

Monitor the ledger until both `trained_registration_pending` and overdue
`retry` counts fall. If Ollama still rejects the adapter architecture, install
a serving runtime that supports that adapter/base-model pairing before
increasing `trainer.max_attempts`.

## Advisor capacity

The production baseline uses three parallel advisors, an 840-second per-call
timeout, a 900-second outer round, and first-valid-response quorum. The values
cover the observed 11–13 minute local-provider latency with scheduling margin.
The outer deadline is normalized to remain at least 30 seconds longer than the
provider timeout. Recalculate these values from current latency metrics after
model or hardware changes.

## Paper-to-live release gate

Live execution defaults to disabled. Paper mode runs the same balance,
freshness, economics, exact-target, immediate-reprice, and intent-deduplication
checks as live mode, stopping immediately before external mutation.

The default qualification requires:

- 100 completed evaluations;
- 5 distinct actionable intents that passed every read-only gate;
- evidence produced by the exact Git commit and source-tree hash being run;
- evidence no older than seven days.

Inspect `/v1/trading/status` for `paper_build_revision`,
`paper_observed_evaluations`, `paper_validated_intents`, and
`paper_qualified_at`. Only then set `live_execution_enabled: true`. A new build
or expired evidence automatically blocks live order placement again.
To renew expired evidence, disable live execution and complete a new paper
window; pre-expiry evaluation and intent counts are not reused.

Keep OctoBot in `BlankTradingMode`; Gail remains the sole execution authority.

## AARNN

Mirrored text now produces a fixed-density sparse fingerprint instead of
saturating every sensory neuron. AARNN waits for a later neural simulation step
and returns those output spikes. It never echoes the supplied LLM response as
an SNN candidate.

Natural-language candidates require a trained neuron-to-token mapping supplied
to AARNN as `AARNN_LLM_OUTPUT_VOCAB_JSON` or
`AARNN_LLM_OUTPUT_VOCAB_PATH`. Without one, the response is honestly marked
`network_output_unmapped` and cannot replace Gail's LLM answer.

The AARNN web runtime reuses loaded workspace handles during reconciliation and
honours `NM_RUNTIME_MAX_LOADED_WORKSPACES`. The deployment baseline keeps two
workspaces resident and provides a 12 GiB web-UI memory limit. Continue
monitoring RSS and OOM termination state after rollout.
