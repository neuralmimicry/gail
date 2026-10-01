# Native ELM verification record

Updated: 1 October 2026. Commands ran in the implementation working tree unless
the deployment evidence below says otherwise. These results verify software
and rollout behaviour; they do not qualify a production model.

## Required verification

| Check | Command | Result |
|---|---|---|
| Formatting | `cargo fmt --all -- --check` | Passed after the final documentation update |
| CPU-only baseline tests | `cargo test --locked --no-default-features` | Passed: 493 library tests; 0 failed |
| ELM CPU-only tests | `cargo test --locked --no-default-features --features elm` | Passed: 515 library tests; 0 failed |
| Trading CI-safe plus ELM | `cargo test --locked --lib trading:: --no-default-features --features ci-trading-tests,elm` | Passed after the final ledger decoder fix: 259 tests; 0 failed |
| Clippy | `cargo clippy --locked --all-targets --no-default-features --features elm -- -D warnings` | Failed: 58 library and 72 library-test lint diagnostics across established modules. No diagnostic points into `src/elm/` or `src/elm_config.rs`; the repository-wide lint gate remains red. Full output was captured at `/tmp/gail-elm-clippy-final.log` for this session. |
| Default-feature tests | `cargo test --locked` | Passed: 493 library tests, 1 QLoRA binary test, and 0 doctests; all passed |
| aarch64 CPU-only compile | `cargo check --target aarch64-unknown-linux-gnu --locked --no-default-features --features elm` | Passed after the final ledger decoder fix. |
| x86_64 CPU | Local host | Passed for fixture tests and benchmark; Intel Core i9-9880H, Linux, 16 reported CPUs |
| aarch64 CPU runtime | qc01 production node | The arm64 Gail, ELM trainer, mirror and existing trainer pods started Ready and served runtime work. Numerical inference parity and target-hardware benchmarking were not run because there is no qualified model. |

The earlier pre-change `cargo test --locked --no-default-features` attempt ran
483 library tests successfully, but its doctest compilation overlapped active
edits and failed to resolve an in-progress module. It is not counted as a clean
baseline.

The final verification also covers a saturating increment for recovered
adaptive-schema semantic-hint counters. This prevents a persisted maximum
counter from panicking while unrelated routing and provider requests update the
hint; its regression test passes in both feature-disabled and ELM-enabled
suites.

## Deployment and live telemetry

The final Gail Ansible play passed syntax validation and completed with 208
tasks OK, 15 changed, no unreachable hosts and no failures. Its persistent-mount
assertion passed. The deployed image is
`ghcr.io/neuralmimicry/gail:2c049b422928`; Gail, the ELM trainer, the mirror
worker and the existing trainer worker were all Ready on qc01 with no pod
restarts at verification time. The mirror worker processed PostgreSQL ledger
batches and recorded comparative-validation results after the `INTEGER` to
Rust `i32` decoder correction.

The Grafana provisioning play passed syntax validation and its initial rollout
completed with 49 tasks OK, 4 changed and no failures. The first Grafana
startup reported a transient SQLite `database is locked` while saving the Gail
dashboard. The same Ansible play was rerun with its targeted rollout-restart
override; recovery completed with 49 tasks OK, 3 changed and no failures. The
new pod is Ready with zero restarts, and its startup log shows dashboard
provisioning started and finished without an error. The live
`grafana-dashboards` ConfigMap contains the **ELM active decision gates by
task**, **ELM effective influence cap**, **ELM canary decisions remaining**,
and **Active decision gate processes by task** panels. The process panel reads
the same bounded process names that Gail writes to each decision's
`active_gate_processes` field, including provider admission, composite paper
qualification, risk/execution rechecks and required mirror delivery. The
dashboard reads live Prometheus series. Its external route requires
authentication, so the provisioned dashboard and metric source were verified,
but an authenticated browser rendering was not independently inspected.

The live Gail `/metrics` response showed `mode="auto"` and
`stage_ceiling="active"` for routing, quant/net-edge and mirror-priority
tasks. For quant/net-edge, `qualified_model_available=0` and `active_stage=0`;
the live-trading-authority gate is configured, but it does not qualify a
model. Gail decision logs recorded `active_gate_processes` with the combined
ELM feature, mode, permission, model-integrity, freshness/calibration,
rollout, inference, fallback and trading-authority gates. Those decisions
selected the baseline with `QualifiedModelUnavailable`, did not call ELM and
reported zero applied influence. The remaining task series likewise report
no qualified model or active ELM stage. `gail_elm_gate_process_active` now
exports the bounded provider, trading-authority and mirror-delivery process
names to Grafana with the same task mode and model-stage context.

Persistence was checked on the deployed host. Gail's `gail-data` claim is
Bound on the `continuum-shared` storage class (5 GiB, RWX); it is mounted at
`/app/data`, backed on qc01 by `/srv/gail-training`. The ELM worker logs its
durable queue at `/app/data/elm/jobs`, and the ELM registry directory exists
on the same mount. The PostgreSQL StatefulSet is Ready with its 20 GiB
`postgres-data-postgres-0` claim Bound on `continuum-shared`; its separate
50 GiB backup claim is also Bound. The approved ELM manifest is absent.

The rollout retained the site's existing operator-armed, auto-gated live
execution configuration; this ELM change did not toggle that setting. Gail
remains able to submit an order when its existing paper qualification, quant,
economics, risk and execution gates permit one. No order was placed. No ELM
model was qualified or applied to a trading decision.

## Fixture training and evaluation

The evaluator trained the candidate and comparators from the checked-in,
explicitly fixture-only manifest, exported the selected model, reloaded it and
verified its content digest. The command intentionally exited `2`, the CLI's
status for an exported but unqualified artefact:

```bash
cargo run --locked --no-default-features --features elm --bin gail-elm-evaluate -- \
  --manifest docs/verification/elm/fixtures/synthetic-routing-v1.json \
  --config docs/verification/elm/fixtures/qualification-blocked.yaml \
  --output-dir docs/verification/elm/artifacts
```

| Evidence | Value |
|---|---|
| Manifest SHA-256 | `e56418752f877525b4f75abbcf3c3469eb2db65ee34c18fb2fbab61c5ad110c4` |
| Model | `bb64190a-e650-41fe-8d9d-4d3852e00398` (`ridge`, selected against ELM challengers) |
| Artefact file SHA-256 | `9b7fffaa6d637b645938bb4d60a62046e05e4066ac1e05e3397e1d15c1635141` |
| Artefact internal content SHA-256 | `8127b21cd3390eb8c7477201beee2437a3b1031e990371447ea04044560936dd` |
| Report | `artifacts/bb64190a-e650-41fe-8d9d-4d3852e00398.evaluation.json` |
| Report file SHA-256 | `0a362dde742ca6bfd8c823e9d5830bbf86548f098224a8cd01ceb4f270deae91` |
| Report's canonical digest | `e4b9fd405378dfc86cbbfd4509357b6499c8d459295471dda4e08d9dc157ade1` |
| CLI exit | `2` (expected: exported, unqualified) |
| Qualification | `false`; reasons: `fixture_only_artifact`, `minimum_practical_improvement_not_met`, `confidence_bound_not_met` |

The selected ridge baseline ties itself on final MSE; the fixture therefore
provides no improvement evidence. The zero-influence fixture policy cannot
authorise live or production use.

## CPU benchmark

The release benchmark ran 72 fixture-only cases on the local x86_64 host with a
two-thread inference pool: 54 warm-inference matrix cases and 18 concurrent
training-contention cases. The recorded matrix covers dimensions 16/64/128,
hidden units 32/128/256, outputs 1/8 and candidate counts 1/8/32. It measures
training, preprocessing, artefact write/load, warm inference and inference
during fitting. Results are in `benchmarks/cpu-x86_64-linux.jsonl`; the run log
is `benchmarks/cpu-x86_64-linux.log`.

| Measurement | Result |
|---|---:|
| Median per-case warm-inference p95 | 0.088648 ms |
| Maximum per-case warm-inference p95 | 0.706344 ms |
| Maximum per-case warm-inference p99 | 0.706994 ms |
| Concurrent training contention wall time | 2.233–64.215 ms |
| Maximum process RSS observed | 13.11 MiB |

These are synthetic fixture measurements from this host and build. They are not
production latency, throughput, or qualification evidence.

## Remaining evidence

The deployment storage contains no approved manifest or permitted real routing
outcomes, point-in-time market and fill data, or timestamped AARNN
activity-window dataset. No production model is qualified. Native arm64
service startup is verified, but ELM inference parity, representative
deployment-hardware latency, and a canary outcome stream remain unavailable.
AARNN readout correctly reports unavailable without timestamped activity
windows. Phase 08 online learning is deferred and remains disabled. Production
Ansible configuration is set to `auto`; without a qualified model, ELM did not
influence a decision. No order was placed.
