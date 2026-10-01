# Requirement and test traceability

This matrix follows the requirements and acceptance scenarios in
`docs/specifications/rust-native-elm-v1.0.md`. **Verified** means the mapped
software check ran successfully; it does not imply production qualification.
**Partial** means code and tests exist but a named acceptance condition still
needs real data, target hardware, or an operational stream. Fixture evidence
is always marked as such.

## Requirement coverage

| Requirements | Current implementation anchors | Verification evidence | Status |
|---|---|---|---|
| BASE-01–04 | `baseline-audit.md`; `Cargo.toml`; `src/config.rs`; `src/orchestration.rs`; `src/specialists.rs`; `src/trading/`; `src/mirror_worker.rs` | Source-anchor audit; default-off and no-default-features suites; specialist fallback regression tests | Verified in software; initial baseline attempt is explicitly excluded because it overlapped edits |
| MATH-01–08 | `src/elm/{math,baselines,features,artifacts}.rs`; `docs/architecture/elm-provenance.md` | ELM feature suite, deterministic/validation/round-trip cases, fixture train/export/reload | Verified in software; fixture-only evaluation |
| ARCH-01–06 | `src/elm/`; `src/elm_config.rs`; bounded worker and registry modules | Feature-on/off tests; aarch64 cross-compile; deployed arm64 startup | Verified in software and runtime startup; native inference parity remains untested |
| GATE-01–08 | `src/elm/gates.rs`; `src/elm/lifecycle.rs`; configuration, status and telemetry integration | Ordered gate/policy cases in ELM suite; fixture report correctly remains unqualified | Verified in software; production evidence gates remain unsatisfied without permitted data |
| LIFE-01–08 | `src/elm/lifecycle.rs`; `src/elm/registry.rs`; `src/elm/worker.rs`; immutable artefact publication | ELM lifecycle/registry tests; genuine fixture artefact export and digest reload | Verified in software; no production champion or replica exercise |
| DATA-01–07 | `src/elm/data.rs`; manifest/schema docs; evaluator | Manifest, timestamp, group-split, maturity and report tests; fixture manifest SHA recorded | Partial: pipeline verified with fixture; permitted domain labels are unavailable |
| ROUTE-01–05 | `src/orchestration.rs`; `src/elm/routing.rs`; provider admission integration | Routing and fallback tests in ELM feature suite; benchmark matrix | Partial: routing integration verified; no permitted outcome replay/canary evidence |
| QUANT-01–06 | `src/elm/quant.rs`; `src/trading/quant.rs`; trading outcome/calibration paths | Net-edge schema, conservative fallback and trading regression tests | Partial: task and gates implemented; no real market/fill data for walk-forward or cost qualification |
| TRADE-01–08 | `src/trading/advisor.rs`; `src/trading/mod.rs`; `src/trading/qualification.rs`; outcome and execution paths | CI-safe trading suite; composite fingerprint and ELM-can-only-demote tests; production policy caps live influence and canary exposure | Verified in software; live ELM authority is explicitly enabled by deployment policy, but no trading model is qualified and the existing Gail paper/risk/execution gates remain mandatory |
| TRAIN-01–07 | `src/elm/worker.rs`; `src/bin/gail-elm-evaluate.rs`; `src/main.rs`; existing trainer kept separate | Feature test suite; fixture evaluation creates and verifies a native model artefact | Partial: local software path verified; no permitted real training job was available to operate |
| MIRROR-01–07 | `src/mirror_worker.rs`; `src/llm_ledger.rs`; `src/elm/mirror.rs`; AARNN readout contract | Mirror-delivery and advisory tests in feature suite | Partial: required delivery remains independent; timestamped AARNN windows and semantic validation are unavailable |
| ONLINE-01–05 | Explicit configuration guard; no update/checkpoint implementation | Phase 08 explicitly deferred; `online_updates` remains false | Deferred optional extension |
| OPS-01–06 | `src/elm_config.rs`; app status routes; `src/orchestration.rs`; Grafana dashboard; `docs/operations/elm.md`; CI workflow | Configuration/status tests; Prometheus gate-state and gate-process tests; provisioned Grafana panels; successful Ansible rollouts; format/build/test checks; runbook and rollback review | Verified in software and deployment except repository-wide Clippy gate; model qualification remains blocked by missing approved data |

## Acceptance test execution map

| Test ID | Requirements | Evidence run | Result |
|---|---|---|---|
| ELM-T01 | BASE-02, ARCH-02, GATE-02 | `cargo test --locked --no-default-features`; `cargo test --locked --no-default-features --features elm` | Passed: 493 and 515 library tests respectively |
| ELM-T02 | MATH-02–06 | ELM numerical unit cases in the feature suite | Passed |
| ELM-T03 | MATH-03, 05, 07 | Invalid-input, numerical stability and bounded-allocation unit cases | Passed |
| ELM-T04 | MATH-04, LIFE-02 | Deterministic projection and artefact digest round trip; evaluator export/reload | Passed; fixture artefact only |
| ELM-T05 | GATE-01–05 | Ordered gate and policy tests in the ELM feature suite | Passed |
| ELM-T06 | GATE-06–08 | Support, freshness, uncertainty and fallback tests in the ELM feature suite | Passed |
| ELM-T07 | ARCH-03–05 | Bounded worker, queue and snapshot tests in the ELM feature suite | Passed |
| ELM-T08 | LIFE-03, 04, 08 | Artefact integrity, atomic publication and revision tests in the ELM feature suite | Passed |
| ELM-T09 | LIFE-05–07, OPS-02 | Lifecycle/policy tests and fixture report with insufficient evidence | Partial: software gates withheld qualification; no real-data champion transition |
| ELM-T10 | DATA-01–07 | Manifest and split tests; 120-row fixture evaluation | Partial: fixture pipeline passed; no real permitted labels/corrections stream |
| ELM-T11 | ROUTE-01–05 | Router eligibility/admission/fallback tests; 72-case fixture benchmark | Partial: software integration passed; no representative paired routing outcomes |
| ELM-T12 | QUANT-01–06 | Quant schema, timestamp and advisory fallback tests | Partial: implementation tested; no market walk-forward/fill evidence |
| ELM-T13 | TRADE-01–04 | `cargo test --locked --lib trading:: --no-default-features --features ci-trading-tests,elm` | Passed: 259 tests, including composite decision fingerprint and baseline authority cases |
| ELM-T14 | TRADE-05–08 | Trading regression suite and ELM conservative-influence tests | Partial: software regressions passed; no canary/rollback outcome stream |
| ELM-T15 | TRAIN-01–07 | Fixture train/evaluate/export/reload; default-feature QLoRA regression test | Partial: artefact pipeline verified; no real permitted worker job executed |
| ELM-T16 | MIRROR-01–07 | Mirror advisory and required-delivery tests in ELM feature suite | Partial: delivery behaviour covered by software tests; no timestamped AARNN runtime input |
| ELM-T17 | ONLINE-01–05 | No test run; extension deferred | Deferred |
| ELM-T18 | OPS-01–06 | Feature/config tests, status tests, docs, and required verification commands | Partial: software checks passed except the all-target Clippy command, which reports repository-wide lint debt |
| ELM-T19 | All runtime areas | x86_64 tests/benchmark, aarch64 cross-compile, and arm64 Gail/worker rollout on qc01 | Partial: arm64 services started Ready; no qualified model was available for inference parity or target-hardware benchmarking |
| ELM-T20 | ROUTE-05, LIFE-05 | 72-case fixture benchmark and fixture evaluation report | Partial: overhead measured on synthetic fixtures; no real replay/canary evidence |
| ELM-T21 | OPS-03–05, GATE-05 | Gail decision logs with `active_gate_processes`; live `/metrics`; deployed Grafana ConfigMap panel inspection | Verified: runtime logs and `gail_elm_gate_process_active` expose the same named process set, while Grafana provisions gate-composition, gate-process, influence-cap and canary-budget panels. The authenticated browser view was not independently inspected. |

## Qualification boundary

The evaluation artefact and report under `docs/verification/elm/artifacts/`
are genuine outputs of the implemented pipeline, but are explicitly
`fixture_only`. The selected ridge baseline ties its comparator and the report
records `qualified: false`. The deployment's approved manifest is absent, and
no permitted real routing outcome, point-in-time market/fill, or timestamped
AARNN activity data was found. Therefore no production model is qualified.
See `verification-results.md` for exact hashes, commands, host measurements,
deployment evidence, the Clippy result and remaining evidence.
