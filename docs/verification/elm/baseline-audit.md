# Native ELM baseline audit

Audit date: 1 October 2026\
Repository: `/home/pbisaacs/Developer/neuralmimicry/gail`\
Specification: `docs/specifications/rust-native-elm-v1.0.md` (provided in the working tree; third-party product wording was generalised to honour the explicit no-reference requirement)

## Checkout and build baseline

| Item | Observed value |
|---|---|
| Gail revision | `2c049b422928421c03d6b4be86170fbc3abf64b5` |
| Branch | `codex/self-hosted-linux-runners-20260927` |
| Initial working tree | Clean except for the user-provided, untracked specification file |
| Rust | `rustc 1.98.1 (48a229cea 2026-09-01)` |
| Cargo | `cargo 1.98.1 (797e8a9bc 2026-08-05)` |
| Installed relevant targets | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`; cross-target compile is not runtime evidence |
| Cargo edition/features | Edition 2024; default feature `training-libtorch`; optional dependency `tch`; no `elm` feature yet |
| Numerical/parallel dependencies | `rayon`, `rand`; no separate linear algebra crate |
| Existing test commands | `cargo test`; `scripts/test-trading-ci-safe.sh` (`cargo test --locked --lib trading:: --no-default-features --features ci-trading-tests`) |
| Baseline CPU-only test | The initial attempt ran 483 library tests successfully, but doctest compilation raced with active edits and failed to resolve an in-progress module; this is not counted as a clean baseline |

No `AGENTS.md` applies at the checkout root or in its parent directories. Repository guidance is in `README.md`, `docs/PROJECT_WORKFLOWS.md`, `docs/INFERENCE_TRADING_RELIABILITY.md`, the existing CI scripts and the crate configuration. Existing `docs/execplans/elm/` and `docs/verification/elm/` directories were empty.

## Current source anchor reconciliation

The specification's reviewed Gail revision (`ea9c65434e9312bb2fabeb0369fa85a29e222ba9`) is older than this checkout. All named integration files still exist, but the live structures have grown. No downgrade to the reviewed revision is appropriate.

| Concern / specified anchor | Current checkout finding and implementation constraint |
|---|---|
| Routing: `src/routing.rs`, `src/orchestration.rs` | Workflow tags and provider capability profiles are in `RoutingProfiles`; hard filtering, provider admission and dispatch live in orchestration/provider admission. Any ELM rank adjustment must happen after eligibility and must not replace admission or explicit-provider behaviour. |
| Admission/telemetry: `src/provider_admission.rs`, `src/nmc_telemetry.rs`, `src/hardware.rs` | Provider/host reservations, workload classes and NMC host state already enforce resource limits. ELM gets an independent bounded queue and may abstain under pressure; it cannot reserve provider capacity or loosen those gates. |
| Configuration/API: `src/config.rs`, `src/models.rs`, `src/app.rs` | `GailConfig` uses serde defaults and a central `normalize`; routes are assembled in `build_router`. Trading mutations require `trading` plus the configured admin identity or `trading_admin` scope. Generic status routes require `status`. ELM configuration needs strict nested parsing while missing configuration remains off. |
| Metrics/issues: `src/metrics.rs`, `src/api_issues.rs`, `src/adaptive_schema.rs` | Provider issues and API schema telemetry are separate from model quality. ELM metrics must be bounded-cardinality and must not affect provider health. |
| Governance/data: `src/governance.rs`, `src/redaction.rs`, `src/llm_ledger.rs` | Governance middleware and redaction are established. Ledger rows can contain prompt/response content and are not presumed permitted for numeric model training. PostgreSQL schema is created additively at runtime in `llm_ledger`; no general migration framework was found. |
| Quant/trading: `src/trading/quant.rs`, `src/trading/quantitative/`, `decision.rs`, `qualification.rs`, `outcomes.rs` | Existing deterministic quant shadow/migration, time-horizon calibration, backtest and paper-qualification paths are active. Paper qualification currently keys to build revision; a composite ELM/trading fingerprint must preserve this gate and invalidate its evidence when decision-affecting models change. |
| Training: `src/trainer_worker.rs`, `src/training/qlora_sft.rs` | Existing worker owns LLM adapter production/registration and durable PostgreSQL job semantics. ELM needs a separate job/artefact family and role; it must not enter adapter import or Ollama registration. |
| Mirroring: `src/aarnn_bridge.rs`, `src/mirror_worker.rs`, `src/aer.rs`, `src/specialists.rs` | AARNN mirroring has bounded in-process queues plus durable ledger retries and explicit acknowledgements. Specialist inference currently falls back to an explicitly named `offline_heuristic` after transport failures (`predict_inputs` → `predict_heuristic`); this is not an ELM prediction and must not be relabelled. |
| Data availability | `data/` contains provider metrics, interaction ledger, adaptive API schema and API issues. No permitted, point-in-time market dataset, AARNN activity-window dataset, or labelled routing outcome dataset was found. The interaction ledger may contain raw text and has no demonstrated ELM training consent/provenance contract, so it is not treated as permitted training data. |
| Existing operating controls | Trading paper/live execution and per-build qualification are in `src/trading`; provider selection and floors are enforced in orchestration; mirror delivery and retry state are in the ledger/worker; trainer registration/recovery are in the LLM trainer. These remain authoritative. |

## Scope and evidence boundary

This delivery implements software, tests, fixture-only numerical examples, worker/CLI surfaces and evidence collection. Available repository data does not establish permission or suitable labels for training routing, trading or AARNN models. Consequently, no production model can be claimed qualified from the checked-in data. The implementation must produce an artefact only from an explicit, validated dataset manifest; fixture artefacts remain marked `fixture_only` and cannot activate.

No live trading, production deployment, exchange mutation, order placement, or production model activation was performed during this audit.

## Initial verification result

`cargo test --locked --no-default-features` was started as the pre-change baseline. The 483 library-test pass is useful context, but the concurrent doctest failure makes the overall command invalid as a clean baseline. Post-implementation commands are tracked in `docs/verification/elm/verification-results.md`.
