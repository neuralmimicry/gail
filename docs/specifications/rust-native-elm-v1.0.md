# Gail: Rust-native ELM requirements and Codex implementation brief

Version: 1.0\
Date: 1 October 2026\
Owner: NeuralMimicry / Paul Isaac’s\
Target: `neuralmimicry/gail`, implemented through Codex in JetBrains RustRover\
Status: implementation specification; no trained model or production validation is claimed by this document.

## 1. Instruction to the implementing Codex agent

Implement the complete capability specified here in the Gail repository. Read the applicable repository instructions and inspect the current checkout before changing code. Create a baseline audit, an implementation plan and requirement-to-test traceability. Work through the phases in section 18, preserving existing behaviour when ELM is disabled. Do not stop at a prototype, scaffolding, synthetic benchmark or configuration-only integration. Complete each integration, lifecycle gate, migration, test and operational document. If real data or target hardware is unavailable, finish the infrastructure and record the exact missing evidence; leave affected models unqualified rather than inventing validation.

This document authorises implementation and local verification. It does not instruct the agent to deploy to production, alter live trading authority, place orders or access additional customer data. Within an already configured operational policy, the implemented system must make routine model lifecycle and per-task decisions automatically without repeated operator approval.

Suggested repository destination: `docs/specifications/rust-native-elm-v1.0.md`. Maintain `docs/execplans/elm/` and `docs/verification/elm/` during implementation. Reconcile these paths with existing conventions. Do not overwrite existing AGENTS.md or unrelated plans. Treat proposed names, endpoints and YAML keys below as a target contract to implement, not APIs that already exist.

Normative language: MUST is required for acceptance; SHOULD requires a documented reason if omitted; MAY is optional. All requirements are mandatory unless explicitly designated as a later extension. Numerical defaults are initial engineering policies, not claims of measured accuracy or speed.

## 2. Objective and boundaries

Gail MUST optionally use small Rust-native Extreme Learning Machines and separately qualified models to assist routing, quantitative analysis, trading, training operations and AARNN mirroring. ELM is a learned decision aid, not a replacement for Gail’s governance, admission, execution or reliability controls.

The system MUST automatically determine:

1. Whether a task is eligible for learned assistance.
2. Whether an applicable, verified model exists and its evidence remains valid.
3. Whether current inputs are compatible, sufficiently fresh and within supported conditions.
4. Whether prediction has sufficient expected benefit after overhead and uncertainty.
5. Whether to apply, shadow, abstain, fall back, retrain, demote or roll back.

A trained ELM is not inherently better than a rule, linear regression or logistic regression. Keeping the existing path is a valid automatic decision. There MUST be no target adoption percentage that incentivises unnecessary ELM use.

Out of scope for the initial release: general text generation, a new LLM serving stack, third-party compatibility layers or source translations, unrestricted online self-modification, mandatory kernel/deep ELM, and replacing trading risk controls. Native batch ELM and the complete integration/lifecycle are in scope. Online ELM is a separately gated later phase described in section 14.

## 3. Source baseline and repository reconciliation

The source review used Gail revision `ea9c65434e9312bb2fabeb0369fa85a29e222ba9` and AARNN revision `fbbbe044ad889eddbef8e7ac863a1e126fd75a61`. These are review anchors, not instructions to downgrade a newer checkout.

| Existing area | Reviewed files | Required integration concern |
|---|---|---|
| Routing and orchestration | `src/routing.rs`, `src/orchestration.rs` | Tagging, eligible candidate ranking, utility and deadline handling |
| Admission and telemetry | `src/provider_admission.rs`, `src/nmc_telemetry.rs`, `src/hardware.rs` | Preserve hard resource and provider constraints |
| Configuration and API | `src/config.rs`, `src/models.rs`, `src/app.rs` | Typed configuration, status and controlled administration |
| Metrics and issues | `src/metrics.rs`, `src/api_issues.rs`, `src/adaptive_schema.rs` | Add ELM provenance and failures without confusing provider health |
| Governance and data | `src/governance.rs`, `src/redaction.rs`, `src/llm_ledger.rs` | Policy, permissions, feature/outcome provenance and retention |
| Quant | `src/trading/quant.rs`, `src/trading/quantitative/` | Existing migration, calibration, selection, portfolio and telemetry |
| Trading | `src/trading/decision.rs`, `advisor.rs`, `fuzzy.rs`, `economics.rs`, `qualification.rs`, `outcomes.rs`, `octobot.rs` | Advisory integration, mature labels, final risk and execution gates |
| Training | `src/trainer_worker.rs`, `src/training/qlora_sft.rs` | Separate ELM jobs/artifacts from LLM adapters and their promotion |
| Mirroring | `src/aarnn_bridge.rs`, `src/mirror_worker.rs`, `src/aer.rs`, `src/specialists.rs` | Preserve AER semantics, delivery and candidate validation |

**BASE-01:** Before implementation, record actual commit, working-tree state, Rust toolchain, dependency features, test commands, data contracts, auth scopes and existing operating modes. Identify the effective execution gates rather than relying on filenames alone.

**BASE-02:** Preserve user changes and existing provider selection, paper/live controls, mirror durability and trainer semantics. Add compatibility tests before introducing new behaviour.

**BASE-03:** The reviewed Cargo configuration defaults to `training-libtorch`; ELM MUST build and operate with `--no-default-features`, independently of libtorch, CUDA, Python and Node.js. CPU-only x86_64 and aarch64 are supported deployment targets. Do not require a new external numerical runtime.

**BASE-04:** In the reviewed specialist code, failed transport can yield an arithmetic heuristic. Do not use this as an ELM prediction. Explicitly identify heuristics and honour their own configuration. Correct the relevant fallback defect with a regression test if present in the actual checkout; never silently relabel heuristic output as inference.

## 4. Native numerical implementation and provenance

**MATH-01 — Provenance.** Implement from academic specifications and mathematical derivations. Record citations, derivation, implementation authorship, dependency licences and relevant design decisions in `docs/architecture/elm-provenance.md`. Do not copy, translate or depend on third-party source code, trained weights, examples or branding. Prior source inspection means this work MUST NOT be described as a formal clean-room implementation. This requirement does not constitute patent clearance.

**MATH-02 — Initial algorithms.** Implement regularised batch ELM for regression and classification, together with linear/ridge regression and logistic classification baselines. Baseline training uses only the training partition and the same declared feature contract. Rule-based Gail behaviour remains another comparator. A model family is selected by evidence, not by preference for ELM.

For row-oriented samples, define:

- `X`: N × d normalised feature matrix; `Y`: N × k targets.
- `W`: d × h seeded fixed random projection; `b`: h biases.
- `H = activation(XW + b)`.
- Solve `min_B ||S^(1/2)(HB − Y)||²_F + lambda ||B||²_F`, with non-negative sample weights S and positive lambda.
- Inference returns raw regression values or class scores with separately fitted calibration.

**MATH-03 — Stability.** Never form an explicit matrix inverse. Prefer a stable augmented QR/SVD ridge solve, or a documented regularised Cholesky approach with conditioning/residual checks and a robust fallback. Record solver choice, effective regularisation, failure reason and residuals. Normal equations square the condition number; account for that in acceptance. Reject failed or numerically suspect fits rather than publishing them.

**MATH-04 — Reproducibility.** Specify and version the PRNG algorithm, seed, projection initialisation and feature order. Store actual weights and biases in artifacts. A seed alone is insufficient across dependency changes. Use f64 training initially; f32 inference is optional only after parity and target-performance validation. Cross-platform results require declared tolerances, not unsupported bitwise-identical guarantees.

**MATH-05 — Validation.** Reject zero/oversized dimensions, ragged arrays, non-finite values, negative sample weights, invalid hyperparameters and arithmetic overflow. Missing feature handling must follow the schema with a missingness indicator where applicable. Do not quietly convert malformed vectors into zeros.

**MATH-06 — Semantics.** Regression returns quantities with units and supported ranges. Classification provides calibrated probabilities, class mapping and calibration identity. Softmax alone is not confidence calibration. Multi-label tasks need their own output semantics. Constrain probability/range violations through validated transforms or abstention, never by hiding a broken model.

**MATH-07 — Bounded complexity.** Cap N, d, h, k, intermediate allocations, training time and concurrent jobs. Account for H storage and h² solver workspace. Record estimates before admission. Initially evaluate h in a small bounded set such as 32, 64, 128 and 256, with a deployment cap of 512. These are search limits, not required winning sizes.

**MATH-08 — No mandatory embeddings.** Prefer already available numeric and categorical features. Any text representation must be versioned and benchmarked including its cost. No network model download or large embedding model may be introduced merely to feed the router.

## 5. Architecture and isolation

**ARCH-01:** Separate numerical code from domain decisions. A suggested structure is:

```text
src/elm/
  mod.rs                 # stable internal façade
  math.rs                # projection and solvers
  baselines.rs           # non-ELM comparators
  features.rs            # schemas and preprocessing
  artifacts.rs           # bounded, versioned artifact decoding
  registry.rs            # lifecycle and immutable snapshots
  gates.rs               # deterministic policy evaluator
  inference.rs           # bounded prediction execution
  dataset.rs             # point-in-time feature/label joining
  training.rs            # native fitting and search
  evaluation.rs          # validation, calibration and reports
  lifecycle.rs           # promotion, rollback and retraining
  telemetry.rs
  integrations/
    routing.rs
    quant.rs
    trading.rs
    training_ops.rs
    mirroring.rs
```

Adapt module granularity to the repository. Do not expand the already large orchestration/trainer files with embedded numerical engines. Extract a reusable crate only if existing workspace conventions justify it.

**ARCH-02:** Use a Cargo feature `elm` for the optional capability. Old configurations load with ELM off. An ELM-enabled config on a build without the feature must yield a clear configuration diagnostic rather than silently suggesting that ELM is operating. Enabled-but-unqualified models use documented baseline fallback.

**ARCH-03:** Inference reads an immutable model snapshot. Model loading, training, persistence, validation and calibration run off the request-handling path. Atomic swaps expose a complete model/preprocessor/calibrator bundle; in-flight work retains its original bundle.

**ARCH-04:** Use bounded workers, queues and per-host CPU/RAM budgets, reusing existing hardware/telemetry controls where appropriate. A timeout around synchronous CPU work does not cancel it: enforce dimensions, bounded work, cooperative cancellation between batches and isolated training processes when hard termination is required. Avoid unbounded `spawn_blocking` jobs and nested Rayon/Tokio oversubscription.

**ARCH-05:** Reserve capacity for interactive requests, trading evaluation, mirror delivery and current LLM trainers. ELM work must respect existing workload priorities. Training yields or pauses under pressure with resumable state where feasible. Inference may abstain immediately when its queue would exceed the remaining deadline.

**ARCH-06:** Maintain separate models per task, target, feature schema, permitted tenant scope and relevant horizon. Reuse numerical code across routing and AARNN readout; never reuse fitted weights merely because vector dimensions match.

## 6. Operational modes and automatic gates

**GATE-01 — Authority envelope.** Distinguish configuration authority from learning. Operators configure allowed tasks, data use, budgets, rollout ceilings and live-trading permission through existing authorised surfaces. ELM can choose within this envelope but cannot change it, weaken it or grant itself permissions. Signed/authorised policy revisions are auditable. Normal training, evaluation, promotion and retirement inside the envelope are automatic.

**GATE-02 — Modes.** Support global and per-task modes. The most restrictive applicable mode wins.

| Mode | Behaviour |
|---|---|
| `off` | No ELM prediction, training or ELM-specific collection; existing Gail behaviour remains |
| `collect` | Permitted feature/outcome collection and bounded offline training/evaluation; no inference influence |
| `shadow` | Eligible predictions recorded alongside baseline; never alter decisions |
| `auto` | Automatically select baseline or qualified learned model, with authorised shadow/canary/active ceilings |

Existing operational logging can continue in `off`; do not add hidden ELM data capture. A build feature, global mode, task mode and model status are distinct concepts.

**GATE-03 — Two gate levels.** A lifecycle gate qualifies a model for a task and operating envelope. A per-decision gate checks that the model is applicable now. Passing one never implies passing the other.

**GATE-04 — Ordered decision process.** Evaluate inexpensive checks before feature extraction or inference:

1. Build support, mode, task permission and tenant/data permission.
2. Existing hard domain eligibility, provider capability, locality and resource admission.
3. Registry readiness, artifact integrity, task/schema compatibility and qualification expiry.
4. Feature availability, freshness, timestamp consistency and supported domain.
5. Resource/deadline feasibility and expected benefit net of ELM overhead.
6. Prediction with a pinned immutable artifact.
7. Finite result, calibration validity, uncertainty, out-of-distribution and drift checks.
8. Decision margin and authorised influence bounds.
9. Final existing policy/admission/execution recheck immediately before the action.

**GATE-05 — Typed outcomes.** Return `Applied`, `Shadowed`, `Abstained`, `Skipped`, `Failed` or `BaselineSelected`, with stable reason codes. Include request/decision ID, task, model digest, policy version, feature schema, baseline decision, learned proposal, actual action, influence cap and timing. Logs must distinguish “model not called” from “model called but rejected”.

Reason codes must cover disabled, task disallowed, model missing/unqualified/expired, artifact invalid, schema mismatch, tenant mismatch, stale/missing input, low support, OOD, drift, uncalibrated, insufficient margin, resource pressure, deadline, inference error, rollout exclusion and hard-policy rejection.

**GATE-06 — Uncertainty.** Calibrate using a separate calibration partition; assess classification Brier/log loss and reliability. For regression use a validated residual/interval method with reported coverage. A tiny ensemble is optional if total cost is measured. Statistical uncertainty is not a guarantee under drift. Model self-confidence cannot certify readiness.

**GATE-07 — Stability.** Add configurable hysteresis, cooldown and minimum dwell periods for ordinary promotion/demotion; hard policy, integrity and severe failure demotions are immediate. Prevent repeated promotion of the same rejected artifact without new evidence. Keep a deterministic qualified baseline path.

**GATE-08 — Failure semantics.** ELM failure alone must not turn a previously successful completion path into an error. Routing falls back to the existing eligible router; training operations use the existing scheduler; mirroring preserves required delivery; trading uses its already permitted baseline or hold, depending on existing gates. No fallback may broaden authority or convert missing evidence into an affirmative prediction.

## 7. Model lifecycle, registry and evidence

**LIFE-01:** Persist transitions: `candidate → trained → evaluated → qualified → shadow → canary → active`, with `rejected`, `quarantined`, `retired` and `rolled_back` outcomes. Policy may require longer shadow stages. No path from candidate directly to active is allowed.

**LIFE-02:** An immutable artifact includes model/task IDs; algorithm family/version; source commit and build fingerprint; dimensions; weights/biases/readout; preprocessing, feature schema and units; target definition and label maturity; calibration; allowed tenant/domain/horizon; training cutoff; dataset digests and provenance; seed/PRNG; hyperparameters; solver diagnostics; validation report digest; qualification envelope; expiry; dependency provenance and content hash. Online checkpoints add covariance/optimizer state and update watermark.

**LIFE-03:** Integrity hashes detect corruption but not authorship. Accept artifacts only from an authorised registry/storage boundary, with signatures where the deployment requires them. Never deserialize executable code, dynamically load model plugins or fetch an arbitrary model URL from request content. Enforce size and dimension limits before allocation.

**LIFE-04:** Publish artifacts with atomic persistence and a compare-and-swap registry revision. Multi-instance Gail requires one logical promotion writer or fenced lease per task/scope. Concurrent trainers cannot overwrite a champion. Followers activate only complete compatible artifacts; mixed rollout versions are observable. Failed publication leaves the prior champion usable.

**LIFE-05:** The verified status requires mathematical tests, artifact round-trip/parity, independent held-out evaluation, baseline comparison, applicable calibration, scope restrictions, performance evidence and domain gates. Successful training, passing unit tests or a high training score is not sufficient.

**LIFE-06:** A frozen release holdout cannot be reused indefinitely for hyperparameter search or repeated model selection. Maintain training, tuning, calibration and final evaluation roles, with rolling unseen windows as data grows. Track all attempted configurations to expose selection bias.

**LIFE-07:** Automatic retraining triggers include enough new mature labels, drift, provider/model changes, scheduled refresh and expiry risk. Deduplicate triggers and impose minimum intervals, concurrency and compute budgets. A drift event first limits use if required; starting a retrain does not extend expired qualification.

**LIFE-08:** Keep last-known-good compatible champions and reconstructable lineage. Rollback is atomic, idempotent, logged and propagated across serving replicas. If no valid predecessor exists, use the baseline. Restart restores the same eligible champion and gate state without replaying completed side effects.

## 8. Data and evaluation contracts

**DATA-01:** Capture features as they existed at decision time, not recomputed using current data. Record event time, observation time, decision time, label availability time and schema version. Normalisation and vocabulary fitting use training data only. Required metadata absent means the sample is ineligible.

**DATA-02:** Separate quality, latency, timeout/failure, monetary/resource cost and domain outcome labels. Latency is not quality; response length is not correctness; a successful HTTP response is not a successful task. Record the label source and reliability. Unresolved/censored observations are never silently labelled negative or successful.

**DATA-03:** Join delayed outcomes by stable IDs and versions with deduplication. Account for retries, cancellations, partial fills, timeouts and multiple horizons. Define corrections/retractions and exclude contaminated training snapshots. Retention and deletion propagate to datasets and require assessment of affected models.

**DATA-04:** Default to minimised numeric metadata. Text-derived features, teacher responses and customer data require the existing permission and retention policy. Do not put raw prompts, secrets or customer identifiers into metrics. Cross-tenant pooling requires explicit configured permission and independent leakage checks.

**DATA-05:** Log eligible action sets, baseline/selected actions, policy/model versions and selection propensities when actual randomisation is used. Do not fabricate propensities for deterministic decisions. Unchosen providers have unknown outcomes, not failures. Counterfactual claims require valid randomised/off-policy evidence and overlap; otherwise report the limitation. Exploration is bounded by existing cost, privacy and execution authority; live trading exploration is off by default.

**DATA-06:** Split by time and group to avoid conversation, duplicate request, strategy, instrument/horizon or shared-outcome leakage. Use purging/embargo where labels overlap in time. Report effective independent samples, class/regime coverage and confidence intervals, not only row counts.

**DATA-07:** Dataset and evaluation reports include exclusions, class balance, coverage, drift, metric definitions, baseline versions, feature cost, subgroup failures and selection history. If any mandatory sample/coverage/threshold policy is unspecified, automatic promotion stays disabled for that task.

## 9. Routing integration

**ROUTE-01:** Integrate after existing hard candidate filtering and before soft ranking/selection. Explicit provider requests retain their meaning; ELM must not redirect them unless existing user-authorised fallback semantics permit it. Recheck admission before dispatch because capacity may change after scoring.

**ROUTE-02:** Use candidate-specific features for request/workflow/role, output requirements, model/provider capabilities, context utilisation, estimated tokens, queue depth, recent latency/failure, host pressure and locality. Unknown provider/model categories require an explicit representation and support gate. Model version changes can invalidate historical support.

**ROUTE-03:** Estimate distinct outputs for task success/quality, latency and cost. The policy may combine them as:

`utility(p,x) = wq * quality_estimate(p,x) − wl * latency_normalised(p,x) − wc * cost_normalised(p,x)`.

Define units, normalisation and policy weights. Use conservative estimates or an uncertainty penalty. Compare against the baseline using a configured minimum benefit/margin; raw incomparable scales cannot be added. Include feature extraction, queueing and inference cost. Never override model floors, context limits, quotas, permissions, locality or circuit breakers for a higher utility.

**ROUTE-04:** Initially bound learned influence, for example via capped score adjustment or choosing among already eligible top candidates. Task classification is a hint; it cannot lower governance classification or remove an explicitly requested capability. Policy sets the maximum influence and canary scope.

**ROUTE-05:** Measure end-to-end task success, quality and p50/p95/p99 latency, failures, cost and ELM overhead against rules and simple baselines. Shadow predictions alone do not prove unchosen-provider quality or savings. Use authorised paired replay or a bounded canary for causal outcome evidence.

## 10. Quantitative analysis integration

**QUANT-01:** Implement independently selectable model tasks for regime classification, horizon-specific expected return/net edge, signal reliability and strategy selection assistance. At least one task must be implemented end-to-end; provide typed extension points and tested gating for the others. Do not claim all tasks are validated from one model.

**QUANT-02:** Integrate with current quantitative selection, multi-horizon calibration and quant migration rather than building a competing controller. An ELM recommendation becomes an optional input to the existing state transition, never an authority to bypass it.

**QUANT-03:** Define each instrument/universe, venue, bar/sampling interval, horizon, feature availability lag and target explicitly. Only information known at decision time is eligible. Disallow final candle values before close, future prices, revised data unavailable at the time and outcomes that have not matured.

**QUANT-04:** Use walk-forward evaluation with purged overlapping labels and suitable embargo. Tune on past training/tuning windows; assess on later unseen windows. Report multiple-search attempts and performance across relevant regimes and instruments. Compare against the existing quant path, a simple baseline and hold/no-position where meaningful.

**QUANT-05:** Evaluate predictions separately from the policy that converts predictions to orders. Strategy tests include actual configured fees, spreads, slippage assumptions, latency, funding/borrow costs where relevant, fill feasibility, turnover, exposure and drawdown. Report sensitivities rather than one optimistic execution assumption. A profitable toy backtest cannot qualify a model.

**QUANT-06:** Drift, stale feeds, unseen regimes, missing market state or insufficient calibration support cause abstention and existing baseline/hold behaviour. Each horizon has independent maturity and qualification. A short-horizon success cannot qualify a long-horizon model.

## 11. Trading integration and immutable authority

**TRADE-01:** ELM may assist expected net edge, regime-aware strategy ranking, advisory confidence and proposed sizing within existing limits. It MUST NOT call exchange/OctoBot mutation endpoints directly, gain new credentials, loosen exposure or loss limits, alter live/paper mode, or bypass freshness, balance, economics, deduplication, qualification, governance and execution gates.

**TRADE-02:** Preserve the current final intent/execution path. Feed ELM through a typed advisory result with model lineage. Final order generation and submission remain owned by existing authorised code. When ELM is off or abstains, behaviour follows the existing configured strategy.

**TRADE-03:** Extend qualification identity beyond the reviewed build-only paper qualification. Compute a composite decision fingerprint including Gail build, active trading-affecting model digests, feature/target schemas, calibration, strategy/risk-relevant policy and execution assumptions. Material changes invalidate affected qualification and start fresh evidence. Even an ELM that changes which advisory provider is selected counts if it can alter trading decisions. Do not silently inherit paper evidence across model swaps.

**TRADE-04:** Automatic activation is allowed only when the operator has already enabled that task and the current trading authority permits the target stage. Require independent model qualification, paper evidence for the composite fingerprint, valid backtest evidence, current data and every existing live gate. In default configuration, no ELM task can increase live-trading influence. Once authorised and qualified, routine promotions within the envelope need no additional manual approval.

**TRADE-05:** Do not count the ELM, its teacher LLM and an AARNN model trained from the same outputs as independent consensus votes. Preserve dependency lineage; initially treat related models as one evidence family or advisory features. Any later ensemble weighting needs a validated dependence-aware design.

**TRADE-06:** Resolve labels from real subsequent prices, fills and costs using current outcome ledgers. A shadow signal’s hypothetical P&L must be marked simulated and kept distinct from paper and realised outcomes. Prevent double counting repeated decisions against the same outcome.

**TRADE-07:** On rollback or timeout, reconcile pending intents using existing idempotency keys; never submit a replacement duplicate. Open positions remain under existing position management. Disabling ELM does not automatically liquidate positions or cancel legitimate orders. A queued order whose decision fingerprint is no longer permitted must be revalidated before submission.

**TRADE-08:** Model outputs are probabilistic predictions, not profit guarantees. Default live exploration is prohibited. Any authorised canary has explicit instrument, allocation, exposure and duration ceilings and automatic demotion criteria. These limits cannot be learned or raised by the model.

## 12. Training components: native model production and operational advice

**TRAIN-01:** Add a native ELM job family to the worker architecture with separate artifact and lifecycle types from LoRA/QLoRA, TorchScript, GGUF and Ollama registration. ELM artifacts never enter the LLM adapter importer, and an unchanged or failed artifact never counts as successful training.

**TRAIN-02:** Provide an `elm-trainer-worker` role or equivalent repository-consistent separate service entry point. Reuse persistence, admission and telemetry infrastructure while preserving separate task queues and resource reservations. Existing LLM trainers continue to operate unchanged when ELM is off.

**TRAIN-03:** Each durable job has task/scope, dataset revision, feature/target schemas, split specification, bounded search space, budget, retry policy and idempotency key. Persist transitions `queued/running/trained/evaluating/completed/failed/cancelled`; `completed` means an evaluation report and artifact are durably available, not that the model is production-active.

**TRAIN-04:** Automatically create challenger models when LIFE-07 triggers. Fit candidate and simple baselines, calibrate, evaluate, record results, then submit to lifecycle gates. Training retries are bounded and cannot repeatedly consume host capacity indefinitely. Cancellation/shutdown preserves completed artifacts and resumes or cleanly restarts unfinished work.

**TRAIN-05:** Optionally use separately trained operational ELMs to estimate training duration, memory demand, probability of job failure, dataset usefulness or scheduling benefit. These models cannot certify their own outputs, change ground-truth labels, skip validation, choose their own resource ceilings or promote the model they helped produce. Bootstrap using existing scheduling until sufficient independent outcomes exist.

**TRAIN-06:** ELM advice may recommend scheduling, batch sizing or bounded hyperparameter choices within enumerated valid configurations. Enforce hard memory/CPU estimates even if the predictor forecasts lower demand. No model-generated shell command or arbitrary executable may be run.

**TRAIN-07:** Support native CPU fitting on individual hosts first. Where existing Slurm/distributed jobs are reused, prefer parallel independent trials/folds. Do not average unrelated ELM weights. A distributed sufficient-statistics method is optional and requires identical projection/preprocessing/schema, correct sample weighting, reproducibility tests and stable solving.

## 13. Mirroring and AARNN readout integration

**MIRROR-01:** Existing required prompt/response mirroring, ledger writes, AER encoding and delivery acknowledgement remain authoritative. ELM may advise worker priority, expected usefulness, retry timing within existing bounds or optional enrichment. It cannot silently drop required mirrors, alter raw AER meaning or mark undelivered events as delivered.

**MIRROR-02:** Maintain correlation IDs, ordering requirements, idempotency and bounded durable retries. If a required write cannot be accepted, expose the existing failure/backpressure state. Lossy dropping is permitted only for explicitly optional ELM analytics and must be counted. Protect low-scored work against starvation.

**MIRROR-03:** Build a separately gated native readout task consuming timestamped AARNN activity windows: population spike counts, temporal bins, first-spike or interval summaries and declared sensor context. Feature aggregation is off the propagation path and bounded. If the runtime cannot provide adequate windows or metadata, report an unavailable capability and preserve existing mirroring.

**MIRROR-04:** Bind readout artifacts to brain/network identity, topology/readout version, population mapping, time units, window duration, preprocessing and target definition. Growth/reindexing that changes semantics invalidates compatibility unless an explicit tested mapping or stable population aggregation preserves it.

**MIRROR-05:** Start with classification/regression outputs. Do not turn a class score into a purported natural-language reply. Any future language-candidate acceptance model must have a separate task specification and independent semantic evaluation; existing AARNN candidate promotion gates remain in force.

**MIRROR-06:** Predict and record before using the current teacher response for any training/update. Evaluation snapshots and earlier state must be preserved. Do not evaluate on the answer just used to update the readout or decoder. LLM agreement is a weak teacher signal, not proof of correctness. Record teacher/model identity and avoid circular ELM→AARNN→ELM self-confirmation.

**MIRROR-07:** The reviewed AARNN README lags its source: current code consumes network output and uses a learned text-fragment mapping. Audit the actual runtime contract rather than assuming either an echo or a general language model. Prefer Gail-side readout integration with existing APIs; any required AARNN change must be separately documented and backward compatible.

## 14. Later extension: bounded online learning

**ONLINE-01:** Implement only after batch baselines, lifecycle and domain gates are complete. Online mode defaults off and is enabled per task when feedback quality and drift justify it. An OS-ELM challenger may update on mature labels; the serving champion is never mutated in place.

**ONLINE-02:** Derive recursive least-squares updates from the cited literature, document the forgetting-factor convention and verify against equivalent weighted batch fits. Cap update batch size and covariance memory. Check conditioning, symmetry and finite values; reset/rebuild a challenger when numerical quality fails.

**ONLINE-03:** Persist projection, output weights, full covariance state, hyperparameters, normaliser version, exact update watermark and schema together. An inference-only artifact is not a resumable online checkpoint. Deduplicate replayed labels and verify restart equivalence.

**ONLINE-04:** Use prequential evaluation: predict first, score when the independent label arrives, update afterwards. Changed normalisation requires coherent transformation or a newly trained challenger. Periodically checkpoint and requalify before publication. Training from biased selected outcomes still requires DATA-05 protections.

**ONLINE-05:** Online updates never bypass trading requalification, qualification expiry or promotion policy. Frequent updates may therefore remain in shadow until enough evidence is available. Kernel and deep variants are optional future ADRs with demonstrated need and bounded complexity; they are not initial acceptance dependencies.

## 15. Configuration, thresholds and management surfaces

**OPS-01:** Implement strict typed configuration with bounds and cross-field validation. Preserve backward compatibility for existing configuration. Reject misspelled/unknown ELM keys rather than silently ignoring them. Merge environment overrides consistently and expose effective redacted configuration.

Illustrative target configuration (agent must implement and round-trip test all fields):

```yaml
elm:
  mode: "off"                  # off | collect | shadow | auto
  registry_path: data/elm
  tasks:
    routing_candidate_utility:
      mode: auto
      max_stage: active
      promotion_policy: routing_v1
    quant_net_edge:
      mode: shadow
      max_stage: shadow
      promotion_policy: quant_v1
    trading_advisory:
      mode: "off"
      max_stage: shadow
      promotion_policy: trading_v1
    training_resource_estimate:
      mode: shadow
      max_stage: shadow
      promotion_policy: training_ops_v1
    mirror_priority:
      mode: shadow
      max_stage: shadow
      promotion_policy: mirror_v1
    aarnn_activity_readout:
      mode: shadow
      max_stage: shadow
      promotion_policy: aarnn_readout_v1
  budgets:
    max_training_jobs: 1
    max_training_threads: 2
    max_training_memory_mb: 512
    max_inference_queue: 128
    max_hidden_units: 512
    inference_deadline_ms: 5
  lifecycle:
    automatic_training: true
    automatic_promotion: true
    automatic_rollback: true
    online_updates: false
  trading:
    allow_live_influence: false
  data:
    allow_raw_text: false
    allow_cross_tenant_training: false
```

Global `off` overrides the illustrative task settings. Memory includes intermediates and process overhead; reject/adapt jobs exceeding the cap. The 5 ms deadline is an initial admission target, not a measured guarantee. A busy host may abstain. Automatic flags do not override mode, permissions, task ceiling or evidence.

**OPS-02:** Each named promotion policy MUST define: task objective; higher/lower-is-better direction; baseline; minimum effective samples and subgroup support; minimum elapsed observation window; confidence level; minimum practical improvement; non-inferiority tolerances; calibration/coverage limits; p95/p99 overhead budget; max input/label age; drift/OOD thresholds; canary fraction/cap; expiry; rollback trigger; cooldown; retrain budget; and authorised influence range. Use explicit units. Default bundled policies are shadow-only until calibrated for deployment data.

For active promotion, require a conservative confidence bound on improvement above the configured practical threshold, with all non-inferiority and hard domain gates passing. Define an appropriate paired/grouped or time-series-aware interval procedure. Avoid naive IID intervals for correlated market samples. Require sufficient canary evidence for claims that shadow evaluation cannot support. Missing policy fields prevent promotion with a reason code.

**OPS-03:** Add read-only status and model/report listing through existing status/auth conventions. Proposed endpoints are `GET /v1/elm/status`, `/models`, `/models/{id}/evaluations`; proposed mutations are authorised train, mode change, quarantine and rollback operations. Reconcile auth scopes with current Gail: read status is not permission to train or mutate models. Mutations need dedicated administration authority, audit logs and idempotency. Never expose artifact paths or sensitive training records to ordinary clients.

**OPS-04:** Distinguish aggregate Prometheus metrics from detailed decision records. Export task/state, predictions, applied/shadow/abstention counts, gate reasons, latency/queue histograms, training pressure, artifact load failures, drift and lifecycle transitions. Keep metric label cardinality bounded; put request IDs, full hashes and tenant-specific detail in access-controlled records instead.

**OPS-05:** Reports must show both model quality and policy outcome, baseline comparisons, uncertainty, data scope, coverage, costs and evidence expiry. A dashboard/status response must explain why Gail did or did not use a model. Do not claim savings by subtracting two hypothetical estimates without observed supporting outcomes.

**OPS-06:** Provide additive, versioned persistence migrations and downgrade behaviour. Use existing storage abstractions; a local-only deployment may use atomic files/JSONL, while shared deployments require shared registry coordination. Database outage must preserve a valid loaded champion or baseline according to evidence policy; it must not silently publish untracked models.

## 16. Acceptance matrix and required evidence

Tests below are required because the feature affects decisions and model lifecycle. Extend existing QA rather than building a disconnected demonstration suite. Record actual command, environment, result and artifact for each row.

| Test ID | Requirements | Acceptance scenario |
|---|---|---|
| ELM-T01 | BASE-02, ARCH-02, GATE-02 | Missing ELM config and feature-disabled builds preserve existing routing/trading/mirroring outcomes; no ELM jobs or collection |
| ELM-T02 | MATH-02–06 | Known small ridge problem matches an independent reference within declared tolerance; classification/regression semantics correct |
| ELM-T03 | MATH-03,05,07 | Ill-conditioned, repeated, ragged, empty, oversized, non-finite and invalid-weight inputs reject or solve with reported valid fallback |
| ELM-T04 | MATH-04, LIFE-02 | Fixed projection replay and export/import preserve predictions; schema, calibrator and normaliser round-trip together |
| ELM-T05 | GATE-01–05 | Table-driven gate tests prove every earlier hard rejection prevents influence, including explicit provider and tenant restrictions |
| ELM-T06 | GATE-06–08 | Low support, OOD, stale/expired evidence and no practical improvement select baseline with distinct reasons |
| ELM-T07 | ARCH-03–05 | Concurrent model swaps, training pressure and inference saturation keep request deadlines and stable snapshots; timeout work cannot grow without bound |
| ELM-T08 | LIFE-03,04,08 | Corrupt/oversized artifact, crash during publication, concurrent promoters and restart cannot activate partial/unverified state |
| ELM-T09 | LIFE-05–07, OPS-02 | Failed holdout/baseline/calibration gates prevent promotion; valid evidence moves through shadow/canary; regression rolls back |
| ELM-T10 | DATA-01–07 | Feature/label timestamps, grouped splits, duplicates, delayed/censored labels and corrected outcomes cannot leak or count twice |
| ELM-T11 | ROUTE-01–05 | Candidate utility can change a qualified soft ranking but cannot bypass admission, governance, floors or explicit selection |
| ELM-T12 | QUANT-01–06 | Walk-forward replay excludes future/overlapping labels; execution costs and horizon/regime restrictions appear in reports |
| ELM-T13 | TRADE-01–04 | ELM cannot directly submit orders or change live authority; model/policy swap invalidates affected paper qualification |
| ELM-T14 | TRADE-05–08 | Correlated votes, repeated outcomes, pending intent rollback and duplicate submission scenarios behave correctly |
| ELM-T15 | TRAIN-01–07 | ELM jobs produce genuine native artifacts/reports; retries/resume work; existing LLM training/import and scheduling remain valid |
| ELM-T16 | MIRROR-01–07 | Required mirrors survive scoring failure; no starvation; predict-before-teach and topology compatibility enforced |
| ELM-T17 | ONLINE-01–05 | If enabled, online update matches weighted batch reference; full checkpoint resumes; inference-only artifact cannot resume silently |
| ELM-T18 | OPS-01–06 | Configuration parsing, scope checks, migrations, metrics bounds, policy precedence and audit records are verified |
| ELM-T19 | All runtime areas | x86_64 and aarch64 CPU-only smoke/parity evidence; no mandatory Python/Node/CUDA/libtorch dependency |
| ELM-T20 | ROUTE-05, LIFE-05 | Representative end-to-end replay/canary report compares rule, simple baseline and ELM, including feature and inference overhead |

Synthetic fixtures are appropriate for numerical, fault and lifecycle tests but cannot qualify market performance, routing quality or AARNN semantic accuracy. Keep demo models visibly marked `fixture_only`, with registry enforcement preventing production activation.

Initial benchmark matrix: d = 16/64/128, h = 32/128/256, k = 1/8, candidate counts = 1/8/32; include cold load, warm inference, feature extraction, concurrent requests and training contention. Record hardware, allocator/peak memory, build flags, data sizes, p50/p95/p99, throughput and baseline overhead. Deployment policy chooses acceptable budgets from results. If a requested platform is unavailable, document the missing hardware evidence and do not claim it passed.

## 17. Implementation commands and deliverables

Agent MUST discover repository-specific commands first. Expected CPU-only gates, after implementing feature `elm`, include:

```bash
cargo fmt --all -- --check
cargo test --locked --no-default-features
cargo test --locked --no-default-features --features elm
cargo test --locked --lib trading:: --no-default-features --features ci-trading-tests,elm
cargo clippy --locked --all-targets --no-default-features --features elm -- -D warnings
```

Also run the existing default-feature tests on an environment with its libtorch prerequisites, or document the specific unavailable prerequisite. Add a feature-combination CI matrix, integration tests, numerical reference tests and performance harness. Never present tests that were only planned as executed.

Required repository deliverables:

1. Native algorithms, simple baselines and reusable inference façade.
2. All five domain integrations: routing, quant/trading, training operations/model production, mirroring and AARNN readout capability handling.
3. Versioned datasets/artifacts, durable jobs, model registry and automated lifecycle.
4. Gate policy engine, typed reasons, defaults, migrations, status and authorised controls.
5. Fault, parity, numerical and integration tests mapped to this specification.
6. An evaluation CLI/role that can train, compare, calibrate, verify and export a model using an explicit dataset manifest; document real commands and exit codes.
7. Generated evaluation reports and fixture artifacts; real-data artifacts only where permitted data and evidence exist.
8. Provenance ADR, operational runbook, model/data schema documentation and rollback instructions.
9. Release notes stating compatibility, resource impact, remaining limitations and actual qualified model scopes.

The model-generation pipeline must produce artifacts by execution. Committing hand-written coefficients as an allegedly trained model does not satisfy this requirement.

## 18. Ordered implementation phases

| Phase | Work | Exit condition |
|---|---|---|
| 00 — Baseline | Audit current checkout, instructions, gates, schemas, tests, data availability and scope | Baseline report, traceability and plan; no live changes |
| 01 — Numerical core | Native batch ELM, baselines, schemas and artifact format | ELM-T02–04 pass; genuine fixture train/export/load |
| 02 — Isolation and gates | Feature/config, immutable snapshots, bounded workers, policy evaluator and telemetry | ELM-T01,05–08 and applicable T18 pass |
| 03 — Data and lifecycle | Durable datasets/jobs, evaluation, calibration, registry and automatic transitions | T09–10 pass; insufficient evidence demonstrably blocks promotion |
| 04 — Routing | Shadow and bounded automatic candidate utility | T11 and representative baseline comparison; automatic fallback works |
| 05 — Quant/trading | Time-aware targets, calibration, advisory integration, fingerprint qualification | T12–14 pass; live influence remains within configured authority |
| 06 — Training/mirroring | ELM worker integration, operational advice, mirror priority and readout adapter | T15–16 pass; required work and LLM trainers unaffected |
| 07 — Operational verification | CI, target hardware, resource tests, runbooks, canary/rollback exercise | T18–20 evidence recorded; outstanding qualification clearly listed |
| 08 — Optional online | OS-ELM challenger updates and resumable checkpoint | T17 passes plus full lifecycle requalification; defaults stay off |

Each phase updates requirement status, decisions, test evidence and remaining work. Architecture may evolve after source audit, but any change to authority, fallback or evidence requirements must be made explicit. Do not quietly reduce scope when data is missing: finish the pipeline and explain the remaining operational qualification.

## 19. Definition of done

Implementation is complete when all initial-release requirements have executable code and verification evidence; disabled mode is compatible; trained artifacts can be automatically produced, independently evaluated and correctly withheld/promoted; every domain has a tested gated integration; failure and rollback preserve existing authority; and the operator can understand every ELM decision from status/audit records.

Production qualification is a separate status per model/task/environment. It requires the configured real-data, paper/canary and target-hardware evidence. A complete software delivery may correctly contain zero production-qualified models if that evidence is absent. The system must then continue to function using its baseline and automatically gather/train/evaluate within the authorised policy until qualification is earned.

## 20. Academic and source references

Use these as starting points; inspect the actual publications when implementing the derivations. Cite additional numerical and calibration methods chosen during implementation.

1. Huang, G.-B., Zhu, Q.-Y., Siew, C.-K. (2006). Extreme learning machine: Theory and applications. Neurocomputing 70, 489–501. https://doi.org/10.1016/j.neucom.2005.12.126
2. Liang, N.-Y., Huang, G.-B., Saratchandran, P., Sundararajan, N. (2006). A fast and accurate online sequential learning algorithm for feedforward networks. IEEE Transactions on Neural Networks 17(6), 1411–1423. https://doi.org/10.1109/TNN.2006.880583
3. Huang, G.-B., Zhou, H., Ding, X., Zhang, R. (2012). Extreme Learning Machine for Regression and Multiclass Classification. IEEE Transactions on Systems, Man, and Cybernetics, Part B 42(2), 513–529. https://doi.org/10.1109/TSMCB.2011.2168604
4. Williams, C. K. I., Seeger, M. (2001). Using the Nyström Method to Speed Up Kernel Machines. Advances in Neural Information Processing Systems 13. Optional future kernel reference. https://proceedings.neurips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html
5. Gail review anchor: https://github.com/neuralmimicry/gail/tree/ea9c65434e9312bb2fabeb0369fa85a29e222ba9
6. AARNN review anchor: https://github.com/neuralmimicry/aarnn_rust/tree/fbbbe044ad889eddbef8e7ac863a1e126fd75a61

## Appendix A — Paste into JetBrains Codex

```text
Read docs/specifications/rust-native-elm-v1.0.md and the applicable repository instructions. Implement the specification in this Gail checkout. First reconcile its source anchors with current code and create the baseline audit, phased execution plan and requirement/test traceability. Then implement phases 00–07 with native Rust numerical code, baselines, bounded background training, immutable model artifacts, automatic evidence-based gates and integrations across routing, quant/trading, training and mirroring. Treat phase 08 online learning as a separate optional extension.

Preserve existing behaviour when disabled and preserve all governance, admission, trading, trainer and mirror-delivery authority. Do not port third-party source. Do not fabricate trained models, test results or production qualification. Automatically produce and evaluate genuine artifacts from permitted data; use fixture-only artifacts for tests when real data is unavailable. Missing data blocks qualification, not implementation of the pipeline.

Run and record the required verification, provide operational/configuration/rollback documentation, and report completed requirements and remaining evidence precisely. Do not deploy to production, enable live trading or place orders as part of this implementation task.
```
