# Rust-native ELM implementation plan

Status: phases 00–07 implemented; verification and evidence limits recorded\
Specification: `docs/specifications/rust-native-elm-v1.0.md`\
Baseline: `docs/verification/elm/baseline-audit.md`

## Phase 00 — Baseline and reconciliation

**State:** complete. The current checkout, build features, toolchain, source
anchors, domain authority gates, test commands and data availability are in the
baseline audit. The older reviewed revision was reconciled with current code.

## Phase 01 — Native numerical core

**State:** complete. Gail now has native regression and classification ELMs,
ridge and logistic comparators, training-only feature normalisation, stable
regularised solving, versioned seeded projections and bounded immutable model
artefacts. Numerical validation, deterministic replay and artefact round-trip
tests pass. A fixture evaluator run trained, exported and reloaded a ridge
artefact; it remained unqualified.

**Exit evidence:** ELM-T02–04 tests passed; fixture digests and report are in
`docs/verification/elm/verification-results.md`.

## Phase 02 — Isolation, configuration and gates

**State:** complete. The `elm` feature is independent of default libtorch.
Strict optional configuration defaults off; bounded workers, immutable
snapshots, ordered evidence gates, decision telemetry and read-only status
surfaces are implemented. Feature-disabled builds reject enabled ELM
configuration clearly, and disabled-mode suites pass.

**Exit evidence:** ELM-T01, T05–08 and relevant T18 software tests passed.

## Phase 03 — Data and lifecycle

**State:** complete. Timestamped manifests, mature-label validation, grouped
chronological splits, evaluation/calibration reports, durable jobs, atomic
artefact publication, revision-checked lifecycle transitions, expiry,
retraining and rollback are implemented. Insufficient evidence blocks
qualification. No permitted real domain dataset was found.

**Exit evidence:** ELM-T09–10 software tests passed; fixture export withheld
qualification as required. Real-data evaluation remains unavailable.

## Phase 04 — Routing

**State:** complete. Gated routing assistance runs after hard eligibility and
before soft selection; existing admission is rechecked before dispatch.
Explicit provider requests and disabled-mode selection retain their existing
meaning. Default influence remains zero.

**Exit evidence:** ELM-T11 tests passed. The fixture harness compares the
candidate with simple baselines, but representative routing outcome replay and
canary evidence are unavailable.

## Phase 05 — Quant and trading

**State:** complete. A time- and horizon-scoped net-edge task feeds the existing
quant/trading path through conservative typed advice. A composite decision
fingerprint invalidates stale paper qualification. ELM advice can demote an
edge to hold; it cannot generate orders, widen risk limits or change live/paper
authority. Live influence defaults off.

**Exit evidence:** ELM-T12–14 software and trading authority tests passed,
including the composite fingerprint and baseline fallback. A permitted market
and fill dataset is unavailable, so no market model is qualified.

## Phase 06 — Training operations, mirroring and AARNN readout

**State:** complete. ELM has a separate bounded worker role, job family and
evaluation/export command. Optional mirror scoring is bounded and parallel;
required ledger delivery and acknowledgement remain independent. A typed
timestamped activity-window readout reports unavailable because the current
runtime does not supply those windows. LLM adapter/trainer semantics remain
separate.

**Exit evidence:** ELM-T15–16 software tests and fixture export passed. No
permitted activity-window dataset or AARNN semantic runtime validation exists.

## Phase 07 — Operational verification and evidence

**State:** complete with limits recorded. Formatting, CPU-only baseline and ELM
tests, trading CI-safe tests, default-feature tests, the x86_64 fixture
benchmark and an aarch64 cross-compile passed. The required all-target Clippy
command ran but failed on repository-wide lint findings; no diagnostic points
into the new numerical or ELM configuration modules. The host cannot provide a
native model-inference parity run. The arm64 Gail and ELM worker services did
start Ready on qc01. Runbooks, configuration, rollback, CI coverage,
benchmarks and evaluation records are checked in.

**Exit evidence:** See `docs/verification/elm/verification-results.md` and the
acceptance map in `docs/verification/elm/requirement-traceability.md`. The
Clippy failure, missing production datasets and canary stream, and unmeasured
deployment-hardware model latency remain explicit evidence gaps.

## Phase 08 — Optional online learning (deferred)

This is a separate optional extension and is outside this delivery. No online
updates or in-place champion mutation are implemented. Configuration requires
`online_updates: false`.

## Cross-phase controls

- Preserve the provided specification and all pre-existing user changes.
- Use only explicit, permitted manifests with recorded provenance for training.
- Fixture, unit, synthetic and unexecuted hardware evidence cannot qualify a
  production model.
- Models cannot change governance, admission, mirror delivery acknowledgement,
  trainer registration semantics, trading risk authority, paper/live mode or
  order execution ownership.
- Keep `requirement-traceability.md` and `verification-results.md` aligned with
  executed evidence and remaining qualification work.
