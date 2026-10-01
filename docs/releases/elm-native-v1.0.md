# Release notes: native ELM foundation

This release adds an opt-in, CPU-native batch ELM capability that builds with
--no-default-features --features elm. Existing deployments remain in
elm.mode: off unless an operator explicitly configures a task and permitted
dataset. ELM uses bounded Rayon pools, bounded queues and immutable JSON model
snapshots; it adds no Python, Node.js, CUDA, libtorch or external numerical
runtime requirement.

Included are regression/classification ELMs, ridge/logistic comparators,
training-only normalisation, explicit numeric manifests, grouped chronological
evaluation, calibration, confidence-bound reporting, a single-host durable
worker, an evaluation/export CLI, status routes, routing and trading advice,
composite paper-qualification fingerprints, bounded mirror-priority advice,
and a timestamped activity-window readout contract.

The mirror worker still sends and acknowledges every required ledger item.
Readout capability is reported unavailable because the current AARNN endpoint
does not expose timestamped population windows. Trading remains under the
existing qualification, freshness, economics, risk, paper/live and execution
gates. The ELM trading hook may only demote a proposed trade to hold.

Fixture data generate fixture-only artefacts for repeatable tests. No trained
production model, market qualification, AARNN semantic validation, target
hardware parity result or production performance claim is included. Online
learning remains a separate, disabled extension.
