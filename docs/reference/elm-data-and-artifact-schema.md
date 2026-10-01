# ELM data and artefact schema

ELM consumes an explicit version-1 JSON manifest. A manifest is eligible only
when its data owner has permitted the named task and scope, its provenance is
recorded, and each row has stable record, group and subgroup identifiers.
Permission is a declaration to validate against the deployment's existing
policy; it does not grant new data access.

Each row records event, observation, decision and label-availability times.
The required ordering is:

    event_time <= observation_time <= decision_time <= label_available_time

Features and targets must be finite numeric arrays matching their declared
schemas. Ragged, duplicated, future-dated, immature or otherwise invalid rows
are excluded or reject the manifest. No text is loaded by this contract.
Missing values are rejected; where missingness is meaningful, represent it as
an explicit numeric feature and schema entry.

group_id identifies the unit that must stay in one time-ordered split
(conversation, strategy, instrument/horizon or shared outcome). subgroup_id
identifies a relevant coverage stratum, such as instrument family or market
regime. Every declared subgroup must have support in the final partition to
qualify. Features and labels from unresolved/censored outcomes must not be
written as successful or failed targets.

The pipeline uses chronological group partitions for training, tuning,
calibration and final evaluation. It fits normalisation on training rows only,
purges labels crossing partition boundaries, chooses among ELM and a simple
baseline on tuning rows, fits calibration on calibration rows, and reports
performance on the untouched final partition. Grouped paired bootstrap bounds
are descriptive evidence; they do not prove causal benefit for unchosen
providers or counterfactual trades.

Version-1 artefacts are immutable JSON bundles containing:

- task/model IDs, algorithm version, build/source revision and content digest;
- feature and target schemas, units, scope, horizon and training cutoff;
- preprocessor, actual projection/readout weights and calibration parameters;
- PRNG name and seed, solver diagnostics and final evaluation summary;
- dataset digest and fixture-only status.

The registry checks size and digest before activation and never deserialises
executable content. A digest detects accidental change, not authorship. Shared
deployments need a fenced single writer and shared durable storage; the current
filesystem registry is single-host.

Any artefact generated from synthetic fixtures carries fixture_only: true.
Registry gates refuse to make it a serving champion. A trained or evaluated
artefact is not by itself a qualified model.
