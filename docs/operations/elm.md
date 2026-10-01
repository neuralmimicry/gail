# Native ELM operations

ELM is off by default in generic Gail configuration. The production Ansible
policy sets global and selected task modes to `auto`, making ELM available to
evidence gates when a qualified output is superior or complementary. At
deployment verification no approved manifest or qualified model was present,
so Gail retained its existing decision functions. A build without the `elm`
Cargo feature remains independent of libtorch; an enabled configuration
reports that the binary lacks support. The model layer cannot alter
governance, provider admission, the LLM trainer, required mirror acknowledgement,
trading limits, live/paper mode or order submission.

## Production rollout

Run from `/home/pbisaacs/Developer/swarmhpc/swarmhpc/ansible` after reviewing
the inventory and site variables:

    ANSIBLE_CONFIG=./ansible.cfg ansible-playbook -i inventory/hosts.ini continuum_tenant_gail_site.yml --syntax-check
    ANSIBLE_CONFIG=./ansible.cfg ansible-playbook -i inventory/hosts.ini continuum_tenant_gail_site.yml
    ANSIBLE_CONFIG=./ansible.cfg ansible-playbook -i inventory/hosts.ini continuum_tenant_grafana_site.yml --syntax-check
    ANSIBLE_CONFIG=./ansible.cfg ansible-playbook -i inventory/hosts.ini continuum_tenant_grafana_site.yml

The Gail play refuses to deploy unless `/srv/gail-training` is mounted from
the configured persistent storage. Verify the Gail, ELM trainer, mirror and
existing trainer rollouts are Ready, then check `/metrics` for each task's
`mode`, `model_stage`, `stage_ceiling`, `qualified_model_available`,
`active_stage` and `live_trading_authority` gate series. Gail's structured
decision log includes `active_gate_processes`, and
`gail_elm_gate_process_active` exposes those bounded process names with task
mode and model-stage labels. The Grafana **Gail LLM Routing** dashboard reports
the task gate composition, provider-admission/trading/mirror process gates,
influence cap and remaining canary budget. A configured authority gate or
`auto` mode does not itself qualify a model.

## Configuration

Existing YAML files need no ELM section. An explicit, non-influential
collection setup can look like:

    elm:
      mode: collect
      registry_path: data/elm
      dataset_manifest: data/elm/permitted-manifest.json
      tasks:
        quant_net_edge:
          mode: collect
          max_stage: shadow
          promotion_policy: quant_v1
          allowed: true
          tenant_scope: local
          rollout_fraction: 0.0
          influence_cap: 0.0
      promotion_policies: {}
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

Unknown keys, unknown task IDs, invalid bounds and incomplete dimensions are
rejected. The global mode and each task mode are separate; the most restrictive
mode wins. A task must be explicitly allowed. The default task stage ceiling
is shadow with zero rollout and influence.

Supported environment overrides are GAIL_ELM_MODE,
GAIL_ELM_REGISTRY_PATH, GAIL_ELM_DATASET_MANIFEST,
GAIL_ELM_MAX_TRAINING_JOBS, GAIL_ELM_MAX_TRAINING_THREADS,
GAIL_ELM_MAX_TRAINING_MEMORY_MB, GAIL_ELM_MAX_TRAINING_SECONDS,
GAIL_ELM_MAX_INFERENCE_QUEUE, GAIL_ELM_MAX_HIDDEN_UNITS,
GAIL_ELM_INFERENCE_DEADLINE_MS, GAIL_ELM_ALLOW_LIVE_INFLUENCE,
GAIL_ELM_ALLOW_RAW_TEXT and GAIL_ELM_ALLOW_CROSS_TENANT_TRAINING. YAML
remains the source for task scopes and promotion policies. Do not set raw-text
or cross-tenant permissions unless the existing data authority separately
permits them.

Every promotion policy must define an objective and direction, simple baseline,
effective and subgroup support, observation window, confidence, practical and
non-inferiority thresholds, calibration/coverage limits, p95/p99 overhead,
feature and label age, drift/OOD bounds, canary limits, expiry, rollback,
cooldown, retraining budget and authorised influence. Current reports support
lower-is-better MSE (regression) and log loss (classification). Other objectives
remain blocked. A confidence setting above the reported 95% grouped bootstrap
bound remains blocked.

## Evaluation and worker

Build the evaluator without default features:

    cargo run --locked --no-default-features --features elm --bin gail-elm-evaluate -- \
      --manifest data/elm/permitted-manifest.json \
      --config gail.yaml \
      --output-dir data/elm/exports

Exit codes are 0 for exported and qualified, 2 for exported but unqualified,
and 1 for an evaluation/export failure. Without --config, the evaluator uses
the safe default policy (off, no tasks allowed), so it can produce an
unqualified artefact but cannot qualify it.

The durable worker polls only an explicitly configured manifest and an
explicitly allowed task:

    cargo run --locked --no-default-features --features elm --bin gail -- \
      --config gail.yaml --role elm-trainer-worker

Jobs and immutable model/evaluation files are stored below the configured
registry path. The worker serialises jobs within one process, bounds fitting
threads, memory estimates, duration and retries, and requeues an interrupted
running job. The registry uses an OS advisory writer lock plus revision checks
for the trainer and serving processes sharing one host-mounted directory. Gail
refreshes complete immutable snapshots in the background; requests already
using an old snapshot finish against that bundle. Multi-host registry
deployments still require a distributed fenced writer.

The worker qualifies only an independent final evaluation that passes every
configured evidence gate. It advances through qualified and shadow, then
enters canary automatically after the configured shadow dwell when the model,
task, policy, rollout and influence ceilings still agree. Canary predictions
are limited by the durable per-model decision budget; the final permitted
claim atomically returns the model to shadow. The worker does not promote
directly to active. A future active-stage promotion requires an independently
verified canary outcome stream and its own evidence gate.

The production Ansible policy uses global and per-task `auto`, allows live
trading influence explicitly, and caps quant/trading influence at 10%, rollout
at 10%, and canary exposure at 1,000 decisions per model. This is an authority
ceiling, not a qualification result. Live ELM use still requires a non-fixture
model with a valid report, shadow dwell, an authorised task and policy, current
feature/calibration checks, the ELM canary budget, and Gail's existing
composite paper qualification, economics, risk and execution checks. The ELM
net-edge integration can only withhold an otherwise permitted trade; it cannot
create an order or widen trading authority.

## Status, monitoring and rollback

Read-only endpoints use Gail's existing status authorisation:

- GET /v1/elm/status reports build support, effective redacted settings,
  readiness and registry states.
- GET /v1/elm/models lists model IDs, digests, stage and fixture status.
- GET /v1/elm/models/{model_id}/evaluations returns the recorded report.

Request IDs, tenant-specific data and artefact paths are not added to metrics.
Decision traces include the effective mode, model stage, gate outcome, reason,
influence cap and the named combination of active ELM, provider-admission,
trading-authority or mirror-delivery checks. Detailed evaluation reports stay
behind the status authorisation. `/metrics` exports bounded-cardinality
`gail_elm_task_gate_active`, `gail_elm_task_influence_cap`,
`gail_elm_canary_decisions_remaining` and `gail_elm_gate_process_active`
series. The last reports the named ELM, provider-admission, trading-authority
and mirror-delivery processes active for each task; structured decision logs
record the same combination in `active_gate_processes`. Grafana's **ELM active
decision gates by task**, **ELM effective influence cap**, **ELM canary
decisions remaining** and **Active decision gate processes by task** panels
show the current mode, model stage, stage ceiling and live gate state without
exposing model digests or tenant records.

If Grafana logs `failed to save dashboard` with `database is locked` after a
dashboard ConfigMap update, rerun the Grafana play through the established
Ansible checkout to force one deployment restart and retry file provisioning:

```bash
cd /home/pbisaacs/Developer/swarmhpc/swarmhpc/ansible
ANSIBLE_CONFIG=./ansible.cfg ansible-playbook -i inventory/hosts.ini \
  continuum_tenant_grafana_site.yml \
  --extra-vars continuum_tenant_k8s_app_force_rollout_restart=true
```

This briefly restarts the single Grafana pod. Confirm the pod is Ready, the
startup log finishes dashboard provisioning without a save error, and the
`grafana-dashboards` ConfigMap contains all four Gail ELM panels.

## Persistent state and data

The Gail deployment and its trainer, mirror and ELM worker pods use the same
`/app/data` mount. On qc01, the `gail-data` claim is Bound on the
`continuum-shared` storage class and backs `/srv/gail-training`; Ansible checks
that it is an active mount before applying workload manifests. A missing volume
therefore fails deployment instead of silently placing state on the node root
filesystem.

Keep transactional interaction-ledger and trainer/mirror coordination records
in Gail's configured PostgreSQL database. The PostgreSQL StatefulSet has its
own 20 GiB `continuum-shared` data claim and a separate 50 GiB backup claim.
Trading state, market data, trainer output, the ELM job queue, immutable model
artefacts, evaluation reports and the content-addressed ELM registry use the
persistent `/app/data` volume. This keeps large immutable bundles atomic on the
filesystem while database-backed transactional state uses PostgreSQL. Include
both PostgreSQL and Gail data-volume backups in recovery drills; a bound backup
claim is not a substitute for an off-node copy and a tested restore.

Keep `/srv/gail-training/elm` and its approved dataset manifest when replacing
images or recreating pods. The current production path is
`/app/data/elm/approved-manifest.json`; it was absent during this rollout, so
automatic training had no permitted input and qualification remained blocked.
Restore a matching registry backup with an older binary during a software
rollback. Never copy a fixture artefact into the production registry.

Serving snapshots are replaced only after a complete immutable artefact and a
revision-checked registry index are durable. A failed registry write leaves the
previous in-memory champion unchanged. Three consecutive numerical, integrity
or deadline failures from a canary/active model schedule an idempotent rollback
off the request path. Queue saturation is treated as resource pressure and
does not trigger model rollback. The rollback selects the newest compatible
prior snapshot; with no compatible predecessor, routing/trading use their
existing baseline.

The local gail-elm-admin binary is the rollback control surface. Filesystem
access to the registry is its authorisation boundary; restrict it to the
existing Gail operator identity. First inspect the revision:

    cargo run --locked --no-default-features --features elm --bin gail-elm-admin -- \
      --registry data/elm status

Then supply that exact revision, a unique stable key and an audit reason:

    cargo run --locked --no-default-features --features elm --bin gail-elm-admin -- \
      --registry data/elm rollback \
      --task routing_candidate_utility \
      --expected-revision 12 \
      --idempotency-key incident-2026-10-01-routing \
      --reason "repeated inference deadline failures"

Verify the returned event and the new champion through GET /v1/elm/status.
Repeating the same key returns the committed event without rolling back the
next model. Do not edit registry JSON by hand. This release exposes no HTTP
model mutation route. Disabling ELM through configuration returns decisions to
the existing baseline; it does not cancel mirrors, orders or positions.

Back up the complete registry directory as a unit. To roll back a failed
software release, redeploy the previous Gail image and its matching registry
backup, or set global and task ELM modes to `off` in the Ansible site policy and
rerun the Gail play while retaining both persistent claims. Never delete the
Gail or PostgreSQL claims as part of an application rollback. Never reuse a
qualification across an artefact, data-schema, policy or trading-decision
fingerprint change.

## Current evidence boundary

No approved manifest, permitted point-in-time routing labels, market evaluation
data or timestamped AARNN activity-window dataset was present on the deployment
volume. The interaction ledger can contain raw text and is not an approved
training source. Consequently this release has no production-qualified model
and no measured production latency claim. Ansible configures the ELM authority
envelope, but live metrics and decision logs show that the gates select Gail's
existing functions while the qualified model is unavailable. The existing
trading execution authority remains mandatory; this implementation did not
place orders.
