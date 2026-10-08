# Gail model intake and evaluation skill

## Purpose and ownership

Use this procedure when Gail, Refiner or Conductor reviews a newly released
language model, a new quantisation, or a provider/runtime change. Gail owns
provider access, routing and model-performance evidence. NMC Continuum owns
infrastructure inventory, resource and dependency checks, downloads, staged
deployment and rollback. Refiner may research candidate metadata and run
approved evaluation work; it must not add an unreviewed model to production
configuration. Home Assistant, nmstt, Conductor, Refiner and OctoBot call Gail
through its authenticated service contract and do not maintain their own model
catalogues or routing rules.

This is an operator/agent procedure, not an automated discovery or approval
service. Gail currently lists models already installed on its Ollama endpoints
and records provider availability, latency and request outcomes. It does not
yet automatically discover upstream releases or qualify general LLM task
quality. Until those functions are implemented and verified, a candidate must
remain `discovered` or `unqualified` and must not be configured for production
selection.

## Skill instructions

When asked to review new models:

1. **Discover without downloading.** Check the official Ollama library,
   Hugging Face model API/model cards, NVIDIA NIM catalogue and relevant
   upstream GitHub releases. Use their official pages/APIs rather than search
   snippets. Follow pagination within provider rate limits and record the
   retrieval time. Treat GitHub as a source only for a release that actually
   contains or links to model artefacts; a repository release is not itself a
   model.
2. **Verify provenance.** Record the exact upstream owner, model ID, immutable
   revision/commit, artefact format, quantisation, architecture, tokenizer and
   chat template, declared context, size, checksum, licence and model card.
   Independently confirm factual claims with at least two reliable sources.
   An absent or unclear licence, mutable/unverifiable artefact, unexpected
   executable payload, unsafe deserialisation format, or disagreement between
   sources blocks the candidate pending review.
3. **Check fit with Continuum.** Ask NMC for current eligible hosts, available
   RAM/VRAM/disk, accelerator/runtime support, shared-host budgets, dependency
   impact and existing model workload. Use measured footprint or a conservative
   estimate; do not assume advertised parameter count maps directly to local
   resource use. A model that competes with critical workloads or lacks a
   bounded rollback target is not eligible for a canary.
4. **Evaluate against a controlled baseline.** Use the same versioned,
   permissioned, task-relevant dataset, runtime settings and output limits for
   the candidate and current production model. Include held-out tasks for
   assistant conversation, planning/structured output, code where applicable,
   research with citations, and authorised household/device questions. Measure
   task-specific correctness and completeness, citation/grounding quality,
   refusal and prompt-injection behaviour, parse/test pass rate, abstention,
   failure rate, latency percentiles, throughput and resource consumption.
   Preserve dataset provenance, splits, exact model/runtime revisions and
   scoring code. Do not send private prompts or customer data to a provider
   without its existing data authority.
5. **Make a comparative decision.** Gail records pass/fail per task and an
   aggregate with uncertainty and subgroup results. A single judge model,
   vendor benchmark, leaderboard position, model size or response speed cannot
   approve a candidate. Reject a candidate that fails any hard safety,
   licensing, privacy or minimum-quality gate, even if it is faster. Keep
   trading/live-execution authority independent from model qualification.
6. **Request a Continuum canary.** Only after review, submit the pinned model
   manifest below to NMC. Continuum validates the source and digest, available
   resources, the named host/namespace, affected dependencies, current health,
   storage headroom, deployment policy and rollback image/model. It downloads
   only the named revision to an isolated canary endpoint. A canary is not
   routable from normal consumers by default.
7. **Qualify, promote or roll back.** Re-run the evaluation against the live
   canary. Gail may add the explicit provider profile only after the canary
   passes every configured quality, safety and reliability threshold. NMC
   verifies deployment health and retains the prior model. Any failed or
   regressed gate leaves the candidate disabled and causes the canary to be
   removed or rolled back. Record a decision and evidence link in Gail's
   model/routing evidence view.

## Candidate manifest

Refiner/Conductor research should produce a review record in this shape before
asking Continuum to act. This describes a proposed contract; the current
Continuum deployment role does not yet consume a Gail qualification record.

```yaml
schema_version: 1
candidate_id: "publisher/model@<immutable-revision>"
source:
  provider: huggingface
  model_id: "publisher/model"
  revision: "<immutable-commit-or-release>"
  model_card_url: "<official model-card URL>"
  independent_sources:
    - "<primary catalogue or model-card URL>"
    - "<independent evaluation or second authoritative source URL>"
  licence: "<SPDX expression or UNKNOWN>"
  artifact_sha256: "<reviewed SHA-256>"
  format: "<safetensors, GGUF or other verified format>"
  quantisation: "<verified quantisation or none>"
  architecture: "<reviewed architecture>"
  context_tokens: null
evaluation:
  dataset_id: "<Gail-approved suite>"
  dataset_revision: "<immutable dataset revision>"
  scoring_revision: "<immutable scoring-code revision>"
  baseline_candidate_id: "<current-production-model@revision>"
  report_sha256: "<reviewed report SHA-256>"
  decision: unqualified
continuum:
  target: "<named canary host or pool>"
  namespace: "<isolated canary namespace>"
  max_disk_bytes: null
  max_ram_bytes: null
  max_vram_bytes: null
  route_from_production: false
  rollback_candidate_id: "<current-production-model@revision>"
```

Never put credentials, bearer tokens, customer prompts, raw benchmark answers
or unredacted private telemetry in this manifest. The candidate's `decision`
field is evidence for review, not an instruction that bypasses Continuum's
policy gates. A checksum proves byte identity, not licensing or model quality.

## Sources and implementation boundaries

- [Ollama model library](https://ollama.com/library)
- [Hugging Face Hub model API](https://huggingface.co/docs/hub/en/api)
- [NVIDIA NIM model catalogue](https://build.nvidia.com/models)
- [GitHub Releases API](https://docs.github.com/en/rest/releases/releases)
- [Gail AARNN mirror workflow](../LLM_SNN_MIRROR_WORKFLOW.md)
- [Gail ELM evaluation operations](./elm.md)

Ollama's `OLLAMA_ALLOW_AUTO_PULL` remains disabled in production. Gail's
Ollama inventory and its provider-performance telemetry remain useful for
runtime operations, but neither substitutes for upstream discovery or the
task-quality evaluation described here. The AARNN mirror path is separately
gated by Gail's scoped service identity, AARNN's local/central authentication,
the explicit peripheral-input grant and a network-scoped peripheral session.
SNN mirroring remains asynchronous and bounded; it must not delay or replace
the primary LLM response unless a separately qualified and authorised policy
allows it.
