Overview: Improve Gail through the Conductor execution loop.

Delivery Context:
- Current stage: development
- Validated stages: none
- Rollout strategy: canary

Requirements Register:
- REQ-001: Inspect and record current repository, runtime, or job evidence before selecting an operation.
- REQ-002: Implement only the scoped change, job update, or progress-monitoring action supported by that evidence.
- REQ-003: Preserve secure, resilient behaviour and avoid destructive commands.
- REQ-004: Update or add tests covering the changed path, or provide the relevant live operational check.
- REQ-005: Run verification commands and report the outcome.
- REQ-006: Leave unrelated files untouched.
- REQ-007: Record rollback/recovery steps and the acceptance signal proving the gap is closed.
- REQ-008: Preserve staged progression and rollout governance metadata.
- REQ-009: Capture a fresh protected-target readiness baseline before any change.
- REQ-010: Use the selected canary or red-green rollout strategy and verify the post-rollout health window.
- REQ-011: Automatically revert the exact produced commit without rewriting history if health or verification degrades.
- REQ-012: Verify rollback readiness and recovery before finalising the delivery.
- REQ-013: When runtime rollout or restart work is needed, use the available Ansible automation context: {"ansible_root":"/srv/swarmhpc/ansible","config_path":"/srv/swarmhpc/ansible/ansible.cfg","host_targets":["rk1"],"hosts":["spirit"],"inventory_path":"/srv/swarmhpc/ansible/inventory/hosts.ini","playbooks":["continuum_tenant_gail_site.yml","continuum_tenant_nmchain_site.yml","continuum_tenant_refiner_site.yml"],"repo_root":"/srv/swarmhpc","roles_path":"/srv/swarmhpc/ansible/roles","secrets_root":"/srv/swarmhpc/ansible/.secrets"}.

Work Item Summary:
gail is linked to live services but no obvious test capability was discovered in the repository inventory. Establish at least a minimal regression or smoke-test baseline before deeper autonomous changes.

Authoritative delivery constraints (mandatory; implement and verify these, do not merely describe them):
- No structured delivery constraints were supplied; follow the work-item summary exactly.

Plan JSON:
{"action":"establish_repository_test_baseline","finding_id":"d3fdf0cb-af6a-4578-a3b6-8e6d0c7dd833","finding_key":"repository_test_baseline:neuralmimicry/gail","linked_services":["gail"],"repository":"neuralmimicry/gail"}

Planner guidance (advisory; it must not weaken or contradict the authoritative work-item requirements):
This work item establishes a test baseline for the gail repository to enable safe autonomous changes. The baseline ensures that live services linked to the repository can be validated without destructive modifications. A focused smoke test will be introduced to validate core service health.

Requirements Register:
- REQ-001: The baseline must be minimal, focusing only on core service health validation.
- REQ-002: No existing production logic or files may be modified during the baseline establishment.
- REQ-003: The solution must pass cargo fmt --check and cargo check before execution.
- REQ-004: A canary rollout strategy must be used, targeting a single host initially.
- REQ-005: Automatic rollback must be enforced if health metrics indicate degradation.
- REQ-006: The team must document evidence of test readiness and uncertainty levels.
- REQ-007: The smoke test must be executable via standard cargo test commands.
- REQ-008: The implementation must be resilient and secure, avoiding destructive commands.


Protected rollout contract (mandatory): capture a fresh readiness baseline before any change; use the selected canary or red_green strategy; verify health throughout the post-rollout window; if health or verification degrades, automatically revert the exact produced commit without rewriting history, rerun tests and GitHub Actions, and verify recovery.