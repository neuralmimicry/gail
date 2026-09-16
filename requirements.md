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
Conductor should improve the live Gail gateway from source, but the local Gail repository path is missing or unreadable.

Authoritative delivery constraints (mandatory; implement and verify these, do not merely describe them):
- No structured delivery constraints were supplied; follow the work-item summary exactly.

Plan JSON:
{"action":"verify_repo_hint","finding_id":"39e658d5-1448-4e90-8451-6740b239f19c","finding_key":"repository_visibility:gail","service":"gail"}

Planner guidance (advisory; it must not weaken or contradict the authoritative work-item requirements):
Overview: This work item restores Gail repository visibility and confirms runtime health prior to further optimisation. The objective is to ensure Conductor can source changes from the repository and that the Gail service is stable.

Requirements Register:
- REQ-001: Confirm the Gail repository at /srv/neuralmimicry/gail is accessible and report the exact obstruction if unavailable.
- REQ-002: Validate K3s control-plane connectivity and worker node probes using kubectl get nodes.
- REQ-003: Inspect service health logs and metrics on host 'spirit' to identify the root cause of degradation.
- REQ-004: Execute ansible-playbook /srv/swarmhpc/ansible/playbooks/continuum_tenant_gail_site.yml --check to verify Ansible syntax and inventory.
- REQ-005: Implement only the smallest safe change supported by evidence, avoiding destructive commands and unrelated file modifications.
- REQ-006: Run cargo fmt --check and cargo check to ensure code quality and compilation integrity.
- REQ-007: Execute relevant self-tests and monitor for degradation during the canary rollout on host 'rk1'.
- REQ-008: Prepare for automatic rollback on degradation and verify recovery procedures.

Rollout notes: Use canary strategy on host 'rk1'. Monitor for service degradation during and after implementation. Ensure protected-target readiness before full rollout.


Protected rollout contract (mandatory): capture a fresh readiness baseline before any change; use the selected canary or red_green strategy; verify health throughout the post-rollout window; if health or verification degrades, automatically revert the exact produced commit without rewriting history, rerun tests and GitHub Actions, and verify recovery.