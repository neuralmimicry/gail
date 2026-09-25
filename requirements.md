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
{"action":"establish_repository_test_baseline","finding_id":"86176ea9-3fb6-4894-8b50-0c82be0edd7a","finding_key":"repository_test_baseline:gail","linked_services":["gail"],"repository":"gail"}

Planner guidance (advisory; it must not weaken or contradict the authoritative work-item requirements):
Overview: This work item establishes a test baseline for the Gail service, which is linked to live services but currently lacks executable project-native validation commands. The objective is to create a minimal, resilient verification infrastructure that supports safe autonomous changes.

Requirements Register:
- REQ-001: Inspect the gail repository and runtime state to identify coverage gaps and uncertainty evidence.
- REQ-002: Validate Ansible syntax and inventory without applying changes using --check mode.
- REQ-003: Generate a minimal test script or CI job definition covering critical smoke-test paths.
- REQ-004: Integrate the new test baseline into the existing CI/CD pipeline for automated verification.
- REQ-005: Ensure code quality and runtime behaviour via cargo fmt, check, and test execution.
- REQ-006: Execute a canary staged rollout to verify live service impact and resilience.
- REQ-007: Implement automatic rollback on degradation and verify recovery procedures.
- REQ-008: Maintain secure, non-destructive changes that leave unrelated files untouched.

Notes:
- The plan prioritises evidence gathering and safe validation before implementation.
- Rollout strategy is canary, requiring careful monitoring of live service metrics.
- Verification steps include both automated tests and manual runtime checks.
- No destructive commands will be executed; all changes are scoped and reversible.


Protected rollout contract (mandatory): capture a fresh readiness baseline before any change; use the selected canary or red_green strategy; verify health throughout the post-rollout window; if health or verification degrades, automatically revert the exact produced commit without rewriting history, rerun tests and GitHub Actions, and verify recovery.