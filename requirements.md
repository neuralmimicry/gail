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
- REQ-013: When runtime rollout or restart work is needed, use the available Ansible automation context: {"ansible_root":"/srv/swarmhpc/ansible","config_path":"/srv/swarmhpc/ansible/ansible.cfg","host_targets":["rk1"],"hosts":["spirit"],"inventory_path":"/srv/swarmhpc/ansible/inventory/hosts.ini","playbooks":["continuum_tenant_gail_site.yml","continuum_tenant_nmchain_site.yml","continuum_tenant_refiner_site.yml"],"repo_root":"/srv/swarmhpc","roles_path":"/srv/swarmhpc/ansible/roles","secrets_root":"/srv/swarmhpc/ansible/.secrets"}.

Work Item Summary:
Prometheus reports 1/1 scrape target(s) down for Gail. Restore exporter coverage or scrape reachability so Conductor can weigh live runtime evidence when prioritising improvement work.

Authoritative delivery constraints (mandatory; implement and verify these, do not merely describe them):
- No structured delivery constraints were supplied; follow the work-item summary exactly.

Plan JSON:
{"action":"restore_observability_coverage","down_targets":1,"finding_id":"088e4136-1985-41e9-8a4e-8b90a8ca77f2","finding_key":"prometheus_target_health:gail","service":"gail","total_targets":1}