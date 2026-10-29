# Business numerical contract

POST /v1/numerical/predict adds bounded compute_demand (cpu_cores) and job_duration (seconds) ingress for Foresight. Authenticate with the existing neuromorphic scope. Requests bind tenant, ordered feature schema/units, feature timestamp, output horizon and exact expected model digest. GAIL_BUSINESS_TENANT defaults to neuralmimicry and must match the deployment realm.

The two new task names remain default-deny. Operators must configure a task permission, tenant scope, promotion policy and rollout ceiling and independently qualify its artefact. The endpoint uses Gail's existing champion registry, model verification, schema/age/OOD/calibration checks, bounded inference executor and per-decision gates. It checks capacity domain, horizon and model identity before calling the model and rejects changed model identities after inference. It does not register, train or promote a model, and has no infrastructure or trading authority. Off, unsupported or unqualified configurations return applied=false with no prediction.

Containerfile.business builds an ARM64 serving overlay on the exact deployed image. Existing trainer/converter/Python assets and deployment configuration remain in the base. The serving binary is built with --no-default-features --features elm; dedicated native trainer executables remain available from the base. An isolated Debian 13 loader/library directory supports the controller cross-compiler's GLIBC requirements without replacing system libraries. The original image entrypoint still performs its startup checks and invokes /usr/bin/gail.

Validate the library suite, build ARM64, publish a digest and use the SwarmHPC business suite's explicit Gail-image rollout option. Keep the previous image digest for rollback. This release does not enable ELM, fit a model, change Gail governance, relax admission or alter trading policy.

