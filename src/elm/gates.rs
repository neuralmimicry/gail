//! Ordered per-decision gates with stable outcome and rejection codes.

use serde::{Deserialize, Serialize};

use super::{ElmArtifact, LifecycleStage, ModelRegistry};
use crate::elm_config::FeatureSchema;
use crate::elm_config::{ElmConfig, ElmMode};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionOutcome {
    Applied,
    Shadowed,
    Abstained,
    Skipped,
    Failed,
    BaselineSelected,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GateReason {
    Disabled,
    CollectOnly,
    TaskDisallowed,
    BuildUnsupported,
    ModelMissing,
    ModelUnqualified,
    ModelExpired,
    ArtifactInvalid,
    SchemaMismatch,
    TenantMismatch,
    StaleInput,
    MissingInput,
    LowSupport,
    OutOfDomain,
    Drift,
    Uncalibrated,
    InsufficientMargin,
    ResourcePressure,
    Deadline,
    InferenceError,
    RolloutExcluded,
    CanaryBudgetExhausted,
    HardPolicyRejection,
    FeatureOff,
    BaselineAvailable,
    QualifiedModelUnavailable,
    TradingInfluenceDisabled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionAudit {
    pub decision_id: String,
    pub task_id: String,
    pub outcome: DecisionOutcome,
    pub reason: Option<GateReason>,
    pub model_digest: Option<String>,
    pub model_stage: Option<LifecycleStage>,
    pub active_gate_processes: Vec<String>,
    pub policy_version: String,
    pub feature_schema: Option<String>,
    pub model_called: bool,
    pub baseline_decision: Option<String>,
    pub learned_proposal: Option<String>,
    pub actual_action: Option<String>,
    pub influence_cap: f64,
    pub elapsed_ms: f64,
}

/// Return the named decision controls evaluated for a task. These are emitted
/// with each ELM decision so operators can distinguish the ELM checks from
/// Gail's still-authoritative downstream controls.
pub fn active_gate_processes(task_id: &str) -> Vec<String> {
    let mut gates = vec![
        "build_feature_support",
        "global_and_task_mode",
        "task_permission_and_scope",
        "model_integrity_and_lifecycle_qualification",
        "feature_schema_calibration_freshness_and_ood",
        "rollout_and_influence_caps",
        "bounded_inference_queue_and_deadline",
        "baseline_fallback",
    ];
    match task_id {
        "routing_candidate_utility" => gates.push("provider_eligibility_and_admission_recheck"),
        "quant_net_edge" | "trading_advisory" => {
            gates.extend([
                "live_trading_influence_authority",
                "composite_paper_qualification",
                "economics_risk_and_execution_recheck",
            ]);
        }
        "mirror_priority" | "aarnn_activity_readout" => {
            gates.push("required_mirror_delivery_and_acknowledgement");
        }
        _ => gates.push("existing_domain_scheduler_and_authority"),
    }
    gates.into_iter().map(str::to_string).collect()
}

#[derive(Clone, Debug)]
pub struct DecisionContext<'a> {
    pub decision_id: &'a str,
    pub task_id: &'a str,
    pub tenant_scope: &'a str,
    pub feature_schema: &'a FeatureSchema,
    pub feature_values: &'a [f64],
    pub feature_timestamp: u64,
    pub now: u64,
    pub remaining_deadline_ms: u64,
    pub baseline_decision: Option<&'a str>,
    pub hard_policy_eligible: bool,
    pub resource_available: bool,
    pub rollout_bucket: u64,
}

#[derive(Clone, Debug)]
pub struct GatedModel {
    pub artifact: Option<std::sync::Arc<ElmArtifact>>,
    pub reason: Option<GateReason>,
    pub influence_cap: f64,
    pub mode: ElmMode,
    pub shadow: bool,
}

/// Run inexpensive applicability gates before the caller extracts additional
/// features or queues numerical work.
pub fn admit<'a>(
    config: &ElmConfig,
    registry: &ModelRegistry,
    context: &DecisionContext<'a>,
) -> GatedModel {
    let mode = config.effective_mode(context.task_id);
    let influence = config
        .tasks
        .get(context.task_id)
        .map_or(0.0, |task| task.influence_cap.clamp(0.0, 1.0));
    let fail = |reason| GatedModel {
        artifact: None,
        reason: Some(reason),
        influence_cap: 0.0,
        mode,
        shadow: mode == ElmMode::Shadow,
    };
    if !cfg!(feature = "elm") {
        return fail(GateReason::BuildUnsupported);
    }
    if mode == ElmMode::Off {
        return fail(GateReason::Disabled);
    }
    if mode == ElmMode::Collect {
        return fail(GateReason::CollectOnly);
    }
    let Some(task) = config.tasks.get(context.task_id) else {
        return fail(GateReason::TaskDisallowed);
    };
    if !task.allowed {
        return fail(GateReason::TaskDisallowed);
    }
    if !context.hard_policy_eligible {
        return fail(GateReason::HardPolicyRejection);
    }
    if !context.resource_available {
        return fail(GateReason::ResourcePressure);
    }
    if context.remaining_deadline_ms == 0 {
        return fail(GateReason::Deadline);
    }
    if context.feature_timestamp > context.now
        || context.now.saturating_sub(context.feature_timestamp)
            > config.data.maximum_feature_age_seconds
    {
        return fail(GateReason::StaleInput);
    }
    if context.feature_values.is_empty()
        || context
            .feature_values
            .iter()
            .any(|value| !value.is_finite())
    {
        return fail(GateReason::MissingInput);
    }
    let Some(artifact) = registry.champion(context.task_id) else {
        return fail(GateReason::QualifiedModelUnavailable);
    };
    let Some(record) = registry.record(&artifact.metadata.model_id) else {
        return fail(GateReason::ModelUnqualified);
    };
    if !matches!(
        record.stage,
        LifecycleStage::Qualified
            | LifecycleStage::Shadow
            | LifecycleStage::Canary
            | LifecycleStage::Active
    ) {
        return fail(GateReason::ModelUnqualified);
    }
    if artifact.metadata.fixture_only {
        return fail(GateReason::ModelUnqualified);
    }
    if artifact
        .metadata
        .expiry_unix_seconds
        .is_some_and(|expiry| context.now > expiry)
    {
        return fail(GateReason::ModelExpired);
    }
    if artifact.feature_schema != *context.feature_schema
        || artifact.feature_schema.names.len() != context.feature_values.len()
    {
        return fail(GateReason::SchemaMismatch);
    }
    if artifact.metadata.tenant_scope != context.tenant_scope {
        return fail(GateReason::TenantMismatch);
    }
    let Some(policy) = config
        .tasks
        .get(context.task_id)
        .and_then(|configured| config.promotion_policies.get(&configured.promotion_policy))
    else {
        return fail(GateReason::ModelUnqualified);
    };
    if !policy.is_complete() {
        return fail(GateReason::ModelUnqualified);
    }
    if record.stage == LifecycleStage::Canary
        && registry.canary_decisions(&artifact.metadata.model_id)
            >= policy.canary_max_decisions.unwrap_or(0) as u64
    {
        return fail(GateReason::CanaryBudgetExhausted);
    }
    let out_of_domain = match artifact
        .preprocessor
        .transform(&[context.feature_values.to_vec()])
    {
        Ok(rows) => rows[0].iter().map(|value| value.abs()).fold(0.0, f64::max),
        Err(_) => return fail(GateReason::SchemaMismatch),
    };
    if out_of_domain > policy.maximum_ood_score.unwrap_or(0.0) {
        return fail(GateReason::OutOfDomain);
    }
    if artifact.calibration.as_ref().is_none_or(|value| {
        if artifact.task_kind == super::TaskKind::Regression {
            value.residual_radius.is_none()
        } else {
            !value.temperature.is_finite() || value.temperature <= 0.0
        }
    }) {
        return fail(GateReason::Uncalibrated);
    }
    if context.now < artifact.metadata.training_cutoff as u64 {
        return fail(GateReason::ModelExpired);
    }
    if task.rollout_fraction < 1.0
        && context.rollout_bucket as f64 / u64::MAX as f64 >= task.rollout_fraction
    {
        return fail(GateReason::RolloutExcluded);
    }
    let stage_authorised = match record.stage {
        LifecycleStage::Canary => task.max_stage >= crate::elm_config::ModelStage::Canary,
        LifecycleStage::Active => task.max_stage >= crate::elm_config::ModelStage::Active,
        _ => false,
    };
    let auto_authorised = mode == ElmMode::Auto
        && stage_authorised
        && influence > 0.0
        && policy.authorised_influence.is_some_and(|cap| cap > 0.0);
    GatedModel {
        artifact: Some(artifact),
        reason: None,
        influence_cap: if auto_authorised {
            influence.min(policy.authorised_influence.unwrap_or(0.0))
        } else {
            0.0
        },
        mode,
        shadow: !auto_authorised,
    }
}

impl DecisionAudit {
    pub fn skipped(
        decision_id: &str,
        task_id: &str,
        reason: GateReason,
        baseline: Option<&str>,
    ) -> Self {
        Self {
            decision_id: decision_id.to_string(),
            task_id: task_id.to_string(),
            outcome: DecisionOutcome::BaselineSelected,
            reason: Some(reason),
            model_digest: None,
            model_stage: None,
            active_gate_processes: active_gate_processes(task_id),
            policy_version: "elm-policy-v1".into(),
            feature_schema: None,
            model_called: false,
            baseline_decision: baseline.map(str::to_string),
            learned_proposal: None,
            actual_action: baseline.map(str::to_string),
            influence_cap: 0.0,
            elapsed_ms: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elm::{
        LifecycleStage,
        model::{AlgorithmFamily, Calibration, FitOptions},
    };
    use crate::elm_config::{ElmTaskConfig, MetricDirection, PromotionPolicy};

    fn schema() -> FeatureSchema {
        FeatureSchema {
            id: "gate-test:v1".into(),
            version: 1,
            names: vec!["x".into()],
            units: vec!["unitless".into()],
        }
    }

    fn policy() -> PromotionPolicy {
        PromotionPolicy {
            task_objective: Some("mse".into()),
            direction: Some(MetricDirection::Lower),
            baseline: Some("ridge".into()),
            minimum_effective_samples: Some(10),
            minimum_subgroup_samples: Some(2),
            minimum_observation_seconds: Some(60),
            confidence_level: Some(0.95),
            minimum_practical_improvement: Some(0.01),
            non_inferiority_tolerance: Some(0.02),
            maximum_calibration_error: Some(0.1),
            minimum_interval_coverage: Some(0.8),
            maximum_p95_overhead_ms: Some(5.0),
            maximum_p99_overhead_ms: Some(10.0),
            maximum_input_age_seconds: Some(300),
            maximum_label_age_seconds: Some(60),
            maximum_drift_score: Some(1.0),
            maximum_ood_score: Some(4.0),
            canary_fraction: Some(0.01),
            canary_max_decisions: Some(100),
            expiry_seconds: Some(3_600),
            rollback_regression: Some(0.1),
            cooldown_seconds: Some(60),
            retrain_budget_per_day: Some(1),
            authorised_influence: Some(0.1),
        }
    }

    fn configured() -> ElmConfig {
        let mut config = ElmConfig {
            mode: ElmMode::Auto,
            ..ElmConfig::default()
        };
        config.tasks.insert(
            "routing_candidate_utility".into(),
            ElmTaskConfig {
                mode: Some(ElmMode::Auto),
                max_stage: crate::elm_config::ModelStage::Active,
                promotion_policy: "routing_v1".into(),
                allowed: true,
                tenant_scope: "local".into(),
                rollout_fraction: 0.01,
                influence_cap: 0.1,
            },
        );
        config
            .promotion_policies
            .insert("routing_v1".into(), policy());
        config
    }

    fn registry_with_stage(target_stage: LifecycleStage) -> (tempfile::TempDir, ModelRegistry) {
        let directory = tempfile::tempdir().unwrap();
        let registry = ModelRegistry::open(directory.path(), 4 * 1024 * 1024).unwrap();
        let features = (0..32)
            .map(|index| vec![index as f64 / 10.0])
            .collect::<Vec<_>>();
        let targets = features.iter().map(|row| vec![row[0]]).collect::<Vec<_>>();
        let mut artifact = crate::elm::ElmArtifact::fit_regression(
            schema(),
            &features,
            &targets,
            None,
            FitOptions {
                task_id: "routing_candidate_utility".into(),
                family: AlgorithmFamily::Ridge,
                hidden_units: 0,
                seed: 7,
                lambda: 1e-3,
                target_names: vec!["utility".into()],
                target_units: vec!["score".into()],
                class_mapping: Vec::new(),
                training_cutoff: 1.0,
                dataset_digest: "gate-test-data".into(),
                tenant_scope: "local".into(),
                domain_scope: "test".into(),
                horizon_seconds: None,
                fixture_only: false,
            },
        )
        .unwrap();
        artifact.calibration = Some(Calibration {
            identity: "test-calibration".into(),
            temperature: 1.0,
            residual_radius: Some(0.1),
            fitted_on: "calibration".into(),
        });
        artifact.refresh_digest().unwrap();
        let id = artifact.metadata.model_id.clone();
        registry
            .publish_candidate(artifact, registry.revision())
            .unwrap();
        registry
            .mark_evaluated(&id, registry.revision(), "test")
            .unwrap();
        registry
            .promote(
                &id,
                LifecycleStage::Qualified,
                registry.revision(),
                "test evidence",
            )
            .unwrap();
        registry
            .promote(
                &id,
                LifecycleStage::Shadow,
                registry.revision(),
                "test shadow",
            )
            .unwrap();
        if target_stage == LifecycleStage::Canary || target_stage == LifecycleStage::Active {
            registry
                .promote(
                    &id,
                    LifecycleStage::Canary,
                    registry.revision(),
                    "test authorised canary stage",
                )
                .unwrap();
        }
        if target_stage == LifecycleStage::Active {
            registry
                .promote(
                    &id,
                    LifecycleStage::Active,
                    registry.revision(),
                    "test authorised stage",
                )
                .unwrap();
        }
        (directory, registry)
    }

    fn registry_with_active_model() -> (tempfile::TempDir, ModelRegistry) {
        registry_with_stage(LifecycleStage::Active)
    }

    fn context<'a>(schema: &'a FeatureSchema, values: &'a [f64]) -> DecisionContext<'a> {
        DecisionContext {
            decision_id: "decision-1",
            task_id: "routing_candidate_utility",
            tenant_scope: "local",
            feature_schema: schema,
            feature_values: values,
            feature_timestamp: 100,
            now: 100,
            remaining_deadline_ms: 5,
            baseline_decision: Some("eligible-baseline"),
            hard_policy_eligible: true,
            resource_available: true,
            rollout_bucket: 0,
        }
    }

    #[test]
    fn earlier_hard_gates_prevent_model_influence() {
        let (_directory, registry) = registry_with_active_model();
        let config = configured();
        let schema = schema();
        let values = [0.2];
        let mut current = context(&schema, &values);
        current.hard_policy_eligible = false;
        assert_eq!(
            admit(&config, &registry, &current).reason,
            Some(GateReason::HardPolicyRejection)
        );
        current.hard_policy_eligible = true;
        current.resource_available = false;
        assert_eq!(
            admit(&config, &registry, &current).reason,
            Some(GateReason::ResourcePressure)
        );
        current.resource_available = true;
        current.remaining_deadline_ms = 0;
        assert_eq!(
            admit(&config, &registry, &current).reason,
            Some(GateReason::Deadline)
        );
        current.remaining_deadline_ms = 5;
        current.feature_timestamp = 0;
        current.now = 301;
        assert_eq!(
            admit(&config, &registry, &current).reason,
            Some(GateReason::StaleInput)
        );
    }

    #[test]
    fn qualified_canary_uses_only_its_authorised_bounded_influence() {
        let (_directory, registry) = registry_with_stage(LifecycleStage::Canary);
        let config = configured();
        let schema = schema();
        let values = [0.2];
        let gated = admit(&config, &registry, &context(&schema, &values));
        assert_eq!(gated.influence_cap, 0.1);
        assert!(!gated.shadow);
    }

    #[test]
    fn schema_and_tenant_mismatches_abstain_after_hard_gates() {
        let (_directory, registry) = registry_with_active_model();
        let config = configured();
        let mismatched_schema = FeatureSchema {
            id: "other-schema:v1".into(),
            ..schema()
        };
        let values = [0.2];
        assert_eq!(
            admit(&config, &registry, &context(&mismatched_schema, &values)).reason,
            Some(GateReason::SchemaMismatch)
        );
        let schema = schema();
        let mut current = context(&schema, &values);
        current.tenant_scope = "other-tenant";
        assert_eq!(
            admit(&config, &registry, &current).reason,
            Some(GateReason::TenantMismatch)
        );
    }
}
