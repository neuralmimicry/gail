//! Deterministic lifecycle transitions and automatic evidence gates.

use serde::{Deserialize, Serialize};

use super::{
    ElmArtifact, LifecycleStage,
    data::TrainingEvaluationReport,
    model::{AlgorithmFamily, EvaluationSummary, TaskKind},
};
use crate::elm_config::{ElmConfig, ElmMode, ModelStage, PromotionPolicy};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualificationResult {
    pub qualified: bool,
    pub reason_codes: Vec<String>,
    pub measured_improvement: Option<f64>,
    pub lower_confidence_bound: Option<f64>,
}

pub fn allowed_transition(from: &LifecycleStage, to: &LifecycleStage) -> bool {
    use LifecycleStage::*;
    matches!(
        (from, to),
        (Candidate, Trained | Rejected | Quarantined)
            | (Trained, Evaluated | Rejected | Quarantined)
            | (Evaluated, Qualified | Rejected | Quarantined)
            | (Qualified, Shadow | Rejected | Quarantined | Retired)
            | (
                Shadow,
                Canary | Active | Rejected | Quarantined | Retired | RolledBack
            )
            | (
                Canary,
                Active | Shadow | Rejected | Quarantined | Retired | RolledBack
            )
            | (
                Active,
                Shadow | Rejected | Quarantined | Retired | RolledBack
            )
            | (RolledBack, Retired)
            | (Rejected | Quarantined, Retired)
    )
}

/// Evaluate final-partition evidence. Missing policy fields and missing data
/// measurements always withhold qualification.
pub fn evaluate_qualification(
    config: &ElmConfig,
    task_id: &str,
    artifact: &ElmArtifact,
    report: &TrainingEvaluationReport,
) -> QualificationResult {
    let mut reasons = Vec::new();
    let task = config.tasks.get(task_id);
    if config.effective_mode(task_id) != ElmMode::Auto {
        reasons.push("task_not_in_auto_mode".into());
    }
    if task.is_none_or(|value| !value.allowed) {
        reasons.push("task_not_permitted".into());
    }
    if task.is_none_or(|value| value.max_stage < ModelStage::Qualified) {
        reasons.push("stage_ceiling_blocks_qualification".into());
    }
    let policy = task.and_then(|value| config.promotion_policies.get(&value.promotion_policy));
    let Some(policy) = policy else {
        reasons.push("promotion_policy_missing".into());
        return result(reasons, report);
    };
    if !policy.is_complete() {
        reasons.push("promotion_policy_incomplete".into());
    }
    if task.is_some_and(|value| value.rollout_fraction > policy.canary_fraction.unwrap_or(0.0)) {
        reasons.push("rollout_exceeds_policy_canary_fraction".into());
    }
    if artifact.metadata.fixture_only || report.fixture_only {
        reasons.push("fixture_only_artifact".into());
    }
    if report.final_candidate.partition != "final" {
        reasons.push("independent_final_evaluation_missing".into());
    }
    if report.final_candidate.effective_sample_count
        < policy.minimum_effective_samples.unwrap_or(usize::MAX) as f64
    {
        reasons.push("insufficient_effective_samples".into());
    }
    if report.minimum_final_subgroup_group_support
        < policy.minimum_subgroup_samples.unwrap_or(usize::MAX)
    {
        reasons.push("insufficient_subgroup_support".into());
    }
    if report.observed_window_seconds < policy.minimum_observation_seconds.unwrap_or(u64::MAX) {
        reasons.push("observation_window_incomplete".into());
    }
    if !report.calibrated {
        reasons.push("calibration_missing".into());
    }

    let expected_baseline = match artifact.task_kind {
        TaskKind::Regression => "ridge",
        TaskKind::Classification => "logistic",
    };
    let expected_objective = match artifact.task_kind {
        TaskKind::Regression => "mse",
        TaskKind::Classification => "log_loss",
    };
    if policy.task_objective.as_deref() != Some(expected_objective)
        || policy.direction != Some(crate::elm_config::MetricDirection::Lower)
    {
        reasons.push("configured_objective_is_not_supported_by_the_report".into());
    }
    let actual_baseline = match report.baseline_algorithm {
        AlgorithmFamily::Ridge => "ridge",
        AlgorithmFamily::Logistic => "logistic",
        AlgorithmFamily::Elm => "elm",
    };
    if policy.baseline.as_deref() != Some(expected_baseline) || actual_baseline != expected_baseline
    {
        reasons.push("configured_simple_baseline_unavailable".into());
    }
    let improvement = report.paired_improvement;
    let bound = report.lower_confidence_bound;
    if report.confidence_level < policy.confidence_level.unwrap_or(1.0) {
        reasons.push("configured_confidence_exceeds_evaluated_bound".into());
    }
    if improvement.is_none_or(|value| {
        !value.is_finite()
            || value
                < policy
                    .minimum_practical_improvement
                    .unwrap_or(f64::INFINITY)
    }) {
        reasons.push("minimum_practical_improvement_not_met".into());
    }
    if bound.is_none_or(|value| {
        !value.is_finite()
            || value
                < policy
                    .minimum_practical_improvement
                    .unwrap_or(f64::INFINITY)
    }) {
        reasons.push("confidence_bound_not_met".into());
    }
    if bound.is_some_and(|value| value < -policy.non_inferiority_tolerance.unwrap_or(f64::INFINITY))
    {
        reasons.push("non_inferiority_gate_failed".into());
    }
    if report
        .maximum_input_age_seconds
        .is_none_or(|age| age > policy.maximum_input_age_seconds.unwrap_or(0))
    {
        reasons.push("input_age_gate_failed".into());
    }
    if report
        .maximum_label_delay_seconds
        .is_none_or(|age| age > policy.maximum_label_age_seconds.unwrap_or(0))
    {
        reasons.push("label_age_gate_failed".into());
    }
    if report
        .drift_score
        .is_none_or(|score| score > policy.maximum_drift_score.unwrap_or(-1.0))
    {
        reasons.push("drift_gate_failed".into());
    }
    if report
        .maximum_ood_score
        .is_none_or(|score| score > policy.maximum_ood_score.unwrap_or(-1.0))
    {
        reasons.push("out_of_domain_gate_failed".into());
    }
    if !calibration_under_limit(&report.final_candidate, policy, artifact.task_kind) {
        reasons.push("calibration_gate_failed".into());
    }
    if !overhead_under_limits(
        report.final_candidate.inference_p95_ms,
        report.final_candidate.inference_p99_ms,
        policy,
    ) {
        reasons.push("inference_overhead_gate_failed".into());
    }
    result(reasons, report)
}

fn result(reasons: Vec<String>, report: &TrainingEvaluationReport) -> QualificationResult {
    QualificationResult {
        qualified: reasons.is_empty(),
        reason_codes: reasons,
        measured_improvement: report.paired_improvement,
        lower_confidence_bound: report.lower_confidence_bound,
    }
}

fn calibration_under_limit(
    report: &EvaluationSummary,
    policy: &PromotionPolicy,
    task_kind: TaskKind,
) -> bool {
    let Some(max_error) = policy.maximum_calibration_error else {
        return false;
    };
    let Some(error) = report.calibration_error else {
        return false;
    };
    if !error.is_finite() || error > max_error {
        return false;
    }
    match task_kind {
        TaskKind::Classification => true,
        TaskKind::Regression => policy.minimum_interval_coverage.is_some_and(|minimum| {
            report
                .residual_coverage
                .is_some_and(|coverage| coverage.is_finite() && coverage >= minimum)
        }),
    }
}

fn overhead_under_limits(p95: Option<f64>, p99: Option<f64>, policy: &PromotionPolicy) -> bool {
    p95.is_some_and(|value| {
        value.is_finite() && value <= policy.maximum_p95_overhead_ms.unwrap_or(-1.0)
    }) && p99.is_some_and(|value| {
        value.is_finite() && value <= policy.maximum_p99_overhead_ms.unwrap_or(-1.0)
    })
}
