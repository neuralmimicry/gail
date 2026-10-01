//! Strict configuration for the optional Rust-native ELM capability.
//!
//! This module is always compiled so configurations can be diagnosed clearly
//! when Gail was built without the `elm` Cargo feature.

use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::errors::{GailError, Result};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ElmMode {
    #[default]
    Off,
    Collect,
    Shadow,
    Auto,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ModelStage {
    #[default]
    Candidate,
    Trained,
    Evaluated,
    Qualified,
    Shadow,
    Canary,
    Active,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElmConfig {
    pub mode: ElmMode,
    pub registry_path: PathBuf,
    pub dataset_manifest: Option<PathBuf>,
    pub tasks: BTreeMap<String, ElmTaskConfig>,
    pub promotion_policies: BTreeMap<String, PromotionPolicy>,
    pub budgets: ElmBudgets,
    pub lifecycle: ElmLifecycleConfig,
    pub trading: ElmTradingConfig,
    pub data: ElmDataConfig,
}

impl Default for ElmConfig {
    fn default() -> Self {
        Self {
            mode: ElmMode::Off,
            registry_path: PathBuf::from("data/elm"),
            dataset_manifest: None,
            tasks: BTreeMap::new(),
            promotion_policies: BTreeMap::new(),
            budgets: ElmBudgets::default(),
            lifecycle: ElmLifecycleConfig::default(),
            trading: ElmTradingConfig::default(),
            data: ElmDataConfig::default(),
        }
    }
}

impl ElmConfig {
    pub fn normalize(&mut self) -> Result<()> {
        apply_env_override("GAIL_ELM_REGISTRY_PATH", &mut self.registry_path)?;
        if let Ok(raw) = std::env::var("GAIL_ELM_DATASET_MANIFEST") {
            self.dataset_manifest = Some(PathBuf::from(raw));
        }
        apply_env_override(
            "GAIL_ELM_MAX_TRAINING_JOBS",
            &mut self.budgets.max_training_jobs,
        )?;
        apply_env_override(
            "GAIL_ELM_MAX_TRAINING_THREADS",
            &mut self.budgets.max_training_threads,
        )?;
        apply_env_override(
            "GAIL_ELM_MAX_TRAINING_MEMORY_MB",
            &mut self.budgets.max_training_memory_mb,
        )?;
        apply_env_override(
            "GAIL_ELM_MAX_TRAINING_SECONDS",
            &mut self.budgets.max_training_seconds,
        )?;
        apply_env_override(
            "GAIL_ELM_MAX_INFERENCE_QUEUE",
            &mut self.budgets.max_inference_queue,
        )?;
        apply_env_override(
            "GAIL_ELM_MAX_HIDDEN_UNITS",
            &mut self.budgets.max_hidden_units,
        )?;
        apply_env_override(
            "GAIL_ELM_INFERENCE_DEADLINE_MS",
            &mut self.budgets.inference_deadline_ms,
        )?;
        apply_env_override(
            "GAIL_ELM_ALLOW_LIVE_INFLUENCE",
            &mut self.trading.allow_live_influence,
        )?;
        apply_env_override("GAIL_ELM_ALLOW_RAW_TEXT", &mut self.data.allow_raw_text)?;
        apply_env_override(
            "GAIL_ELM_ALLOW_CROSS_TENANT_TRAINING",
            &mut self.data.allow_cross_tenant_training,
        )?;
        if self.registry_path.as_os_str().is_empty() {
            return Err(GailError::invalid_config(
                "elm.registry_path must not be empty",
            ));
        }
        if let Ok(mode) = std::env::var("GAIL_ELM_MODE") {
            self.mode = serde_yaml::from_str::<ElmMode>(&mode).map_err(|_| {
                GailError::invalid_config("GAIL_ELM_MODE must be off, collect, shadow, or auto")
            })?;
        }
        for (name, task) in &self.tasks {
            if !KNOWN_TASKS.contains(&name.as_str()) {
                return Err(GailError::invalid_config(format!(
                    "unknown ELM task `{name}`"
                )));
            }
            if task.promotion_policy.trim().is_empty() {
                return Err(GailError::invalid_config(format!(
                    "elm.tasks.{name}.promotion_policy must not be empty"
                )));
            }
            if !task.rollout_fraction.is_finite()
                || !(0.0..=1.0).contains(&task.rollout_fraction)
                || !task.influence_cap.is_finite()
                || !(0.0..=1.0).contains(&task.influence_cap)
            {
                return Err(GailError::invalid_config(format!(
                    "elm.tasks.{name} rollout_fraction and influence_cap must be finite values in [0, 1]"
                )));
            }
        }
        self.budgets.validate()?;
        if !(1..=86_400).contains(&self.lifecycle.lease_seconds)
            || !(1..=31_536_000).contains(&self.lifecycle.cooldown_seconds)
            || !(0..=31_536_000).contains(&self.lifecycle.minimum_dwell_seconds)
            || !(1_048_576..=1_073_741_824).contains(&self.lifecycle.artifact_max_bytes)
        {
            return Err(GailError::invalid_config(
                "elm.lifecycle values fall outside supported bounds",
            ));
        }
        if !self.trading.maximum_influence.is_finite()
            || !(0.0..=1.0).contains(&self.trading.maximum_influence)
            || self.data.maximum_feature_age_seconds > 604_800
            || self.data.minimum_label_age_seconds > 31_536_000
        {
            return Err(GailError::invalid_config(
                "elm.trading or elm.data values fall outside supported bounds",
            ));
        }
        for (name, policy) in &self.promotion_policies {
            if name.trim().is_empty() {
                return Err(GailError::invalid_config(
                    "ELM promotion policy names must not be empty",
                ));
            }
            policy.validate(name)?;
        }
        if self.lifecycle.online_updates {
            return Err(GailError::invalid_config(
                "elm.lifecycle.online_updates is a deferred extension and must remain false",
            ));
        }
        if self.enabled() && !cfg!(feature = "elm") {
            return Err(GailError::invalid_config(
                "ELM is enabled in configuration but this Gail binary was built without the `elm` Cargo feature",
            ));
        }
        Ok(())
    }

    pub fn enabled(&self) -> bool {
        self.mode != ElmMode::Off
    }

    /// Apply restrictive precedence. `off` always disables a task; otherwise
    /// the lower mode in the capability order wins.
    pub fn effective_mode(&self, task_name: &str) -> ElmMode {
        let task_mode = self
            .tasks
            .get(task_name)
            .and_then(|task| task.mode)
            .unwrap_or(self.mode);
        self.mode.min(task_mode)
    }

    pub fn max_stage(&self, task_name: &str) -> ModelStage {
        self.tasks
            .get(task_name)
            .map(|task| task.max_stage)
            .unwrap_or(ModelStage::Shadow)
    }
}

fn apply_env_override<T>(name: &str, field: &mut T) -> Result<()>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    if let Ok(raw) = std::env::var(name) {
        *field = raw.parse().map_err(|error| {
            GailError::invalid_config(format!("{name} has an invalid value: {error}"))
        })?;
    }
    Ok(())
}

const KNOWN_TASKS: &[&str] = &[
    "routing_candidate_utility",
    "quant_net_edge",
    "trading_advisory",
    "training_resource_estimate",
    "mirror_priority",
    "aarnn_activity_readout",
];

/// Versioned numeric feature order, also available to feature-disabled builds
/// so their configuration and request contracts remain type-checkable.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct FeatureSchema {
    pub id: String,
    pub version: u32,
    pub names: Vec<String>,
    pub units: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElmTaskConfig {
    pub mode: Option<ElmMode>,
    pub max_stage: ModelStage,
    pub promotion_policy: String,
    pub allowed: bool,
    pub tenant_scope: String,
    pub rollout_fraction: f64,
    pub influence_cap: f64,
}

impl Default for ElmTaskConfig {
    fn default() -> Self {
        Self {
            mode: None,
            max_stage: ModelStage::Shadow,
            promotion_policy: "unconfigured".to_string(),
            allowed: false,
            tenant_scope: "local".to_string(),
            rollout_fraction: 0.0,
            influence_cap: 0.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElmBudgets {
    pub max_training_jobs: usize,
    pub max_training_threads: usize,
    pub max_training_memory_mb: usize,
    pub max_training_seconds: u64,
    pub max_training_retries: usize,
    pub max_inference_queue: usize,
    pub max_hidden_units: usize,
    pub max_samples: usize,
    pub max_features: usize,
    pub inference_deadline_ms: u64,
}

impl Default for ElmBudgets {
    fn default() -> Self {
        Self {
            max_training_jobs: 1,
            max_training_threads: 2,
            max_training_memory_mb: 512,
            max_training_seconds: 900,
            max_training_retries: 3,
            max_inference_queue: 128,
            max_hidden_units: 512,
            max_samples: 100_000,
            max_features: 2_048,
            inference_deadline_ms: 5,
        }
    }
}

impl ElmBudgets {
    fn validate(&self) -> Result<()> {
        if !(1..=32).contains(&self.max_training_jobs)
            || !(1..=64).contains(&self.max_training_threads)
            || !(64..=65_536).contains(&self.max_training_memory_mb)
            || !(1..=86_400).contains(&self.max_training_seconds)
            || !(0..=10).contains(&self.max_training_retries)
            || !(1..=65_536).contains(&self.max_inference_queue)
            || !(1..=512).contains(&self.max_hidden_units)
            || !(1..=1_000_000).contains(&self.max_samples)
            || !(1..=16_384).contains(&self.max_features)
            || !(1..=60_000).contains(&self.inference_deadline_ms)
        {
            return Err(GailError::invalid_config(
                "one or more elm.budgets values fall outside supported bounds",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElmLifecycleConfig {
    pub automatic_training: bool,
    pub automatic_promotion: bool,
    pub automatic_rollback: bool,
    pub online_updates: bool,
    pub minimum_dwell_seconds: u64,
    pub cooldown_seconds: u64,
    pub artifact_max_bytes: usize,
    pub lease_seconds: u64,
}

impl Default for ElmLifecycleConfig {
    fn default() -> Self {
        Self {
            automatic_training: false,
            automatic_promotion: false,
            automatic_rollback: true,
            online_updates: false,
            minimum_dwell_seconds: 3_600,
            cooldown_seconds: 3_600,
            artifact_max_bytes: 16 * 1024 * 1024,
            lease_seconds: 60,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElmTradingConfig {
    pub allow_live_influence: bool,
    pub maximum_influence: f64,
}

impl Default for ElmTradingConfig {
    fn default() -> Self {
        Self {
            allow_live_influence: false,
            maximum_influence: 0.0,
        }
    }
}

impl ElmTradingConfig {
    /// Bound trading influence by both the task evidence policy and the
    /// operator's trading envelope. Live decisions remain unchanged unless
    /// the operator explicitly permits live ELM influence.
    pub fn effective_influence_cap(&self, requested: f64, live_execution: bool) -> f64 {
        if !requested.is_finite()
            || requested <= 0.0
            || !self.maximum_influence.is_finite()
            || self.maximum_influence <= 0.0
            || (live_execution && !self.allow_live_influence)
        {
            0.0
        } else {
            requested.min(self.maximum_influence).clamp(0.0, 1.0)
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElmDataConfig {
    pub allow_raw_text: bool,
    pub allow_cross_tenant_training: bool,
    pub minimum_label_age_seconds: u64,
    pub maximum_feature_age_seconds: u64,
}

impl Default for ElmDataConfig {
    fn default() -> Self {
        Self {
            allow_raw_text: false,
            allow_cross_tenant_training: false,
            minimum_label_age_seconds: 0,
            maximum_feature_age_seconds: 300,
        }
    }
}

/// Every item is optional in YAML so an incomplete policy can be rejected as
/// evidence rather than accidentally receiving permissive defaults.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionPolicy {
    pub task_objective: Option<String>,
    pub direction: Option<MetricDirection>,
    pub baseline: Option<String>,
    pub minimum_effective_samples: Option<usize>,
    pub minimum_subgroup_samples: Option<usize>,
    pub minimum_observation_seconds: Option<u64>,
    pub confidence_level: Option<f64>,
    pub minimum_practical_improvement: Option<f64>,
    pub non_inferiority_tolerance: Option<f64>,
    pub maximum_calibration_error: Option<f64>,
    pub minimum_interval_coverage: Option<f64>,
    pub maximum_p95_overhead_ms: Option<f64>,
    pub maximum_p99_overhead_ms: Option<f64>,
    pub maximum_input_age_seconds: Option<u64>,
    pub maximum_label_age_seconds: Option<u64>,
    pub maximum_drift_score: Option<f64>,
    pub maximum_ood_score: Option<f64>,
    pub canary_fraction: Option<f64>,
    pub canary_max_decisions: Option<usize>,
    pub expiry_seconds: Option<u64>,
    pub rollback_regression: Option<f64>,
    pub cooldown_seconds: Option<u64>,
    pub retrain_budget_per_day: Option<usize>,
    pub authorised_influence: Option<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    Higher,
    Lower,
}

impl PromotionPolicy {
    pub fn is_complete(&self) -> bool {
        self.task_objective
            .as_ref()
            .is_some_and(|x| !x.trim().is_empty())
            && self.direction.is_some()
            && self.baseline.as_ref().is_some_and(|x| !x.trim().is_empty())
            && self.minimum_effective_samples.is_some_and(|x| x > 0)
            && self.minimum_subgroup_samples.is_some_and(|x| x > 0)
            && self.minimum_observation_seconds.is_some_and(|x| x > 0)
            && self.confidence_level.is_some()
            && self.minimum_practical_improvement.is_some()
            && self.non_inferiority_tolerance.is_some()
            && self.maximum_calibration_error.is_some()
            && self.minimum_interval_coverage.is_some()
            && self.maximum_p95_overhead_ms.is_some()
            && self.maximum_p99_overhead_ms.is_some()
            && self.maximum_input_age_seconds.is_some()
            && self.maximum_label_age_seconds.is_some()
            && self.maximum_drift_score.is_some()
            && self.maximum_ood_score.is_some()
            && self.canary_fraction.is_some()
            && self.canary_max_decisions.is_some()
            && self.expiry_seconds.is_some()
            && self.rollback_regression.is_some()
            && self.cooldown_seconds.is_some()
            && self.retrain_budget_per_day.is_some()
            && self.authorised_influence.is_some()
    }

    fn validate(&self, name: &str) -> Result<()> {
        for value in [
            self.confidence_level,
            self.canary_fraction,
            self.minimum_interval_coverage,
            self.authorised_influence,
        ]
        .into_iter()
        .flatten()
        {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(GailError::invalid_config(format!(
                    "elm.promotion_policies.{name} contains a value outside [0, 1]"
                )));
            }
        }
        if self
            .confidence_level
            .is_some_and(|value| value <= 0.5 || value >= 1.0)
            || self
                .maximum_calibration_error
                .is_some_and(|value| value > 1.0)
            || self
                .minimum_effective_samples
                .is_some_and(|value| value == 0)
            || self
                .minimum_subgroup_samples
                .is_some_and(|value| value == 0)
            || self
                .minimum_observation_seconds
                .is_some_and(|value| value == 0)
            || self.canary_max_decisions.is_some_and(|value| value == 0)
            || self.expiry_seconds.is_some_and(|value| value == 0)
            || self.retrain_budget_per_day.is_some_and(|value| value == 0)
        {
            return Err(GailError::invalid_config(format!(
                "elm.promotion_policies.{name} has invalid confidence or a zero support/budget value"
            )));
        }
        for value in [
            self.minimum_practical_improvement,
            self.non_inferiority_tolerance,
            self.maximum_calibration_error,
            self.maximum_p95_overhead_ms,
            self.maximum_p99_overhead_ms,
            self.maximum_drift_score,
            self.maximum_ood_score,
            self.rollback_regression,
        ]
        .into_iter()
        .flatten()
        {
            if !value.is_finite() || value < 0.0 {
                return Err(GailError::invalid_config(format!(
                    "elm.promotion_policies.{name} contains a negative or non-finite threshold"
                )));
            }
        }
        if self.is_complete() && self.maximum_p95_overhead_ms > self.maximum_p99_overhead_ms {
            return Err(GailError::invalid_config(format!(
                "elm.promotion_policies.{name} p95 overhead budget must not exceed p99"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_off_and_deny_unknown_elm_keys() {
        let cfg = ElmConfig::default();
        assert_eq!(cfg.mode, ElmMode::Off);
        assert!(!cfg.enabled());
        assert!(serde_yaml::from_str::<ElmConfig>("mode: shadow\nmode_typo: auto").is_err());
    }

    #[test]
    fn most_restrictive_task_mode_wins_and_stage_defaults_to_shadow() {
        let mut cfg = ElmConfig {
            mode: ElmMode::Auto,
            ..ElmConfig::default()
        };
        cfg.tasks.insert(
            "routing_candidate_utility".into(),
            ElmTaskConfig {
                mode: Some(ElmMode::Shadow),
                ..ElmTaskConfig::default()
            },
        );
        assert_eq!(
            cfg.effective_mode("routing_candidate_utility"),
            ElmMode::Shadow
        );
        assert_eq!(
            cfg.max_stage("routing_candidate_utility"),
            ModelStage::Shadow
        );
        assert_eq!(cfg.effective_mode("quant_net_edge"), ElmMode::Auto);
    }

    #[test]
    fn trading_influence_requires_an_explicit_live_allowance_and_respects_cap() {
        let mut policy = ElmTradingConfig {
            allow_live_influence: false,
            maximum_influence: 0.1,
        };
        assert_eq!(policy.effective_influence_cap(0.5, false), 0.1);
        assert_eq!(policy.effective_influence_cap(0.5, true), 0.0);
        policy.allow_live_influence = true;
        assert_eq!(policy.effective_influence_cap(0.5, true), 0.1);
        assert_eq!(policy.effective_influence_cap(f64::NAN, false), 0.0);
    }

    #[test]
    fn online_updates_stay_disabled_and_live_authority_defaults_to_false() {
        let mut cfg = ElmConfig::default();
        cfg.lifecycle.online_updates = true;
        assert!(cfg.normalize().is_err());
        cfg.lifecycle.online_updates = false;
        assert!(!cfg.trading.allow_live_influence);
        assert!(cfg.normalize().is_ok());
    }

    #[test]
    fn enabled_configuration_reports_a_feature_disabled_build() {
        if cfg!(feature = "elm") {
            return;
        }
        let mut cfg = ElmConfig {
            mode: ElmMode::Shadow,
            ..ElmConfig::default()
        };
        assert!(
            cfg.normalize()
                .unwrap_err()
                .to_string()
                .contains("without the `elm` Cargo feature")
        );
    }
}
