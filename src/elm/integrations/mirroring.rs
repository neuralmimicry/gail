//! Optional mirror queue advice and a separately gated activity-readout contract.

use std::{sync::Arc, time::Duration};

use crate::{
    elm_config::{ElmConfig, ElmMode, FeatureSchema},
    llm_ledger::LedgerInteraction,
};
use futures::future::join_all;

use super::super::{
    ElmError, ModelRegistry,
    executor::InferenceExecutor,
    gates::{DecisionContext, DecisionOutcome, GateReason, admit},
};

const MIRROR_TASK: &str = "mirror_priority";
const READOUT_TASK: &str = "aarnn_activity_readout";

pub fn mirror_priority_schema() -> FeatureSchema {
    FeatureSchema {
        id: "mirror-priority-v1".into(),
        version: 1,
        names: vec![
            "age_seconds".into(),
            "retry_attempts".into(),
            "last_latency_ms".into(),
            "latency_missing".into(),
            "completion_failed".into(),
        ],
        units: vec![
            "seconds".into(),
            "count".into(),
            "milliseconds".into(),
            "boolean".into(),
            "boolean".into(),
        ],
    }
}

/// Owns a pinned registry snapshot source and a bounded inference executor.
pub struct MirrorPriorityAdvisor {
    config: ElmConfig,
    registry: Arc<ModelRegistry>,
    inference: InferenceExecutor,
}

impl MirrorPriorityAdvisor {
    pub async fn open(config: &ElmConfig) -> crate::elm::Result<Option<Self>> {
        if config.effective_mode(MIRROR_TASK) == ElmMode::Off
            || config
                .tasks
                .get(MIRROR_TASK)
                .is_none_or(|task| !task.allowed)
        {
            return Ok(None);
        }
        let root = config.registry_path.clone();
        let max_bytes = config.lifecycle.artifact_max_bytes;
        let registry = tokio::task::spawn_blocking(move || ModelRegistry::open(root, max_bytes))
            .await
            .map_err(|error| ElmError::Artifact(format!("open mirror ELM registry: {error}")))??;
        let inference = InferenceExecutor::new(
            config.budgets.max_training_threads.min(4),
            config.budgets.max_inference_queue,
        )?;
        Ok(Some(Self {
            config: config.clone(),
            registry: Arc::new(registry),
            inference,
        }))
    }

    /// Reorders only eligible records and always returns every required mirror.
    /// Shadow predictions are recorded but retain the ledger's original order.
    pub async fn order_batch(&self, entries: Vec<LedgerInteraction>) -> Vec<LedgerInteraction> {
        if self.config.effective_mode(MIRROR_TASK) == ElmMode::Off || entries.len() < 2 {
            return entries;
        }
        let now = unix_seconds();
        let schema = mirror_priority_schema();
        // Keep the optional scorer bounded even when the durable ledger returns
        // a large batch. Required mirror rows beyond this limit remain ordered
        // normally and are still delivered.
        let scored_count = entries
            .len()
            .min(self.config.budgets.max_inference_queue)
            .min(256);
        if scored_count < entries.len() {
            tracing::debug!(
                task = MIRROR_TASK,
                batch_size = entries.len(),
                scored_count,
                omitted_advisories = entries.len() - scored_count,
                "bounded optional mirror scoring; required delivery remains queued"
            );
        }
        let mut scores = join_all(
            entries
                .iter()
                .take(scored_count)
                .map(|entry| self.score_entry(entry, now, &schema)),
        )
        .await;
        scores.resize(entries.len(), None);
        let mut sortable = entries.into_iter().enumerate().collect::<Vec<_>>();

        let mut eligible_positions = scores
            .iter()
            .enumerate()
            .filter_map(|(index, score)| score.map(|_| index))
            .collect::<Vec<_>>();
        eligible_positions.sort_by(|left, right| {
            scores[*right]
                .unwrap_or_default()
                .total_cmp(&scores[*left].unwrap_or_default())
                .then_with(|| left.cmp(right))
        });
        let mut ranked = eligible_positions
            .into_iter()
            .map(|index| sortable[index].1.clone())
            .collect::<Vec<_>>()
            .into_iter();
        for (index, score) in scores.iter().enumerate() {
            if score.is_some() {
                sortable[index].1 = ranked.next().unwrap_or_else(|| sortable[index].1.clone());
            }
        }
        sortable.sort_by_key(|(original_position, _)| *original_position);
        sortable.into_iter().map(|(_, entry)| entry).collect()
    }

    async fn score_entry(
        &self,
        entry: &LedgerInteraction,
        now: u64,
        schema: &FeatureSchema,
    ) -> Option<f64> {
        let age = now.saturating_sub(entry.created_ts.max(0.0) as u64);
        let features = [
            age as f64,
            entry.mirror_attempts as f64,
            entry.latency_ms.unwrap_or(0) as f64,
            f64::from(entry.latency_ms.is_none()),
            f64::from(entry.status != "ok"),
        ];
        let context = DecisionContext {
            decision_id: &entry.request_id,
            task_id: MIRROR_TASK,
            tenant_scope: self
                .config
                .tasks
                .get(MIRROR_TASK)
                .map_or("local", |task| task.tenant_scope.as_str()),
            feature_schema: schema,
            feature_values: &features,
            feature_timestamp: entry.created_ts.max(0.0) as u64,
            now,
            remaining_deadline_ms: self.config.budgets.inference_deadline_ms,
            baseline_decision: Some("ledger_id_order"),
            hard_policy_eligible: true,
            resource_available: true,
            rollout_bucket: entry.id.max(0) as u64,
        };
        let gated = admit(&self.config, &self.registry, &context);
        let Some(artifact) = gated.artifact else {
            tracing::debug!(ledger_id = entry.id, reason = ?gated.reason, "mirror ELM retained ledger order");
            return None;
        };
        let semantic_match = artifact.task_kind == super::super::TaskKind::Regression
            && artifact.metadata.target_names.first().map(String::as_str) == Some("priority")
            && artifact.metadata.target_units.first().map(String::as_str) == Some("unit_interval");
        if !semantic_match {
            tracing::warn!(ledger_id = entry.id, model_digest = %artifact.content_sha256, "mirror ELM artefact has incompatible priority semantics");
            return None;
        }
        let prediction = self
            .inference
            .predict(
                artifact.clone(),
                features.to_vec(),
                Duration::from_millis(self.config.budgets.inference_deadline_ms),
            )
            .await;
        let score = match prediction {
            Ok(value) => value
                .values
                .first()
                .copied()
                .filter(|score| score.is_finite() && (0.0..=1.0).contains(score)),
            Err(error) => {
                tracing::warn!(ledger_id = entry.id, error = %error, "mirror ELM abstained; required delivery continues");
                None
            }
        };
        let applied = score.is_some() && !gated.shadow && gated.influence_cap > 0.0;
        tracing::debug!(
            ledger_id = entry.id,
            outcome = if applied { "applied" } else { "shadowed" },
            model_digest = %artifact.content_sha256,
            model_called = true,
            "mirror priority advisory"
        );
        if applied { score } else { None }
    }
}

#[derive(Clone, Debug)]
pub struct ActivityWindow {
    pub brain_id: String,
    pub topology_version: String,
    pub population_mapping_version: String,
    pub window_start_us: u64,
    pub window_end_us: u64,
    pub population_spike_counts: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActivityReadoutCapability {
    Available { schema: String },
    Unavailable { reason: String },
}

/// The current mirror endpoint exposes text exchange results, not timestamped
/// activity windows. Callers report that missing capability without changing
/// required mirroring or interpreting text as neural activity.
pub fn activity_readout_capability(
    runtime_has_timestamped_windows: bool,
) -> ActivityReadoutCapability {
    if runtime_has_timestamped_windows {
        ActivityReadoutCapability::Available {
            schema: format!("{READOUT_TASK}:activity-window-v1"),
        }
    } else {
        ActivityReadoutCapability::Unavailable {
            reason: "AARNN runtime does not expose timestamped population activity windows".into(),
        }
    }
}

pub fn activity_window_features(window: &ActivityWindow) -> crate::elm::Result<Vec<f64>> {
    if window.brain_id.trim().is_empty()
        || window.topology_version.trim().is_empty()
        || window.population_mapping_version.trim().is_empty()
        || window.window_start_us >= window.window_end_us
        || window.population_spike_counts.is_empty()
        || window.population_spike_counts.len() > 65_536
    {
        return Err(ElmError::InvalidInput(
            "AARNN activity window is incomplete or outside its bounded contract".into(),
        ));
    }
    Ok(window
        .population_spike_counts
        .iter()
        .map(|count| *count as f64)
        .collect())
}

pub fn activity_window_schema(window: &ActivityWindow) -> crate::elm::Result<FeatureSchema> {
    if window.brain_id.trim().is_empty()
        || window.topology_version.trim().is_empty()
        || window.population_mapping_version.trim().is_empty()
        || window.window_start_us >= window.window_end_us
        || window.population_spike_counts.is_empty()
        || window.population_spike_counts.len() > 65_536
    {
        return Err(ElmError::InvalidInput(
            "AARNN activity window is incomplete or outside its bounded contract".into(),
        ));
    }
    let duration_us = window.window_end_us - window.window_start_us;
    let schema = FeatureSchema {
        id: format!(
            "aarnn-activity:{}:{}:{}:{}us:v1",
            window.brain_id,
            window.topology_version,
            window.population_mapping_version,
            duration_us
        ),
        version: 1,
        names: (0..window.population_spike_counts.len())
            .map(|index| format!("population_{index}_spike_count"))
            .collect(),
        units: vec![format!("spikes_per_{duration_us}us"); window.population_spike_counts.len()],
    };
    schema.validate()?;
    Ok(schema)
}

pub fn activity_artifact_compatible(
    artifact: &super::super::ElmArtifact,
    window: &ActivityWindow,
) -> crate::elm::Result<()> {
    let schema = activity_window_schema(window)?;
    if artifact.feature_schema != schema
        || artifact.metadata.task_id != READOUT_TASK
        || artifact.metadata.domain_scope != window.brain_id
        || artifact.target_schema_id != "aarnn-activity-readout:v1"
        || artifact.task_kind == super::super::TaskKind::Classification
            && artifact.metadata.class_mapping.len() < 2
    {
        return Err(ElmError::SchemaMismatch);
    }
    Ok(())
}

pub fn mirror_failure_outcome() -> (DecisionOutcome, GateReason) {
    (
        DecisionOutcome::BaselineSelected,
        GateReason::InferenceError,
    )
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{elm::executor::InferenceExecutor, elm_config::ElmTaskConfig};

    #[test]
    fn activity_capability_is_explicit_when_runtime_has_no_window_contract() {
        assert!(matches!(
            activity_readout_capability(false),
            ActivityReadoutCapability::Unavailable { .. }
        ));
    }

    #[test]
    fn activity_window_rejects_missing_topology_and_invalid_time() {
        let window = ActivityWindow {
            brain_id: "brain-a".into(),
            topology_version: String::new(),
            population_mapping_version: "pop-v1".into(),
            window_start_us: 10,
            window_end_us: 20,
            population_spike_counts: vec![1, 0, 2],
        };
        assert!(activity_window_features(&window).is_err());
    }

    #[test]
    fn activity_artifact_schema_binds_topology_population_and_window() {
        let window = ActivityWindow {
            brain_id: "brain-a".into(),
            topology_version: "topology-3".into(),
            population_mapping_version: "mapping-2".into(),
            window_start_us: 10,
            window_end_us: 20,
            population_spike_counts: vec![1, 0, 2],
        };
        let schema = activity_window_schema(&window).unwrap();
        let changed = ActivityWindow {
            topology_version: "topology-4".into(),
            ..window.clone()
        };
        assert_ne!(schema, activity_window_schema(&changed).unwrap());
    }

    #[test]
    fn mirror_priority_features_have_a_fixed_numeric_contract() {
        let schema = mirror_priority_schema();
        schema.validate().unwrap();
        assert_eq!(schema.names.len(), schema.units.len());
    }

    #[tokio::test]
    async fn missing_model_keeps_all_required_mirror_rows_in_ledger_order() {
        let directory = tempfile::tempdir().unwrap();
        let registry = ModelRegistry::open(directory.path(), 1024 * 1024).unwrap();
        let mut config = ElmConfig {
            mode: ElmMode::Auto,
            ..ElmConfig::default()
        };
        config.tasks.insert(
            MIRROR_TASK.into(),
            crate::elm_config::ElmTaskConfig {
                mode: Some(ElmMode::Auto),
                allowed: true,
                ..ElmTaskConfig::default()
            },
        );
        let advisor = MirrorPriorityAdvisor {
            config,
            registry: Arc::new(registry),
            inference: InferenceExecutor::new(1, 1).unwrap(),
        };
        let now = unix_seconds() as f64;
        let make_entry = |id| LedgerInteraction {
            id,
            request_id: format!("request-{id}"),
            conversation_id: "conversation".into(),
            workflow: "general".into(),
            role: "user".into(),
            provider_requested: None,
            model_requested: None,
            provider_resolved: None,
            model_resolved: None,
            request_category: None,
            system_prompt: None,
            prompt_text: "fixture prompt".into(),
            response_text: Some("fixture response".into()),
            message_roles: vec!["user".into()],
            status: "ok".into(),
            error_text: None,
            latency_ms: Some(5),
            usage: None,
            raw: None,
            metadata: None,
            mirror_status: Some("retry".into()),
            mirror_attempts: 1,
            validation_status: None,
            created_ts: now,
        };
        let result = advisor
            .order_batch(vec![make_entry(1), make_entry(2), make_entry(3)])
            .await;
        assert_eq!(result.len(), 3);
        assert_eq!(
            result.iter().map(|row| row.id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }
}
