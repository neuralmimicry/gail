//! Explicit numeric datasets, point-in-time checks and time/group evaluation.

use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    ElmArtifact, ElmError, Result,
    model::{AlgorithmFamily, EvaluationSummary, FeatureSchema, FitOptions, TaskKind},
};

pub const DATASET_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetManifest {
    pub format_version: u32,
    pub dataset_id: String,
    pub permitted: bool,
    pub fixture_only: bool,
    pub tenant_scope: String,
    pub domain_scope: String,
    pub task_id: String,
    pub task_kind: TaskKind,
    pub feature_schema: FeatureSchema,
    pub target_schema_id: String,
    pub target_names: Vec<String>,
    pub target_units: Vec<String>,
    pub class_mapping: Vec<String>,
    pub horizon_seconds: Option<u64>,
    pub provenance: String,
    pub records: Vec<DatasetRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetRecord {
    pub record_id: String,
    pub group_id: String,
    pub subgroup_id: String,
    pub event_time: f64,
    pub observation_time: f64,
    pub decision_time: f64,
    pub label_available_time: f64,
    pub features: Vec<f64>,
    pub target: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataQualityReport {
    pub dataset_id: String,
    pub dataset_digest: String,
    pub eligible_samples: usize,
    pub excluded_samples: usize,
    pub group_count: usize,
    pub subgroup_group_counts: BTreeMap<String, usize>,
    pub class_balance: BTreeMap<String, usize>,
    pub training_cutoff: f64,
    pub tuning_cutoff: f64,
    pub calibration_cutoff: f64,
    pub final_cutoff: f64,
    pub fixture_only: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingEvaluationReport {
    pub report_version: u32,
    pub task_id: String,
    pub dataset_id: String,
    pub dataset_digest: String,
    pub fixture_only: bool,
    pub selected_algorithm: AlgorithmFamily,
    pub baseline_algorithm: AlgorithmFamily,
    pub attempted_configurations: Vec<String>,
    pub tuning_metrics: BTreeMap<String, f64>,
    pub final_candidate: EvaluationSummary,
    pub final_baseline: EvaluationSummary,
    pub quality: DataQualityReport,
    pub final_independent_groups: usize,
    pub minimum_final_subgroup_group_support: usize,
    pub observed_window_seconds: u64,
    pub maximum_input_age_seconds: Option<u64>,
    pub maximum_label_delay_seconds: Option<u64>,
    pub drift_score: Option<f64>,
    pub maximum_ood_score: Option<f64>,
    pub paired_improvement: Option<f64>,
    pub lower_confidence_bound: Option<f64>,
    pub confidence_level: f64,
    pub calibrated: bool,
    pub qualified: bool,
    pub qualification_reasons: Vec<String>,
    pub report_sha256: String,
}

pub struct TrainedEvaluation {
    pub artifact: ElmArtifact,
    pub report: TrainingEvaluationReport,
}

/// Hard limits for one batch fit; the values are checked before fitting starts.
#[derive(Clone, Copy, Debug)]
pub struct TrainingLimits {
    pub max_samples: usize,
    pub max_features: usize,
    pub max_hidden_units: usize,
    pub max_memory_mb: usize,
    pub max_training_threads: usize,
    pub max_training_seconds: u64,
    pub minimum_label_age_seconds: u64,
}

impl DatasetManifest {
    pub fn load(path: &Path, max_bytes: usize) -> Result<Self> {
        let metadata =
            std::fs::metadata(path).map_err(|error| ElmError::Artifact(error.to_string()))?;
        if metadata.len() as usize > max_bytes {
            return Err(ElmError::InvalidInput(
                "dataset manifest exceeds configured size limit".into(),
            ));
        }
        let bytes = std::fs::read(path).map_err(|error| ElmError::Artifact(error.to_string()))?;
        let manifest = serde_json::from_slice(&bytes).map_err(|error| {
            ElmError::InvalidInput(format!("invalid dataset manifest: {error}"))
        })?;
        Ok(manifest)
    }

    pub fn digest(&self) -> Result<String> {
        let bytes =
            serde_json::to_vec(self).map_err(|error| ElmError::Artifact(error.to_string()))?;
        Ok(hex::encode(Sha256::digest(bytes)))
    }

    pub fn validate(
        &self,
        now: f64,
        max_samples: usize,
        max_features: usize,
        minimum_label_age: u64,
    ) -> Result<(Vec<usize>, DataQualityReport)> {
        if self.format_version != DATASET_FORMAT_VERSION || !self.permitted {
            return Err(ElmError::InvalidInput(
                "dataset format unsupported or data-use permission is absent".into(),
            ));
        }
        if self.dataset_id.trim().is_empty()
            || self.task_id.trim().is_empty()
            || self.provenance.trim().is_empty()
            || self.tenant_scope.trim().is_empty()
            || self.domain_scope.trim().is_empty()
        {
            return Err(ElmError::InvalidInput(
                "dataset identity, scope and provenance are required".into(),
            ));
        }
        self.feature_schema.validate()?;
        if self.records.len() > max_samples || self.feature_schema.names.len() > max_features {
            return Err(ElmError::InvalidInput(
                "dataset exceeds configured sample or feature bounds".into(),
            ));
        }
        if self.target_names.is_empty()
            || self.target_names.len() != self.target_units.len()
            || self.target_schema_id.trim().is_empty()
        {
            return Err(ElmError::InvalidInput("target schema is incomplete".into()));
        }
        if self.task_kind == TaskKind::Classification && self.class_mapping.len() < 2 {
            return Err(ElmError::InvalidInput(
                "classification dataset needs an explicit class mapping".into(),
            ));
        }
        let mut seen = HashSet::new();
        let mut eligible = Vec::new();
        let mut groups = HashSet::new();
        let mut subgroup_groups: BTreeMap<String, HashSet<String>> = BTreeMap::new();
        let mut balance = BTreeMap::new();
        for (index, record) in self.records.iter().enumerate() {
            if !seen.insert(record.record_id.as_str()) {
                return Err(ElmError::InvalidInput(
                    "duplicate record ID in dataset".into(),
                ));
            }
            let timestamp_valid = [
                record.event_time,
                record.observation_time,
                record.decision_time,
                record.label_available_time,
            ]
            .iter()
            .all(|value| value.is_finite())
                && record.event_time <= record.observation_time
                && record.observation_time <= record.decision_time
                && record.label_available_time >= record.decision_time;
            let shape_valid = record.features.len() == self.feature_schema.names.len()
                && record.target.len() == self.target_names.len()
                && record
                    .features
                    .iter()
                    .chain(&record.target)
                    .all(|value| value.is_finite());
            let mature = record.label_available_time + minimum_label_age as f64 <= now;
            if record.record_id.trim().is_empty()
                || record.group_id.trim().is_empty()
                || record.subgroup_id.trim().is_empty()
                || !timestamp_valid
                || !shape_valid
            {
                continue;
            }
            if !mature {
                continue;
            }
            if self.task_kind == TaskKind::Classification {
                let class = record.target.first().copied().unwrap_or(-1.0);
                if class < 0.0 || class.fract() != 0.0 || class as usize >= self.class_mapping.len()
                {
                    continue;
                }
                *balance
                    .entry(self.class_mapping[class as usize].clone())
                    .or_default() += 1;
            }
            groups.insert(record.group_id.as_str());
            subgroup_groups
                .entry(record.subgroup_id.clone())
                .or_default()
                .insert(record.group_id.clone());
            eligible.push(index);
        }
        if eligible.len() < 10 || groups.len() < 10 {
            return Err(ElmError::InvalidInput(format!(
                "insufficient mature independent support: {} samples across {} groups; need at least 10 of each",
                eligible.len(),
                groups.len()
            )));
        }
        let digest = self.digest()?;
        let mut ordered = eligible
            .iter()
            .map(|index| self.records[*index].decision_time)
            .collect::<Vec<_>>();
        ordered.sort_by(f64::total_cmp);
        let quantile =
            |fraction: f64| ordered[((ordered.len() - 1) as f64 * fraction).floor() as usize];
        let latest_training_time = quantile(0.59);
        let tuning_cutoff = quantile(0.74);
        let calibration_cutoff = quantile(0.89);
        let final_cutoff = quantile(0.99);
        Ok((
            eligible.clone(),
            DataQualityReport {
                dataset_id: self.dataset_id.clone(),
                dataset_digest: digest,
                eligible_samples: eligible.len(),
                excluded_samples: self.records.len().saturating_sub(eligible.len()),
                group_count: groups.len(),
                subgroup_group_counts: subgroup_groups
                    .into_iter()
                    .map(|(name, values)| (name, values.len()))
                    .collect(),
                class_balance: balance,
                training_cutoff: latest_training_time,
                tuning_cutoff,
                calibration_cutoff,
                final_cutoff,
                fixture_only: self.fixture_only,
            },
        ))
    }
}

/// Group-preserving chronological partition with four independent data roles.
/// Groups are assigned as a whole; label maturity across each boundary is
/// purged to reduce time leakage.
fn split_groups(manifest: &DatasetManifest, eligible: &[usize]) -> Result<[Vec<usize>; 4]> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for index in eligible {
        groups
            .entry(manifest.records[*index].group_id.clone())
            .or_default()
            .push(*index);
    }
    let mut groups: Vec<Vec<usize>> = groups.into_values().collect();
    groups.sort_by(|left, right| {
        let left_time = left
            .iter()
            .map(|index| manifest.records[*index].decision_time)
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        let right_time = right
            .iter()
            .map(|index| manifest.records[*index].decision_time)
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        left_time.total_cmp(&right_time)
    });
    let count = groups.len();
    let train_end = (count * 60 / 100).max(1);
    let tune_end = (count * 75 / 100).max(train_end + 1);
    let calibration_end = (count * 90 / 100).max(tune_end + 1).min(count - 1);
    if train_end >= tune_end || tune_end >= calibration_end || calibration_end >= count {
        return Err(ElmError::InvalidInput(
            "not enough independent groups for train/tune/calibration/final partitions".into(),
        ));
    }
    let boundaries = [train_end, tune_end, calibration_end].map(|point| {
        groups[point]
            .iter()
            .map(|index| manifest.records[*index].decision_time)
            .min_by(f64::total_cmp)
            .unwrap_or(f64::INFINITY)
    });
    let mut output = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (group_index, group) in groups.iter().enumerate() {
        let partition = if group_index < train_end {
            0
        } else if group_index < tune_end {
            1
        } else if group_index < calibration_end {
            2
        } else {
            3
        };
        let first_time = group
            .iter()
            .map(|index| manifest.records[*index].decision_time)
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        for index in group {
            let record = &manifest.records[*index];
            let purged = match partition {
                0 => record.label_available_time > boundaries[0],
                1 => record.label_available_time > boundaries[1],
                2 => record.label_available_time > boundaries[2],
                _ => false,
            } || (partition > 0 && record.decision_time < first_time);
            if !purged {
                output[partition].push(*index);
            }
        }
    }
    if output.iter().any(|partition| partition.is_empty()) {
        return Err(ElmError::InvalidInput(
            "time/group purging left an empty evaluation partition".into(),
        ));
    }
    Ok(output)
}

pub fn train_and_evaluate(
    manifest: &DatasetManifest,
    now: f64,
    limits: &TrainingLimits,
) -> Result<TrainedEvaluation> {
    let training_started = std::time::Instant::now();
    let (eligible, quality) = manifest.validate(
        now,
        limits.max_samples,
        limits.max_features,
        limits.minimum_label_age_seconds,
    )?;
    if manifest.fixture_only {
        // Fixtures exercise fitting but are permanently ineligible for promotion.
    }
    let partitions = split_groups(manifest, &eligible)?;
    let training = &partitions[0];
    let tuning = &partitions[1];
    let calibration = &partitions[2];
    let final_test = &partitions[3];
    let feature_dims = manifest.feature_schema.names.len();
    let requested_hidden = [32_usize, 64, 128, 256]
        .into_iter()
        .filter(|value| *value <= limits.max_hidden_units)
        .collect::<Vec<_>>();
    let hidden_search = if requested_hidden.is_empty() {
        vec![limits.max_hidden_units.clamp(1, 32)]
    } else {
        requested_hidden
    };
    let memory_estimate = (training
        .len()
        .saturating_mul(feature_dims.saturating_add(limits.max_hidden_units))
        .saturating_mul(8)
        .saturating_add(
            limits
                .max_hidden_units
                .saturating_mul(limits.max_hidden_units)
                .saturating_mul(8),
        )
        .saturating_mul(limits.max_training_threads.clamp(1, 64)))
        / (1024 * 1024);
    if memory_estimate > limits.max_memory_mb {
        return Err(ElmError::InvalidInput(format!(
            "estimated ELM workspace {memory_estimate} MiB exceeds training memory budget"
        )));
    }

    let families = match manifest.task_kind {
        TaskKind::Regression => vec![AlgorithmFamily::Ridge, AlgorithmFamily::Elm],
        TaskKind::Classification => vec![AlgorithmFamily::Logistic, AlgorithmFamily::Elm],
    };
    let mut trials = Vec::new();
    for family in &families {
        let widths: Vec<usize> = if *family == AlgorithmFamily::Elm {
            hidden_search.clone()
        } else {
            vec![0]
        };
        for hidden in widths {
            let name = format!("{}:h={hidden}:lambda=1e-3", family_name(*family));
            trials.push((*family, hidden, name));
        }
    }
    let attempts = trials
        .iter()
        .map(|(_, _, name)| name.clone())
        .collect::<Vec<_>>();
    let (x_train, y_train, labels_train) = rows(manifest, training);
    let (x_tune, y_tune, labels_tune) = rows(manifest, tuning);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(limits.max_training_threads.clamp(1, 64))
        .thread_name(|index| format!("gail-elm-train-{index}"))
        .build()
        .map_err(|error| ElmError::Artifact(format!("create bounded training pool: {error}")))?;
    let fitted = pool.install(|| {
        trials
            .par_iter()
            .map(|(family, hidden, name)| {
                if training_started.elapsed().as_secs() >= limits.max_training_seconds {
                    return Err(ElmError::Numerical(
                        "configured training time budget exhausted between trials".into(),
                    ));
                }
                let options = fit_options(
                    manifest,
                    quality.training_cutoff,
                    &quality.dataset_digest,
                    *family,
                    *hidden,
                );
                let result = match manifest.task_kind {
                    TaskKind::Regression => ElmArtifact::fit_regression(
                        manifest.feature_schema.clone(),
                        &x_train,
                        &y_train,
                        None,
                        options,
                    ),
                    TaskKind::Classification => ElmArtifact::fit_classifier(
                        manifest.feature_schema.clone(),
                        &x_train,
                        &labels_train,
                        None,
                        options,
                    ),
                };
                result.and_then(|artifact| {
                    let metric =
                        measure(&artifact, &x_tune, &y_tune, &labels_tune, false)?.primary_metric;
                    Ok((name.clone(), artifact, metric))
                })
            })
            .collect::<Vec<_>>()
    });
    let mut candidates = Vec::new();
    let mut attempts = attempts;
    for result in fitted {
        match result {
            Ok(candidate) => candidates.push(candidate),
            Err(error) => attempts.push(format!("candidate_rejected:{error}")),
        }
    }
    if candidates.is_empty() {
        return Err(ElmError::Numerical(
            "no candidate estimator completed a fit".into(),
        ));
    }
    // MSE/log loss are lower-is-better. Ridge/logistic win exact ties by order.
    candidates.sort_by(|left, right| left.2.total_cmp(&right.2));
    let baseline = candidates
        .iter()
        .filter(|(_, item, _)| item.metadata.algorithm != AlgorithmFamily::Elm)
        .min_by(|left, right| left.2.total_cmp(&right.2))
        .map(|(_, item, _)| item.clone())
        .unwrap_or_else(|| candidates[0].1.clone());
    let tuning_metrics = candidates
        .iter()
        .map(|(name, _, metric)| (name.clone(), *metric))
        .collect::<BTreeMap<_, _>>();
    let (selected_name, mut artifact, selected_tuning) = candidates.remove(0);
    let selected_family = artifact.metadata.algorithm;
    let _selected_training_metric = (selected_name, selected_tuning);

    let (x_calibration, y_calibration, labels_calibration) = rows(manifest, calibration);
    match manifest.task_kind {
        TaskKind::Regression => {
            artifact.fit_residual_radius(&x_calibration, &y_calibration, 0.9)?
        }
        TaskKind::Classification => {
            artifact.fit_temperature(&x_calibration, &labels_calibration)?
        }
    }
    let (x_final, y_final, labels_final) = rows(manifest, final_test);
    let mut final_candidate = measure(&artifact, &x_final, &y_final, &labels_final, true)?;
    let (base_x_cal, base_y_cal, base_labels_cal) = rows(manifest, calibration);
    let mut calibrated_baseline = baseline;
    match manifest.task_kind {
        TaskKind::Regression => {
            calibrated_baseline.fit_residual_radius(&base_x_cal, &base_y_cal, 0.9)?
        }
        TaskKind::Classification => {
            calibrated_baseline.fit_temperature(&base_x_cal, &base_labels_cal)?
        }
    }
    let mut final_baseline = measure(
        &calibrated_baseline,
        &x_final,
        &y_final,
        &labels_final,
        true,
    )?;
    let final_records = final_test
        .iter()
        .map(|index| &manifest.records[*index])
        .collect::<Vec<_>>();
    let group_count = final_records
        .iter()
        .map(|record| record.group_id.as_str())
        .collect::<HashSet<_>>()
        .len();
    let mut final_subgroups: BTreeMap<&str, HashSet<&str>> = BTreeMap::new();
    for record in &final_records {
        final_subgroups
            .entry(record.subgroup_id.as_str())
            .or_default()
            .insert(record.group_id.as_str());
    }
    let minimum_final_subgroup_group_support = quality
        .subgroup_group_counts
        .keys()
        .map(|subgroup| {
            final_subgroups
                .get(subgroup.as_str())
                .map_or(0, HashSet::len)
        })
        .min()
        .unwrap_or(0);
    final_candidate.summary.effective_sample_count = group_count as f64;
    final_baseline.summary.effective_sample_count = group_count as f64;
    let earliest = final_records
        .iter()
        .map(|record| record.decision_time)
        .fold(f64::INFINITY, f64::min);
    let latest = final_records
        .iter()
        .map(|record| record.decision_time)
        .fold(f64::NEG_INFINITY, f64::max);
    let observed_window_seconds = (latest - earliest).max(0.0).min(u64::MAX as f64) as u64;
    let maximum_input_age_seconds = final_records
        .iter()
        .map(|record| {
            (record.decision_time - record.observation_time)
                .max(0.0)
                .min(u64::MAX as f64) as u64
        })
        .max();
    let maximum_label_delay_seconds = final_records
        .iter()
        .map(|record| {
            (record.label_available_time - record.decision_time)
                .max(0.0)
                .min(u64::MAX as f64) as u64
        })
        .max();
    let (drift_score, maximum_ood_score) = distribution_diagnostics(&artifact, &x_train, &x_final)?;
    let (paired_improvement, lower_confidence_bound) = grouped_paired_improvement(
        &artifact,
        &calibrated_baseline,
        &x_final,
        &y_final,
        &labels_final,
        &final_records,
        0.95,
    )?;
    (
        final_candidate.summary.inference_p95_ms,
        final_candidate.summary.inference_p99_ms,
    ) = measure_latency_percentiles(&artifact, &x_final)?;
    (
        final_baseline.summary.inference_p95_ms,
        final_baseline.summary.inference_p99_ms,
    ) = measure_latency_percentiles(&calibrated_baseline, &x_final)?;
    artifact.evaluation = Some(final_candidate.summary.clone());
    artifact.refresh_digest()?;
    let mut report = TrainingEvaluationReport {
        report_version: 1,
        task_id: manifest.task_id.clone(),
        dataset_id: manifest.dataset_id.clone(),
        dataset_digest: quality.dataset_digest.clone(),
        fixture_only: manifest.fixture_only,
        selected_algorithm: selected_family,
        baseline_algorithm: calibrated_baseline.metadata.algorithm,
        attempted_configurations: attempts,
        tuning_metrics,
        final_candidate: final_candidate.summary,
        final_baseline: final_baseline.summary,
        quality,
        final_independent_groups: group_count,
        minimum_final_subgroup_group_support,
        observed_window_seconds,
        maximum_input_age_seconds,
        maximum_label_delay_seconds,
        drift_score: Some(drift_score),
        maximum_ood_score: Some(maximum_ood_score),
        paired_improvement,
        lower_confidence_bound,
        confidence_level: 0.95,
        calibrated: artifact.calibration.is_some(),
        qualified: false,
        qualification_reasons: if manifest.fixture_only {
            vec!["fixture_only_artifact".into()]
        } else {
            vec!["promotion_policy_and_domain_evidence_not_supplied".into()]
        },
        report_sha256: String::new(),
    };
    let bytes =
        serde_json::to_vec(&report).map_err(|error| ElmError::Artifact(error.to_string()))?;
    report.report_sha256 = hex::encode(Sha256::digest(bytes));
    Ok(TrainedEvaluation { artifact, report })
}

struct Measured {
    summary: EvaluationSummary,
    primary_metric: f64,
}

fn measure(
    artifact: &ElmArtifact,
    features: &[Vec<f64>],
    targets: &[Vec<f64>],
    labels: &[usize],
    calibrated: bool,
) -> Result<Measured> {
    let started = std::time::Instant::now();
    let raw = artifact.raw_predict(features)?;
    let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut summary = EvaluationSummary {
        partition: if calibrated { "final" } else { "tuning" }.into(),
        sample_count: features.len(),
        effective_sample_count: features.len() as f64,
        mse: None,
        brier_score: None,
        log_loss: None,
        accuracy: None,
        calibration_error: None,
        residual_coverage: None,
        inference_p95_ms: Some(duration_ms / features.len().max(1) as f64),
        inference_p99_ms: Some(duration_ms / features.len().max(1) as f64),
    };
    let primary_metric = match artifact.task_kind {
        TaskKind::Regression => {
            let squared: Vec<f64> = raw
                .iter()
                .zip(targets)
                .flat_map(|(prediction, target)| {
                    prediction
                        .iter()
                        .zip(target)
                        .map(|(left, right)| (left - right).powi(2))
                })
                .collect();
            let mse = squared.iter().sum::<f64>() / squared.len().max(1) as f64;
            summary.mse = Some(mse);
            let radius = if calibrated {
                artifact
                    .calibration
                    .as_ref()
                    .and_then(|value| value.residual_radius)
            } else {
                None
            };
            if let Some(radius) = radius {
                let errors = raw.iter().zip(targets).flat_map(|(prediction, target)| {
                    prediction
                        .iter()
                        .zip(target)
                        .map(|(left, right)| (left - right).abs())
                });
                let covered = errors.filter(|error| *error <= radius).count();
                let dimensions = targets.iter().map(Vec::len).sum::<usize>().max(1);
                let coverage = covered as f64 / dimensions as f64;
                summary.residual_coverage = Some(coverage);
                summary.calibration_error = Some((coverage - 0.9).abs());
            }
            mse
        }
        TaskKind::Classification => {
            let probabilities = if calibrated {
                artifact.predict_probabilities(features)?
            } else {
                raw.iter()
                    .map(|scores| stable_softmax(scores))
                    .collect::<Result<Vec<_>>>()?
            };
            let mut brier = 0.0;
            let mut loss = 0.0;
            let mut correct = 0;
            let mut calibration_gap = 0.0;
            for (probability, label) in probabilities.iter().zip(labels) {
                if *label >= probability.len() {
                    return Err(ElmError::SchemaMismatch);
                }
                for (class, score) in probability.iter().enumerate() {
                    brier += (score - if class == *label { 1.0 } else { 0.0 }).powi(2);
                }
                loss -= probability[*label].max(f64::MIN_POSITIVE).ln();
                let predicted = probability
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                if predicted == *label {
                    correct += 1;
                }
                calibration_gap +=
                    (probability[predicted] - if predicted == *label { 1.0 } else { 0.0 }).abs();
            }
            let total = labels.len().max(1) as f64;
            let classes = probabilities[0].len() as f64;
            summary.brier_score = Some(brier / (total * classes));
            summary.log_loss = Some(loss / total);
            summary.accuracy = Some(correct as f64 / total);
            summary.calibration_error = Some(calibration_gap / total);
            loss / total
        }
    };
    Ok(Measured {
        summary,
        primary_metric,
    })
}

fn measure_latency_percentiles(
    artifact: &ElmArtifact,
    features: &[Vec<f64>],
) -> Result<(Option<f64>, Option<f64>)> {
    if features.is_empty() {
        return Ok((None, None));
    }
    // Keep evaluation overhead bounded on large manifests while retaining a
    // deterministic sample of the final partition for per-request timings.
    let count = features.len().min(4_096);
    let stride = features.len().div_ceil(count);
    let mut samples = Vec::with_capacity(count);
    for row in features.iter().step_by(stride).take(count) {
        let started = std::time::Instant::now();
        if artifact.task_kind == TaskKind::Classification {
            artifact.predict_probabilities(std::slice::from_ref(row))?;
        } else {
            artifact.raw_predict(std::slice::from_ref(row))?;
        }
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let index = ((samples.len().saturating_sub(1)) as f64 * fraction).ceil() as usize;
        samples.get(index).copied()
    };
    Ok((percentile(0.95), percentile(0.99)))
}

fn distribution_diagnostics(
    artifact: &ElmArtifact,
    training: &[Vec<f64>],
    final_rows: &[Vec<f64>],
) -> Result<(f64, f64)> {
    let preprocessor = &artifact.preprocessor;
    let training = preprocessor.transform(training)?;
    let final_rows = preprocessor.transform(final_rows)?;
    let mut maximum_ood: f64 = 0.0;
    let mut maximum_drift: f64 = 0.0;
    for column in 0..preprocessor.means.len() {
        let absolute = final_rows
            .iter()
            .map(|row| row[column].abs())
            .fold(0.0, f64::max);
        let train_mean =
            training.iter().map(|row| row[column]).sum::<f64>() / training.len().max(1) as f64;
        let final_mean =
            final_rows.iter().map(|row| row[column]).sum::<f64>() / final_rows.len().max(1) as f64;
        maximum_ood = maximum_ood.max(absolute);
        maximum_drift = maximum_drift.max((final_mean - train_mean).abs());
    }
    if !maximum_ood.is_finite() || !maximum_drift.is_finite() {
        return Err(ElmError::Numerical(
            "non-finite distribution diagnostic".into(),
        ));
    }
    Ok((maximum_drift, maximum_ood))
}

fn grouped_paired_improvement(
    candidate: &ElmArtifact,
    baseline: &ElmArtifact,
    features: &[Vec<f64>],
    targets: &[Vec<f64>],
    labels: &[usize],
    records: &[&DatasetRecord],
    confidence_level: f64,
) -> Result<(Option<f64>, Option<f64>)> {
    if features.is_empty() || features.len() != records.len() {
        return Ok((None, None));
    }
    let losses = |artifact: &ElmArtifact| -> Result<Vec<f64>> {
        match artifact.task_kind {
            TaskKind::Regression => {
                let predicted = artifact.raw_predict(features)?;
                Ok(predicted
                    .iter()
                    .zip(targets)
                    .map(|(prediction, target)| {
                        prediction
                            .iter()
                            .zip(target)
                            .map(|(left, right)| (left - right).powi(2))
                            .sum::<f64>()
                            / target.len().max(1) as f64
                    })
                    .collect())
            }
            TaskKind::Classification => {
                let probabilities = artifact.predict_probabilities(features)?;
                probabilities
                    .iter()
                    .zip(labels)
                    .map(|(row, label)| {
                        row.get(*label)
                            .copied()
                            .filter(|value| value.is_finite() && *value > 0.0)
                            .map(|value| -value.ln())
                            .ok_or_else(|| {
                                ElmError::Numerical("invalid calibrated class probability".into())
                            })
                    })
                    .collect()
            }
        }
    };
    let candidate_loss = losses(candidate)?;
    let baseline_loss = losses(baseline)?;
    let mut grouped: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for ((candidate, baseline), record) in candidate_loss.iter().zip(&baseline_loss).zip(records) {
        grouped
            .entry(record.group_id.as_str())
            .or_default()
            .push(baseline - candidate);
    }
    if grouped.len() < 2 {
        return Ok((None, None));
    }
    let group_differences = grouped
        .values()
        .map(|values| values.iter().sum::<f64>() / values.len().max(1) as f64)
        .collect::<Vec<_>>();
    let improvement = group_differences.iter().sum::<f64>() / group_differences.len() as f64;
    let mut random = 0x9e37_79b9_7f4a_7c15_u64;
    let mut bootstrap = Vec::with_capacity(2_000);
    for _ in 0..2_000 {
        let mut total = 0.0;
        for _ in 0..group_differences.len() {
            random = random.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = random;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            let index = (value as usize) % group_differences.len();
            total += group_differences[index];
        }
        bootstrap.push(total / group_differences.len() as f64);
    }
    bootstrap.sort_by(f64::total_cmp);
    let tail = ((1.0 - confidence_level).clamp(0.0, 0.5) * bootstrap.len() as f64).floor() as usize;
    Ok((Some(improvement), bootstrap.get(tail).copied()))
}

fn stable_softmax(scores: &[f64]) -> Result<Vec<f64>> {
    if scores.is_empty() || scores.iter().any(|value| !value.is_finite()) {
        return Err(ElmError::Numerical("invalid classifier output".into()));
    }
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut values: Vec<f64> = scores.iter().map(|score| (score - max).exp()).collect();
    let sum = values.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(ElmError::Numerical("invalid softmax output".into()));
    }
    values.iter_mut().for_each(|value| *value /= sum);
    Ok(values)
}

fn rows(
    manifest: &DatasetManifest,
    indices: &[usize],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>) {
    let mut features = Vec::with_capacity(indices.len());
    let mut targets = Vec::with_capacity(indices.len());
    let mut labels = Vec::with_capacity(indices.len());
    for index in indices {
        let record = &manifest.records[*index];
        features.push(record.features.clone());
        targets.push(record.target.clone());
        labels.push(record.target.first().copied().unwrap_or(0.0).max(0.0) as usize);
    }
    (features, targets, labels)
}

fn fit_options(
    manifest: &DatasetManifest,
    cutoff: f64,
    digest: &str,
    family: AlgorithmFamily,
    hidden: usize,
) -> FitOptions {
    FitOptions {
        task_id: manifest.task_id.clone(),
        family,
        hidden_units: hidden,
        seed: 20261001,
        lambda: 1e-3,
        target_names: manifest.target_names.clone(),
        target_units: manifest.target_units.clone(),
        class_mapping: manifest.class_mapping.clone(),
        training_cutoff: cutoff,
        dataset_digest: digest.into(),
        tenant_scope: manifest.tenant_scope.clone(),
        domain_scope: manifest.domain_scope.clone(),
        horizon_seconds: manifest.horizon_seconds,
        fixture_only: manifest.fixture_only,
    }
}

fn family_name(family: AlgorithmFamily) -> &'static str {
    match family {
        AlgorithmFamily::Elm => "elm",
        AlgorithmFamily::Ridge => "ridge",
        AlgorithmFamily::Logistic => "logistic",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> DatasetManifest {
        let schema = FeatureSchema {
            id: "fixture:v1".into(),
            version: 1,
            names: vec!["x".into()],
            units: vec!["unitless".into()],
        };
        let records = (0..100)
            .map(|index| {
                let time = index as f64 + 1.0;
                DatasetRecord {
                    record_id: format!("row-{index}"),
                    group_id: format!("group-{index}"),
                    subgroup_id: if index % 2 == 0 { "even" } else { "odd" }.into(),
                    event_time: time,
                    observation_time: time,
                    decision_time: time,
                    label_available_time: time + 1.0,
                    features: vec![index as f64 / 10.0],
                    target: vec![index as f64 / 5.0 + 1.0],
                }
            })
            .collect();
        DatasetManifest {
            format_version: 1,
            dataset_id: "fixture-linear".into(),
            permitted: true,
            fixture_only: true,
            tenant_scope: "local".into(),
            domain_scope: "test".into(),
            task_id: "routing_candidate_utility".into(),
            task_kind: TaskKind::Regression,
            feature_schema: schema,
            target_schema_id: "utility:v1".into(),
            target_names: vec!["utility".into()],
            target_units: vec!["score".into()],
            class_mapping: Vec::new(),
            horizon_seconds: None,
            provenance: "unit test fixture".into(),
            records,
        }
    }

    #[test]
    fn manifest_pipeline_makes_evaluated_fixture_artifact_and_withholds_qualification() {
        let result = train_and_evaluate(
            &fixture(),
            1_000.0,
            &TrainingLimits {
                max_samples: 1_000,
                max_features: 8,
                max_hidden_units: 32,
                max_memory_mb: 64,
                max_training_threads: 2,
                max_training_seconds: 900,
                minimum_label_age_seconds: 0,
            },
        )
        .unwrap();
        assert!(result.artifact.metadata.fixture_only);
        assert!(result.artifact.evaluation.is_some());
        assert!(!result.report.qualified);
        assert!(result.report.final_candidate.inference_p95_ms.is_some());
        assert!(result.report.final_candidate.inference_p99_ms.is_some());
        assert!(
            result.report.final_candidate.inference_p95_ms
                <= result.report.final_candidate.inference_p99_ms
        );
        assert!(
            result
                .report
                .qualification_reasons
                .contains(&"fixture_only_artifact".into())
        );
        result.artifact.verify().unwrap();
    }

    #[test]
    fn data_permission_and_point_in_time_contract_are_required() {
        let mut dataset = fixture();
        dataset.permitted = false;
        assert!(dataset.validate(1_000.0, 1000, 8, 0).is_err());
        dataset = fixture();
        dataset.records[0].observation_time = dataset.records[0].decision_time + 1.0;
        let (_, quality) = dataset.validate(1_000.0, 1000, 8, 0).unwrap();
        assert_eq!(quality.excluded_samples, 1);
    }
}
