//! Reproducible batch ELM and simple baseline estimators.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub use crate::elm_config::FeatureSchema;

use super::{
    ElmError, Result,
    math::{SolverDiagnostics, validate_matrix, weighted_ridge},
};

pub const ARTIFACT_FORMAT_VERSION: u32 = 1;
pub const PRNG_ALGORITHM: &str = "splitmix64-v1";

impl FeatureSchema {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty()
            || self.version == 0
            || self.names.is_empty()
            || self.names.len() != self.units.len()
            || self.names.iter().any(|name| name.trim().is_empty())
        {
            return Err(ElmError::InvalidInput(
                "feature schema needs an ID, version, unique named features and one unit per feature".into(),
            ));
        }
        let mut names = self.names.clone();
        names.sort();
        names.dedup();
        if names.len() != self.names.len() {
            return Err(ElmError::InvalidInput(
                "feature names must be unique".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Preprocessor {
    pub means: Vec<f64>,
    pub scales: Vec<f64>,
    pub fit_partition: String,
}

impl Preprocessor {
    pub fn fit(rows: &[Vec<f64>]) -> Result<Self> {
        let (count, dimensions) = validate_matrix(rows, "preprocessor features")?;
        let mut means = vec![0.0; dimensions];
        let mut scales = vec![0.0; dimensions];
        for row in rows {
            for (index, value) in row.iter().enumerate() {
                means[index] += value / count as f64;
            }
        }
        for row in rows {
            for (index, value) in row.iter().enumerate() {
                scales[index] += (value - means[index]).powi(2) / count as f64;
            }
        }
        for scale in &mut scales {
            *scale = scale.sqrt();
            if *scale <= 1e-12 {
                *scale = 1.0;
            }
        }
        Ok(Self {
            means,
            scales,
            fit_partition: "training".into(),
        })
    }

    pub fn transform(&self, rows: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if self.means.is_empty()
            || self.means.len() != self.scales.len()
            || self
                .scales
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ElmError::Artifact("invalid fitted preprocessor".into()));
        }
        let (count, dimensions) = validate_matrix(rows, "features")?;
        if dimensions != self.means.len() {
            return Err(ElmError::SchemaMismatch);
        }
        Ok((0..count)
            .map(|row| {
                (0..dimensions)
                    .map(|column| (rows[row][column] - self.means[column]) / self.scales[column])
                    .collect()
            })
            .collect())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Regression,
    Classification,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmFamily {
    Elm,
    Ridge,
    Logistic,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Estimator {
    Elm {
        input_weights: Vec<Vec<f64>>,
        biases: Vec<f64>,
        output_weights: Vec<Vec<f64>>,
    },
    Linear {
        weights: Vec<Vec<f64>>,
        intercepts: Vec<f64>,
    },
    Logistic {
        weights: Vec<Vec<f64>>,
        intercepts: Vec<f64>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calibration {
    pub identity: String,
    pub temperature: f64,
    pub residual_radius: Option<f64>,
    pub fitted_on: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: String,
    pub task_id: String,
    pub algorithm: AlgorithmFamily,
    pub algorithm_version: u32,
    pub source_revision: String,
    pub seed: u64,
    pub prng: String,
    pub hidden_units: usize,
    pub target_names: Vec<String>,
    pub target_units: Vec<String>,
    pub class_mapping: Vec<String>,
    pub training_cutoff: f64,
    pub dataset_digest: String,
    pub tenant_scope: String,
    pub domain_scope: String,
    pub horizon_seconds: Option<u64>,
    pub expiry_unix_seconds: Option<u64>,
    pub fixture_only: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationSummary {
    pub partition: String,
    pub sample_count: usize,
    pub effective_sample_count: f64,
    pub mse: Option<f64>,
    pub brier_score: Option<f64>,
    pub log_loss: Option<f64>,
    pub accuracy: Option<f64>,
    pub calibration_error: Option<f64>,
    pub residual_coverage: Option<f64>,
    pub inference_p95_ms: Option<f64>,
    #[serde(default)]
    pub inference_p99_ms: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElmArtifact {
    pub format_version: u32,
    pub metadata: ModelMetadata,
    pub feature_schema: FeatureSchema,
    pub target_schema_id: String,
    pub task_kind: TaskKind,
    pub preprocessor: Preprocessor,
    pub estimator: Estimator,
    pub calibration: Option<Calibration>,
    pub solver: SolverDiagnostics,
    pub evaluation: Option<EvaluationSummary>,
    pub content_sha256: String,
}

#[derive(Clone, Debug)]
pub struct FitOptions {
    pub task_id: String,
    pub family: AlgorithmFamily,
    pub hidden_units: usize,
    pub seed: u64,
    pub lambda: f64,
    pub target_names: Vec<String>,
    pub target_units: Vec<String>,
    pub class_mapping: Vec<String>,
    pub training_cutoff: f64,
    pub dataset_digest: String,
    pub tenant_scope: String,
    pub domain_scope: String,
    pub horizon_seconds: Option<u64>,
    pub fixture_only: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            task_id: "unspecified".into(),
            family: AlgorithmFamily::Elm,
            hidden_units: 32,
            seed: 1,
            lambda: 1e-3,
            target_names: vec!["target".into()],
            target_units: vec!["unitless".into()],
            class_mapping: Vec::new(),
            training_cutoff: 0.0,
            dataset_digest: String::new(),
            tenant_scope: "local".into(),
            domain_scope: "unspecified".into(),
            horizon_seconds: None,
            fixture_only: true,
        }
    }
}

impl ElmArtifact {
    pub fn fit_regression(
        schema: FeatureSchema,
        features: &[Vec<f64>],
        targets: &[Vec<f64>],
        sample_weights: Option<&[f64]>,
        options: FitOptions,
    ) -> Result<Self> {
        schema.validate()?;
        let (rows, dimensions) = validate_matrix(features, "training features")?;
        let (target_rows, outputs) = validate_matrix(targets, "training targets")?;
        validate_fit(&options, rows, dimensions, target_rows, outputs, &schema)?;
        if options.family == AlgorithmFamily::Logistic || !options.class_mapping.is_empty() {
            return Err(ElmError::InvalidInput(
                "classification options used for regression".into(),
            ));
        }
        let preprocessor = Preprocessor::fit(features)?;
        let x = preprocessor.transform(features)?;
        let mut design = Vec::with_capacity(rows);
        let (estimator, solver) = match options.family {
            AlgorithmFamily::Elm => {
                if options.hidden_units == 0 || options.hidden_units > 512 {
                    return Err(ElmError::InvalidInput(
                        "hidden units must be in 1..=512".into(),
                    ));
                }
                let mut rng = SplitMix64::new(options.seed);
                let input_weights: Vec<Vec<f64>> = (0..dimensions)
                    .map(|_| {
                        (0..options.hidden_units)
                            .map(|_| rng.range(-1.0, 1.0))
                            .collect()
                    })
                    .collect();
                let biases: Vec<f64> = (0..options.hidden_units)
                    .map(|_| rng.range(-std::f64::consts::PI, std::f64::consts::PI))
                    .collect();
                for row in &x {
                    design.push(project(row, &input_weights, &biases)?);
                }
                let (readout, diagnostics) =
                    weighted_ridge(&design, targets, sample_weights, options.lambda)?;
                (
                    Estimator::Elm {
                        input_weights,
                        biases,
                        output_weights: readout,
                    },
                    diagnostics,
                )
            }
            AlgorithmFamily::Ridge => {
                for row in &x {
                    let mut with_intercept = Vec::with_capacity(dimensions + 1);
                    with_intercept.push(1.0);
                    with_intercept.extend(row.iter().copied());
                    design.push(with_intercept);
                }
                let (readout, diagnostics) =
                    weighted_ridge(&design, targets, sample_weights, options.lambda)?;
                let intercepts = readout[0].clone();
                let weights = readout[1..].to_vec();
                (
                    Estimator::Linear {
                        weights,
                        intercepts,
                    },
                    diagnostics,
                )
            }
            AlgorithmFamily::Logistic => unreachable!(),
        };
        Self::new(
            schema,
            preprocessor,
            estimator,
            solver,
            TaskKind::Regression,
            options,
        )
    }

    pub fn fit_classifier(
        schema: FeatureSchema,
        features: &[Vec<f64>],
        labels: &[usize],
        sample_weights: Option<&[f64]>,
        options: FitOptions,
    ) -> Result<Self> {
        schema.validate()?;
        let (rows, dimensions) = validate_matrix(features, "training features")?;
        if rows != labels.len() || options.class_mapping.len() < 2 {
            return Err(ElmError::InvalidInput(
                "classifier labels and class map must match and define at least two classes".into(),
            ));
        }
        if labels
            .iter()
            .any(|label| *label >= options.class_mapping.len())
        {
            return Err(ElmError::InvalidInput(
                "class index is outside the declared class mapping".into(),
            ));
        }
        if options.family == AlgorithmFamily::Ridge {
            return Err(ElmError::InvalidInput(
                "regression ridge family used for classification".into(),
            ));
        }
        if options.lambda <= 0.0 || !options.lambda.is_finite() || dimensions != schema.names.len()
        {
            return Err(ElmError::InvalidInput(
                "invalid classifier dimensions or regularisation".into(),
            ));
        }
        validate_weights(sample_weights, rows)?;
        let preprocessor = Preprocessor::fit(features)?;
        let x = preprocessor.transform(features)?;
        let outputs = options.class_mapping.len();
        let targets: Vec<Vec<f64>> = labels
            .iter()
            .map(|label| {
                (0..outputs)
                    .map(|index| if index == *label { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let (estimator, solver) = if options.family == AlgorithmFamily::Logistic {
            let (weights, intercepts, diagnostics) =
                fit_logistic(&x, labels, outputs, sample_weights, options.lambda)?;
            (
                Estimator::Logistic {
                    weights,
                    intercepts,
                },
                diagnostics,
            )
        } else {
            if options.hidden_units == 0 || options.hidden_units > 512 {
                return Err(ElmError::InvalidInput(
                    "hidden units must be in 1..=512".into(),
                ));
            }
            let mut rng = SplitMix64::new(options.seed);
            let projection_weights: Vec<Vec<f64>> = (0..dimensions)
                .map(|_| {
                    (0..options.hidden_units)
                        .map(|_| rng.range(-1.0, 1.0))
                        .collect()
                })
                .collect();
            let biases: Vec<f64> = (0..options.hidden_units)
                .map(|_| rng.range(-std::f64::consts::PI, std::f64::consts::PI))
                .collect();
            let design: Vec<Vec<f64>> = x
                .iter()
                .map(|row| project(row, &projection_weights, &biases))
                .collect::<Result<_>>()?;
            let (readout, diagnostics) =
                weighted_ridge(&design, &targets, sample_weights, options.lambda)?;
            (
                Estimator::Elm {
                    input_weights: projection_weights,
                    biases,
                    output_weights: readout,
                },
                diagnostics,
            )
        };
        Self::new(
            schema,
            preprocessor,
            estimator,
            solver,
            TaskKind::Classification,
            options,
        )
    }

    fn new(
        schema: FeatureSchema,
        preprocessor: Preprocessor,
        estimator: Estimator,
        solver: SolverDiagnostics,
        task_kind: TaskKind,
        options: FitOptions,
    ) -> Result<Self> {
        let task_id = options.task_id.clone();
        let mut artifact = Self {
            format_version: ARTIFACT_FORMAT_VERSION,
            metadata: ModelMetadata {
                model_id: Uuid::new_v4().to_string(),
                task_id: options.task_id,
                algorithm: options.family,
                algorithm_version: 1,
                source_revision: crate::build_info::revision().to_string(),
                seed: options.seed,
                prng: PRNG_ALGORITHM.into(),
                hidden_units: if options.family == AlgorithmFamily::Elm {
                    options.hidden_units
                } else {
                    0
                },
                target_names: options.target_names,
                target_units: options.target_units,
                class_mapping: options.class_mapping,
                training_cutoff: options.training_cutoff,
                dataset_digest: options.dataset_digest,
                tenant_scope: options.tenant_scope,
                domain_scope: options.domain_scope,
                horizon_seconds: options.horizon_seconds,
                expiry_unix_seconds: None,
                fixture_only: options.fixture_only,
            },
            target_schema_id: format!("{task_id}:v1"),
            feature_schema: schema,
            task_kind,
            preprocessor,
            estimator,
            calibration: None,
            solver,
            evaluation: None,
            content_sha256: String::new(),
        };
        artifact.refresh_digest()?;
        Ok(artifact)
    }

    pub fn raw_predict(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let transformed = self.preprocessor.transform(features)?;
        let outputs = match &self.estimator {
            Estimator::Elm {
                input_weights,
                biases,
                output_weights,
            } => transformed
                .iter()
                .map(|row| {
                    project(row, input_weights, biases)
                        .and_then(|hidden| dot_rows(&hidden, output_weights))
                })
                .collect::<Result<Vec<_>>>()?,
            Estimator::Linear {
                weights,
                intercepts,
            }
            | Estimator::Logistic {
                weights,
                intercepts,
            } => transformed
                .iter()
                .map(|row| linear_output(row, weights, intercepts))
                .collect::<Result<Vec<_>>>()?,
        };
        Ok(outputs)
    }

    pub fn predict_probabilities(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if self.task_kind != TaskKind::Classification || self.calibration.is_none() {
            return Err(ElmError::Uncalibrated);
        }
        let temperature = self.calibration.as_ref().expect("checked").temperature;
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(ElmError::Artifact("invalid calibration temperature".into()));
        }
        self.raw_predict(features)?
            .iter()
            .map(|scores| softmax(scores, temperature))
            .collect()
    }

    pub fn fit_temperature(&mut self, features: &[Vec<f64>], labels: &[usize]) -> Result<()> {
        if self.task_kind != TaskKind::Classification
            || features.len() != labels.len()
            || labels.is_empty()
        {
            return Err(ElmError::InvalidInput(
                "calibration requires classifier scores and non-empty labels".into(),
            ));
        }
        let scores = self.raw_predict(features)?;
        if labels.iter().any(|label| *label >= scores[0].len()) {
            return Err(ElmError::InvalidInput(
                "calibration label outside class map".into(),
            ));
        }
        let mut best = (f64::INFINITY, 1.0);
        for step in 0..121 {
            let temperature = 0.25 + step as f64 * 0.05;
            let loss = scores
                .iter()
                .zip(labels)
                .map(|(row, label)| {
                    let probability = softmax(row, temperature).unwrap_or_default();
                    -probability
                        .get(*label)
                        .copied()
                        .unwrap_or(f64::MIN_POSITIVE)
                        .max(f64::MIN_POSITIVE)
                        .ln()
                })
                .sum::<f64>()
                / labels.len() as f64;
            if loss < best.0 {
                best = (loss, temperature);
            }
        }
        self.calibration = Some(Calibration {
            identity: format!("temperature-grid-v1:{}", best.1),
            temperature: best.1,
            residual_radius: None,
            fitted_on: "calibration".into(),
        });
        self.refresh_digest()?;
        Ok(())
    }

    pub fn fit_residual_radius(
        &mut self,
        features: &[Vec<f64>],
        targets: &[Vec<f64>],
        coverage: f64,
    ) -> Result<()> {
        if self.task_kind != TaskKind::Regression
            || !(0.5..1.0).contains(&coverage)
            || features.len() != targets.len()
            || features.is_empty()
        {
            return Err(ElmError::InvalidInput(
                "regression interval calibration inputs are invalid".into(),
            ));
        }
        let predictions = self.raw_predict(features)?;
        if predictions[0].len() != targets[0].len() {
            return Err(ElmError::SchemaMismatch);
        }
        let mut residuals: Vec<f64> = predictions
            .iter()
            .zip(targets)
            .flat_map(|(prediction, target)| {
                prediction
                    .iter()
                    .zip(target)
                    .map(|(left, right)| (left - right).abs())
            })
            .collect();
        residuals.sort_by(f64::total_cmp);
        let index =
            (((residuals.len() - 1) as f64 * coverage).ceil() as usize).min(residuals.len() - 1);
        self.calibration = Some(Calibration {
            identity: format!("absolute-residual-quantile-v1:{coverage:.4}"),
            temperature: 1.0,
            residual_radius: Some(residuals[index]),
            fitted_on: "calibration".into(),
        });
        self.refresh_digest()?;
        Ok(())
    }

    pub fn refresh_digest(&mut self) -> Result<()> {
        self.content_sha256.clear();
        let bytes =
            serde_json::to_vec(self).map_err(|error| ElmError::Artifact(error.to_string()))?;
        self.content_sha256 = hex::encode(Sha256::digest(bytes));
        Ok(())
    }

    pub fn verify(&self) -> Result<()> {
        if self.format_version != ARTIFACT_FORMAT_VERSION {
            return Err(ElmError::Artifact(
                "unsupported artifact format version".into(),
            ));
        }
        if self.metadata.prng != PRNG_ALGORITHM || self.feature_schema.validate().is_err() {
            return Err(ElmError::Artifact(
                "artifact provenance or feature schema is invalid".into(),
            ));
        }
        let expected = self.content_sha256.clone();
        let mut copy = self.clone();
        copy.content_sha256.clear();
        let bytes =
            serde_json::to_vec(&copy).map_err(|error| ElmError::Artifact(error.to_string()))?;
        let actual = hex::encode(Sha256::digest(bytes));
        if actual != expected {
            return Err(ElmError::Artifact(format!(
                "artifact content digest mismatch (expected {expected}, computed {actual})"
            )));
        }
        validate_estimator(&self.estimator, self.feature_schema.names.len())?;
        Ok(())
    }
}

fn validate_estimator(estimator: &Estimator, dimensions: usize) -> Result<()> {
    match estimator {
        Estimator::Elm {
            input_weights,
            biases,
            output_weights,
        } => {
            let (input_dims, hidden) = validate_matrix(input_weights, "projection weights")?;
            let (_, outputs) = validate_matrix(output_weights, "output weights")?;
            if input_dims != dimensions
                || hidden != biases.len()
                || output_weights.len() != hidden
                || outputs == 0
            {
                return Err(ElmError::Artifact(
                    "artifact estimator dimensions do not match".into(),
                ));
            }
        }
        Estimator::Linear {
            weights,
            intercepts,
        }
        | Estimator::Logistic {
            weights,
            intercepts,
        } => {
            let (_, outputs) = validate_matrix(weights, "baseline weights")?;
            if weights.len() != dimensions || intercepts.len() != outputs {
                return Err(ElmError::Artifact(
                    "artifact baseline dimensions do not match".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_fit(
    options: &FitOptions,
    rows: usize,
    dimensions: usize,
    target_rows: usize,
    outputs: usize,
    schema: &FeatureSchema,
) -> Result<()> {
    if rows != target_rows
        || dimensions != schema.names.len()
        || outputs != options.target_names.len()
        || outputs != options.target_units.len()
        || options.lambda <= 0.0
        || !options.lambda.is_finite()
        || options.task_id.trim().is_empty()
        || options.hidden_units > 512
    {
        return Err(ElmError::InvalidInput(
            "fit dimensions, targets, task ID or regularisation are invalid".into(),
        ));
    }
    Ok(())
}

fn validate_weights(weights: Option<&[f64]>, rows: usize) -> Result<()> {
    if let Some(weights) = weights
        && (weights.len() != rows
            || weights.iter().any(|x| !x.is_finite() || *x < 0.0)
            || weights.iter().all(|x| *x == 0.0))
    {
        return Err(ElmError::InvalidInput("invalid sample weights".into()));
    }
    Ok(())
}

fn project(row: &[f64], weights: &[Vec<f64>], biases: &[f64]) -> Result<Vec<f64>> {
    if weights.len() != row.len() || weights.is_empty() || weights[0].len() != biases.len() {
        return Err(ElmError::SchemaMismatch);
    }
    let mut hidden = vec![0.0; biases.len()];
    for (feature, projection) in row.iter().zip(weights) {
        for (index, coefficient) in projection.iter().enumerate() {
            hidden[index] += feature * coefficient;
        }
    }
    for (value, bias) in hidden.iter_mut().zip(biases) {
        *value = (*value + bias).tanh();
        if !value.is_finite() {
            return Err(ElmError::Numerical(
                "non-finite projected activation".into(),
            ));
        }
    }
    Ok(hidden)
}

fn dot_rows(row: &[f64], weights: &[Vec<f64>]) -> Result<Vec<f64>> {
    if row.len() != weights.len() {
        return Err(ElmError::SchemaMismatch);
    }
    let outputs = weights.first().map_or(0, Vec::len);
    if outputs == 0 || weights.iter().any(|values| values.len() != outputs) {
        return Err(ElmError::Artifact("invalid readout dimensions".into()));
    }
    let values: Vec<f64> = (0..outputs)
        .map(|output| {
            row.iter()
                .zip(weights)
                .map(|(value, column)| value * column[output])
                .sum()
        })
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ElmError::Numerical("non-finite prediction".into()));
    }
    Ok(values)
}

fn linear_output(row: &[f64], weights: &[Vec<f64>], intercepts: &[f64]) -> Result<Vec<f64>> {
    let mut output = dot_rows(row, weights)?;
    if output.len() != intercepts.len() {
        return Err(ElmError::SchemaMismatch);
    }
    for (value, intercept) in output.iter_mut().zip(intercepts) {
        *value += intercept;
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(ElmError::Numerical("non-finite prediction".into()));
    }
    Ok(output)
}

fn softmax(scores: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if scores.is_empty()
        || scores.iter().any(|value| !value.is_finite())
        || temperature <= 0.0
        || !temperature.is_finite()
    {
        return Err(ElmError::Numerical(
            "invalid classification score or temperature".into(),
        ));
    }
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max) / temperature;
    let mut probabilities: Vec<f64> = scores
        .iter()
        .map(|score| (score / temperature - maximum).exp())
        .collect();
    let total: f64 = probabilities.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(ElmError::Numerical("invalid softmax normaliser".into()));
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    Ok(probabilities)
}

fn fit_logistic(
    x: &[Vec<f64>],
    labels: &[usize],
    classes: usize,
    sample_weights: Option<&[f64]>,
    lambda: f64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, SolverDiagnostics)> {
    let rows = x.len();
    let dimensions = x[0].len();
    if rows < 2 || classes < 2 {
        return Err(ElmError::InvalidInput(
            "logistic fit needs at least two rows and classes".into(),
        ));
    }
    let mut weights = vec![vec![0.0; classes]; dimensions];
    let mut biases = vec![0.0; classes];
    let step = 0.1 / (dimensions as f64).sqrt().max(1.0);
    // Bounded deterministic full-batch gradient descent on multinomial log loss.
    for _ in 0..800 {
        let mut grad_w = vec![vec![0.0; classes]; dimensions];
        let mut grad_b = vec![0.0; classes];
        let total_weight: f64 = sample_weights.map_or(rows as f64, |values| values.iter().sum());
        for (row_index, row) in x.iter().enumerate() {
            let scores: Vec<f64> = (0..classes)
                .map(|class| {
                    biases[class]
                        + row
                            .iter()
                            .enumerate()
                            .map(|(feature, value)| value * weights[feature][class])
                            .sum::<f64>()
                })
                .collect();
            let probs = softmax(&scores, 1.0)?;
            let sample_weight = sample_weights.map_or(1.0, |values| values[row_index]);
            for class in 0..classes {
                let error = (probs[class] - if labels[row_index] == class { 1.0 } else { 0.0 })
                    * sample_weight;
                grad_b[class] += error / total_weight;
                for feature in 0..dimensions {
                    grad_w[feature][class] +=
                        (error * row[feature] + lambda * weights[feature][class]) / total_weight;
                }
            }
        }
        for feature in 0..dimensions {
            for class in 0..classes {
                weights[feature][class] -= step * grad_w[feature][class];
            }
        }
        for class in 0..classes {
            biases[class] -= step * grad_b[class];
        }
    }
    if weights
        .iter()
        .flatten()
        .chain(&biases)
        .any(|value| !value.is_finite())
    {
        return Err(ElmError::Numerical(
            "logistic optimiser produced non-finite values".into(),
        ));
    }
    Ok((
        weights,
        biases,
        SolverDiagnostics {
            solver: "bounded_full_batch_multinomial_logistic_gradient_descent".into(),
            regularisation: lambda,
            diagonal_condition_estimate: 1.0,
            relative_residual: 0.0,
            retries: 0,
        },
    ))
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }
    fn range(&mut self, minimum: f64, maximum: f64) -> f64 {
        let unit = (self.next() >> 11) as f64 / ((1_u64 << 53) as f64);
        minimum + (maximum - minimum) * unit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> FeatureSchema {
        FeatureSchema {
            id: "test:v1".into(),
            version: 1,
            names: vec!["x".into()],
            units: vec!["unitless".into()],
        }
    }

    fn options(family: AlgorithmFamily) -> FitOptions {
        FitOptions {
            family,
            target_names: vec!["y".into()],
            target_units: vec!["unitless".into()],
            ..FitOptions::default()
        }
    }

    #[test]
    fn seeded_elm_and_artifact_round_trip_keep_predictions() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![vec![1.0], vec![3.0], vec![5.0], vec![7.0]];
        let model =
            ElmArtifact::fit_regression(schema(), &x, &y, None, options(AlgorithmFamily::Elm))
                .unwrap();
        let before = model.raw_predict(&x).unwrap();
        model.verify().unwrap();
        let encoded = serde_json::to_vec(&model).unwrap();
        let decoded: ElmArtifact = serde_json::from_slice(&encoded).unwrap();
        decoded.verify().unwrap();
        assert_eq!(before, decoded.raw_predict(&x).unwrap());
    }

    #[test]
    fn classification_requires_separate_calibration_partition() {
        let x = vec![vec![-2.0], vec![-1.0], vec![1.0], vec![2.0]];
        let labels = vec![0, 0, 1, 1];
        let mut opts = options(AlgorithmFamily::Elm);
        opts.target_names = vec!["class".into()];
        opts.target_units = vec!["class".into()];
        opts.class_mapping = vec!["low".into(), "high".into()];
        let mut model = ElmArtifact::fit_classifier(schema(), &x, &labels, None, opts).unwrap();
        assert!(matches!(
            model.predict_probabilities(&x),
            Err(ElmError::Uncalibrated)
        ));
        model.fit_temperature(&x, &labels).unwrap();
        let predictions = model.predict_probabilities(&x).unwrap();
        assert!(
            predictions
                .iter()
                .all(|row| (row.iter().sum::<f64>() - 1.0).abs() < 1e-12)
        );
    }

    #[test]
    fn logistic_baseline_is_a_separate_family() {
        let x = vec![vec![-2.0], vec![-1.0], vec![1.0], vec![2.0]];
        let labels = vec![0, 0, 1, 1];
        let mut opts = options(AlgorithmFamily::Logistic);
        opts.target_names = vec!["class".into()];
        opts.target_units = vec!["class".into()];
        opts.class_mapping = vec!["low".into(), "high".into()];
        let mut model = ElmArtifact::fit_classifier(schema(), &x, &labels, None, opts).unwrap();
        model.fit_temperature(&x, &labels).unwrap();
        assert_eq!(model.metadata.algorithm, AlgorithmFamily::Logistic);
    }
}
