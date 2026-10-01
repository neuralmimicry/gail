//! Reproducible fixture-only CPU timing harness for ELM training and inference.

use std::{
    fs,
    time::{Duration, Instant},
};

use clap::Parser;
use futures::future::join_all;
use gail::elm::{
    ElmArtifact,
    executor::InferenceExecutor,
    load_artifact,
    model::{AlgorithmFamily, FeatureSchema, FitOptions},
    save_artifact,
};

#[derive(Debug, Parser)]
#[command(name = "gail-elm-benchmark")]
#[command(about = "Measure bounded native ELM fitting and concurrent inference")]
struct Cli {
    #[arg(long, default_value = "16,64,128")]
    dimensions: String,
    #[arg(long, default_value = "32,128,256")]
    hidden_units: String,
    #[arg(long, default_value = "1,8")]
    outputs: String,
    #[arg(long, default_value = "1,8,32")]
    candidate_counts: String,
    #[arg(long, default_value_t = 512)]
    samples: usize,
    #[arg(long, default_value_t = 2)]
    threads: usize,
    /// Conservative maximum memory budget, including concurrent fitting copies.
    #[arg(long, default_value_t = 512)]
    memory_limit_mb: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let dimensions = parse_list(&cli.dimensions, 1, 2_048)?;
    let hidden_units = parse_list(&cli.hidden_units, 1, 512)?;
    let outputs = parse_list(&cli.outputs, 1, 64)?;
    let candidate_counts = parse_list(&cli.candidate_counts, 1, 32)?;
    if !(16..=100_000).contains(&cli.samples)
        || !(1..=64).contains(&cli.threads)
        || !(64..=16_384).contains(&cli.memory_limit_mb)
    {
        anyhow::bail!(
            "samples, threads, or memory limit are outside the supported benchmark bounds"
        );
    }
    let mut reports = Vec::new();
    for dimension in dimensions {
        for hidden in hidden_units.iter().copied() {
            for output_count in outputs.iter().copied() {
                let estimated_workspace_bytes =
                    workspace_estimate(cli.samples, dimension, hidden, output_count);
                if estimated_workspace_bytes.saturating_mul(3)
                    > cli.memory_limit_mb.saturating_mul(1024 * 1024)
                {
                    anyhow::bail!(
                        "estimated benchmark workspace exceeds --memory-limit-mb before allocation"
                    );
                }
                let features = fixture_matrix(cli.samples, dimension);
                let targets = fixture_targets(&features, output_count);
                let schema = FeatureSchema {
                    id: format!("benchmark-fixture:d{dimension}:v1"),
                    version: 1,
                    names: (0..dimension).map(|index| format!("x{index}")).collect(),
                    units: vec!["unitless".into(); dimension],
                };
                let fit_options = FitOptions {
                    task_id: "benchmark_fixture".into(),
                    family: AlgorithmFamily::Elm,
                    hidden_units: hidden,
                    seed: 20261001,
                    lambda: 1e-3,
                    target_names: (0..output_count).map(|index| format!("y{index}")).collect(),
                    target_units: vec!["unitless".into(); output_count],
                    class_mapping: Vec::new(),
                    training_cutoff: 0.0,
                    dataset_digest: "synthetic-benchmark-only".into(),
                    tenant_scope: "local".into(),
                    domain_scope: "benchmark-fixture".into(),
                    horizon_seconds: None,
                    fixture_only: true,
                };
                let started = Instant::now();
                let model = ElmArtifact::fit_regression(
                    schema.clone(),
                    &features,
                    &targets,
                    None,
                    fit_options.clone(),
                )?;
                let training_ms = started.elapsed().as_secs_f64() * 1_000.0;
                model.verify()?;
                let preprocessing_started = Instant::now();
                let _normalised = model.preprocessor.transform(&features)?;
                let preprocessing_ms = preprocessing_started.elapsed().as_secs_f64() * 1_000.0;
                drop(_normalised);
                let artifact_directory = std::env::temp_dir()
                    .join(format!("gail-elm-benchmark-{}", uuid::Uuid::new_v4()));
                fs::create_dir_all(&artifact_directory)?;
                let artifact_path = artifact_directory.join("candidate.elm.json");
                let artifact_write_started = Instant::now();
                save_artifact(&artifact_path, &model, 256 * 1024 * 1024)?;
                let artifact_write_ms = artifact_write_started.elapsed().as_secs_f64() * 1_000.0;
                let artifact_load_started = Instant::now();
                let loaded = load_artifact(&artifact_path, 256 * 1024 * 1024)?;
                let artifact_load_ms = artifact_load_started.elapsed().as_secs_f64() * 1_000.0;
                loaded.verify()?;
                fs::remove_dir_all(&artifact_directory)?;
                let model = std::sync::Arc::new(model);
                let inference = InferenceExecutor::new(
                    cli.threads,
                    *candidate_counts.iter().max().unwrap_or(&1),
                )?;
                let probes = (0..candidate_counts.iter().copied().max().unwrap_or(1))
                    .map(|index| features[index % features.len()].clone())
                    .collect::<Vec<_>>();
                for candidate_count in &candidate_counts {
                    let start = Instant::now();
                    let tasks = probes.iter().take(*candidate_count).map(|row| {
                        inference.predict(model.clone(), row.clone(), Duration::from_secs(30))
                    });
                    let results = join_all(tasks).await;
                    let batch_elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
                    let mut latencies_ms = results
                        .into_iter()
                        .map(|result| result.map(|item| item.elapsed.as_secs_f64() * 1_000.0))
                        .collect::<Result<Vec<_>, _>>()?;
                    latencies_ms.sort_by(f64::total_cmp);
                    let report = serde_json::json!({
                        "fixture_only": true,
                        "architecture": std::env::consts::ARCH,
                        "os": std::env::consts::OS,
                        "available_parallelism": std::thread::available_parallelism().map_or(1, usize::from),
                        "threads": cli.threads,
                        "samples": cli.samples,
                        "dimensions": dimension,
                        "hidden_units": hidden,
                        "outputs": output_count,
                        "candidate_count": candidate_count,
                        "estimated_workspace_bytes": estimated_workspace_bytes,
                        "training_ms": training_ms,
                        "preprocessing_ms": preprocessing_ms,
                        "artifact_write_ms": artifact_write_ms,
                        "artifact_load_ms": artifact_load_ms,
                        "process_peak_rss_bytes": process_peak_rss_bytes(),
                        "batch_inference_ms": batch_elapsed_ms,
                        "inference_p50_ms": percentile(&latencies_ms, 0.50),
                        "inference_p95_ms": percentile(&latencies_ms, 0.95),
                        "inference_p99_ms": percentile(&latencies_ms, 0.99),
                        "requests_per_second": *candidate_count as f64 / (batch_elapsed_ms / 1_000.0).max(f64::MIN_POSITIVE),
                        "build_feature": "elm",
                    });
                    println!("{}", serde_json::to_string(&report)?);
                    reports.push(report);
                }
                let training_features = features.clone();
                let training_targets = targets.clone();
                let contention_schema = schema.clone();
                let contention_options = fit_options.clone();
                let contention_started = Instant::now();
                let training = tokio::task::spawn_blocking(move || {
                    ElmArtifact::fit_regression(
                        contention_schema,
                        &training_features,
                        &training_targets,
                        None,
                        contention_options,
                    )
                });
                let tasks = probes.iter().map(|row| {
                    inference.predict(model.clone(), row.clone(), Duration::from_secs(30))
                });
                let (results, trained) = tokio::join!(join_all(tasks), training);
                let trained = trained??;
                trained.verify()?;
                let contention_wall_ms = contention_started.elapsed().as_secs_f64() * 1_000.0;
                let mut contention_latencies_ms = results
                    .into_iter()
                    .map(|result| result.map(|item| item.elapsed.as_secs_f64() * 1_000.0))
                    .collect::<Result<Vec<_>, _>>()?;
                contention_latencies_ms.sort_by(f64::total_cmp);
                let contention_report = serde_json::json!({
                    "case": "training_contention",
                    "fixture_only": true,
                    "architecture": std::env::consts::ARCH,
                    "os": std::env::consts::OS,
                    "threads": cli.threads,
                    "samples": cli.samples,
                    "dimensions": dimension,
                    "hidden_units": hidden,
                    "outputs": output_count,
                    "concurrent_inference_requests": probes.len(),
                    "training_contention_wall_ms": contention_wall_ms,
                    "inference_p50_ms": percentile(&contention_latencies_ms, 0.50),
                    "inference_p95_ms": percentile(&contention_latencies_ms, 0.95),
                    "inference_p99_ms": percentile(&contention_latencies_ms, 0.99),
                    "process_peak_rss_bytes": process_peak_rss_bytes(),
                    "build_feature": "elm",
                });
                println!("{}", serde_json::to_string(&contention_report)?);
                reports.push(contention_report);
            }
        }
    }
    eprintln!(
        "completed {} fixture-only benchmark cases; these timings are not production qualification",
        reports.len()
    );
    Ok(())
}

fn process_peak_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let kilobytes = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:")?.split_whitespace().next())?
        .parse::<u64>()
        .ok()?;
    kilobytes.checked_mul(1024)
}

fn parse_list(raw: &str, minimum: usize, maximum: usize) -> anyhow::Result<Vec<usize>> {
    let values = raw
        .split(',')
        .map(|item| item.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty()
        || values
            .iter()
            .any(|value| !(minimum..=maximum).contains(value))
    {
        anyhow::bail!("benchmark list contains an empty or out-of-range value");
    }
    Ok(values)
}

fn fixture_matrix(rows: usize, columns: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|row| {
            (0..columns)
                .map(|column| {
                    let phase = (row as f64 * 0.017) + (column as f64 * 0.31);
                    phase.sin() + 0.25 * phase.cos()
                })
                .collect()
        })
        .collect()
}

fn fixture_targets(features: &[Vec<f64>], outputs: usize) -> Vec<Vec<f64>> {
    features
        .iter()
        .enumerate()
        .map(|(row, values)| {
            (0..outputs)
                .map(|output| {
                    values[output % values.len()] * (output + 1) as f64
                        + (row as f64 * 0.013).sin() * 0.1
                })
                .collect()
        })
        .collect()
}

fn workspace_estimate(samples: usize, dimensions: usize, hidden: usize, outputs: usize) -> usize {
    samples
        .saturating_mul(dimensions.saturating_add(hidden))
        .saturating_mul(8)
        .saturating_add(hidden.saturating_mul(hidden).saturating_mul(8))
        .saturating_add(hidden.saturating_mul(outputs).saturating_mul(8))
}

fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let index = ((values.len() - 1) as f64 * fraction).ceil() as usize;
    values.get(index).copied()
}
