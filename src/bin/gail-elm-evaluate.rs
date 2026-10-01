//! Evaluate one explicitly permitted numeric manifest and export its evidence.

use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use clap::Parser;
use gail::{config::GailConfig, elm};
use sha2::Digest;

#[derive(Debug, Parser)]
#[command(name = "gail-elm-evaluate")]
#[command(
    about = "Train, compare and export a native ELM candidate from a permitted numeric manifest"
)]
struct Cli {
    /// Versioned JSON dataset manifest with explicit permission and provenance.
    #[arg(long)]
    manifest: PathBuf,
    /// Directory in which the immutable candidate and evaluation report are written.
    #[arg(long)]
    output_dir: PathBuf,
    /// Gail configuration containing the task authority and promotion policy.
    #[arg(long)]
    config: Option<PathBuf>,
}

fn main() {
    let status = match run(Cli::parse()) {
        Ok(qualified) => {
            if qualified {
                0
            } else {
                2
            }
        }
        Err(error) => {
            eprintln!("ELM evaluation failed: {error}");
            1
        }
    };
    std::process::exit(status);
}

fn run(cli: Cli) -> anyhow::Result<bool> {
    let default = gail::elm_config::ElmConfig::default();
    let elm_config = if let Some(path) = cli.config.as_deref() {
        GailConfig::load(path)?.elm
    } else {
        default
    };
    let budgets = &elm_config.budgets;
    let manifest =
        elm::data::DatasetManifest::load(&cli.manifest, elm_config.lifecycle.artifact_max_bytes)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as f64;
    let limits = elm::data::TrainingLimits {
        max_samples: budgets.max_samples,
        max_features: budgets.max_features,
        max_hidden_units: budgets.max_hidden_units,
        max_memory_mb: budgets.max_training_memory_mb,
        max_training_threads: budgets.max_training_threads,
        max_training_seconds: budgets.max_training_seconds,
        minimum_label_age_seconds: elm_config.data.minimum_label_age_seconds,
    };
    let mut evaluation = elm::data::train_and_evaluate(&manifest, now, &limits)?;
    let qualification = elm::lifecycle::evaluate_qualification(
        &elm_config,
        &manifest.task_id,
        &evaluation.artifact,
        &evaluation.report,
    );
    evaluation.report.qualified = qualification.qualified;
    evaluation.report.qualification_reasons = qualification.reason_codes;
    evaluation.report.report_sha256.clear();
    let canonical = serde_json::to_vec(&evaluation.report)?;
    evaluation.report.report_sha256 = hex::encode(sha2::Sha256::digest(canonical));

    fs::create_dir_all(&cli.output_dir)?;
    let artifact_path = cli.output_dir.join(format!(
        "{}.elm.json",
        evaluation.artifact.metadata.model_id
    ));
    let report_path = cli.output_dir.join(format!(
        "{}.evaluation.json",
        evaluation.artifact.metadata.model_id
    ));
    elm::save_artifact(
        &artifact_path,
        &evaluation.artifact,
        elm_config.lifecycle.artifact_max_bytes,
    )?;
    let report = serde_json::to_vec_pretty(&evaluation.report)?;
    write_atomic(&report_path, &report)?;

    let loaded = elm::load_artifact(&artifact_path, elm_config.lifecycle.artifact_max_bytes)?;
    if loaded.content_sha256 != evaluation.artifact.content_sha256 {
        anyhow::bail!("exported artifact failed content-hash round-trip verification");
    }
    println!("artifact={}", artifact_path.display());
    println!("report={}", report_path.display());
    println!("qualified={}", evaluation.report.qualified);
    if !evaluation.report.qualification_reasons.is_empty() {
        println!(
            "qualification_reasons={}",
            evaluation.report.qualification_reasons.join(",")
        );
    }
    Ok(evaluation.report.qualified)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("report path has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        uuid::Uuid::new_v4()
    ));
    let mut file = File::create(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}
