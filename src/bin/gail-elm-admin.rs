//! Local, OS-authorised inspection and rollback for a single-host ELM registry.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use gail::elm::ModelRegistry;

#[derive(Debug, Parser)]
#[command(name = "gail-elm-admin")]
#[command(about = "Inspect or roll back a local native ELM registry")]
struct Cli {
    #[arg(long, default_value = "data/elm")]
    registry: PathBuf,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    maximum_artifact_bytes: usize,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Status,
    Rollback {
        #[arg(long)]
        task: String,
        #[arg(long)]
        expected_revision: u64,
        #[arg(long)]
        idempotency_key: String,
        #[arg(long)]
        reason: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let registry = ModelRegistry::open(cli.registry, cli.maximum_artifact_bytes)?;
    match cli.command {
        Command::Status => println!("{}", serde_json::to_string_pretty(&registry.status_json())?),
        Command::Rollback {
            task,
            expected_revision,
            idempotency_key,
            reason,
        } => {
            let event =
                registry.rollback_champion(&task, &idempotency_key, expected_revision, &reason)?;
            println!("{}", serde_json::to_string_pretty(&event)?);
        }
    }
    Ok(())
}
