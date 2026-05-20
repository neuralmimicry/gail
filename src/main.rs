use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use gail::{app, config::GailConfig, mirror_worker, orchestration::GailService, trainer_worker};
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Debug, Parser)]
#[command(name = "gail")]
#[command(about = "Gateway AI and neuromorphic middleware")]
#[command(version)]
struct Cli {
    #[arg(long, env = "GAIL_CONFIG", default_value = "gail.yaml")]
    config: PathBuf,
    #[arg(long, env = "GAIL_ROLE", value_enum, default_value_t = RuntimeRole::Serve)]
    role: RuntimeRole,
}

#[derive(Clone, Debug, ValueEnum)]
enum RuntimeRole {
    Serve,
    MirrorWorker,
    TrainerWorker,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app_version = env!("CARGO_PKG_VERSION");
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).with_target(false).init();

    let cli = Cli::parse();
    tracing::info!(
        version = app_version,
        config = %cli.config.display(),
        role = ?cli.role,
        "Gail starting"
    );
    let config = GailConfig::load(&cli.config)?;
    match cli.role {
        RuntimeRole::Serve => {
            let service = GailService::new(config.clone()).await?;
            let router = app::build_router(service);
            let listener = tokio::net::TcpListener::bind(config.server.bind_addr.as_str()).await?;
            tracing::info!(
                version = app_version,
                bind_addr = %config.server.bind_addr,
                "Gail listening"
            );
            axum::serve(listener, router)
                .with_graceful_shutdown(app::shutdown_signal())
                .await?;
        }
        RuntimeRole::MirrorWorker => {
            if !config.mirror_worker.enabled {
                tracing::warn!(
                    "mirror worker role selected but mirror_worker.enabled=false; exiting"
                );
                return Ok(());
            }
            mirror_worker::run(config).await?;
        }
        RuntimeRole::TrainerWorker => {
            if !config.trainer.enabled {
                tracing::warn!("trainer worker role selected but trainer.enabled=false; exiting");
                return Ok(());
            }
            trainer_worker::run(config).await?;
        }
    }
    Ok(())
}
