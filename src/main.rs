use std::path::PathBuf;

use clap::Parser;
use gail::{app, config::GailConfig, orchestration::GailService};
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Debug, Parser)]
#[command(name = "gail")]
#[command(about = "Gateway AI and neuromorphic middleware")]
#[command(version)]
struct Cli {
    #[arg(long, env = "GAIL_CONFIG", default_value = "gail.yaml")]
    config: PathBuf,
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
        "Gail starting"
    );
    let config = GailConfig::load(&cli.config)?;
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
    Ok(())
}
