use std::time::Duration;

use reqwest::Client;

use crate::{
    aarnn_bridge::{AarnnMirrorClient, AarnnMirrorExchange},
    config::GailConfig,
    errors::{GailError, Result},
    hardware::{detect_hardware, log_hardware_profile},
    llm_ledger,
    models::AarnnMirrorDirection,
    specialists::build_specialist_engines,
};

pub async fn run(config: GailConfig) -> Result<()> {
    let Some(dsn) = config.storage.postgres_dsn.clone() else {
        return Err(GailError::invalid_config(
            "mirror worker requires storage.postgres_dsn (or GAIL_POSTGRES_DSN)",
        ));
    };
    llm_ledger::initialize_schema(&dsn).await.map_err(|error| {
        GailError::invalid_config(format!("failed to initialise LLM ledger schema: {error}"))
    })?;
    let client = Client::builder()
        .use_rustls_tls()
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(8)
        .tcp_keepalive(Duration::from_secs(30))
        .user_agent(format!("gail-mirror-worker/{}", env!("CARGO_PKG_VERSION")))
        .build()?;
    let specialists = build_specialist_engines(&config, client.clone());
    let bridge =
        AarnnMirrorClient::from_config(&config, client, &specialists).ok_or_else(|| {
            GailError::invalid_config(
                "mirror worker requires aarnn_bridge.enabled=true and a valid bridge endpoint",
            )
        })?;
    let hardware = detect_hardware().await;
    log_hardware_profile("mirror_worker", &hardware);
    tracing::info!(
        poll_interval_ms = config.mirror_worker.poll_interval_ms,
        batch_size = config.mirror_worker.batch_size,
        max_attempts = config.mirror_worker.max_attempts,
        retry_backoff_seconds = config.mirror_worker.retry_backoff_seconds,
        aarnn_endpoint = %bridge.endpoint(),
        "Gail mirror worker started"
    );
    let poll_interval = Duration::from_millis(config.mirror_worker.poll_interval_ms);
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("mirror worker received shutdown signal");
                break;
            }
            _ = tokio::time::sleep(poll_interval) => {}
        }
        let entries = match llm_ledger::fetch_pending_mirror(&dsn, config.mirror_worker.batch_size)
            .await
        {
            Ok(entries) => entries,
            Err(error) => {
                tracing::warn!(error = %error, "mirror worker failed to fetch pending ledger rows");
                continue;
            }
        };
        if entries.is_empty() {
            continue;
        }
        tracing::info!(
            count = entries.len(),
            "mirror worker processing ledger batch"
        );
        for entry in entries {
            let mut errors = Vec::new();
            // Replay prompt-side stimulation for durability/recovery, even when
            // inline mirroring was skipped or timed out on the request path.
            if bridge.should_mirror_input() && !entry.prompt_text.trim().is_empty() {
                let trace = bridge
                    .mirror(build_exchange(
                        &entry,
                        AarnnMirrorDirection::Input,
                        entry.provider_requested.as_deref(),
                        entry.model_requested.as_deref(),
                        entry.prompt_text.as_str(),
                    ))
                    .await;
                if let Some(error) = trace.error {
                    errors.push(format!("input mirror: {error}"));
                }
            }
            // Replay response-side stimulation and candidate request flow using
            // the resolved provider/model when available.
            if bridge.should_mirror_output()
                && let Some(response_text) = entry.response_text.as_deref()
                && !response_text.trim().is_empty()
            {
                let trace = bridge
                    .mirror(build_exchange(
                        &entry,
                        AarnnMirrorDirection::Output,
                        entry
                            .provider_resolved
                            .as_deref()
                            .or(entry.provider_requested.as_deref()),
                        entry
                            .model_resolved
                            .as_deref()
                            .or(entry.model_requested.as_deref()),
                        response_text,
                    ))
                    .await;
                if let Some(error) = trace.error {
                    errors.push(format!("output mirror: {error}"));
                }
            }
            if errors.is_empty() {
                if let Err(error) =
                    llm_ledger::mark_mirror_success(&dsn, entry.id, "mirrored").await
                {
                    tracing::warn!(
                        error = %error,
                        ledger_id = entry.id,
                        "mirror worker failed to mark ledger row as mirrored"
                    );
                }
                continue;
            }
            let reason = errors.join(" | ");
            tracing::warn!(
                ledger_id = entry.id,
                request_id = %entry.request_id,
                error = %reason,
                "mirror worker failed to mirror one or more exchanges"
            );
            if let Err(error) = llm_ledger::mark_mirror_retry(
                &dsn,
                entry.id,
                reason.as_str(),
                config.mirror_worker.max_attempts,
                config.mirror_worker.retry_backoff_seconds,
            )
            .await
            {
                tracing::warn!(
                    error = %error,
                    ledger_id = entry.id,
                    "mirror worker failed to mark mirror retry state"
                );
            }
        }
    }
    Ok(())
}

fn build_exchange(
    entry: &llm_ledger::LedgerInteraction,
    direction: AarnnMirrorDirection,
    provider: Option<&str>,
    model: Option<&str>,
    text: &str,
) -> AarnnMirrorExchange {
    AarnnMirrorExchange {
        request_id: entry.request_id.clone(),
        // Keep replay rows usable even when upstream callers omitted a stable
        // conversation id by falling back to request id.
        conversation_id: if entry.conversation_id.trim().is_empty() {
            entry.request_id.clone()
        } else {
            entry.conversation_id.clone()
        },
        workflow: entry.workflow.clone(),
        role: entry.role.clone(),
        direction,
        provider: provider.map(ToOwned::to_owned),
        model: model.map(ToOwned::to_owned),
        request_category: entry.request_category.clone(),
        system: entry.system_prompt.clone(),
        prompt_text: Some(entry.prompt_text.clone()),
        text: text.to_string(),
        message_roles: entry.message_roles.clone(),
    }
}
