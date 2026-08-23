use std::{env, path::PathBuf, time::Duration};

use reqwest::Client;

use crate::{
    aarnn_bridge::{AarnnMirrorClient, AarnnMirrorExchange},
    config::GailConfig,
    errors::{GailError, Result},
    hardware::{detect_hardware, log_hardware_profile},
    llm_ledger,
    models::{AarnnMirrorDirection, ChatMessage, MessageContent, ProviderCompletionRequest},
    provider_admission::{
        AdmissionObservation, admission_endpoint_matches, record_observation, response_quality,
        token_similarity,
    },
    providers::build_adapter,
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
        AarnnMirrorClient::from_config(&config, client.clone(), &specialists).ok_or_else(|| {
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
            let mirror_pending = !matches!(
                entry.mirror_status.as_deref(),
                Some("mirrored") | Some("failed")
            );
            // Replay prompt-side stimulation for durability/recovery, even when
            // inline mirroring was skipped or timed out on the request path.
            if mirror_pending
                && bridge.should_mirror_input()
                && !entry.prompt_text.trim().is_empty()
            {
                let trace = bridge
                    .mirror(build_exchange(
                        &entry,
                        AarnnMirrorDirection::Input,
                        entry.provider_requested.as_deref(),
                        entry.model_requested.as_deref(),
                        entry.prompt_text.as_str(),
                    ))
                    .await;
                if let Some(error) = mirror_semantic_error(&trace) {
                    errors.push(format!("input mirror: {error}"));
                }
            }
            // Replay response-side stimulation and candidate request flow using
            // the resolved provider/model when available.
            let mut aarnn_trace = None;
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
                aarnn_trace = Some(trace.clone());
                if mirror_pending && let Some(error) = mirror_semantic_error(&trace) {
                    errors.push(format!("output mirror: {error}"));
                }
            }
            let validation_error = validate_and_record(
                &config,
                &client,
                &dsn,
                &bridge,
                &entry,
                aarnn_trace.as_ref(),
            )
            .await;
            if let Err(error) = validation_error {
                tracing::warn!(
                    ledger_id = entry.id,
                    error = %error,
                    "comparative validation failed; the next scheduled sample will retry"
                );
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

async fn validate_and_record(
    config: &GailConfig,
    client: &Client,
    dsn: &str,
    bridge: &AarnnMirrorClient,
    entry: &llm_ledger::LedgerInteraction,
    aarnn_trace: Option<&crate::models::AarnnMirrorInvocationTrace>,
) -> std::result::Result<(), String> {
    if !config.comparative_validation.enabled {
        llm_ledger::mark_validation(
            dsn,
            entry.id,
            "disabled",
            None,
            config
                .comparative_validation
                .sample_interval_seconds
                .max(60),
        )
        .await
        .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let Some(baseline) = entry
        .response_text
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(());
    };
    if baseline.starts_with("<redacted chars=") {
        llm_ledger::mark_validation(
            dsn,
            entry.id,
            "skipped_no_content",
            Some("ledger response content is redacted; enable retain_content_for_validation"),
            config.comparative_validation.sample_interval_seconds,
        )
        .await
        .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let prompt = entry.prompt_text.trim();
    if prompt.is_empty() || prompt.starts_with("<redacted chars=") {
        llm_ledger::mark_validation(
            dsn,
            entry.id,
            "skipped_no_content",
            Some("ledger prompt content is redacted or empty"),
            config.comparative_validation.sample_interval_seconds,
        )
        .await
        .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let expected_json = entry
        .request_category
        .as_deref()
        .is_some_and(|value| value.to_ascii_lowercase().contains("json"));
    let max_tokens = entry
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("request_max_tokens"))
        .and_then(serde_json::Value::as_u64)
        .map(|value| value.clamp(1, 4096) as u32)
        .unwrap_or(512);
    let temperature = entry
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("request_temperature"))
        .and_then(serde_json::Value::as_f64)
        .map(|value| value.clamp(0.0, 2.0) as f32)
        .unwrap_or(0.2);
    let baseline_quality = response_quality(baseline, expected_json);
    let mut sampled = false;
    let mut errors = Vec::new();

    if let Some((model_version, endpoint, model, api_key)) = active_trained_target(config) {
        let profile = crate::config::ProviderProfile {
            name: "comparative-trained".to_string(),
            provider_type: "openai".to_string(),
            model: Some(model.clone()),
            api_key,
            base_url: Some(endpoint.clone()),
            source: Some("comparative_validation_trained".to_string()),
            ..crate::config::ProviderProfile::default()
        };
        let adapter = build_adapter(client.clone(), &profile).map_err(|error| error.to_string())?;
        let request = ProviderCompletionRequest {
            provider: "openai".to_string(),
            model: Some(model.clone()),
            api_key: profile.api_key.clone(),
            access_token: None,
            base_url: Some(endpoint.clone()),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text(prompt.to_string()),
            }],
            system: entry.system_prompt.clone(),
            max_tokens: Some(max_tokens),
            temperature: Some(temperature),
            timeout_seconds: Some(90),
            reasoning_effort: None,
            request_category: entry.request_category.clone(),
            workflow: Some(entry.workflow.clone()),
            role: Some(entry.role.clone()),
            min_model_size_b: None,
            strict_no_downgrade: Some(false),
            source: Some("comparative_validation".to_string()),
            request_profile: None,
        };
        sampled = true;
        match adapter.complete(&request).await {
            Ok(response) => {
                let candidate_quality = response_quality(&response.text, expected_json);
                let similarity = token_similarity(&response.text, baseline);
                let useful =
                    candidate_quality >= config.comparative_validation.minimum_candidate_quality;
                let decision = record_observation(
                    dsn,
                    &AdmissionObservation {
                        kind: "trained".to_string(),
                        endpoint,
                        model,
                        model_version,
                        useful,
                        candidate_quality,
                        baseline_quality,
                        similarity,
                        latency_ms: Some(response.latency_ms),
                        error: (!useful)
                            .then(|| "candidate response failed usefulness gate".to_string()),
                    },
                    &config.comparative_validation,
                )
                .await
                .map_err(|error| error.to_string())?;
                tracing::info!(
                    ledger_id = entry.id,
                    admitted = decision.admitted,
                    samples = decision.sample_count,
                    candidate_quality,
                    baseline_quality,
                    similarity,
                    "recorded trained-model comparative validation"
                );
            }
            Err(error) => {
                let (model_version, endpoint, model, _) = active_trained_target(config).unwrap();
                record_observation(
                    dsn,
                    &AdmissionObservation {
                        kind: "trained".to_string(),
                        endpoint,
                        model,
                        model_version,
                        useful: false,
                        candidate_quality: 0.0,
                        baseline_quality,
                        similarity: 0.0,
                        latency_ms: None,
                        error: Some(error.to_string()),
                    },
                    &config.comparative_validation,
                )
                .await
                .map_err(|error| error.to_string())?;
                errors.push(error.to_string());
            }
        }
    }

    if let Some(trace) = aarnn_trace {
        sampled = true;
        let evaluation = bridge.evaluate_candidate(trace, baseline, prompt);
        let candidate_text = trace
            .candidate
            .as_ref()
            .and_then(|candidate| candidate.reply_text.as_deref())
            .unwrap_or("");
        let useful = trace.accepted
            && trace.error.is_none()
            && trace
                .candidate
                .as_ref()
                .is_some_and(|candidate| candidate.usable)
            && response_quality(candidate_text, expected_json)
                >= config.comparative_validation.minimum_candidate_quality;
        let decision = record_observation(
            dsn,
            &AdmissionObservation {
                kind: "aarnn".to_string(),
                endpoint: trace.endpoint.clone(),
                model: bridge.response_model().to_string(),
                model_version: format!("decoder-{}", evaluation.decoder_version),
                useful,
                candidate_quality: evaluation.quality_score,
                baseline_quality,
                similarity: evaluation.agreement_score,
                latency_ms: Some(trace.latency_ms),
                error: trace.error.clone(),
            },
            &config.comparative_validation,
        )
        .await
        .map_err(|error| error.to_string())?;
        tracing::info!(
            ledger_id = entry.id,
            admitted = decision.admitted,
            samples = decision.sample_count,
            quality = evaluation.quality_score,
            similarity = evaluation.agreement_score,
            "recorded AARNN comparative validation"
        );
    }

    let status = if sampled {
        "sampled"
    } else {
        "skipped_no_target"
    };
    llm_ledger::mark_validation(
        dsn,
        entry.id,
        status,
        (!errors.is_empty()).then(|| errors.join(" | ")).as_deref(),
        config.comparative_validation.sample_interval_seconds,
    )
    .await
    .map_err(|error| error.to_string())?;
    Ok(())
}

fn active_trained_target(config: &GailConfig) -> Option<(String, String, String, Option<String>)> {
    let path = PathBuf::from(&config.trainer.output_root).join("active_snapshot.json");
    let raw = std::fs::read_to_string(path).ok()?;
    let pointer: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let version = pointer.get("snapshot_id")?.as_str()?.trim();
    let target = pointer.get("serving_target")?.as_object()?;
    let endpoint = target.get("endpoint")?.as_str()?.trim();
    let model = target
        .get("model_alias")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            pointer
                .get("model_alias")
                .and_then(serde_json::Value::as_str)
        })?
        .trim();
    if version.is_empty() || endpoint.is_empty() || model.is_empty() {
        return None;
    }
    let api_key = config
        .providers
        .iter()
        .find(|profile| {
            profile
                .base_url
                .as_deref()
                .is_some_and(|base| admission_endpoint_matches(base, endpoint))
                && profile
                    .api_key
                    .as_deref()
                    .is_some_and(|key| !key.trim().is_empty())
        })
        .and_then(|profile| profile.api_key.clone())
        .or_else(|| env::var("GAIL_LLAMACPP_API_KEY").ok());
    Some((
        version.to_string(),
        endpoint.to_string(),
        model.to_string(),
        api_key,
    ))
}

/// Transport acceptance is not semantic success.  In particular, AARNN can
/// acknowledge an output request while its decoder reports
/// `network_output_unavailable`; closing the ledger row in that case would
/// permanently discard the only replay opportunity for that interaction.
fn mirror_semantic_error(trace: &crate::models::AarnnMirrorInvocationTrace) -> Option<String> {
    if let Some(error) = trace.error.as_deref() {
        return Some(error.to_string());
    }
    if !trace.accepted {
        return Some("AARNN transport response was not accepted".to_string());
    }
    if matches!(trace.direction, AarnnMirrorDirection::Output) {
        let Some(candidate) = trace.candidate.as_ref() else {
            return Some("output candidate missing after accepted transport".to_string());
        };
        if !candidate.usable {
            return Some(format!(
                "output candidate unusable: {}",
                candidate
                    .source
                    .as_deref()
                    .unwrap_or("network_output_unavailable")
            ));
        }
        if candidate
            .reply_text
            .as_deref()
            .is_none_or(|reply| reply.trim().is_empty())
        {
            return Some("output candidate has no reply text".to_string());
        }
    }
    None
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
