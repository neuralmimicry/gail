use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use reqwest::Client;
use serde_json::{Value, json};
use tokio::task;
use uuid::Uuid;

use crate::{
    adaptive_schema,
    aer::{decode_spikes, decode_spikes_auto, encode_spikes, payload_hex, spikes_from_floats},
    config::{GailConfig, SpecialistProfile},
    errors::{GailError, Result},
    models::{
        NeuromorphicAnalyzeRequest, NeuromorphicPredictRequest, NeuromorphicPredictResponse,
        SpecialistAnalysisResponse,
    },
};

const DEFAULT_AARNN_REPO_ROOT: &str = "/home/pbisaacs/Developer/neuralmimicry/aarnn_rust";
const DEFAULT_SOCKET_PATH: &str = "/tmp/aarnn_rust.nn";
const NEUROMORPHIC_KEYWORDS: &[&str] = &[
    "aarnn",
    "aer",
    "spiking neural",
    "spiking-neural",
    "spike train",
    "snn",
    "neuromorphic",
    "celegans",
    "drosophila",
];

const AARNN_GUIDANCE_LINES: &[&str] = &[
    "Prefer AARNN-grown SNN designs when the task explicitly calls for spiking or neuromorphic networks.",
    "Use `AER1` payloads for spike exchange with the AARNN UDS/runtime path.",
];

const GENERIC_AER_GUIDANCE_LINES: &[&str] = &[
    "Prefer spike-native or neuromorphic designs when the task explicitly calls for SNN/AER systems.",
    "Use `AER1` payloads when the attached runtime or translation layer expects AER-based communication.",
];

#[derive(Clone, Debug)]
struct AarnnPrediction {
    score: f64,
    fired: bool,
    mode: String,
    threshold: f64,
    input_spikes: Vec<u8>,
    output_spikes: Vec<u8>,
    aer_payload_hex: String,
    raw: Value,
}

#[derive(Clone, Debug)]
pub struct SpecialistEngine {
    client: Client,
    profile: SpecialistProfile,
}

impl SpecialistEngine {
    pub fn new(client: Client, mut profile: SpecialistProfile) -> Self {
        normalize_specialist_profile(&mut profile);
        Self { client, profile }
    }

    pub fn name(&self) -> &str {
        self.profile.name.as_str()
    }

    pub fn engine_type(&self) -> &str {
        self.profile.engine_type.as_str()
    }

    pub fn profile(&self) -> &SpecialistProfile {
        &self.profile
    }

    pub fn matches_name(&self, name: &str) -> bool {
        let normalized = name.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return false;
        }
        self.profile.name.trim().eq_ignore_ascii_case(&normalized)
            || self
                .profile
                .engine_type
                .trim()
                .eq_ignore_ascii_case(&normalized)
    }

    pub fn supports_role(&self, role: &str) -> bool {
        if self.profile.roles.is_empty() {
            return true;
        }
        self.profile
            .roles
            .iter()
            .any(|item| item.eq_ignore_ascii_case(role))
    }

    pub fn is_available(&self) -> bool {
        self.profile
            .endpoint
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
            || self
                .profile
                .socket_path
                .as_deref()
                .is_some_and(|value| Path::new(value).exists())
            || self
                .profile
                .repo_root
                .as_deref()
                .is_some_and(|value| Path::new(value).exists())
    }

    pub async fn health_check(&self) -> Value {
        let started = std::time::Instant::now();
        let health = if let Some(endpoint) = normalized_url(self.profile.endpoint.as_deref()) {
            let url = format!("{endpoint}/healthz");
            match self
                .client
                .get(&url)
                .timeout(Duration::from_secs_f64(
                    self.profile.timeout_seconds.max(0.2),
                ))
                .send()
                .await
            {
                Ok(response) => {
                    let status = response.status();
                    let payload = response
                        .json::<Value>()
                        .await
                        .unwrap_or_else(|_| Value::Object(Default::default()));
                    if status.is_success() {
                        adaptive_schema::observe_success(
                            &format!("specialist:{}", self.profile.name),
                            "GET",
                            &url,
                            "health",
                            &payload,
                        )
                        .await;
                    } else {
                        adaptive_schema::observe_failure(
                            &format!("specialist:{}", self.profile.name),
                            "GET",
                            &url,
                            "health",
                            Some(status.as_u16()),
                            &payload.to_string(),
                        )
                        .await;
                    }
                    json!({
                        "ok": status.is_success(),
                        "mode": if status.is_success() { "http" } else { "http_error" },
                        "endpoint": endpoint,
                        "details": payload,
                    })
                }
                Err(error) => {
                    adaptive_schema::observe_failure(
                        &format!("specialist:{}", self.profile.name),
                        "GET",
                        &url,
                        "health",
                        None,
                        &error.to_string(),
                    )
                    .await;
                    self.heuristic_health(Some(error.to_string()))
                }
            }
        } else if let Some(socket_path) = self.profile.socket_path.as_ref() {
            match self.socket_handshake(socket_path.clone()).await {
                Ok(payload) => json!({
                    "ok": true,
                    "mode": "uds",
                    "socket_path": socket_path,
                    "details": payload,
                }),
                Err(error) => self.heuristic_health(Some(error.to_string())),
            }
        } else {
            self.heuristic_health(None)
        };

        let latency_ms = started.elapsed().as_millis() as u64;
        let mut object = as_object(health);
        object.insert("latency_ms".to_string(), json!(latency_ms));
        Value::Object(object)
    }

    pub async fn summary(&self, probe_health: bool) -> Value {
        let health = if probe_health {
            self.health_check().await
        } else {
            self.basic_health_snapshot()
        };
        json!({
            "name": self.profile.name,
            "type": self.profile.engine_type,
            "enabled": true,
            "available": self.is_available(),
            "roles": self.profile.roles,
            "specialties": self.profile.specialties,
            "repo_root": self.profile.repo_root,
            "endpoint": normalized_url(self.profile.endpoint.as_deref()),
            "socket_path": self.profile.socket_path,
            "sensory_size": self.profile.sensory_size,
            "output_size": self.profile.output_size,
            "aer_sensory_base": self.profile.aer_sensory_base,
            "aer_output_base": self.profile.aer_output_base,
            "description": self.profile.description,
            "weight": self.profile.weight,
            "health": merge_probed_flag(health, probe_health),
        })
    }

    pub async fn summary_from_profile(
        client: Client,
        profile: SpecialistProfile,
        probe_health: bool,
    ) -> Value {
        let engine = Self::new(client, profile);
        if engine.is_available() {
            return engine.summary(probe_health).await;
        }
        json!({
            "name": engine.profile.name,
            "type": engine.profile.engine_type,
            "enabled": true,
            "available": false,
            "roles": engine.profile.roles,
            "specialties": engine.profile.specialties,
            "repo_root": engine.profile.repo_root,
            "endpoint": normalized_url(engine.profile.endpoint.as_deref()),
            "socket_path": engine.profile.socket_path,
            "sensory_size": engine.profile.sensory_size,
            "output_size": engine.profile.output_size,
            "aer_sensory_base": engine.profile.aer_sensory_base,
            "aer_output_base": engine.profile.aer_output_base,
            "description": engine.profile.description,
            "weight": engine.profile.weight,
            "health": {
                "ok": false,
                "mode": "unavailable",
                "details": {"reason": "No endpoint, socket, or repository was detected for this engine"},
                "probed": probe_health,
            },
        })
    }

    pub async fn predict_request(
        &self,
        request: &NeuromorphicPredictRequest,
    ) -> Result<NeuromorphicPredictResponse> {
        let prediction = self.predict_inputs(&request.inputs).await?;
        Ok(NeuromorphicPredictResponse {
            score: round_to(prediction.score, 6),
            fired: prediction.fired,
            mode: prediction.mode,
            threshold: prediction.threshold,
            input_spikes: prediction.input_spikes,
            output_spikes: prediction.output_spikes,
            aer_payload_hex: prediction.aer_payload_hex,
            raw: prediction.raw,
        })
    }

    pub async fn analyze_task(&self, request: &NeuromorphicAnalyzeRequest) -> Result<Value> {
        let cleaned = compact_text(request.text.as_str());
        let lowered = cleaned.to_ascii_lowercase();
        let hits = keyword_hits(&cleaned, &self.profile.keyword_hints);
        let aer_hits = hits
            .iter()
            .filter(|(keyword, _)| keyword.contains("aer") || keyword.contains("spike"))
            .map(|(_, count)| *count as f64)
            .sum::<f64>();
        let workflow = request.workflow.as_deref().unwrap_or("general");
        let role = request.role.as_deref().unwrap_or("general");
        let feature_inputs = vec![
            (cleaned.len() as f64 / 1200.0).min(1.0) as f32,
            (hits.values().sum::<usize>() as f64 / 6.0).min(1.0) as f32,
            if matches!(
                workflow,
                "project_solver" | "playground_plan" | "assistant_requirements"
            ) {
                1.0
            } else {
                0.4
            },
            if matches!(role, "planner" | "reviewer" | "researcher" | "assistant") {
                1.0
            } else {
                0.3
            },
            (aer_hits / 3.0).min(1.0) as f32,
        ];
        let prediction = self.predict_inputs(&feature_inputs).await?;
        let relevant = !hits.is_empty()
            || prediction.fired
            || lowered.contains("neuromorphic")
            || lowered.contains("spiking");
        Ok(json!({
            "engine": self.profile.engine_type,
            "engine_name": self.profile.name,
            "relevant": relevant,
            "score": round_to(prediction.score, 6),
            "fired": prediction.fired,
            "mode": prediction.mode,
            "threshold": prediction.threshold,
            "keyword_hits": hits,
            "inputs": feature_inputs.iter().map(|value| round_to(*value as f64, 6)).collect::<Vec<_>>(),
            "input_spikes": prediction.input_spikes,
            "output_spikes": prediction.output_spikes,
            "aer_payload_hex": prediction.aer_payload_hex,
            "workflow": workflow,
            "role": role,
            "repo_root": self.profile.repo_root,
            "endpoint": normalized_url(self.profile.endpoint.as_deref()),
            "socket_path": self.profile.socket_path,
            "aer_sensory_base": self.profile.aer_sensory_base,
            "aer_output_base": self.profile.aer_output_base,
            "roles": self.profile.roles,
            "specialties": self.profile.specialties,
            "description": self.profile.description,
            "weight": self.profile.weight,
            "raw": prediction.raw,
        }))
    }

    pub fn format_prompt_context(&self, analysis: &Value) -> String {
        if !analysis
            .get("relevant")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return String::new();
        }
        let mode = analysis
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("offline_heuristic");
        let location = normalized_url(self.profile.endpoint.as_deref())
            .or_else(|| self.profile.socket_path.clone())
            .or_else(|| self.profile.repo_root.clone())
            .unwrap_or_else(|| "not configured".to_string());
        let mut lines = vec![
            "Neuromorphic engine support is available for this task.".to_string(),
            format!("- Engine: {} ({mode})", self.profile.name),
            format!("- Engine type: {}", self.profile.engine_type),
            format!("- Location: {location}"),
            format!(
                "- AER sensory/output bases: {}/{}",
                self.profile.aer_sensory_base, self.profile.aer_output_base
            ),
        ];
        if let Some(description) = self.profile.description.as_deref() {
            if !description.trim().is_empty() {
                lines.push(format!("- Description: {description}"));
            }
        }
        for guidance in &self.profile.guidance_lines {
            let cleaned = guidance.trim();
            if cleaned.is_empty() {
                continue;
            }
            if cleaned.starts_with('-') {
                lines.push(cleaned.to_string());
            } else {
                lines.push(format!("- {cleaned}"));
            }
        }
        if self.profile.prefer_aarnn_designs
            && self
                .profile
                .specialties
                .iter()
                .any(|item| item.eq_ignore_ascii_case("aarnn"))
            && !lines
                .iter()
                .any(|line| line.to_ascii_uppercase().contains("AARNN"))
        {
            lines.push(
                "- Prefer AARNN-grown SNN designs when the task explicitly calls for spiking or neuromorphic networks.".to_string(),
            );
        }
        lines.push(format!(
            "- Routing score: {}",
            analysis.get("score").cloned().unwrap_or(Value::Null)
        ));
        lines.push(format!(
            "- Input AER payload sample (hex): {}",
            analysis
                .get("aer_payload_hex")
                .and_then(Value::as_str)
                .unwrap_or_default()
        ));
        lines.join("\n").trim().to_string()
    }

    async fn predict_inputs(&self, inputs: &[f32]) -> Result<AarnnPrediction> {
        if let Some(endpoint) = normalized_url(self.profile.endpoint.as_deref()) {
            match self.predict_http(&endpoint, inputs).await {
                Ok(prediction) => return Ok(prediction),
                Err(error) => {
                    tracing::debug!(engine = %self.profile.name, error = %error, "specialist HTTP predict failed; falling back")
                }
            }
        }
        if let Some(socket_path) = self.profile.socket_path.as_ref() {
            match self.predict_socket(socket_path.clone(), inputs).await {
                Ok(prediction) => return Ok(prediction),
                Err(error) => {
                    tracing::debug!(engine = %self.profile.name, error = %error, "specialist UDS predict failed; falling back")
                }
            }
        }
        Ok(self.predict_heuristic(inputs))
    }

    async fn predict_http(&self, endpoint: &str, inputs: &[f32]) -> Result<AarnnPrediction> {
        let url = format!("{endpoint}/predict");
        let response = match self
            .client
            .post(&url)
            .timeout(Duration::from_secs_f64(
                self.profile.timeout_seconds.max(0.2),
            ))
            .json(&json!({"inputs": inputs}))
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                adaptive_schema::observe_failure(
                    &format!("specialist:{}", self.profile.name),
                    "POST",
                    &url,
                    "predict",
                    None,
                    &error.to_string(),
                )
                .await;
                return Err(error.into());
            }
        };
        let status = response.status();
        let payload = response
            .json::<Value>()
            .await
            .unwrap_or_else(|_| Value::Object(Default::default()));
        if !status.is_success() {
            adaptive_schema::observe_failure(
                &format!("specialist:{}", self.profile.name),
                "POST",
                &url,
                "predict",
                Some(status.as_u16()),
                &payload.to_string(),
            )
            .await;
            return Err(GailError::upstream(
                self.profile.name.clone(),
                Some(status),
                payload.to_string(),
            ));
        }
        adaptive_schema::observe_success(
            &format!("specialist:{}", self.profile.name),
            "POST",
            &url,
            "predict",
            &payload,
        )
        .await;
        let input_spikes = spikes_from_floats(inputs, self.profile.spike_threshold);
        let aer_payload = encode_spikes(now_ts_us(), self.profile.aer_sensory_base, &input_spikes);
        let score = payload
            .get("score")
            .and_then(Value::as_f64)
            .unwrap_or_else(|| average(inputs));
        let threshold = payload
            .get("threshold")
            .and_then(Value::as_f64)
            .unwrap_or(self.profile.spike_threshold);
        let fired = payload
            .get("fired")
            .and_then(Value::as_bool)
            .unwrap_or(score >= threshold);
        let output_spikes = if fired { vec![1] } else { Vec::new() };
        Ok(AarnnPrediction {
            score,
            fired,
            mode: "http".to_string(),
            threshold,
            input_spikes,
            output_spikes,
            aer_payload_hex: payload_hex(&aer_payload),
            raw: payload,
        })
    }

    async fn predict_socket(&self, socket_path: String, inputs: &[f32]) -> Result<AarnnPrediction> {
        let timeout = self.profile.timeout_seconds.max(0.2);
        let input_spikes = spikes_from_floats(inputs, self.profile.spike_threshold);
        let payload = encode_spikes(now_ts_us(), self.profile.aer_sensory_base, &input_spikes);
        let aer_payload_hex = payload_hex(&payload);
        let output_base = self.profile.aer_output_base;
        let output_size = self.profile.output_size;
        let response = task::spawn_blocking(move || socket_predict(socket_path, timeout, payload))
            .await
            .map_err(|error| GailError::upstream("specialist", None, error.to_string()))??;
        let mut output_spikes = decode_spikes(&response, output_base, output_size)
            .or_else(|_| decode_spikes_auto(&response, output_base))?;
        if !output_spikes.iter().any(|value| *value > 0) {
            output_spikes = decode_spikes_auto(&response, output_base).unwrap_or(output_spikes);
        }
        let spike_total = output_spikes.iter().filter(|value| **value > 0).count();
        let score =
            spike_total as f64 / output_spikes.len().max(self.profile.output_size).max(1) as f64;
        Ok(AarnnPrediction {
            score,
            fired: spike_total > 0,
            mode: "uds".to_string(),
            threshold: self.profile.spike_threshold,
            input_spikes,
            output_spikes,
            aer_payload_hex,
            raw: json!({
                "socket_path": self.profile.socket_path,
                "response_bytes": response.len(),
            }),
        })
    }

    fn predict_heuristic(&self, inputs: &[f32]) -> AarnnPrediction {
        let input_spikes = spikes_from_floats(inputs, self.profile.spike_threshold);
        let aer_payload = encode_spikes(now_ts_us(), self.profile.aer_sensory_base, &input_spikes);
        let score = average(inputs);
        let fired = score >= self.profile.spike_threshold;
        AarnnPrediction {
            score,
            fired,
            mode: "offline_heuristic".to_string(),
            threshold: self.profile.spike_threshold,
            input_spikes,
            output_spikes: if fired { vec![1] } else { Vec::new() },
            aer_payload_hex: payload_hex(&aer_payload),
            raw: json!({"repo_root": self.profile.repo_root}),
        }
    }

    fn heuristic_health(&self, reason: Option<String>) -> Value {
        if self
            .profile
            .repo_root
            .as_deref()
            .is_some_and(|value| Path::new(value).exists())
        {
            json!({
                "ok": true,
                "mode": "offline_heuristic",
                "details": {
                    "reason": reason.unwrap_or_else(|| {
                        "AARNN repository present; using offline heuristic routing and AER translation".to_string()
                    }),
                },
            })
        } else if self.profile.socket_path.is_some() {
            json!({
                "ok": false,
                "mode": "uds_missing",
                "details": {
                    "socket_path": self.profile.socket_path,
                    "reason": reason,
                },
            })
        } else {
            json!({
                "ok": false,
                "mode": "unavailable",
                "details": {
                    "reason": reason.unwrap_or_else(|| {
                        "No AARNN endpoint, socket, or repository detected".to_string()
                    }),
                },
            })
        }
    }

    fn basic_health_snapshot(&self) -> Value {
        if let Some(endpoint) = normalized_url(self.profile.endpoint.as_deref()) {
            return json!({
                "ok": true,
                "mode": "http_configured",
                "details": {"endpoint": endpoint},
            });
        }
        if self
            .profile
            .socket_path
            .as_deref()
            .is_some_and(|value| Path::new(value).exists())
        {
            return json!({
                "ok": true,
                "mode": "uds_ready",
                "details": {"socket_path": self.profile.socket_path},
            });
        }
        if self
            .profile
            .repo_root
            .as_deref()
            .is_some_and(|value| Path::new(value).exists())
        {
            return json!({
                "ok": true,
                "mode": "offline_heuristic",
                "details": {"repo_root": self.profile.repo_root},
            });
        }
        self.heuristic_health(None)
    }

    async fn socket_handshake(&self, socket_path: String) -> Result<Value> {
        let sensory_size = self.profile.sensory_size;
        let output_size = self.profile.output_size;
        let timeout = self.profile.timeout_seconds.max(0.2);
        task::spawn_blocking(move || {
            socket_handshake(socket_path, sensory_size, output_size, timeout)
        })
        .await
        .map_err(|error| GailError::upstream("specialist", None, error.to_string()))?
    }
}

pub fn build_specialist_engines(config: &GailConfig, client: Client) -> Vec<SpecialistEngine> {
    let mut engines = config
        .specialists
        .iter()
        .cloned()
        .map(|profile| SpecialistEngine::new(client.clone(), profile))
        .filter(|engine| engine.is_available())
        .collect::<Vec<_>>();

    if !engines
        .iter()
        .any(|engine| engine.profile.engine_type.eq_ignore_ascii_case("aarnn"))
    {
        if let Some(legacy) = legacy_aarnn_engine(client) {
            engines.push(legacy);
        }
    }
    engines
}

pub async fn specialist_engine_summaries(
    config: &GailConfig,
    client: Client,
    probe_health: bool,
) -> Vec<Value> {
    let mut summaries = Vec::new();
    for profile in &config.specialists {
        summaries.push(
            SpecialistEngine::summary_from_profile(client.clone(), profile.clone(), probe_health)
                .await,
        );
    }
    if !config
        .specialists
        .iter()
        .any(|profile| profile.engine_type.eq_ignore_ascii_case("aarnn"))
    {
        if let Some(legacy) = legacy_aarnn_engine(client.clone()) {
            summaries.push(legacy.summary(probe_health).await);
        }
    }
    summaries
}

pub async fn analyze_specialist_engines(
    engines: &[SpecialistEngine],
    request: &NeuromorphicAnalyzeRequest,
) -> SpecialistAnalysisResponse {
    let _workflow = request.workflow.as_deref().unwrap_or("general");
    let role = request.role.as_deref().unwrap_or("general");
    let active = engines
        .iter()
        .filter(|engine| engine.supports_role(role))
        .cloned()
        .collect::<Vec<_>>();
    if active.is_empty() {
        return SpecialistAnalysisResponse {
            relevant: false,
            engine_count: 0,
            engines: Vec::new(),
            selected: None,
            combined_specialties: Vec::new(),
            context_blocks: Vec::new(),
            context: String::new(),
        };
    }

    let mut join_set = tokio::task::JoinSet::new();
    for engine in active.clone() {
        let request = request.clone();
        join_set.spawn(async move {
            let analysis = engine.analyze_task(&request).await.unwrap_or_else(|error| {
                json!({
                    "engine": engine.engine_type(),
                    "engine_name": engine.name(),
                    "relevant": false,
                    "error": error.to_string(),
                    "specialties": engine.profile.specialties,
                    "roles": engine.profile.roles,
                    "weight": engine.profile.weight,
                })
            });
            (engine, analysis)
        });
    }

    let mut paired = Vec::new();
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(pair) => paired.push(pair),
            Err(error) => tracing::warn!(error = %error, "specialist analysis task join failed"),
        }
    }

    paired.sort_by(|(left_engine, left), (right_engine, right)| {
        let left_score = score_analysis(left_engine, left);
        let right_score = score_analysis(right_engine, right);
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .get("relevant")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                    .cmp(
                        &left
                            .get("relevant")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                    )
            })
            .then_with(|| right_engine.name().cmp(left_engine.name()))
    });

    let analyses = paired
        .iter()
        .map(|(_, analysis)| analysis.clone())
        .collect::<Vec<_>>();
    let relevant = analyses
        .iter()
        .filter(|analysis| {
            analysis
                .get("relevant")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut context_blocks = Vec::new();
    let mut seen_contexts = HashSet::new();
    for (engine, analysis) in &paired {
        if !analysis
            .get("relevant")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            continue;
        }
        let context = engine.format_prompt_context(analysis);
        if !context.is_empty() && seen_contexts.insert(context.clone()) {
            context_blocks.push(context);
        }
    }

    let combined_specialties = relevant
        .iter()
        .flat_map(|analysis| {
            analysis
                .get("specialties")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|value| value.as_str().map(|item| item.trim().to_ascii_lowercase()))
        .filter(|item| !item.is_empty())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let mut combined_specialties = combined_specialties;
    combined_specialties.sort();

    SpecialistAnalysisResponse {
        relevant: !relevant.is_empty(),
        engine_count: analyses.len(),
        engines: analyses.clone(),
        selected: relevant
            .first()
            .cloned()
            .or_else(|| analyses.first().cloned()),
        combined_specialties,
        context_blocks: context_blocks.clone(),
        context: context_blocks.join("\n\n"),
    }
}

fn normalize_specialist_profile(profile: &mut SpecialistProfile) {
    profile.name = if profile.name.trim().is_empty() {
        if profile.engine_type.eq_ignore_ascii_case("aarnn") {
            "AARNN".to_string()
        } else {
            "SNN/AER Specialist".to_string()
        }
    } else {
        profile.name.trim().to_string()
    };
    profile.engine_type = if profile.engine_type.trim().is_empty() {
        if profile
            .specialties
            .iter()
            .any(|item| item.eq_ignore_ascii_case("aarnn"))
            || profile.name.to_ascii_lowercase().contains("aarnn")
        {
            "aarnn".to_string()
        } else {
            "snn_aer".to_string()
        }
    } else {
        profile.engine_type.trim().to_ascii_lowercase()
    };
    profile.roles = normalized_tags(&profile.roles);
    if profile.specialties.is_empty() {
        profile.specialties = if profile.engine_type == "aarnn" {
            vec!["aarnn", "snn", "neuromorphic", "aer"]
                .into_iter()
                .map(ToOwned::to_owned)
                .collect()
        } else {
            vec!["snn", "neuromorphic", "aer"]
                .into_iter()
                .map(ToOwned::to_owned)
                .collect()
        };
    }
    profile.specialties = normalized_tags(&profile.specialties);
    if profile.keyword_hints.is_empty() {
        profile.keyword_hints = profile.specialties.clone();
        profile.keyword_hints.push(profile.name.clone());
        profile.keyword_hints.push(profile.engine_type.clone());
    }
    profile.keyword_hints = normalized_tags(&profile.keyword_hints);
    if profile.guidance_lines.is_empty() {
        profile.guidance_lines = if profile.engine_type == "aarnn" {
            AARNN_GUIDANCE_LINES
                .iter()
                .map(|item| (*item).to_string())
                .collect()
        } else {
            GENERIC_AER_GUIDANCE_LINES
                .iter()
                .map(|item| (*item).to_string())
                .collect()
        };
    }
    profile.endpoint = normalized_url(profile.endpoint.as_deref());
    profile.socket_path = profile
        .socket_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    profile.repo_root = profile
        .repo_root
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    profile.timeout_seconds = profile.timeout_seconds.max(0.2);
    profile.health_ttl_seconds = profile.health_ttl_seconds.max(5.0);
    profile.spike_threshold = profile.spike_threshold.clamp(0.0, 1.0);
    profile.weight = round_to(profile.weight, 6);
}

fn normalized_tags(items: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    for item in items {
        let cleaned = item.trim().to_ascii_lowercase();
        if cleaned.is_empty() || !seen.insert(cleaned.clone()) {
            continue;
        }
        normalized.push(cleaned);
    }
    normalized
}

fn normalized_url(raw: Option<&str>) -> Option<String> {
    let value = raw?.trim();
    if value.is_empty() {
        return None;
    }
    if value.contains("://") {
        Some(value.trim_end_matches('/').to_string())
    } else {
        Some(format!("http://{}", value.trim_end_matches('/')))
    }
}

fn keyword_hits(text: &str, keywords: &[String]) -> HashMap<String, usize> {
    let lowered = text.to_ascii_lowercase();
    let configured = if keywords.is_empty() {
        NEUROMORPHIC_KEYWORDS
            .iter()
            .map(|item| (*item).to_string())
            .collect::<Vec<_>>()
    } else {
        keywords.to_vec()
    };
    configured
        .into_iter()
        .filter_map(|keyword| {
            let count = lowered.matches(keyword.as_str()).count();
            (count > 0).then_some((keyword, count))
        })
        .collect()
}

fn compact_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn average(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64
}

fn round_to(value: f64, precision: i32) -> f64 {
    let factor = 10_f64.powi(precision);
    (value * factor).round() / factor
}

fn now_ts_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_micros() as u64)
        .unwrap_or(0)
}

fn merge_probed_flag(health: Value, probed: bool) -> Value {
    let mut object = as_object(health);
    object.insert("probed".to_string(), json!(probed));
    Value::Object(object)
}

fn as_object(value: Value) -> serde_json::Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => {
            let mut map = serde_json::Map::new();
            map.insert("value".to_string(), other);
            map
        }
    }
}

fn score_analysis(engine: &SpecialistEngine, analysis: &Value) -> f64 {
    let relevant = if analysis
        .get("relevant")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        1.0
    } else {
        0.0
    };
    let score = analysis.get("score").and_then(Value::as_f64).unwrap_or(0.0);
    relevant * 1000.0 + score + engine.profile.weight
}

fn legacy_aarnn_engine(client: Client) -> Option<SpecialistEngine> {
    let enabled = env_bool_any(&["GAIL_AARNN_ENABLED", "REFINER_AARNN_ENABLED"], true);
    if !enabled {
        return None;
    }
    let repo_root = env::var("GAIL_AARNN_REPO_ROOT")
        .ok()
        .or_else(|| env::var("REFINER_AARNN_REPO_ROOT").ok())
        .or_else(|| {
            Path::new(DEFAULT_AARNN_REPO_ROOT)
                .exists()
                .then(|| DEFAULT_AARNN_REPO_ROOT.to_string())
        });
    let endpoint = env::var("GAIL_AARNN_ENDPOINT")
        .ok()
        .or_else(|| env::var("REFINER_AARNN_ENDPOINT").ok());
    let socket_path = env::var("GAIL_AARNN_SOCKET")
        .ok()
        .or_else(|| env::var("REFINER_AARNN_SOCKET").ok())
        .or_else(|| {
            Path::new(DEFAULT_SOCKET_PATH)
                .exists()
                .then(|| DEFAULT_SOCKET_PATH.to_string())
        });
    let profile = SpecialistProfile {
        name: "AARNN".to_string(),
        engine_type: "aarnn".to_string(),
        endpoint,
        socket_path,
        repo_root,
        sensory_size: env_int_any(
            &["GAIL_AARNN_SENSORY_SIZE", "REFINER_AARNN_SENSORY_SIZE"],
            32,
        ) as usize,
        output_size: env_int_any(&["GAIL_AARNN_OUTPUT_SIZE", "REFINER_AARNN_OUTPUT_SIZE"], 16)
            as usize,
        aer_sensory_base: env_int_any(
            &[
                "GAIL_AARNN_AER_SENSORY_BASE",
                "REFINER_AARNN_AER_SENSORY_BASE",
            ],
            4096,
        ) as u32,
        aer_output_base: env_int_any(
            &[
                "GAIL_AARNN_AER_OUTPUT_BASE",
                "REFINER_AARNN_AER_OUTPUT_BASE",
            ],
            16384,
        ) as u32,
        timeout_seconds: env_float_any(&["GAIL_AARNN_TIMEOUT", "REFINER_AARNN_TIMEOUT"], 2.0),
        health_ttl_seconds: 300.0,
        spike_threshold: env_float_any(
            &[
                "GAIL_AARNN_SPIKE_THRESHOLD",
                "REFINER_AARNN_SPIKE_THRESHOLD",
            ],
            0.5,
        ),
        roles: vec!["planner", "reviewer", "researcher", "assistant"]
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        specialties: vec!["aarnn", "snn", "neuromorphic", "aer"]
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        keyword_hints: Vec::new(),
        guidance_lines: Vec::new(),
        description: None,
        weight: 0.0,
        prefer_aarnn_designs: true,
        allow_offline_heuristic: true,
    };
    let engine = SpecialistEngine::new(client, profile);
    engine.is_available().then_some(engine)
}

fn env_bool_any(names: &[&str], default: bool) -> bool {
    for name in names {
        if let Ok(value) = env::var(name) {
            return matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }
    }
    default
}

fn env_int_any(names: &[&str], default: u64) -> u64 {
    for name in names {
        if let Ok(value) = env::var(name) {
            if let Ok(parsed) = value.trim().parse::<u64>() {
                return parsed;
            }
        }
    }
    default
}

fn env_float_any(names: &[&str], default: f64) -> f64 {
    for name in names {
        if let Ok(value) = env::var(name) {
            if let Ok(parsed) = value.trim().parse::<f64>() {
                return parsed;
            }
        }
    }
    default
}

#[cfg(unix)]
fn socket_handshake(
    socket_path: String,
    sensory_size: usize,
    output_size: usize,
    timeout_seconds: f64,
) -> Result<Value> {
    use std::os::unix::net::UnixDatagram;

    if !Path::new(&socket_path).exists() {
        return Err(GailError::not_found(format!(
            "AARNN socket does not exist: {socket_path}"
        )));
    }
    let client_path = format!("/tmp/gail_aarnn_{}_health.sock", Uuid::new_v4());
    let timeout = Duration::from_secs_f64(timeout_seconds.max(0.2));
    let datagram = UnixDatagram::bind(&client_path)?;
    datagram.set_read_timeout(Some(timeout))?;
    datagram.set_write_timeout(Some(timeout))?;
    let payload = json!({"expected_s": sensory_size, "expected_o": output_size}).to_string();
    datagram.send_to(payload.as_bytes(), &socket_path)?;
    let mut buffer = vec![0u8; 4096];
    let size = datagram.recv(&mut buffer)?;
    let _ = fs::remove_file(&client_path);
    let response = String::from_utf8_lossy(&buffer[..size]).to_string();
    let payload =
        serde_json::from_str::<Value>(&response).unwrap_or_else(|_| json!({"raw": response}));
    Ok(payload)
}

#[cfg(not(unix))]
fn socket_handshake(
    _socket_path: String,
    _sensory_size: usize,
    _output_size: usize,
    _timeout_seconds: f64,
) -> Result<Value> {
    Err(GailError::bad_request(
        "Unix datagram sockets are not supported on this platform",
    ))
}

#[cfg(unix)]
fn socket_predict(socket_path: String, timeout_seconds: f64, payload: Vec<u8>) -> Result<Vec<u8>> {
    use std::os::unix::net::UnixDatagram;

    if !Path::new(&socket_path).exists() {
        return Err(GailError::not_found(format!(
            "AARNN socket does not exist: {socket_path}"
        )));
    }
    let client_path = format!("/tmp/gail_aarnn_{}_predict.sock", Uuid::new_v4());
    let timeout = Duration::from_secs_f64(timeout_seconds.max(0.2));
    let datagram = UnixDatagram::bind(&client_path)?;
    datagram.set_read_timeout(Some(timeout))?;
    datagram.set_write_timeout(Some(timeout))?;
    datagram.send_to(&payload, &socket_path)?;
    let mut buffer = vec![0u8; 65_536];
    let size = datagram.recv(&mut buffer)?;
    let _ = fs::remove_file(&client_path);
    buffer.truncate(size);
    Ok(buffer)
}

#[cfg(not(unix))]
fn socket_predict(
    _socket_path: String,
    _timeout_seconds: f64,
    _payload: Vec<u8>,
) -> Result<Vec<u8>> {
    Err(GailError::bad_request(
        "Unix datagram sockets are not supported on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn heuristic_analysis_exposes_aer_context() {
        let client = Client::builder().build().expect("client");
        let engine = SpecialistEngine::new(
            client,
            SpecialistProfile {
                repo_root: Some("/tmp".to_string()),
                sensory_size: 8,
                output_size: 4,
                ..SpecialistProfile::default()
            },
        );
        let analysis = engine
            .analyze_task(&NeuromorphicAnalyzeRequest {
                text: "Implement an AARNN-generated SNN with an AER communication layer."
                    .to_string(),
                workflow: Some("project_solver".to_string()),
                role: Some("planner".to_string()),
            })
            .await
            .expect("analysis");
        assert!(
            analysis
                .get("relevant")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        );
        assert_eq!(
            analysis.get("mode").and_then(Value::as_str),
            Some("offline_heuristic")
        );
        let context = engine.format_prompt_context(&analysis);
        assert!(context.contains("AER"));
        assert!(context.contains("AARNN"));
    }
}
