//! Gail LLM<->SNN mirror bridge.
//!
//! This module mirrors LLM prompt/response text into AARNN by projecting text
//! into sensory spikes, encoding them as AER payloads, and posting the exchange
//! to `/api/llm/mirror`.

use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use reqwest::{
    Client,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use serde::Serialize;
use serde_json::Value;
use tokio::{
    sync::{Mutex as AsyncMutex, Semaphore, mpsc, oneshot},
    time::{sleep, timeout},
};
use tracing::{debug, info, warn};

use crate::{
    adaptive_schema,
    aer::{AerEvent, encode_events, encode_spikes, payload_hex},
    config::{AarnnBridgeConfig, GailConfig, SpecialistProfile},
    models::{
        AarnnBridgeStatus, AarnnMirrorCandidate, AarnnMirrorDirection, AarnnMirrorInvocationTrace,
        AarnnMirrorRequest, AarnnMirrorResponse, AarnnMirrorTrace, AarnnResponsePreference,
    },
    specialists::SpecialistEngine,
};

const AARNN_MIRROR_PATH: &str = "/api/llm/mirror";
const AARNN_RESPONSE_MODEL: &str = "aarnn-snn-aer-bridge";
const AARNN_PERIPHERAL_SESSIONS_PATH: &str = "/api/peripheral/sessions";
const PERIPHERAL_SESSION_HEADER: &str = "x-aarnn-peripheral-session";
/// AARNN caps session TTL at 900 s and offers no renewal.
const PERIPHERAL_SESSION_TTL_SECS: u64 = 900;
/// Re-create a session once fewer than this many seconds remain.
const PERIPHERAL_SESSION_REFRESH_MARGIN: Duration = Duration::from_secs(60);
/// How long to remember that AARNN refused/does not support sessions
/// (auth mode `none` -> 403 "disabled", older AARNN -> 404/405).
const PERIPHERAL_SESSION_UNSUPPORTED_TTL: Duration = Duration::from_secs(60);
/// Back-off after a transient session-creation failure (timeout, 5xx).
const PERIPHERAL_SESSION_FAILURE_TTL: Duration = Duration::from_secs(5);
/// AER base for the auditory sensory region fed by nmstt speech mirroring:
/// between the text sensory region (4096 + sensory_size) and outputs (16384).
pub const SPEECH_AER_BASE: u32 = 8192;
/// Upper bounds that keep one utterance a single, bounded AER batch.
pub const SPEECH_MAX_BANDS: u32 = 256;
pub const SPEECH_MAX_FRAMES: usize = 6000;
pub const SPEECH_MAX_EVENTS: usize = 60_000;

/// One utterance of audio (as cochlea-like band spike frames) paired with the
/// text that goes with it, so AARNN can learn sound-to-word associations.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct SpeechMirrorPair {
    pub pair_id: String,
    /// "stt" (heard audio + transcript) or "tts" (spoken audio + source text).
    pub source: String,
    pub text: String,
    pub frame_ms: u32,
    pub bands: u32,
    /// Sparse frames: band indices that spiked in each frame, in time order.
    pub frames: Vec<Vec<u16>>,
    #[serde(default)]
    pub lang: Option<String>,
}

impl SpeechMirrorPair {
    pub fn validate(&self) -> Result<(), String> {
        if self.pair_id.trim().is_empty() || self.pair_id.len() > 128 {
            return Err("pair_id must be 1-128 characters".into());
        }
        if !matches!(self.source.as_str(), "stt" | "tts") {
            return Err("source must be stt or tts".into());
        }
        if self.text.trim().is_empty() {
            return Err("text is required".into());
        }
        if !(1..=100).contains(&self.frame_ms) || self.bands == 0 || self.bands > SPEECH_MAX_BANDS {
            return Err("frame_ms must be 1-100 and bands 1-256".into());
        }
        if self.frames.len() > SPEECH_MAX_FRAMES {
            return Err(format!("at most {SPEECH_MAX_FRAMES} frames"));
        }
        let events: usize = self.frames.iter().map(Vec::len).sum();
        if events == 0 || events > SPEECH_MAX_EVENTS {
            return Err(format!("event count must be 1-{SPEECH_MAX_EVENTS}"));
        }
        if self
            .frames
            .iter()
            .flatten()
            .any(|b| u32::from(*b) >= self.bands)
        {
            return Err("band index out of range".into());
        }
        Ok(())
    }

    /// Timed AER events rooted at the auditory base, plus per-band activity
    /// (spike counts, saturating) as the record's sensory vector.
    pub fn to_aer(&self, t0_us: u64) -> (Vec<AerEvent>, Vec<u8>) {
        let mut events = Vec::new();
        let mut activity = vec![0u8; self.bands as usize];
        for (i, frame) in self.frames.iter().enumerate() {
            let ts_us = t0_us + i as u64 * u64::from(self.frame_ms) * 1000;
            for band in frame {
                events.push(AerEvent {
                    ts_us,
                    addr: SPEECH_AER_BASE + u32::from(*band),
                    value: 1,
                });
                let a = &mut activity[*band as usize];
                *a = a.saturating_add(1);
            }
        }
        (events, activity)
    }
}

#[derive(Debug)]
struct AarnnMirrorJob {
    exchange: AarnnMirrorExchange,
    response: Option<oneshot::Sender<AarnnMirrorInvocationTrace>>,
}

#[derive(Clone, Debug)]
pub struct AarnnMirrorExchange {
    // LLM-side context captured from orchestration/ledger before translation.
    pub request_id: String,
    pub conversation_id: String,
    pub workflow: String,
    pub role: String,
    pub direction: AarnnMirrorDirection,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub request_category: Option<String>,
    pub system: Option<String>,
    pub prompt_text: Option<String>,
    pub text: String,
    pub message_roles: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct AarnnMirrorClient {
    client: Client,
    endpoint: String,
    access_token: Option<String>,
    timeout: Duration,
    queue_tx: mpsc::Sender<AarnnMirrorJob>,
    queue_capacity: usize,
    worker_count: usize,
    enqueue_timeout: Duration,
    candidate_wait_timeout: Duration,
    mirror_input: bool,
    mirror_output: bool,
    request_candidate_reply: bool,
    response_preference: AarnnResponsePreference,
    candidate_confidence_threshold: f64,
    candidate_min_reply_chars: usize,
    network_id: Option<String>,
    node_id: Option<String>,
    max_text_chars: usize,
    sensory_size: usize,
    aer_sensory_base: u32,
    aer_output_base: u32,
    request_max_attempts: usize,
    request_backoff: Duration,
    request_backoff_max: Duration,
    audit_enabled: bool,
    audit_log_llm_prompts: bool,
    audit_log_llm_responses: bool,
    audit_store_llm_content: bool,
    audit_log_aer_payloads: bool,
    audit_max_chars: usize,
    peripheral_sessions: Arc<PeripheralSessions>,
}

/// Cached state of the AARNN peripheral session for one network_id.
#[derive(Debug, Default)]
enum PeripheralSessionSlot {
    #[default]
    Empty,
    Active {
        session_id: String,
        expires_at: Instant,
    },
    /// Sessions are disabled (auth mode `none`), denied, or unsupported by an
    /// older AARNN: mirror without the header until `until`.
    Unsupported { until: Instant },
    /// Creation failed transiently; mirror without the header until `until`.
    Failed { until: Instant },
}

/// Per-network peripheral session cache. Each network has its own async
/// mutex which is held across session creation, so concurrent mirrors for the
/// same network single-flight onto one `POST /api/peripheral/sessions`.
#[derive(Debug, Default)]
struct PeripheralSessions {
    slots: Mutex<HashMap<String, Arc<AsyncMutex<PeripheralSessionSlot>>>>,
    /// Set once the "sessions unsupported" warning was emitted; later
    /// occurrences log at debug to avoid spamming every 60 s.
    warned_unsupported: std::sync::atomic::AtomicBool,
}

impl PeripheralSessions {
    fn slot(&self, network_id: &str) -> Arc<AsyncMutex<PeripheralSessionSlot>> {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        slots.entry(network_id.to_string()).or_default().clone()
    }
}

#[derive(Debug, serde::Deserialize)]
struct PeripheralSessionReply {
    session_id: String,
    #[serde(default)]
    expires_at_unix_secs: Option<u64>,
}

static PROMOTION_HISTORY: Lazy<Mutex<HashMap<String, VecDeque<f64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static EVALUATION_METRICS: Lazy<Mutex<HashMap<String, (u64, f64, f64, f64, f64, u64)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static DECODER_METRICS: Lazy<Mutex<(u64, u64, u64)>> = Lazy::new(|| Mutex::new((0, 0, 0)));
static MIRROR_RECONCILIATION: Lazy<Mutex<(HashMap<String, bool>, u64, u64, u64)>> =
    Lazy::new(|| Mutex::new((HashMap::new(), 0, 0, 0)));

#[derive(Clone, Debug, Serialize)]
pub struct AarnnPairedEvaluation {
    pub task: String,
    pub observed: bool,
    pub agreement_score: f64,
    pub relevance_score: f64,
    pub confidence_score: f64,
    pub quality_score: f64,
    pub sustained_samples: usize,
    pub eligible: bool,
    pub decoder_version: u64,
    pub decoder_mapped_neurons: usize,
    pub network_neurons: u64,
}

impl AarnnMirrorClient {
    pub fn from_config(
        config: &GailConfig,
        client: Client,
        specialists: &[SpecialistEngine],
    ) -> Option<Self> {
        let bridge = &config.aarnn_bridge;
        if !bridge.enabled {
            return None;
        }
        let endpoint = resolve_endpoint(bridge, specialists, &config.specialists)?;
        let transport = resolve_transport_profile(specialists, &config.specialists);
        let queue_capacity = bridge.queue_capacity.clamp(8, 32_768);
        let worker_count = bridge.worker_count.clamp(1, 128);
        let request_max_attempts = env_usize_clamped("GAIL_AARNN_MIRROR_MAX_ATTEMPTS", 3, 1, 10);
        let request_backoff_ms = env_u64_clamped("GAIL_AARNN_MIRROR_BACKOFF_MS", 300, 10, 30_000);
        let request_backoff_max_ms = env_u64_clamped(
            "GAIL_AARNN_MIRROR_BACKOFF_MAX_MS",
            request_backoff_ms.saturating_mul(8),
            request_backoff_ms,
            120_000,
        );
        let (queue_tx, queue_rx) = mpsc::channel(queue_capacity);
        let mirror = Self {
            client,
            endpoint,
            access_token: bridge.access_token.clone(),
            timeout: Duration::from_secs_f64(bridge.timeout_seconds.max(0.2)),
            queue_tx,
            queue_capacity,
            worker_count,
            enqueue_timeout: Duration::from_millis(bridge.enqueue_timeout_ms.clamp(1, 10_000)),
            candidate_wait_timeout: Duration::from_millis(
                bridge.candidate_wait_timeout_ms.min(30_000),
            ),
            mirror_input: bridge.mirror_input,
            mirror_output: bridge.mirror_output,
            request_candidate_reply: bridge.request_candidate_reply,
            response_preference: bridge.response_preference.clone(),
            candidate_confidence_threshold: bridge.candidate_confidence_threshold.clamp(0.0, 1.0),
            candidate_min_reply_chars: bridge.candidate_min_reply_chars.max(1),
            network_id: bridge.network_id.clone(),
            node_id: bridge.node_id.clone(),
            max_text_chars: bridge.max_text_chars.clamp(128, 65_536),
            sensory_size: transport.sensory_size,
            aer_sensory_base: transport.aer_sensory_base,
            aer_output_base: transport.aer_output_base,
            request_max_attempts,
            request_backoff: Duration::from_millis(request_backoff_ms),
            request_backoff_max: Duration::from_millis(request_backoff_max_ms),
            audit_enabled: config.audit_logging.enabled,
            audit_log_llm_prompts: config.audit_logging.log_llm_prompts,
            audit_log_llm_responses: config.audit_logging.log_llm_responses,
            audit_store_llm_content: config.audit_logging.store_llm_content,
            audit_log_aer_payloads: config.audit_logging.log_aer_payloads,
            audit_max_chars: config.audit_logging.max_chars.clamp(1, 262_144),
            peripheral_sessions: Arc::new(PeripheralSessions::default()),
        };
        mirror.start_worker_bus(queue_rx);
        Some(mirror)
    }

    pub fn status(config: &GailConfig, specialists: &[SpecialistEngine]) -> AarnnBridgeStatus {
        let bridge = &config.aarnn_bridge;
        let transport = resolve_transport_profile(specialists, &config.specialists);
        let endpoint = resolve_endpoint(bridge, specialists, &config.specialists);
        let reason = if !bridge.enabled {
            Some("AARNN mirrored LLM exchange support is disabled.".to_string())
        } else if endpoint.is_none() {
            Some("No AARNN HTTP endpoint is configured for mirrored LLM exchanges.".to_string())
        } else {
            None
        };
        AarnnBridgeStatus {
            enabled: bridge.enabled,
            available: bridge.enabled && endpoint.is_some(),
            endpoint,
            timeout_seconds: bridge.timeout_seconds.max(0.2),
            queue_capacity: bridge.queue_capacity.clamp(8, 32_768),
            worker_count: bridge.worker_count.clamp(1, 128),
            enqueue_timeout_ms: bridge.enqueue_timeout_ms.clamp(1, 10_000),
            candidate_wait_timeout_ms: bridge.candidate_wait_timeout_ms.min(30_000),
            mirror_input: bridge.mirror_input,
            mirror_output: bridge.mirror_output,
            request_candidate_reply: bridge.request_candidate_reply,
            response_preference: bridge.response_preference.clone(),
            candidate_confidence_threshold: bridge.candidate_confidence_threshold.clamp(0.0, 1.0),
            candidate_min_reply_chars: bridge.candidate_min_reply_chars.max(1),
            network_id: bridge.network_id.clone(),
            node_id: bridge.node_id.clone(),
            sensory_size: transport.sensory_size,
            output_size: transport.output_size,
            aer_sensory_base: transport.aer_sensory_base,
            aer_output_base: transport.aer_output_base,
            max_text_chars: bridge.max_text_chars.clamp(128, 65_536),
            reason,
        }
    }

    pub fn should_mirror_input(&self) -> bool {
        self.mirror_input
    }

    pub fn should_mirror_output(&self) -> bool {
        self.mirror_output
    }

    pub fn response_model(&self) -> &'static str {
        AARNN_RESPONSE_MODEL
    }

    pub fn response_preference(&self) -> &AarnnResponsePreference {
        &self.response_preference
    }

    pub fn endpoint(&self) -> &str {
        self.endpoint.as_str()
    }

    pub fn candidate_confidence_threshold(&self) -> f64 {
        self.candidate_confidence_threshold
    }

    pub fn candidate_min_reply_chars(&self) -> usize {
        self.candidate_min_reply_chars
    }

    pub fn candidate_wait_timeout(&self) -> Duration {
        self.candidate_wait_timeout
    }

    pub fn build_trace(
        &self,
        input: Option<AarnnMirrorInvocationTrace>,
        output: Option<AarnnMirrorInvocationTrace>,
    ) -> AarnnMirrorTrace {
        AarnnMirrorTrace {
            enabled: true,
            endpoint: self.endpoint.clone(),
            mirror_input: self.mirror_input,
            mirror_output: self.mirror_output,
            response_preference: self.response_preference.clone(),
            candidate_confidence_threshold: self.candidate_confidence_threshold,
            candidate_min_reply_chars: self.candidate_min_reply_chars,
            input,
            output,
        }
    }

    pub fn should_promote_candidate(
        &self,
        trace: &AarnnMirrorInvocationTrace,
        llm_text: &str,
        prompt_text: &str,
    ) -> bool {
        // Candidate promotion is intentionally strict so Gail only replaces the
        // selected LLM text when AARNN returned a distinct, high-confidence
        // response that cleared configured quality gates.
        let evaluation = self.evaluate_candidate(trace, llm_text, prompt_text);
        if self.response_preference != AarnnResponsePreference::PreferAarnnWhenConfident
            || !evaluation.eligible
        {
            return false;
        }
        let Some(candidate) = trace.candidate.as_ref() else {
            return false;
        };
        if !candidate.usable {
            return false;
        }
        if candidate.source.as_deref() != Some("network_output_decoder") {
            return false;
        }
        let Some(reply_text) = candidate
            .reply_text
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return false;
        };
        if reply_text.chars().count() < self.candidate_min_reply_chars {
            return false;
        }
        let promoted = normalise_for_compare(reply_text) != normalise_for_compare(llm_text);
        if promoted {
            if let Ok(mut metrics) = EVALUATION_METRICS.lock() {
                if let Some(entry) =
                    metrics.get_mut(trace.request_category.as_deref().unwrap_or("general"))
                {
                    entry.5 = entry.5.saturating_add(1);
                }
            }
        }
        promoted
    }

    /// Validate one response before using a durably admitted AARNN decoder.
    /// Durable admission supplies the sustained history; this check still
    /// rejects a bad/empty response on the current request.
    pub fn should_promote_admitted_candidate(
        &self,
        trace: &AarnnMirrorInvocationTrace,
        llm_text: &str,
        prompt_text: &str,
    ) -> bool {
        let evaluation = self.evaluate_candidate(trace, llm_text, prompt_text);
        let Some(candidate) = trace.candidate.as_ref() else {
            return false;
        };
        let Some(reply) = candidate.reply_text.as_deref().map(str::trim) else {
            return false;
        };
        candidate.usable
            && trace.accepted
            && trace.error.is_none()
            && candidate.source.as_deref() == Some("network_output_decoder")
            && reply.chars().count() >= self.candidate_min_reply_chars
            && evaluation.quality_score >= 0.5
            && (evaluation.agreement_score >= 0.35 || evaluation.quality_score >= 0.80)
            && normalise_for_compare(reply) != normalise_for_compare(llm_text)
    }

    pub fn evaluate_candidate(
        &self,
        trace: &AarnnMirrorInvocationTrace,
        llm_text: &str,
        prompt_text: &str,
    ) -> AarnnPairedEvaluation {
        let task = trace
            .request_category
            .clone()
            .unwrap_or_else(|| "general".to_string());
        let candidate = trace.candidate.as_ref();
        let candidate_text = candidate
            .and_then(|value| value.reply_text.as_deref())
            .unwrap_or("");
        let agreement_score = token_overlap(candidate_text, llm_text);
        let relevance_score = token_overlap(candidate_text, prompt_text);
        let confidence_score = candidate.and_then(|value| value.confidence).unwrap_or(0.0);
        if let Some(candidate) = candidate {
            if let Ok(mut decoder) = DECODER_METRICS.lock() {
                decoder.0 = decoder.0.max(candidate.decoder_version);
                decoder.1 = decoder.1.max(candidate.decoder_mapped_neurons as u64);
                decoder.2 = decoder.2.max(candidate.network_neurons);
            }
        }
        let quality_score = if candidate.is_some_and(|value| value.usable)
            && candidate_text.chars().count() >= self.candidate_min_reply_chars
            && candidate
                .is_some_and(|value| value.source.as_deref() == Some("network_output_decoder"))
        {
            (agreement_score * 0.45 + relevance_score * 0.25 + confidence_score * 0.30)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        {
            let mut metrics = EVALUATION_METRICS.lock().expect("evaluation metrics lock");
            let entry = metrics.entry(task.clone()).or_default();
            entry.0 = entry.0.saturating_add(1);
            entry.1 += quality_score;
            entry.2 += agreement_score;
            entry.3 += relevance_score;
            entry.4 += confidence_score;
        }
        let mut history = PROMOTION_HISTORY.lock().expect("promotion history lock");
        let samples = history.entry(task.clone()).or_default();
        samples.push_back(quality_score);
        while samples.len() > 32 {
            samples.pop_front();
        }
        let sustained_samples = samples.len();
        let average = samples.iter().sum::<f64>() / sustained_samples.max(1) as f64;
        let recent =
            samples.iter().rev().take(10).sum::<f64>() / samples.len().min(10).max(1) as f64;
        AarnnPairedEvaluation {
            task,
            observed: candidate.is_some(),
            agreement_score,
            relevance_score,
            confidence_score,
            quality_score,
            sustained_samples,
            eligible: sustained_samples >= 20
                && average >= 0.80
                && recent >= 0.75
                && confidence_score >= self.candidate_confidence_threshold,
            decoder_version: candidate.map(|value| value.decoder_version).unwrap_or(0),
            decoder_mapped_neurons: candidate
                .map(|value| value.decoder_mapped_neurons)
                .unwrap_or(0),
            network_neurons: candidate.map(|value| value.network_neurons).unwrap_or(0),
        }
    }

    pub fn evaluation_prometheus_metrics() -> String {
        let metrics = EVALUATION_METRICS.lock().expect("evaluation metrics lock");
        let mut out = String::from(
            "# HELP gail_aarnn_paired_evaluations_total Paired LLM/AARNN evaluations.\n\
# TYPE gail_aarnn_paired_evaluations_total counter\n\
# HELP gail_aarnn_quality_score_average Average guarded AARNN quality score.\n\
# TYPE gail_aarnn_quality_score_average gauge\n\
# HELP gail_aarnn_agreement_score_average Average AARNN/LLM agreement score.\n\
# TYPE gail_aarnn_agreement_score_average gauge\n\
# HELP gail_aarnn_relevance_score_average Average AARNN prompt relevance score.\n\
# TYPE gail_aarnn_relevance_score_average gauge\n\
# HELP gail_aarnn_confidence_score_average Average AARNN decoder confidence score.\n\
# TYPE gail_aarnn_confidence_score_average gauge\n\
# HELP gail_aarnn_promotions_total AARNN responses promoted after sustained gates.\n\
# TYPE gail_aarnn_promotions_total counter\n",
        );
        for (task, (count, quality, agreement, relevance, confidence, promotions)) in metrics.iter()
        {
            let task = task.replace('\\', "\\\\").replace('"', "\\\"");
            let labels = format!("task=\"{}\"", task);
            let denominator = (*count).max(1) as f64;
            out.push_str(&format!(
                "gail_aarnn_paired_evaluations_total{{{labels}}} {count}\n\
gail_aarnn_quality_score_average{{{labels}}} {}\n\
gail_aarnn_agreement_score_average{{{labels}}} {}\n\
gail_aarnn_relevance_score_average{{{labels}}} {}\n\
gail_aarnn_confidence_score_average{{{labels}}} {}\n\
gail_aarnn_promotions_total{{{labels}}} {promotions}\n",
                quality / denominator,
                agreement / denominator,
                relevance / denominator,
                confidence / denominator
            ));
        }
        if let Ok(decoder) = DECODER_METRICS.lock() {
            out.push_str(&format!(
                "gail_aarnn_decoder_version {}\n\
gail_aarnn_decoder_mapped_neurons {}\n\
gail_aarnn_network_neurons {}\n",
                decoder.0, decoder.1, decoder.2
            ));
        }
        if let Ok(reconciliation) = MIRROR_RECONCILIATION.lock() {
            let (ids, inputs, outputs, usable) = &*reconciliation;
            out.push_str(&format!(
                "gail_aarnn_mirror_inputs_total {}\n\
gail_aarnn_mirror_outputs_total {}\n\
gail_aarnn_mirror_usable_outputs_total {}\n\
gail_aarnn_mirror_unmatched_inputs {}\n",
                inputs,
                outputs,
                usable,
                inputs.saturating_sub(*outputs)
            ));
            let _ = ids;
        }
        out
    }

    pub fn promoted_reply(&self, trace: &AarnnMirrorInvocationTrace) -> Option<String> {
        trace
            .candidate
            .as_ref()?
            .reply_text
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    }

    fn truncate_audit_text(&self, value: &str) -> String {
        truncate_chars(value, self.audit_max_chars.max(1))
    }

    fn optional_audit_text(&self, value: Option<&str>) -> Option<String> {
        value
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(|item| self.truncate_audit_text(item))
    }

    fn audit_text_for_direction(
        &self,
        direction: &AarnnMirrorDirection,
        value: Option<&str>,
    ) -> Option<String> {
        match direction {
            AarnnMirrorDirection::Input
                if self.audit_log_llm_prompts && self.audit_store_llm_content =>
            {
                self.optional_audit_text(value)
            }
            AarnnMirrorDirection::Output
                if self.audit_log_llm_responses && self.audit_store_llm_content =>
            {
                self.optional_audit_text(value)
            }
            _ => None,
        }
    }

    fn log_mirror_request_audit(&self, request: &AarnnMirrorRequest, spike_count: usize) {
        if let Ok(mut reconciliation) = MIRROR_RECONCILIATION.lock() {
            if reconciliation.0.len() >= 100_000 {
                reconciliation.0.clear();
            }
            if reconciliation
                .0
                .insert(request.request_id.clone(), false)
                .is_none()
            {
                reconciliation.1 = reconciliation.1.saturating_add(1);
            }
        }
        if !self.audit_enabled {
            return;
        }
        let text = self.audit_text_for_direction(&request.direction, Some(request.text.as_str()));
        let prompt_text =
            self.audit_text_for_direction(&request.direction, request.prompt_text.as_deref());
        let system_text = if self.audit_log_llm_prompts && self.audit_store_llm_content {
            self.optional_audit_text(request.system.as_deref())
        } else {
            None
        };
        let payload_hex = if self.audit_log_aer_payloads {
            self.optional_audit_text(Some(request.aer_payload_hex.as_str()))
        } else {
            None
        };
        let active_spike_indices = if self.audit_log_aer_payloads {
            Some(
                request
                    .sensory_spikes
                    .iter()
                    .enumerate()
                    .filter_map(|(index, spike)| (*spike > 0).then_some(index as u32))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        info!(
            audit_stream = "aarnn",
            direction = ?request.direction,
            request_id = %request.request_id,
            conversation_id = %request.conversation_id,
            workflow = %request.workflow,
            role = %request.role,
            provider = ?request.provider,
            model = ?request.model,
            request_category = ?request.request_category,
            system_prompt = ?system_text,
            prompt_text = ?prompt_text,
            text = ?text,
            aer_base = request.aer_base,
            output_base = request.output_base,
            aer_payload_hex = ?payload_hex,
            sensory_spike_count = spike_count,
            sensory_spike_indices = ?active_spike_indices,
            "GAIL_AUDIT_AARNN_MIRROR_REQUEST"
        );
    }

    fn log_mirror_response_audit(
        &self,
        request: &AarnnMirrorRequest,
        response: &AarnnMirrorResponse,
        text_chars: usize,
        spike_count: usize,
    ) {
        if let Ok(mut reconciliation) = MIRROR_RECONCILIATION.lock() {
            if reconciliation
                .0
                .get(&request.request_id)
                .is_some_and(|seen| !*seen)
            {
                if let Some(seen) = reconciliation.0.get_mut(&request.request_id) {
                    *seen = true;
                }
                reconciliation.2 = reconciliation.2.saturating_add(1);
                if response
                    .candidate
                    .as_ref()
                    .is_some_and(|candidate| candidate.usable)
                {
                    reconciliation.3 = reconciliation.3.saturating_add(1);
                }
            }
        }
        if !self.audit_enabled {
            return;
        }
        let candidate_reply_text = if self.audit_log_llm_responses && self.audit_store_llm_content {
            self.optional_audit_text(
                response
                    .candidate
                    .as_ref()
                    .and_then(|candidate| candidate.reply_text.as_deref()),
            )
        } else {
            None
        };
        let response_payload_hex = if self.audit_log_aer_payloads {
            self.optional_audit_text(response.aer_payload_hex.as_deref())
        } else {
            None
        };
        let candidate_payload_hex = if self.audit_log_aer_payloads {
            self.optional_audit_text(
                response
                    .candidate
                    .as_ref()
                    .and_then(|candidate| candidate.output_aer_payload_hex.as_deref()),
            )
        } else {
            None
        };
        let candidate_spike_indices = if self.audit_log_aer_payloads {
            response
                .candidate
                .as_ref()
                .map(|candidate| candidate.output_spike_indices.clone())
        } else {
            None
        };
        info!(
            audit_stream = "aarnn",
            direction = ?request.direction,
            request_id = %request.request_id,
            accepted = response.accepted,
            endpoint = %self.endpoint,
            text_chars = response.text_chars.max(text_chars),
            spike_count = response.spike_count.max(spike_count),
            aer_payload_hex = ?response_payload_hex,
            candidate_usable = ?response.candidate.as_ref().map(|candidate| candidate.usable),
            candidate_confidence = ?response.candidate.as_ref().and_then(|candidate| candidate.confidence),
            candidate_source = ?response.candidate.as_ref().and_then(|candidate| candidate.source.as_deref()),
            candidate_reply_text = ?candidate_reply_text,
            candidate_output_aer_payload_hex = ?candidate_payload_hex,
            candidate_output_spike_indices = ?candidate_spike_indices,
            stimulation = ?response.stimulation,
            "GAIL_AUDIT_AARNN_MIRROR_RESPONSE"
        );
    }

    fn log_mirror_error_audit(
        &self,
        request: &AarnnMirrorRequest,
        error: &str,
        text_chars: usize,
        spike_count: usize,
    ) {
        if !self.audit_enabled {
            return;
        }
        let text = self.audit_text_for_direction(&request.direction, Some(request.text.as_str()));
        let payload_hex = if self.audit_log_aer_payloads {
            self.optional_audit_text(Some(request.aer_payload_hex.as_str()))
        } else {
            None
        };
        warn!(
            audit_stream = "aarnn",
            direction = ?request.direction,
            request_id = %request.request_id,
            endpoint = %self.endpoint,
            text_chars,
            spike_count,
            text = ?text,
            aer_payload_hex = ?payload_hex,
            error = %self.truncate_audit_text(error),
            "GAIL_AUDIT_AARNN_MIRROR_ERROR"
        );
    }

    fn start_worker_bus(&self, mut rx: mpsc::Receiver<AarnnMirrorJob>) {
        let worker = self.clone();
        let concurrency = self.worker_count.max(1);
        tokio::spawn(async move {
            let permits = Arc::new(Semaphore::new(concurrency));
            while let Some(job) = rx.recv().await {
                // Bound in-flight mirror requests so queue consumers do not
                // overwhelm AARNN or local runtime resources.
                let permit = match permits.clone().acquire_owned().await {
                    Ok(permit) => permit,
                    Err(_) => break,
                };
                let worker = worker.clone();
                tokio::spawn(async move {
                    let _permit = permit;
                    let trace = worker.mirror(job.exchange).await;
                    if let Some(response) = job.response {
                        let _ = response.send(trace);
                    }
                });
            }
        });
    }

    pub async fn enqueue(
        &self,
        exchange: AarnnMirrorExchange,
        wait_for_trace: bool,
    ) -> Option<oneshot::Receiver<AarnnMirrorInvocationTrace>> {
        let (response_tx, response_rx) = if wait_for_trace {
            let (tx, rx) = oneshot::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        let direction = exchange.direction.clone();
        let send = self.queue_tx.send(AarnnMirrorJob {
            exchange,
            response: response_tx,
        });
        match timeout(self.enqueue_timeout, send).await {
            Ok(Ok(())) => response_rx,
            Ok(Err(error)) => {
                warn!(
                    endpoint = %self.endpoint,
                    direction = ?direction,
                    error = %error,
                    "AARNN mirror queue is closed; dropping exchange"
                );
                None
            }
            Err(_) => {
                warn!(
                    endpoint = %self.endpoint,
                    direction = ?direction,
                    queue_capacity = self.queue_capacity,
                    enqueue_timeout_ms = self.enqueue_timeout.as_millis(),
                    "AARNN mirror queue is saturated; dropping exchange"
                );
                None
            }
        }
    }

    pub async fn mirror(&self, exchange: AarnnMirrorExchange) -> AarnnMirrorInvocationTrace {
        let started = Instant::now();
        let text_chars = exchange.text.chars().count();
        // LLM text/context -> deterministic sensory spikes -> AER hex payload.
        let request = self.build_request(exchange);
        let spike_count = request
            .sensory_spikes
            .iter()
            .filter(|value| **value > 0)
            .count();
        self.log_mirror_request_audit(&request, spike_count);
        match self.mirror_once(&request).await {
            Ok(response) => {
                self.log_mirror_response_audit(&request, &response, text_chars, spike_count);
                AarnnMirrorInvocationTrace {
                    direction: request.direction.clone(),
                    request_category: request.request_category.clone(),
                    accepted: response.accepted,
                    endpoint: self.endpoint.clone(),
                    latency_ms: started.elapsed().as_millis() as u64,
                    text_chars: response.text_chars.max(text_chars),
                    spike_count: response.spike_count.max(spike_count),
                    candidate: response
                        .candidate
                        .map(|candidate| sanitize_candidate(candidate, self.max_text_chars)),
                    stimulation: response.stimulation,
                    error: None,
                }
            }
            Err(error) => {
                self.log_mirror_error_audit(&request, error.as_str(), text_chars, spike_count);
                AarnnMirrorInvocationTrace {
                    direction: request.direction,
                    request_category: request.request_category,
                    accepted: false,
                    endpoint: self.endpoint.clone(),
                    latency_ms: started.elapsed().as_millis() as u64,
                    text_chars,
                    spike_count,
                    candidate: None,
                    stimulation: None,
                    error: Some(error),
                }
            }
        }
    }

    /// Mirror a speech utterance (auditory spikes + paired text) into AARNN as
    /// one timed AER batch at the auditory base, via the authorised mirror path
    /// (same retries and audit logging as text mirrors).
    pub async fn mirror_speech(&self, pair: SpeechMirrorPair) -> AarnnMirrorInvocationTrace {
        let started = Instant::now();
        let (events, activity) = pair.to_aer(now_ts_us());
        let spike_count = events.len();
        let text = truncate_chars(&compact_text(&pair.text), self.max_text_chars);
        let request = AarnnMirrorRequest {
            request_id: pair.pair_id.clone(),
            conversation_id: format!("speech:{}", pair.pair_id),
            workflow: "speech".into(),
            role: "user".into(),
            direction: AarnnMirrorDirection::Input,
            provider: Some("nmstt".into()),
            model: Some(format!("nmstt-{}", pair.source)),
            request_category: Some(format!("speech_{}", pair.source)),
            system: None,
            prompt_text: pair.lang.clone(),
            text: text.clone(),
            message_roles: Vec::new(),
            aer_base: SPEECH_AER_BASE,
            output_base: self.aer_output_base,
            aer_payload_hex: payload_hex(&encode_events(&events)),
            sensory_spikes: activity,
            network_id: self.network_id.clone(),
            node_id: self.node_id.clone(),
            request_candidate_reply: false,
        };
        self.log_mirror_request_audit(&request, spike_count);
        let text_chars = text.chars().count();
        let (accepted, error, stimulation) = match self.mirror_once(&request).await {
            Ok(response) => {
                self.log_mirror_response_audit(&request, &response, text_chars, spike_count);
                (response.accepted, None, response.stimulation)
            }
            Err(error) => {
                self.log_mirror_error_audit(&request, error.as_str(), text_chars, spike_count);
                (false, Some(error), None)
            }
        };
        AarnnMirrorInvocationTrace {
            direction: AarnnMirrorDirection::Input,
            request_category: request.request_category,
            accepted,
            endpoint: self.endpoint.clone(),
            latency_ms: started.elapsed().as_millis() as u64,
            text_chars,
            spike_count,
            candidate: None,
            stimulation,
            error,
        }
    }

    fn build_request(&self, exchange: AarnnMirrorExchange) -> AarnnMirrorRequest {
        // Normalize textual context before projecting into the sensory layer.
        let text = truncate_chars(&compact_text(&exchange.text), self.max_text_chars);
        let system = exchange
            .system
            .as_deref()
            .map(compact_text)
            .map(|value| truncate_chars(&value, self.max_text_chars));
        let prompt_text = exchange
            .prompt_text
            .as_deref()
            .map(compact_text)
            .map(|value| truncate_chars(&value, self.max_text_chars));
        // Deterministically map normalized text to binary spikes, then encode
        // active spike indices into AER events rooted at the sensory base.
        let sensory_spikes = text_to_spikes(text.as_str(), self.sensory_size);
        let aer_payload = encode_spikes(now_ts_us(), self.aer_sensory_base, &sensory_spikes);
        AarnnMirrorRequest {
            request_id: exchange.request_id,
            conversation_id: exchange.conversation_id,
            workflow: exchange.workflow,
            role: exchange.role,
            direction: exchange.direction.clone(),
            provider: exchange.provider,
            model: exchange.model,
            request_category: exchange.request_category,
            system,
            prompt_text,
            text,
            message_roles: exchange.message_roles,
            aer_base: self.aer_sensory_base,
            output_base: self.aer_output_base,
            aer_payload_hex: payload_hex(&aer_payload),
            sensory_spikes,
            network_id: self.network_id.clone(),
            node_id: self.node_id.clone(),
            request_candidate_reply: self.request_candidate_reply
                && matches!(exchange.direction, AarnnMirrorDirection::Output),
        }
    }

    async fn mirror_once(
        &self,
        request: &AarnnMirrorRequest,
    ) -> Result<AarnnMirrorResponse, String> {
        let network_id = request
            .network_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let session_id = match network_id {
            Some(network_id) => self.peripheral_session(network_id).await,
            None => None,
        };
        match self
            .mirror_with_retries(request, session_id.as_deref())
            .await
        {
            Err(error) if network_id.is_some() && error.is_peripheral_session_error() => {
                // The session expired, was revoked, or AARNN restarted (sessions
                // are in-memory). Drop it, create a fresh one and retry once.
                let network_id = network_id.unwrap_or_default();
                self.invalidate_peripheral_session(network_id, session_id.as_deref())
                    .await;
                match self.peripheral_session(network_id).await {
                    Some(fresh) => {
                        info!(
                            endpoint = %self.endpoint,
                            network_id,
                            status = error.status,
                            "AARNN rejected peripheral session; retrying mirror with a new session"
                        );
                        self.mirror_with_retries(request, Some(fresh.as_str()))
                            .await
                            .map_err(|error| error.message)
                    }
                    None => Err(error.message),
                }
            }
            other => other.map_err(|error| error.message),
        }
    }

    async fn mirror_with_retries(
        &self,
        request: &AarnnMirrorRequest,
        session_id: Option<&str>,
    ) -> Result<AarnnMirrorResponse, MirrorHttpError> {
        // Retries are only used for transient transport/upstream failures.
        let max_attempts = self.request_max_attempts.max(1);
        let mut last_error = MirrorHttpError::retryable(String::new());
        for attempt in 1..=max_attempts {
            match self.mirror_once_attempt(request, session_id).await {
                Ok(response) => return Ok(response),
                Err(error) => {
                    let retryable = error.retryable && attempt < max_attempts;
                    if !retryable {
                        return Err(error);
                    }
                    let delay = retry_delay(
                        self.request_backoff,
                        self.request_backoff_max,
                        attempt.saturating_sub(1),
                    );
                    warn!(
                        endpoint = %self.endpoint,
                        attempt,
                        max_attempts,
                        backoff_ms = delay.as_millis() as u64,
                        error = %error.message,
                        "AARNN mirror request failed; retrying"
                    );
                    last_error = error;
                    sleep(delay).await;
                }
            }
        }
        Err(last_error)
    }

    /// Return a valid peripheral session id for `network_id`, creating one if
    /// none is cached or the cached one has under 60 s left. Returns `None`
    /// when AARNN has sessions disabled/unsupported (or creation failed), in
    /// which case the mirror is sent without the session header.
    async fn peripheral_session(&self, network_id: &str) -> Option<String> {
        let slot = self.peripheral_sessions.slot(network_id);
        // Held across creation: concurrent callers wait here and then reuse
        // the session the first caller created (single-flight).
        let mut guard = slot.lock().await;
        let now = Instant::now();
        match &*guard {
            PeripheralSessionSlot::Active {
                session_id,
                expires_at,
            } if expires_at.saturating_duration_since(now) > PERIPHERAL_SESSION_REFRESH_MARGIN => {
                return Some(session_id.clone());
            }
            PeripheralSessionSlot::Unsupported { until }
            | PeripheralSessionSlot::Failed { until }
                if *until > now =>
            {
                return None;
            }
            _ => {}
        }
        let (next, session_id) = self.create_peripheral_session(network_id).await;
        *guard = next;
        session_id
    }

    /// Drop the cached session for `network_id`, but only if it is still the
    /// one that was rejected (a concurrent caller may already have replaced
    /// it). Negative/back-off entries are cleared so a retry re-probes.
    async fn invalidate_peripheral_session(&self, network_id: &str, stale: Option<&str>) {
        let slot = self.peripheral_sessions.slot(network_id);
        let mut guard = slot.lock().await;
        let clear = match (&*guard, stale) {
            (PeripheralSessionSlot::Active { session_id, .. }, Some(stale)) => session_id == stale,
            (PeripheralSessionSlot::Active { .. }, None) => false,
            (PeripheralSessionSlot::Unsupported { .. }, _) => false,
            _ => true,
        };
        if clear {
            *guard = PeripheralSessionSlot::Empty;
        }
    }

    async fn create_peripheral_session(
        &self,
        network_id: &str,
    ) -> (PeripheralSessionSlot, Option<String>) {
        let url = format!("{}{}", self.endpoint, AARNN_PERIPHERAL_SESSIONS_PATH);
        let headers = match self.headers() {
            Ok(headers) => headers,
            Err(error) => {
                warn!(error = %error, "AARNN peripheral session headers invalid");
                return self.session_failed();
            }
        };
        let body = serde_json::json!({
            "brain_id": network_id,
            "local_consent": true,
            "ttl_secs": PERIPHERAL_SESSION_TTL_SECS,
        });
        let started = SystemTime::now();
        let response = match self
            .client
            .post(url)
            .headers(headers)
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                warn!(
                    endpoint = %self.endpoint,
                    network_id,
                    error = %error,
                    "AARNN peripheral session creation failed; mirroring without session"
                );
                return self.session_failed();
            }
        };
        let status = response.status().as_u16();
        if !response.status().is_success() {
            let text = timeout(self.timeout, response.text())
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or_default();
            let text = truncate_chars(text.trim(), 256);
            return match status {
                403..=405 => {
                    let reason = if status == 403 && text.to_ascii_lowercase().contains("disabled")
                    {
                        "peripheral access disabled (auth mode none)"
                    } else if status == 403 {
                        "peripheral session creation forbidden"
                    } else {
                        "peripheral sessions not supported by this AARNN"
                    };
                    if !self
                        .peripheral_sessions
                        .warned_unsupported
                        .swap(true, std::sync::atomic::Ordering::Relaxed)
                    {
                        warn!(
                            endpoint = %self.endpoint,
                            network_id,
                            status,
                            body = %text,
                            reason,
                            "AARNN peripheral session unavailable; mirroring without session header"
                        );
                    } else {
                        debug!(
                            endpoint = %self.endpoint,
                            network_id,
                            status,
                            reason,
                            "AARNN peripheral session still unavailable"
                        );
                    }
                    (
                        PeripheralSessionSlot::Unsupported {
                            until: Instant::now() + PERIPHERAL_SESSION_UNSUPPORTED_TTL,
                        },
                        None,
                    )
                }
                _ => {
                    warn!(
                        endpoint = %self.endpoint,
                        network_id,
                        status,
                        body = %text,
                        "AARNN peripheral session creation rejected; mirroring without session"
                    );
                    self.session_failed()
                }
            };
        }
        let reply = match timeout(self.timeout, response.json::<PeripheralSessionReply>()).await {
            Ok(Ok(reply)) if !reply.session_id.trim().is_empty() => reply,
            Ok(Ok(_)) => {
                warn!(
                    network_id,
                    "AARNN peripheral session reply had empty session_id"
                );
                return self.session_failed();
            }
            Ok(Err(error)) => {
                warn!(network_id, error = %error, "AARNN peripheral session reply unparseable");
                return self.session_failed();
            }
            Err(_) => {
                warn!(network_id, "AARNN peripheral session reply timed out");
                return self.session_failed();
            }
        };
        let lifetime = session_lifetime(reply.expires_at_unix_secs, started);
        if HeaderValue::from_str(reply.session_id.as_str()).is_err() {
            warn!(
                network_id,
                "AARNN peripheral session id is not a valid header value"
            );
            return self.session_failed();
        }
        debug!(
            endpoint = %self.endpoint,
            network_id,
            lifetime_secs = lifetime.as_secs(),
            "AARNN peripheral session created"
        );
        (
            PeripheralSessionSlot::Active {
                session_id: reply.session_id.clone(),
                expires_at: Instant::now() + lifetime,
            },
            Some(reply.session_id),
        )
    }

    fn session_failed(&self) -> (PeripheralSessionSlot, Option<String>) {
        (
            PeripheralSessionSlot::Failed {
                until: Instant::now() + PERIPHERAL_SESSION_FAILURE_TTL,
            },
            None,
        )
    }

    async fn mirror_once_attempt(
        &self,
        request: &AarnnMirrorRequest,
        session_id: Option<&str>,
    ) -> Result<AarnnMirrorResponse, MirrorHttpError> {
        let url = format!("{}{}", self.endpoint, AARNN_MIRROR_PATH);
        let mut headers = self
            .headers()
            .map_err(|error| MirrorHttpError::non_retryable(error.as_str()))?;
        if let Some(session_id) = session_id {
            let value = HeaderValue::from_str(session_id)
                .map_err(|error| MirrorHttpError::non_retryable(&error.to_string()))?;
            headers.insert(HeaderName::from_static(PERIPHERAL_SESSION_HEADER), value);
        }
        let response = match self
            .client
            .post(url)
            .headers(headers)
            .timeout(self.timeout)
            .json(request)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                let message = error.to_string();
                adaptive_schema::observe_failure(
                    "aarnn_bridge",
                    "POST",
                    AARNN_MIRROR_PATH,
                    "mirror",
                    None,
                    &message,
                )
                .await;
                return Err(MirrorHttpError::retryable(message));
            }
        };
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let message = if body.trim().is_empty() {
                status.to_string()
            } else {
                format!("{status}: {body}")
            };
            let retryable = mirror_status_retryable(status.as_u16());
            let session_error = matches!(status.as_u16(), 403 | 404) && {
                let lower = body.to_ascii_lowercase();
                lower.contains("peripheral") || lower.contains("session")
            };
            adaptive_schema::observe_failure(
                "aarnn_bridge",
                "POST",
                AARNN_MIRROR_PATH,
                "mirror",
                Some(status.as_u16()),
                &message,
            )
            .await;
            let mut error = if retryable {
                MirrorHttpError::retryable(message)
            } else {
                MirrorHttpError::non_retryable(message.as_str())
            };
            error.status = Some(status.as_u16());
            error.session_error = session_error;
            return Err(error);
        }
        match response.json::<AarnnMirrorResponse>().await {
            Ok(parsed) => {
                let body = serde_json::to_value(&parsed).unwrap_or(Value::Null);
                adaptive_schema::observe_success(
                    "aarnn_bridge",
                    "POST",
                    AARNN_MIRROR_PATH,
                    "mirror",
                    &body,
                )
                .await;
                Ok(parsed)
            }
            Err(error) => {
                adaptive_schema::observe_failure(
                    "aarnn_bridge",
                    "POST",
                    AARNN_MIRROR_PATH,
                    "mirror",
                    Some(status.as_u16()),
                    &error.to_string(),
                )
                .await;
                Err(MirrorHttpError::retryable(error.to_string()))
            }
        }
    }

    fn headers(&self) -> Result<HeaderMap, String> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(token) = self.access_token.as_deref() {
            let value = format!("Bearer {token}");
            let header = HeaderValue::from_str(&value).map_err(|error| error.to_string())?;
            headers.insert(AUTHORIZATION, header);
        }
        Ok(headers)
    }
}

#[derive(Debug)]
struct MirrorHttpError {
    message: String,
    retryable: bool,
    status: Option<u16>,
    /// 403/404 whose body mentions "peripheral" or "session".
    session_error: bool,
}

impl MirrorHttpError {
    fn retryable(message: String) -> Self {
        Self {
            message,
            retryable: true,
            status: None,
            session_error: false,
        }
    }

    fn non_retryable(message: &str) -> Self {
        Self {
            message: message.to_string(),
            retryable: false,
            status: None,
            session_error: false,
        }
    }

    fn is_peripheral_session_error(&self) -> bool {
        self.session_error
    }
}

/// Lifetime of a freshly created session: from AARNN's absolute expiry when
/// present (measured against the time the request was sent), else the TTL we
/// asked for. Never longer than the 900 s cap.
fn session_lifetime(expires_at_unix_secs: Option<u64>, requested_at: SystemTime) -> Duration {
    let cap = Duration::from_secs(PERIPHERAL_SESSION_TTL_SECS);
    let Some(expires_at) = expires_at_unix_secs else {
        return cap;
    };
    let requested = requested_at
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Duration::from_secs(expires_at.saturating_sub(requested)).min(cap)
}

#[derive(Clone, Debug)]
struct TransportProfile {
    sensory_size: usize,
    output_size: usize,
    aer_sensory_base: u32,
    aer_output_base: u32,
}

fn mirror_status_retryable(status: u16) -> bool {
    matches!(status, 408 | 425 | 429 | 500 | 502 | 503 | 504)
}

fn retry_delay(base: Duration, max: Duration, attempt_index: usize) -> Duration {
    let shift = attempt_index.min(8) as u32;
    let factor = 2_u32.saturating_pow(shift).max(1);
    let expanded = base.saturating_mul(factor);
    if expanded > max { max } else { expanded }
}

fn env_usize_clamped(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(min, max.max(min))
}

fn env_u64_clamped(name: &str, default: u64, min: u64, max: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
        .clamp(min, max.max(min))
}

fn resolve_endpoint(
    bridge: &AarnnBridgeConfig,
    active_specialists: &[SpecialistEngine],
    configured_specialists: &[SpecialistProfile],
) -> Option<String> {
    bridge
        .endpoint
        .clone()
        .or_else(|| {
            active_specialists
                .iter()
                .find(|engine| engine.engine_type().eq_ignore_ascii_case("aarnn"))
                .and_then(|engine| engine.profile().endpoint.clone())
        })
        .or_else(|| {
            configured_specialists
                .iter()
                .find(|profile| profile.engine_type.eq_ignore_ascii_case("aarnn"))
                .and_then(|profile| profile.endpoint.clone())
        })
        .or_else(|| legacy_env("GAIL_AARNN_ENDPOINT"))
        .and_then(|value| normalized_url(value.as_str()))
}

fn resolve_transport_profile(
    active_specialists: &[SpecialistEngine],
    configured_specialists: &[SpecialistProfile],
) -> TransportProfile {
    let profile = active_specialists
        .iter()
        .find(|engine| engine.engine_type().eq_ignore_ascii_case("aarnn"))
        .map(|engine| engine.profile().clone())
        .or_else(|| {
            configured_specialists
                .iter()
                .find(|profile| profile.engine_type.eq_ignore_ascii_case("aarnn"))
                .cloned()
        })
        .unwrap_or_default();
    TransportProfile {
        sensory_size: profile.sensory_size.max(8),
        output_size: profile.output_size.max(8),
        aer_sensory_base: profile.aer_sensory_base,
        aer_output_base: profile.aer_output_base,
    }
}

fn text_to_spikes(text: &str, sensory_size: usize) -> Vec<u8> {
    // Fixed-density bottom-hash projection. Unlike the former set-every-hit
    // encoder, long inputs cannot saturate the sensory layer and distinct text
    // retains a discriminative whole-document fingerprint.
    let sensory_size = sensory_size.max(8);
    let mut spikes = vec![0u8; sensory_size];
    let compact = compact_text(text).to_ascii_lowercase();
    let bytes = compact.as_bytes();
    if bytes.is_empty() {
        return spikes;
    }
    let target = (sensory_size / 4).clamp(2, 64).min(sensory_size);
    let mut ranked = (0..bytes.len())
        .map(|index| {
            let end = (index + 5).min(bytes.len());
            let mut hash = 0xcbf29ce484222325u64;
            for byte in &bytes[index..end] {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash ^= (index as u64).wrapping_mul(0x9e3779b97f4a7c15);
            (hash, (hash as usize) % sensory_size)
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    let mut active = 0usize;
    for (_, neuron) in ranked {
        if spikes[neuron] == 0 {
            spikes[neuron] = 1;
            active += 1;
        }
        if active >= target {
            break;
        }
    }
    spikes
}

fn sanitize_candidate(
    mut candidate: AarnnMirrorCandidate,
    max_text_chars: usize,
) -> AarnnMirrorCandidate {
    candidate.reply_text = candidate
        .reply_text
        .as_deref()
        .map(compact_text)
        .map(|value| truncate_chars(&value, max_text_chars))
        .filter(|value| !value.is_empty());
    if candidate.reply_text.is_none() {
        candidate.usable = false;
    }
    candidate
}

fn compact_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(value: &str, limit: usize) -> String {
    value.chars().take(limit.max(1)).collect()
}

fn normalise_for_compare(value: &str) -> String {
    compact_text(value).to_ascii_lowercase()
}

fn token_overlap(left: &str, right: &str) -> f64 {
    use std::collections::HashSet;

    let left_tokens = left
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    let right_tokens = right
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }
    let intersection = left_tokens.intersection(&right_tokens).count() as f64;
    let union = left_tokens.union(&right_tokens).count() as f64;
    (intersection / union).clamp(0.0, 1.0)
}

fn normalized_url(raw: &str) -> Option<String> {
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    if value.contains("://") {
        Some(value.trim_end_matches('/').to_string())
    } else {
        Some(format!("http://{}", value.trim_end_matches('/')))
    }
}

fn legacy_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn now_ts_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    use crate::{
        config::{GailConfig, SpecialistProfile},
        models::{AarnnMirrorDirection, AarnnResponsePreference},
    };

    #[test]
    fn status_uses_specialist_endpoint_when_bridge_endpoint_is_unset() {
        let mut config = GailConfig::default();
        config.aarnn_bridge.enabled = true;
        config.specialists.push(SpecialistProfile {
            endpoint: Some("http://aarnn.internal:8080".to_string()),
            ..SpecialistProfile::default()
        });

        let status = AarnnMirrorClient::status(&config, &[]);
        assert!(status.enabled);
        assert!(status.available);
        assert_eq!(
            status.endpoint.as_deref(),
            Some("http://aarnn.internal:8080")
        );
    }

    #[test]
    fn candidate_promotion_requires_confident_non_duplicate_reply() {
        let (queue_tx, _queue_rx) = mpsc::channel(1);
        let client = AarnnMirrorClient {
            client: Client::builder().build().expect("client"),
            endpoint: "http://example.invalid".to_string(),
            access_token: None,
            timeout: Duration::from_secs(1),
            queue_tx,
            queue_capacity: 1,
            worker_count: 1,
            enqueue_timeout: Duration::from_millis(10),
            candidate_wait_timeout: Duration::from_millis(100),
            mirror_input: true,
            mirror_output: true,
            request_candidate_reply: true,
            response_preference: AarnnResponsePreference::PreferAarnnWhenConfident,
            candidate_confidence_threshold: 0.8,
            candidate_min_reply_chars: 10,
            network_id: None,
            node_id: None,
            max_text_chars: 2048,
            sensory_size: 32,
            aer_sensory_base: 4096,
            aer_output_base: 16384,
            request_max_attempts: 1,
            request_backoff: Duration::from_millis(10),
            request_backoff_max: Duration::from_millis(50),
            audit_enabled: false,
            audit_log_llm_prompts: true,
            audit_log_llm_responses: true,
            audit_store_llm_content: false,
            audit_log_aer_payloads: true,
            audit_max_chars: 2048,
            peripheral_sessions: Arc::new(PeripheralSessions::default()),
        };
        let promoted = AarnnMirrorInvocationTrace {
            direction: AarnnMirrorDirection::Output,
            request_category: Some("general".to_string()),
            accepted: true,
            endpoint: client.endpoint.clone(),
            latency_ms: 5,
            text_chars: 20,
            spike_count: 8,
            candidate: Some(AarnnMirrorCandidate {
                reply_text: Some("Alternative SNN answer".to_string()),
                confidence: Some(0.9),
                usable: true,
                source: Some("network_output_decoder".to_string()),
                output_spike_indices: vec![1, 2],
                output_aer_payload_hex: Some("41455231".to_string()),
                decoder_version: 1,
                decoder_mapped_neurons: 2,
                network_neurons: 2,
            }),
            stimulation: None,
            error: None,
        };
        for _ in 0..20 {
            client.evaluate_candidate(&promoted, "Alternative answer", "Alternative SNN answer");
        }
        assert!(client.should_promote_candidate(
            &promoted,
            "Alternative answer",
            "Alternative SNN answer"
        ));

        let duplicate = AarnnMirrorInvocationTrace {
            candidate: Some(AarnnMirrorCandidate {
                reply_text: Some("LLM answer".to_string()),
                confidence: Some(0.95),
                usable: true,
                source: Some("network_output_decoder".to_string()),
                output_spike_indices: vec![],
                output_aer_payload_hex: None,
                decoder_version: 1,
                decoder_mapped_neurons: 2,
                network_neurons: 2,
            }),
            ..promoted.clone()
        };
        assert!(!client.should_promote_candidate(&duplicate, "LLM answer", "prompt"));

        let legacy_echo = AarnnMirrorInvocationTrace {
            candidate: Some(AarnnMirrorCandidate {
                reply_text: Some("Plausible but echoed answer".to_string()),
                confidence: Some(1.0),
                usable: true,
                source: Some("stimulated_transport_echo".to_string()),
                output_spike_indices: vec![1],
                output_aer_payload_hex: None,
                decoder_version: 1,
                decoder_mapped_neurons: 2,
                network_neurons: 2,
            }),
            ..promoted
        };
        assert!(!client.should_promote_candidate(&legacy_echo, "LLM answer", "prompt"));
    }

    #[test]
    fn long_text_projection_is_sparse_and_discriminative() {
        let first = text_to_spikes(&"alpha beta gamma ".repeat(100), 32);
        let second = text_to_spikes(&"delta epsilon zeta ".repeat(100), 32);
        assert_eq!(first.iter().filter(|value| **value > 0).count(), 8);
        assert_eq!(second.iter().filter(|value| **value > 0).count(), 8);
        assert_ne!(first, second);
    }

    #[tokio::test]
    async fn mirror_posts_bearer_authenticated_payload() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/llm/mirror"))
            .and(header("authorization", "Bearer bridge-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "accepted": true,
                "text_chars": 12,
                "spike_count": 4
            })))
            .mount(&server)
            .await;

        let mut config = GailConfig::default();
        config.aarnn_bridge.enabled = true;
        config.aarnn_bridge.endpoint = Some(server.uri());
        config.aarnn_bridge.access_token = Some("bridge-token".to_string());
        let client = AarnnMirrorClient::from_config(
            &config,
            Client::builder().build().expect("client"),
            &[],
        )
        .expect("bridge client");

        let trace = client
            .mirror(AarnnMirrorExchange {
                request_id: "req-1".to_string(),
                conversation_id: "conv-1".to_string(),
                workflow: "assistant".to_string(),
                role: "assistant".to_string(),
                direction: AarnnMirrorDirection::Input,
                provider: Some("openai".to_string()),
                model: Some("gpt-5.3-codex".to_string()),
                request_category: None,
                system: Some("Keep it concise.".to_string()),
                prompt_text: None,
                text: "Hello world".to_string(),
                message_roles: vec!["system".to_string(), "user".to_string()],
            })
            .await;

        assert!(trace.accepted);
        assert!(trace.error.is_none());
    }

    #[tokio::test]
    async fn enqueue_dispatches_non_blocking_worker_job() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/llm/mirror"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "accepted": true,
                "text_chars": 16,
                "spike_count": 4
            })))
            .mount(&server)
            .await;

        let mut config = GailConfig::default();
        config.aarnn_bridge.enabled = true;
        config.aarnn_bridge.endpoint = Some(server.uri());
        config.aarnn_bridge.queue_capacity = 8;
        config.aarnn_bridge.worker_count = 2;
        config.aarnn_bridge.enqueue_timeout_ms = 25;
        let client = AarnnMirrorClient::from_config(
            &config,
            Client::builder().build().expect("client"),
            &[],
        )
        .expect("bridge client");
        let receiver = client
            .enqueue(
                AarnnMirrorExchange {
                    request_id: "req-2".to_string(),
                    conversation_id: "conv-2".to_string(),
                    workflow: "assistant".to_string(),
                    role: "assistant".to_string(),
                    direction: AarnnMirrorDirection::Input,
                    provider: Some("openai".to_string()),
                    model: Some("gpt-5.3-codex".to_string()),
                    request_category: None,
                    system: None,
                    prompt_text: None,
                    text: "queued hello".to_string(),
                    message_roles: vec!["user".to_string()],
                },
                true,
            )
            .await
            .expect("queued receiver");
        let trace = timeout(Duration::from_secs(1), receiver)
            .await
            .expect("worker timeout")
            .expect("worker trace");
        assert!(trace.accepted);
        assert!(trace.error.is_none());
    }
}

#[cfg(test)]
mod peripheral_session_tests {
    use super::*;
    use wiremock::{
        Mock, MockServer, Request, ResponseTemplate,
        matchers::{header, method, path},
    };

    use crate::{config::GailConfig, models::AarnnMirrorDirection};

    const NETWORK: &str = "shared-snn";

    fn client_for(server: &MockServer, network_id: Option<&str>) -> AarnnMirrorClient {
        let mut config = GailConfig::default();
        config.aarnn_bridge.enabled = true;
        config.aarnn_bridge.endpoint = Some(server.uri());
        config.aarnn_bridge.access_token = Some("bridge-token".to_string());
        config.aarnn_bridge.network_id = network_id.map(str::to_string);
        AarnnMirrorClient::from_config(&config, Client::builder().build().expect("client"), &[])
            .expect("bridge client")
    }

    fn exchange(id: &str) -> AarnnMirrorExchange {
        AarnnMirrorExchange {
            request_id: id.to_string(),
            conversation_id: "conv".to_string(),
            workflow: "assistant".to_string(),
            role: "user".to_string(),
            direction: AarnnMirrorDirection::Input,
            provider: None,
            model: None,
            request_category: None,
            system: None,
            prompt_text: None,
            text: "hello peripheral".to_string(),
            message_roles: vec!["user".to_string()],
        }
    }

    fn unix_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn session_reply(id: &str, ttl: u64) -> ResponseTemplate {
        ResponseTemplate::new(201).set_body_json(serde_json::json!({
            "session_id": id,
            "brain_id": NETWORK,
            "channel": "aer",
            "direction": "input",
            "expires_at_unix_secs": unix_now() + ttl,
        }))
    }

    fn mirror_ok() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "accepted": true,
            "text_chars": 16,
            "spike_count": 4
        }))
    }

    async fn requests_to(server: &MockServer, route: &str) -> Vec<Request> {
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|request| request.url.path() == route)
            .collect()
    }

    fn session_header(request: &Request) -> Option<String> {
        request
            .headers
            .get(PERIPHERAL_SESSION_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
    }

    #[tokio::test]
    async fn peripheral_session_created_then_reused() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(AARNN_PERIPHERAL_SESSIONS_PATH))
            .and(header("authorization", "Bearer bridge-token"))
            .respond_with(session_reply("sess-1", 900))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(AARNN_MIRROR_PATH))
            .and(header(PERIPHERAL_SESSION_HEADER, "sess-1"))
            .respond_with(mirror_ok())
            .expect(3)
            .mount(&server)
            .await;
        let client = client_for(&server, Some(NETWORK));
        for i in 0..3 {
            let trace = client.mirror(exchange(&format!("r{i}"))).await;
            assert!(trace.accepted, "{:?}", trace.error);
        }
        let sessions = requests_to(&server, AARNN_PERIPHERAL_SESSIONS_PATH).await;
        let body: Value = serde_json::from_slice(&sessions[0].body).expect("json");
        assert_eq!(body["brain_id"], NETWORK);
        assert_eq!(body["local_consent"], true);
        assert_eq!(body["ttl_secs"], 900);
    }

    #[tokio::test]
    async fn peripheral_session_refreshed_near_expiry() {
        let server = MockServer::start().await;
        // First session expires in 30 s (< 60 s margin), so the next mirror
        // must create a fresh one.
        Mock::given(method("POST"))
            .and(path(AARNN_PERIPHERAL_SESSIONS_PATH))
            .respond_with(session_reply("sess-short", 30))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(AARNN_PERIPHERAL_SESSIONS_PATH))
            .respond_with(session_reply("sess-long", 900))
            .with_priority(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(AARNN_MIRROR_PATH))
            .respond_with(mirror_ok())
            .mount(&server)
            .await;
        let client = client_for(&server, Some(NETWORK));
        for i in 0..3 {
            assert!(client.mirror(exchange(&format!("r{i}"))).await.accepted);
        }
        assert_eq!(
            requests_to(&server, AARNN_PERIPHERAL_SESSIONS_PATH)
                .await
                .len(),
            2
        );
        let used: Vec<_> = requests_to(&server, AARNN_MIRROR_PATH)
            .await
            .iter()
            .map(session_header)
            .collect();
        assert_eq!(
            used,
            vec![
                Some("sess-short".to_string()),
                Some("sess-long".to_string()),
                Some("sess-long".to_string())
            ]
        );
    }

    #[tokio::test]
    async fn peripheral_session_recreated_after_403_with_one_retry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(AARNN_PERIPHERAL_SESSIONS_PATH))
            .respond_with(session_reply("sess-old", 900))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(AARNN_PERIPHERAL_SESSIONS_PATH))
            .respond_with(session_reply("sess-new", 900))
            .with_priority(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(AARNN_MIRROR_PATH))
            .and(header(PERIPHERAL_SESSION_HEADER, "sess-old"))
            .respond_with(ResponseTemplate::new(403).set_body_string("unknown peripheral session"))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(AARNN_MIRROR_PATH))
            .and(header(PERIPHERAL_SESSION_HEADER, "sess-new"))
            .respond_with(mirror_ok())
            .mount(&server)
            .await;
        let client = client_for(&server, Some(NETWORK));
        let trace = client.mirror(exchange("r1")).await;
        assert!(trace.accepted, "{:?}", trace.error);
        assert_eq!(
            requests_to(&server, AARNN_PERIPHERAL_SESSIONS_PATH)
                .await
                .len(),
            2
        );
        assert_eq!(requests_to(&server, AARNN_MIRROR_PATH).await.len(), 2);
    }

    #[tokio::test]
    async fn peripheral_session_error_retries_exactly_once() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(AARNN_PERIPHERAL_SESSIONS_PATH))
            .respond_with(session_reply("sess-x", 900))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(AARNN_MIRROR_PATH))
            .respond_with(
                ResponseTemplate::new(404).set_body_string("peripheral session not found"),
            )
            .mount(&server)
            .await;
        let client = client_for(&server, Some(NETWORK));
        let trace = client.mirror(exchange("r1")).await;
        assert!(!trace.accepted);
        assert!(trace.error.expect("error").contains("404"));
        assert_eq!(requests_to(&server, AARNN_MIRROR_PATH).await.len(), 2);
        assert_eq!(
            requests_to(&server, AARNN_PERIPHERAL_SESSIONS_PATH)
                .await
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn peripheral_disabled_mirrors_without_header_and_caches() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(AARNN_PERIPHERAL_SESSIONS_PATH))
            .respond_with(
                ResponseTemplate::new(403)
                    .set_body_string("peripheral access is disabled (auth mode none)"),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(AARNN_MIRROR_PATH))
            .respond_with(mirror_ok())
            .mount(&server)
            .await;
        let client = client_for(&server, Some(NETWORK));
        for i in 0..3 {
            assert!(client.mirror(exchange(&format!("r{i}"))).await.accepted);
        }
        let mirrors = requests_to(&server, AARNN_MIRROR_PATH).await;
        assert_eq!(mirrors.len(), 3);
        assert!(
            mirrors
                .iter()
                .all(|request| session_header(request).is_none())
        );
    }

    #[tokio::test]
    async fn peripheral_unsupported_404_mirrors_without_header() {
        for status in [404u16, 405] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path(AARNN_PERIPHERAL_SESSIONS_PATH))
                .respond_with(ResponseTemplate::new(status))
                .expect(1)
                .mount(&server)
                .await;
            Mock::given(method("POST"))
                .and(path(AARNN_MIRROR_PATH))
                .respond_with(mirror_ok())
                .mount(&server)
                .await;
            let client = client_for(&server, Some(NETWORK));
            assert!(client.mirror(exchange("r1")).await.accepted);
            assert!(client.mirror(exchange("r2")).await.accepted);
            let mirrors = requests_to(&server, AARNN_MIRROR_PATH).await;
            assert!(
                mirrors
                    .iter()
                    .all(|request| session_header(request).is_none())
            );
        }
    }

    #[tokio::test]
    async fn no_network_id_skips_peripheral_session() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(AARNN_PERIPHERAL_SESSIONS_PATH))
            .respond_with(session_reply("never", 900))
            .expect(0)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(AARNN_MIRROR_PATH))
            .respond_with(mirror_ok())
            .mount(&server)
            .await;
        let client = client_for(&server, None);
        assert!(client.mirror(exchange("r1")).await.accepted);
    }

    #[tokio::test]
    async fn concurrent_mirrors_create_single_session() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(AARNN_PERIPHERAL_SESSIONS_PATH))
            .respond_with(session_reply("sess-shared", 900).set_delay(Duration::from_millis(200)))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(AARNN_MIRROR_PATH))
            .and(header(PERIPHERAL_SESSION_HEADER, "sess-shared"))
            .respond_with(mirror_ok())
            .expect(16)
            .mount(&server)
            .await;
        let client = client_for(&server, Some(NETWORK));
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let client = client.clone();
                tokio::spawn(async move { client.mirror(exchange(&format!("r{i}"))).await })
            })
            .collect();
        for handle in handles {
            assert!(handle.await.expect("join").accepted);
        }
    }

    #[test]
    fn session_lifetime_is_capped_and_relative() {
        let now = SystemTime::now();
        let secs = now.duration_since(UNIX_EPOCH).unwrap().as_secs();
        assert_eq!(session_lifetime(None, now), Duration::from_secs(900));
        assert_eq!(
            session_lifetime(Some(secs + 5000), now),
            Duration::from_secs(900)
        );
        assert_eq!(
            session_lifetime(Some(secs + 120), now),
            Duration::from_secs(120)
        );
        assert_eq!(session_lifetime(Some(secs - 10), now), Duration::ZERO);
    }
}

#[cfg(test)]
mod speech_mirror_tests {
    use super::*;
    use crate::aer::decode_events;

    fn pair(frames: Vec<Vec<u16>>) -> SpeechMirrorPair {
        SpeechMirrorPair {
            pair_id: "p1".into(),
            source: "stt".into(),
            text: "hello".into(),
            frame_ms: 10,
            bands: 32,
            frames,
            lang: Some("en-GB".into()),
        }
    }

    #[test]
    fn validates_bounds() {
        assert!(pair(vec![vec![0, 31]]).validate().is_ok());
        assert!(pair(vec![vec![32]]).validate().is_err()); // band out of range
        assert!(pair(vec![vec![], vec![]]).validate().is_err()); // no events
        let mut p = pair(vec![vec![1]]);
        p.source = "radio".into();
        assert!(p.validate().is_err());
        p = pair(vec![vec![1]]);
        p.text = "  ".into();
        assert!(p.validate().is_err());
    }

    #[test]
    fn encodes_timed_events_at_auditory_base() {
        let p = pair(vec![vec![2], vec![], vec![2, 5]]);
        let (events, activity) = p.to_aer(1_000_000);
        let decoded = decode_events(&encode_events(&events)).unwrap();
        let got: Vec<(u64, u32)> = decoded.iter().map(|e| (e.ts_us, e.addr)).collect();
        assert_eq!(
            got,
            vec![
                (1_000_000, SPEECH_AER_BASE + 2),
                (1_020_000, SPEECH_AER_BASE + 2),
                (1_020_000, SPEECH_AER_BASE + 5)
            ]
        );
        assert_eq!(activity[2], 2);
        assert_eq!(activity[5], 1);
        assert!(
            SPEECH_AER_BASE + SPEECH_MAX_BANDS <= 16384,
            "must not overlap the output region"
        );
    }
}
