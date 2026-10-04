//! Alexa custom-skill backend ("Alexa, ask Aaron ...").
//!
//! Alexa cannot present a Gail bearer token, so every request is
//! authenticated with Amazon's mandatory verification for self-hosted HTTPS
//! skills before anything else happens:
//!
//! 1. `SignatureCertChainUrl` must point at `https://s3.amazonaws.com/echo.api/...`.
//! 2. The PEM chain at that URL must be currently valid, name
//!    `echo-api.amazon.com` and chain to a trusted root (webpki roots).
//! 3. `Signature-256` (RSA-SHA256 over the raw body) must verify against the
//!    leaf key. The legacy SHA-1 `Signature` header is only used when the
//!    SHA-256 header is absent.
//! 4. `request.timestamp` must be within 150 seconds of now.
//! 5. The skill application id must be one of the configured skill ids.
//!
//! Verified questions are answered through Gail's ordinary orchestrated chat
//! path (`gail-auto`), which applies Aria governance and LLM ledger auditing
//! under the client id [`CLIENT_ID`].

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    Json,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use base64::Engine;
use futures::{StreamExt, future::BoxFuture};
use rustls_pki_types::{
    AlgorithmIdentifier, CertificateDer, InvalidSignature, ServerName,
    SignatureVerificationAlgorithm, TrustAnchor, UnixTime, alg_id, pem::PemObject,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::task::JoinHandle;
use url::Url;

use crate::{
    errors::GailError,
    models::{ChatMessage, MessageContent},
};

/// Client id used for governance, ledger and audit attribution.
pub const CLIENT_ID: &str = "alexa";
/// Scope Alexa traffic is treated as holding (equivalent to an `llm` token).
pub const CLIENT_SCOPE: &str = "llm";
/// Public route served by Gail.
pub const ROUTE: &str = "/v1/integrations/alexa";
/// Subject alternative name Amazon's signing certificate must carry.
pub const ECHO_API_SUBJECT: &str = "echo-api.amazon.com";
/// Amazon's hard limit on request timestamp skew.
pub const MAX_TIMESTAMP_TOLERANCE_SECONDS: u64 = 150;
/// Alexa request bodies are small; anything larger is not a skill request.
pub const MAX_REQUEST_BYTES: usize = 128 * 1024;
/// Alexa rejects output speech longer than 8000 characters.
pub const MAX_SSML_CHARS: usize = 8000;

const MAX_CERT_CHAIN_BYTES: usize = 64 * 1024;
const MAX_CACHED_CHAINS: usize = 16;
const MAX_PENDING_ANSWERS: usize = 64;
const PENDING_ANSWER_TTL: Duration = Duration::from_secs(120);
const MAX_HISTORY_TURNS: usize = 3;
const MAX_HISTORY_CHARS: usize = 400;
const DEFAULT_PROGRESSIVE_DELAY: Duration = Duration::from_millis(1500);

pub const VOICE_SYSTEM_PROMPT: &str = "You are Aaron, a voice assistant answering through an Amazon Echo speaker. \
Reply in plain, natural spoken English: usually one to three short sentences and under 80 words, unless the user asks for more detail. \
Never use markdown, bullet points, numbered lists, tables, code blocks, URLs or emoji. \
Write numbers, symbols and abbreviations the way a person would say them aloud. \
If you are not sure of something, say so briefly.";

const GREETING: &str = "Hi, I'm Aaron. What would you like to ask?";
const REPROMPT: &str = "You can ask me anything. What would you like to know?";
const FOLLOW_UP_REPROMPT: &str = "Anything else you'd like to ask?";
const HELP: &str = "Ask me a question in your own words, for example, what is the tallest mountain in Scotland? \
Say stop when you're finished. What would you like to ask?";
const FALLBACK: &str = "Sorry, I didn't catch that. Try asking me a question in your own words.";
const GOODBYE: &str = "Goodbye.";
const EMPTY_QUERY: &str = "What would you like to ask?";
const STILL_THINKING: &str = "I'm still thinking about that one. Say yes to hear the answer when it's ready, or ask me something else.";
const STILL_THINKING_REPROMPT: &str = "Say yes to hear my answer, or ask me something else.";
const NOTHING_PENDING: &str = "I don't have an answer waiting. What would you like to ask?";
const BLOCKED: &str =
    "Sorry, I can't help with that one. Is there something else you'd like to ask?";
const FAILED: &str = "Sorry, I couldn't get an answer just now. Please try again in a moment.";
const PROGRESSIVE_SPEECH: &str = "One moment, let me think.";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// `alexa:` section of `gail.yaml`. Disabled by default; when disabled the
/// route answers 404 as though it did not exist.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct AlexaConfig {
    pub enabled: bool,
    /// Accepted `amzn1.ask.skill.*` application ids.
    pub skill_ids: Vec<String>,
    /// Allowed request timestamp skew; clamped to Amazon's 150 s maximum.
    pub timestamp_tolerance_seconds: u64,
    /// Deadline for the Gail completion (Alexa waits about 8 s in total).
    pub answer_deadline_ms: u64,
    /// How long a downloaded signing chain is cached.
    pub cert_cache_ttl_seconds: u64,
    /// Send a Progressive Response ("One moment...") when an answer is slow.
    pub progressive_response: bool,
    /// Upper bound on generated tokens for a spoken answer.
    pub max_answer_tokens: u32,
}

impl Default for AlexaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            skill_ids: Vec::new(),
            timestamp_tolerance_seconds: MAX_TIMESTAMP_TOLERANCE_SECONDS,
            answer_deadline_ms: 6500,
            cert_cache_ttl_seconds: 3600,
            progressive_response: true,
            max_answer_tokens: 350,
        }
    }
}

impl AlexaConfig {
    /// Apply `GAIL_ALEXA_ENABLED` / `GAIL_ALEXA_SKILL_IDS` and clamp limits.
    pub fn normalize(&mut self) {
        self.apply_overrides(
            std::env::var("GAIL_ALEXA_ENABLED").ok(),
            std::env::var("GAIL_ALEXA_SKILL_IDS").ok(),
        );
    }

    pub fn apply_overrides(&mut self, enabled: Option<String>, skill_ids: Option<String>) {
        if let Some(value) = enabled.map(|v| v.trim().to_ascii_lowercase())
            && !value.is_empty()
        {
            self.enabled = matches!(value.as_str(), "1" | "true" | "yes" | "on");
        }
        if let Some(value) = skill_ids.filter(|v| !v.trim().is_empty()) {
            self.skill_ids = value
                .split(|c: char| c == ',' || c.is_whitespace())
                .map(str::to_string)
                .collect();
        }
        self.skill_ids = self
            .skill_ids
            .iter()
            .map(|id| id.trim().to_string())
            .filter(|id| !id.is_empty())
            .collect();
        self.skill_ids.dedup();
        self.timestamp_tolerance_seconds = self
            .timestamp_tolerance_seconds
            .clamp(1, MAX_TIMESTAMP_TOLERANCE_SECONDS);
        self.answer_deadline_ms = self.answer_deadline_ms.clamp(500, 7500);
        self.cert_cache_ttl_seconds = self.cert_cache_ttl_seconds.clamp(60, 86_400);
        self.max_answer_tokens = self.max_answer_tokens.clamp(32, 2048);
    }
}

// ---------------------------------------------------------------------------
// Request verification
// ---------------------------------------------------------------------------

/// A verification failure, mapped to a 400/401 response without detail
/// beyond a short reason code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rejection {
    pub status: StatusCode,
    pub reason: &'static str,
}

impl Rejection {
    const fn bad_request(reason: &'static str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            reason,
        }
    }
    const fn unauthorized(reason: &'static str) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            reason,
        }
    }
}

impl IntoResponse for Rejection {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.reason }))).into_response()
    }
}

/// Validate `SignatureCertChainUrl` per Amazon's rules. The `url` crate
/// lower-cases the scheme and host, removes `..` segments and drops an
/// explicit default port, so the checks run on the normalised form.
pub fn validate_cert_chain_url(raw: &str) -> Result<Url, Rejection> {
    let invalid = Rejection::bad_request("invalid_signature_cert_chain_url");
    let url = Url::parse(raw.trim()).map_err(|_| invalid)?;
    let host_ok = url
        .host_str()
        .is_some_and(|host| host.eq_ignore_ascii_case("s3.amazonaws.com"));
    if url.scheme() != "https"
        || !host_ok
        || matches!(url.port(), Some(port) if port != 443)
        || !url.username().is_empty()
        || url.password().is_some()
        || !url.path().starts_with("/echo.api/")
    {
        return Err(invalid);
    }
    Ok(url)
}

/// Parse a PEM bundle (leaf first) into DER certificates.
pub fn parse_pem_chain(pem: &[u8]) -> Result<Vec<CertificateDer<'static>>, Rejection> {
    let invalid = Rejection::unauthorized("invalid_certificate_chain");
    if pem.len() > MAX_CERT_CHAIN_BYTES {
        return Err(invalid);
    }
    let certs = CertificateDer::pem_slice_iter(pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid)?;
    if certs.is_empty() || certs.len() > 8 {
        return Err(invalid);
    }
    Ok(certs)
}

/// Verify the chain at `now`: validity period, path to one of `anchors`, and
/// the `echo-api.amazon.com` SAN on the leaf.
pub fn verify_chain(
    certs: &[CertificateDer<'_>],
    anchors: &[TrustAnchor<'_>],
    now: UnixTime,
) -> Result<(), Rejection> {
    let (leaf, intermediates) = certs
        .split_first()
        .ok_or(Rejection::unauthorized("invalid_certificate_chain"))?;
    let end_entity = webpki::EndEntityCert::try_from(leaf)
        .map_err(|_| Rejection::unauthorized("invalid_certificate"))?;
    end_entity
        .verify_for_usage(
            webpki::ALL_VERIFICATION_ALGS,
            anchors,
            intermediates,
            now,
            webpki::KeyUsage::server_auth(),
            None,
            None,
        )
        .map_err(|error| {
            Rejection::unauthorized(match error {
                webpki::Error::CertExpired { .. } => "certificate_expired",
                webpki::Error::CertNotValidYet { .. } => "certificate_not_yet_valid",
                _ => "untrusted_certificate_chain",
            })
        })?;
    let subject = ServerName::try_from(ECHO_API_SUBJECT).expect("static DNS name is valid");
    end_entity
        .verify_is_valid_for_subject_name(&subject)
        .map_err(|_| Rejection::unauthorized("certificate_subject_mismatch"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignatureKind {
    Sha256,
    LegacySha1,
}

/// Prefer `Signature-256`; fall back to `Signature` only when it is absent.
pub fn select_signature(headers: &HeaderMap) -> Result<(SignatureKind, Vec<u8>), Rejection> {
    let (kind, value) = if let Some(value) = headers.get("signature-256") {
        (SignatureKind::Sha256, value)
    } else if let Some(value) = headers.get("signature") {
        (SignatureKind::LegacySha1, value)
    } else {
        return Err(Rejection::bad_request("missing_signature"));
    };
    let text = value
        .to_str()
        .map_err(|_| Rejection::bad_request("invalid_signature_encoding"))?;
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(compact)
        .map_err(|_| Rejection::bad_request("invalid_signature_encoding"))?;
    Ok((kind, bytes))
}

/// RSA PKCS#1 v1.5 with SHA-1, used only for Amazon's legacy `Signature`
/// header. Never offered for certificate path validation.
#[derive(Debug)]
struct RsaPkcs1Sha1Legacy;

impl SignatureVerificationAlgorithm for RsaPkcs1Sha1Legacy {
    fn verify_signature(
        &self,
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), InvalidSignature> {
        ring::signature::UnparsedPublicKey::new(
            &ring::signature::RSA_PKCS1_2048_8192_SHA1_FOR_LEGACY_USE_ONLY,
            public_key,
        )
        .verify(message, signature)
        .map_err(|_| InvalidSignature)
    }

    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::RSA_ENCRYPTION
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
        // Only used when verifying certificate signatures, which this
        // algorithm is never offered for.
        alg_id::RSA_PKCS1_SHA256
    }
}

static RSA_PKCS1_SHA1_LEGACY: RsaPkcs1Sha1Legacy = RsaPkcs1Sha1Legacy;

/// Verify the request-body signature with the leaf certificate's key.
pub fn verify_body_signature(
    leaf: &CertificateDer<'_>,
    body: &[u8],
    kind: SignatureKind,
    signature: &[u8],
) -> Result<(), Rejection> {
    let end_entity = webpki::EndEntityCert::try_from(leaf)
        .map_err(|_| Rejection::unauthorized("invalid_certificate"))?;
    let algorithm: &dyn SignatureVerificationAlgorithm = match kind {
        SignatureKind::Sha256 => webpki::ring::RSA_PKCS1_2048_8192_SHA256,
        SignatureKind::LegacySha1 => &RSA_PKCS1_SHA1_LEGACY,
    };
    end_entity
        .verify_signature(algorithm, body, signature)
        .map_err(|_| Rejection::unauthorized("signature_mismatch"))
}

/// Parse `request.timestamp` (ISO 8601, or epoch seconds/milliseconds) into
/// Unix seconds.
pub fn parse_timestamp(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => {
            let raw = number.as_i64()?;
            Some(if raw > 100_000_000_000 {
                raw / 1000
            } else {
                raw
            })
        }
        Value::String(text) => parse_iso8601(text.trim()),
        _ => None,
    }
}

fn parse_iso8601(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    if bytes.len() < 20 || bytes[4] != b'-' || bytes[7] != b'-' || bytes[13] != b':' {
        return None;
    }
    if !matches!(bytes[10], b'T' | b't' | b' ') || bytes[16] != b':' {
        return None;
    }
    let num = |range: std::ops::Range<usize>| -> Option<i64> {
        let part = text.get(range)?;
        part.bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| part.parse().ok())?
    };
    let (year, month, day) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (hour, minute, second) = (num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let mut rest = &text[19..];
    if let Some(fraction) = rest.strip_prefix('.') {
        let digits = fraction.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return None;
        }
        rest = &fraction[digits..];
    }
    let offset = match rest {
        "Z" | "z" => 0,
        _ if rest.len() == 6 && matches!(&rest[3..4], ":") => {
            let sign = match &rest[0..1] {
                "+" => 1,
                "-" => -1,
                _ => return None,
            };
            let hours: i64 = rest[1..3].parse().ok()?;
            let minutes: i64 = rest[4..6].parse().ok()?;
            sign * (hours * 3600 + minutes * 60)
        }
        _ => return None,
    };
    Some(days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second - offset)
}

/// Howard Hinnant's days-from-civil algorithm (proleptic Gregorian).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub fn check_timestamp(envelope: &Value, now_unix: i64, tolerance: u64) -> Result<(), Rejection> {
    let timestamp = envelope
        .pointer("/request/timestamp")
        .and_then(parse_timestamp)
        .ok_or(Rejection::bad_request("invalid_timestamp"))?;
    if timestamp.abs_diff(now_unix) > tolerance.min(MAX_TIMESTAMP_TOLERANCE_SECONDS) {
        return Err(Rejection::bad_request("timestamp_out_of_range"));
    }
    Ok(())
}

pub fn check_application_id(envelope: &Value, allowed: &[String]) -> Result<(), Rejection> {
    let ids: Vec<&str> = [
        "/session/application/applicationId",
        "/context/System/application/applicationId",
    ]
    .iter()
    .filter_map(|pointer| envelope.pointer(pointer).and_then(Value::as_str))
    .collect();
    if ids.is_empty() {
        return Err(Rejection::bad_request("missing_application_id"));
    }
    if allowed.is_empty() || !ids.iter().all(|id| allowed.iter().any(|ok| ok == id)) {
        return Err(Rejection::bad_request("application_id_mismatch"));
    }
    Ok(())
}

struct CachedChain {
    certs: Arc<Vec<CertificateDer<'static>>>,
    fetched: Instant,
}

/// Downloads, caches and verifies Amazon signing chains.
pub struct AlexaVerifier {
    client: reqwest::Client,
    anchors: Vec<TrustAnchor<'static>>,
    cache: Mutex<HashMap<String, CachedChain>>,
    ttl: Duration,
}

impl AlexaVerifier {
    /// Verifier trusting the bundled webpki (Mozilla) roots.
    pub fn new(cache_ttl: Duration) -> Self {
        Self::with_trust_anchors(webpki_roots::TLS_SERVER_ROOTS.to_vec(), cache_ttl)
    }

    pub fn with_trust_anchors(anchors: Vec<TrustAnchor<'static>>, cache_ttl: Duration) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(3))
            .redirect(reqwest::redirect::Policy::none())
            .https_only(true)
            .build()
            .unwrap_or_default();
        Self {
            client,
            anchors,
            cache: Mutex::new(HashMap::new()),
            ttl: cache_ttl,
        }
    }

    /// Seed the cache (used by tests and offline fixtures).
    pub fn insert_chain(&self, url: &str, pem: &[u8]) -> Result<(), Rejection> {
        let url = validate_cert_chain_url(url)?;
        let certs = parse_pem_chain(pem)?;
        self.store(url.as_str(), Arc::new(certs));
        Ok(())
    }

    fn store(&self, key: &str, certs: Arc<Vec<CertificateDer<'static>>>) {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= MAX_CACHED_CHAINS && !cache.contains_key(key) {
            let ttl = self.ttl;
            cache.retain(|_, entry| entry.fetched.elapsed() < ttl);
            if cache.len() >= MAX_CACHED_CHAINS
                && let Some(oldest) = cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.fetched)
                    .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest);
            }
        }
        cache.insert(
            key.to_string(),
            CachedChain {
                certs,
                fetched: Instant::now(),
            },
        );
    }

    async fn chain_for(&self, url: &Url) -> Result<Arc<Vec<CertificateDer<'static>>>, Rejection> {
        {
            let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = cache.get(url.as_str())
                && entry.fetched.elapsed() < self.ttl
            {
                return Ok(entry.certs.clone());
            }
        }
        let fetch_failed = Rejection::unauthorized("certificate_chain_unavailable");
        let response = self
            .client
            .get(url.clone())
            .send()
            .await
            .map_err(|_| fetch_failed)?;
        if !response.status().is_success()
            || response
                .content_length()
                .is_some_and(|len| len > MAX_CERT_CHAIN_BYTES as u64)
        {
            return Err(fetch_failed);
        }
        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| fetch_failed)?;
            if body.len() + chunk.len() > MAX_CERT_CHAIN_BYTES {
                return Err(Rejection::unauthorized("invalid_certificate_chain"));
            }
            body.extend_from_slice(&chunk);
        }
        let certs = Arc::new(parse_pem_chain(&body)?);
        self.store(url.as_str(), certs.clone());
        Ok(certs)
    }

    /// Run every check and return the parsed request envelope.
    pub async fn verify_request(
        &self,
        headers: &HeaderMap,
        body: &[u8],
        now: SystemTime,
        config: &AlexaConfig,
    ) -> Result<Value, Rejection> {
        let cert_url = headers
            .get("signaturecertchainurl")
            .and_then(|value| value.to_str().ok())
            .ok_or(Rejection::bad_request("missing_signature_cert_chain_url"))?;
        let cert_url = validate_cert_chain_url(cert_url)?;
        let (kind, signature) = select_signature(headers)?;
        let certs = self.chain_for(&cert_url).await?;
        let since_epoch = now
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Rejection::unauthorized("clock_error"))?;
        verify_chain(
            &certs,
            &self.anchors,
            UnixTime::since_unix_epoch(since_epoch),
        )?;
        verify_body_signature(&certs[0], body, kind, &signature)?;
        let envelope: Value =
            serde_json::from_slice(body).map_err(|_| Rejection::bad_request("invalid_json"))?;
        check_timestamp(
            &envelope,
            since_epoch.as_secs() as i64,
            config.timestamp_tolerance_seconds,
        )?;
        check_application_id(&envelope, &config.skill_ids)?;
        Ok(envelope)
    }
}

// ---------------------------------------------------------------------------
// Speech output
// ---------------------------------------------------------------------------

/// Escape text for inclusion in SSML (XML) and drop characters XML forbids.
pub fn escape_ssml(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        push_escaped(&mut out, c);
    }
    out
}

fn push_escaped(out: &mut String, c: char) {
    match c {
        '&' => out.push_str("&amp;"),
        '<' => out.push_str("&lt;"),
        '>' => out.push_str("&gt;"),
        '"' => out.push_str("&quot;"),
        '\'' => out.push_str("&apos;"),
        '\n' | '\r' | '\t' => out.push(' '),
        c if c.is_control() => {}
        c => out.push(c),
    }
}

/// Wrap text in `<speak>`, escaping it and keeping the whole document within
/// Alexa's 8000-character limit (truncating at a word boundary if needed).
pub fn to_ssml(text: &str) -> String {
    const OPEN: &str = "<speak>";
    const CLOSE: &str = "</speak>";
    const ELLIPSIS: &str = "...";
    let budget = MAX_SSML_CHARS - OPEN.len() - CLOSE.len();
    let escaped = escape_ssml(text);
    if escaped.chars().count() <= budget {
        return format!("{OPEN}{escaped}{CLOSE}");
    }
    let limit = budget - ELLIPSIS.len();
    let mut body = String::new();
    let mut count = 0;
    let mut last_space = 0;
    for c in text.chars() {
        let mut piece = String::new();
        push_escaped(&mut piece, c);
        let len = piece.chars().count();
        if count + len > limit {
            break;
        }
        if piece == " " {
            last_space = body.len();
        }
        body.push_str(&piece);
        count += len;
    }
    if last_space > 0 {
        body.truncate(last_space);
    }
    format!("{OPEN}{}{ELLIPSIS}{CLOSE}", body.trim_end())
}

/// Make model output suitable for speech: strip markdown and collapse
/// whitespace.
pub fn spoken_text(text: &str) -> String {
    static LINK: once_cell::sync::Lazy<regex::Regex> = once_cell::sync::Lazy::new(|| {
        regex::Regex::new(r"!?\[([^\]]*)\]\([^)]*\)").expect("valid regex")
    });
    static LIST_MARKER: once_cell::sync::Lazy<regex::Regex> = once_cell::sync::Lazy::new(|| {
        regex::Regex::new(r"(?m)^\s*(?:[-*+•]|#{1,6}|>)\s+").expect("valid regex")
    });
    let without_links = LINK.replace_all(text, "$1");
    let without_markers = LIST_MARKER.replace_all(&without_links, "");
    let cleaned: String = without_markers
        .replace("```", " ")
        .chars()
        .filter(|c| !matches!(c, '*' | '`' | '#' | '|'))
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn ssml_speech(text: &str) -> Value {
    json!({ "type": "SSML", "ssml": to_ssml(text) })
}

/// Build an Alexa response envelope.
pub fn speech_response(
    text: &str,
    reprompt: Option<&str>,
    end_session: bool,
    session_attributes: Option<Value>,
) -> Value {
    let mut response = json!({
        "outputSpeech": ssml_speech(text),
        "shouldEndSession": end_session,
    });
    if let Some(reprompt) = reprompt {
        response["reprompt"] = json!({ "outputSpeech": ssml_speech(reprompt) });
    }
    let mut envelope = json!({ "version": "1.0", "response": response });
    if let Some(attributes) = session_attributes {
        envelope["sessionAttributes"] = attributes;
    }
    envelope
}

fn empty_response() -> Value {
    json!({ "version": "1.0", "response": {} })
}

// ---------------------------------------------------------------------------
// Conversation handling
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AskError {
    /// Governance (Aria) refused the request or response.
    Blocked,
    Failed,
}

impl AskError {
    pub fn from_gail(error: &GailError) -> Self {
        match error {
            GailError::Upstream {
                provider, status, ..
            } if provider == "aria" && *status == Some(StatusCode::FORBIDDEN) => Self::Blocked,
            _ => Self::Failed,
        }
    }
}

pub type AskFuture = BoxFuture<'static, Result<String, AskError>>;
/// Sends a chat transcript through Gail and returns the answer text.
pub type Asker = Arc<dyn Fn(Vec<ChatMessage>) -> AskFuture + Send + Sync>;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    pub q: String,
    pub a: String,
}

struct PendingAnswer {
    handle: JoinHandle<Result<String, AskError>>,
    query: String,
    history: Vec<Turn>,
    created: Instant,
}

pub struct AlexaRuntime {
    config: AlexaConfig,
    verifier: AlexaVerifier,
    asker: Asker,
    http: reqwest::Client,
    pending: Mutex<HashMap<String, PendingAnswer>>,
    progressive_delay: Duration,
    allow_any_directive_endpoint: bool,
}

fn text_message(role: &str, text: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: MessageContent::Text(text.to_string()),
    }
}

fn truncate_chars(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}

/// `AMAZON.SearchQuery` slots must be preceded by a carrier phrase, so the
/// interaction model splits questions across `AskAaron*` intents by their
/// leading words. The carrier is not part of the slot value; this table
/// restores it so "what is the tallest mountain" reaches Gail intact.
/// `AskAaronIntent` itself is the catch-all whose carriers ("about",
/// "to", ...) carry no meaning of their own.
pub const QUESTION_INTENTS: &[(&str, &str)] = &[
    ("AskAaronIntent", ""),
    ("AskAaronTellMeIntent", "tell me"),
    ("AskAaronExplainIntent", "explain"),
    ("AskAaronDefineIntent", "define"),
    ("AskAaronWhatIntent", "what"),
    ("AskAaronWhatIsIntent", "what is"),
    ("AskAaronWhoIntent", "who"),
    ("AskAaronWhoIsIntent", "who is"),
    ("AskAaronWhereIntent", "where"),
    ("AskAaronWhereIsIntent", "where is"),
    ("AskAaronWhenIntent", "when"),
    ("AskAaronWhyIntent", "why"),
    ("AskAaronHowIntent", "how"),
    ("AskAaronHowIsIntent", "how is"),
    ("AskAaronWhichIntent", "which"),
    ("AskAaronIsIntent", "is"),
    ("AskAaronAreIntent", "are"),
    ("AskAaronWhetherIntent", "whether"),
    ("AskAaronCanIntent", "can"),
    ("AskAaronCouldIntent", "could"),
    ("AskAaronDoIntent", "do"),
    ("AskAaronDoesIntent", "does"),
    ("AskAaronDidIntent", "did"),
    ("AskAaronShouldIntent", "should"),
    ("AskAaronWillIntent", "will"),
    ("AskAaronWouldIntent", "would"),
];

/// Carrier words to restore in front of the slot for a question intent.
pub fn question_prefix(intent: &str) -> Option<&'static str> {
    QUESTION_INTENTS
        .iter()
        .find(|(name, _)| *name == intent)
        .map(|(_, prefix)| *prefix)
}

/// Build the chat transcript sent to Gail for a spoken question.
pub fn build_messages(history: &[Turn], query: &str) -> Vec<ChatMessage> {
    let mut messages = vec![text_message("system", VOICE_SYSTEM_PROMPT)];
    for turn in history {
        messages.push(text_message("user", &turn.q));
        messages.push(text_message("assistant", &turn.a));
    }
    messages.push(text_message("user", query));
    messages
}

fn history_from(envelope: &Value) -> Vec<Turn> {
    envelope
        .pointer("/session/attributes/history")
        .cloned()
        .and_then(|value| serde_json::from_value::<Vec<Turn>>(value).ok())
        .unwrap_or_default()
        .into_iter()
        .rev()
        .take(MAX_HISTORY_TURNS)
        .rev()
        .map(|turn| Turn {
            q: truncate_chars(&turn.q, MAX_HISTORY_CHARS),
            a: truncate_chars(&turn.a, MAX_HISTORY_CHARS),
        })
        .collect()
}

fn attributes_with(history: &[Turn]) -> Value {
    json!({ "history": history })
}

/// Only Amazon's Alexa API hosts may receive the request's API access token.
pub fn directive_url(api_endpoint: &str, allow_any: bool) -> Option<Url> {
    let base = Url::parse(api_endpoint).ok()?;
    if !allow_any {
        let host = base.host_str()?.to_ascii_lowercase();
        if base.scheme() != "https"
            || !(host == "api.amazonalexa.com" || host.ends_with(".amazonalexa.com"))
            || base.port().is_some()
        {
            return None;
        }
    }
    base.join("/v1/directives").ok()
}

impl AlexaRuntime {
    pub fn new(config: AlexaConfig, asker: Asker) -> Self {
        let verifier = AlexaVerifier::new(Duration::from_secs(config.cert_cache_ttl_seconds));
        Self::with_verifier(config, verifier, asker)
    }

    pub fn with_verifier(config: AlexaConfig, verifier: AlexaVerifier, asker: Asker) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(1))
            .timeout(Duration::from_secs(3))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_default();
        Self {
            config,
            verifier,
            asker,
            http,
            pending: Mutex::new(HashMap::new()),
            progressive_delay: DEFAULT_PROGRESSIVE_DELAY,
            allow_any_directive_endpoint: false,
        }
    }

    /// Test hook: shorten the progressive-response delay and allow a mock
    /// directive endpoint.
    #[cfg(test)]
    pub(crate) fn with_test_directives(mut self, delay: Duration) -> Self {
        self.progressive_delay = delay;
        self.allow_any_directive_endpoint = true;
        self
    }

    pub fn config(&self) -> &AlexaConfig {
        &self.config
    }

    pub fn verifier(&self) -> &AlexaVerifier {
        &self.verifier
    }

    /// Full HTTP handling: 404 when disabled, 400/401 when verification
    /// fails, otherwise an Alexa response envelope.
    pub async fn handle(&self, headers: &HeaderMap, body: &[u8]) -> Response {
        if !self.config.enabled {
            return StatusCode::NOT_FOUND.into_response();
        }
        let envelope = match self
            .verifier
            .verify_request(headers, body, SystemTime::now(), &self.config)
            .await
        {
            Ok(envelope) => envelope,
            Err(rejection) => {
                tracing::warn!(
                    reason = rejection.reason,
                    status = rejection.status.as_u16(),
                    "GAIL_ALEXA_REJECTED"
                );
                return rejection.into_response();
            }
        };
        Json(self.respond(&envelope).await).into_response()
    }

    /// Dispatch a verified request envelope.
    pub async fn respond(&self, envelope: &Value) -> Value {
        let request_type = envelope
            .pointer("/request/type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let intent = envelope
            .pointer("/request/intent/name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        tracing::info!(
            client_id = CLIENT_ID,
            request_type,
            intent,
            "GAIL_ALEXA_REQUEST"
        );
        let history = history_from(envelope);
        let attributes = Some(attributes_with(&history));
        match request_type {
            "LaunchRequest" => speech_response(GREETING, Some(REPROMPT), false, attributes),
            "IntentRequest" => self.intent(envelope, intent, history).await,
            "SessionEndedRequest" => {
                self.clear_pending(envelope);
                empty_response()
            }
            "System.ExceptionEncountered" => empty_response(),
            _ => speech_response(FALLBACK, Some(REPROMPT), false, attributes),
        }
    }

    async fn intent(&self, envelope: &Value, intent: &str, history: Vec<Turn>) -> Value {
        let attributes = Some(attributes_with(&history));
        match intent {
            name if question_prefix(name).is_some() => {
                let prefix = question_prefix(name).unwrap_or_default();
                let slot = envelope
                    .pointer("/request/intent/slots/query/value")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or_default();
                if slot.is_empty() {
                    return speech_response(EMPTY_QUERY, Some(REPROMPT), false, attributes);
                }
                let query = if prefix.is_empty() {
                    slot.to_string()
                } else {
                    format!("{prefix} {slot}")
                };
                self.clear_pending(envelope);
                let messages = build_messages(&history, &query);
                let handle = tokio::spawn((self.asker)(messages));
                self.await_answer(envelope, handle, query, history).await
            }
            "AMAZON.YesIntent" => match self.take_pending(envelope) {
                Some(pending) => {
                    self.await_answer(envelope, pending.handle, pending.query, pending.history)
                        .await
                }
                None => speech_response(NOTHING_PENDING, Some(REPROMPT), false, attributes),
            },
            "AMAZON.NoIntent" => {
                self.clear_pending(envelope);
                speech_response(
                    "Okay. What else would you like to ask?",
                    Some(REPROMPT),
                    false,
                    attributes,
                )
            }
            "AMAZON.HelpIntent" => speech_response(HELP, Some(REPROMPT), false, attributes),
            "AMAZON.NavigateHomeIntent" => {
                speech_response(GREETING, Some(REPROMPT), false, attributes)
            }
            "AMAZON.StopIntent" | "AMAZON.CancelIntent" => {
                self.clear_pending(envelope);
                speech_response(GOODBYE, None, true, None)
            }
            _ => speech_response(FALLBACK, Some(REPROMPT), false, attributes),
        }
    }

    async fn await_answer(
        &self,
        envelope: &Value,
        mut handle: JoinHandle<Result<String, AskError>>,
        query: String,
        mut history: Vec<Turn>,
    ) -> Value {
        let deadline = Duration::from_millis(self.config.answer_deadline_ms);
        let progress = self.spawn_progressive_response(envelope);
        let outcome = tokio::time::timeout(deadline, &mut handle).await;
        if let Some(progress) = progress {
            progress.abort();
        }
        match outcome {
            Ok(Ok(Ok(text))) => {
                let spoken = spoken_text(&text);
                let spoken = if spoken.is_empty() {
                    FAILED.to_string()
                } else {
                    spoken
                };
                history.push(Turn {
                    q: truncate_chars(&query, MAX_HISTORY_CHARS),
                    a: truncate_chars(&spoken, MAX_HISTORY_CHARS),
                });
                let excess = history.len().saturating_sub(MAX_HISTORY_TURNS);
                history.drain(..excess);
                speech_response(
                    &spoken,
                    Some(FOLLOW_UP_REPROMPT),
                    false,
                    Some(attributes_with(&history)),
                )
            }
            Ok(Ok(Err(AskError::Blocked))) => {
                tracing::info!(client_id = CLIENT_ID, "GAIL_ALEXA_GOVERNANCE_BLOCKED");
                speech_response(
                    BLOCKED,
                    Some(REPROMPT),
                    false,
                    Some(attributes_with(&history)),
                )
            }
            Ok(Ok(Err(AskError::Failed))) | Ok(Err(_)) => speech_response(
                FAILED,
                Some(REPROMPT),
                false,
                Some(attributes_with(&history)),
            ),
            Err(_) => {
                let attributes = Some(attributes_with(&history));
                if let Some(session_id) = session_id(envelope) {
                    self.store_pending(
                        session_id,
                        PendingAnswer {
                            handle,
                            query,
                            history,
                            created: Instant::now(),
                        },
                    );
                } else {
                    handle.abort();
                }
                speech_response(
                    STILL_THINKING,
                    Some(STILL_THINKING_REPROMPT),
                    false,
                    attributes,
                )
            }
        }
    }

    fn spawn_progressive_response(&self, envelope: &Value) -> Option<JoinHandle<()>> {
        if !self.config.progressive_response {
            return None;
        }
        let token = envelope
            .pointer("/context/System/apiAccessToken")
            .and_then(Value::as_str)?
            .to_string();
        let endpoint = envelope
            .pointer("/context/System/apiEndpoint")
            .and_then(Value::as_str)?;
        let request_id = envelope
            .pointer("/request/requestId")
            .and_then(Value::as_str)?
            .to_string();
        let url = directive_url(endpoint, self.allow_any_directive_endpoint)?;
        let client = self.http.clone();
        let delay = self.progressive_delay;
        Some(tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            let body = json!({
                "header": { "requestId": request_id },
                "directive": {
                    "type": "VoicePlayer.Speak",
                    "speech": to_ssml(PROGRESSIVE_SPEECH),
                },
            });
            if let Err(error) = client.post(url).bearer_auth(token).json(&body).send().await {
                tracing::debug!(error = %error, "Alexa progressive response failed");
            }
        }))
    }

    fn store_pending(&self, session_id: String, pending: PendingAnswer) {
        let mut map = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, entry| {
            let keep = entry.created.elapsed() < PENDING_ANSWER_TTL;
            if !keep {
                entry.handle.abort();
            }
            keep
        });
        if map.len() >= MAX_PENDING_ANSWERS
            && let Some(oldest) = map
                .iter()
                .min_by_key(|(_, entry)| entry.created)
                .map(|(key, _)| key.clone())
            && let Some(entry) = map.remove(&oldest)
        {
            entry.handle.abort();
        }
        if let Some(previous) = map.insert(session_id, pending) {
            previous.handle.abort();
        }
    }

    fn take_pending(&self, envelope: &Value) -> Option<PendingAnswer> {
        let session_id = session_id(envelope)?;
        let mut map = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        map.remove(&session_id)
            .filter(|entry| entry.created.elapsed() < PENDING_ANSWER_TTL)
    }

    fn clear_pending(&self, envelope: &Value) {
        if let Some(entry) = self.take_pending(envelope) {
            entry.handle.abort();
        }
    }
}

fn session_id(envelope: &Value) -> Option<String> {
    envelope
        .pointer("/session/sessionId")
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
#[path = "alexa_tests.rs"]
pub(crate) mod tests;
