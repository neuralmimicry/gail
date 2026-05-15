/// Multi-AI advisory system for trading decisions.
///
/// Fires all configured AI providers in parallel via `GailService::direct_complete()`,
/// sends each a structured prompt describing the current market state, and aggregates
/// their responses into a consensus recommendation.
use std::{
    collections::{HashMap, HashSet},
    env,
    time::Duration,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::task::JoinSet;
use tracing::{debug, warn};

use crate::{
    config::ProviderProfile,
    errors::message_indicates_quota,
    models::{ChatMessage, MessageContent, ProviderCompletionRequest},
    orchestration::GailService,
    providers::normalize_provider_type,
};

use super::octobot::{MarketSnapshot, OctobotPortfolio};
use super::refiner::ResearchContext;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Advice returned by a single AI provider.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AiAdvice {
    pub provider: String,
    pub model: Option<String>,
    /// Action recommended: "buy", "sell", "hold", "strong_buy", "strong_sell"
    pub action: String,
    /// Provider's confidence in its recommendation (0.0–1.0).
    pub confidence: f64,
    /// Brief reasoning text.
    pub reasoning: String,
    /// Suggested USD amount (may be None if provider didn't specify).
    pub suggested_amount_usd: Option<f64>,
    /// Provider-estimated risk in [0, 1], where 1 is highest risk.
    pub risk_score: f64,
    /// Provider-supplied risk flags that explain uncertainty or trade hazards.
    pub risk_flags: Vec<String>,
    /// Provider-suggested target symbol when it differs from the highest-ranked market.
    pub target_symbol: Option<String>,
    /// Raw LLM response (for logging).
    pub raw_response: String,
    /// Whether parsing the structured JSON succeeded.
    pub parsed_ok: bool,
    /// Provider quality weight applied during aggregation (from MetricsStore EWMA).
    pub weight: f64,
}

/// Aggregated consensus from all AI providers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AiConsensus {
    /// Majority-weighted action label.
    pub action: String,
    /// Blended confidence score (0.0–1.0).
    pub confidence: f64,
    /// Scalar consensus in [−1, +1]: −1 = unanimous sell, +1 = unanimous buy.
    pub signal: f64,
    /// Distribution of weighted votes per action.
    pub vote_distribution: Value,
    /// Individual provider advices (for logging / transparency).
    pub advices: Vec<AiAdvice>,
    /// Number of providers that responded.
    pub responders: usize,
    /// Number of providers that failed / timed out.
    pub failures: usize,
}

impl AiConsensus {
    pub fn uncertain() -> Self {
        Self {
            action: "hold".to_string(),
            confidence: 0.0,
            signal: 0.0,
            vote_distribution: json!({}),
            advices: Vec::new(),
            responders: 0,
            failures: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Advisor
// ---------------------------------------------------------------------------

pub struct TradingAdvisor {
    service: GailService,
    timeout: Duration,
}

impl TradingAdvisor {
    pub fn new(service: GailService, advisor_timeout_seconds: f64) -> Self {
        Self {
            service,
            timeout: Duration::from_secs_f64(advisor_timeout_seconds),
        }
    }

    /// Consult all configured AI providers in parallel and aggregate their advice.
    pub async fn consult_all(
        &self,
        market_snapshots: &[MarketSnapshot],
        research: &ResearchContext,
        portfolio: &OctobotPortfolio,
        max_advisors: usize,
    ) -> AiConsensus {
        let providers = self.select_providers(max_advisors);
        if providers.is_empty() {
            warn!("trading: no providers available for AI advisory");
            return AiConsensus::uncertain();
        }

        let prompt = build_advisory_prompt(market_snapshots, research, portfolio);
        let system = advisory_system_prompt();
        let timeout_secs = self.timeout.as_secs();

        let mut join_set: JoinSet<AiAdvice> = JoinSet::new();
        for profile in providers {
            let svc = self.service.clone();
            let prompt_clone = prompt.clone();
            let system_clone = system.clone();
            let profile_clone = profile.clone();
            join_set.spawn(async move {
                query_provider(svc, profile_clone, prompt_clone, system_clone, timeout_secs).await
            });
        }

        let mut advices: Vec<AiAdvice> = Vec::new();
        let mut failures = 0usize;
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(advice) => {
                    if advice.parsed_ok {
                        advices.push(advice);
                    } else {
                        failures += 1;
                        advices.push(advice); // keep it for logs even if unparsed
                    }
                }
                Err(_) => failures += 1,
            }
        }

        aggregate_consensus(advices, failures)
    }

    /// Pick providers to consult, up to max_advisors, ordered by quality weight.
    fn select_providers(&self, max: usize) -> Vec<ProviderProfile> {
        select_trading_profiles(&self.service.config().providers, max)
    }
}

fn select_trading_profiles(profiles: &[ProviderProfile], max: usize) -> Vec<ProviderProfile> {
    let max = max.max(1);
    let mut ranked = profiles
        .iter()
        .filter(|profile| provider_can_advise(profile))
        .cloned()
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        provider_advisor_score(right)
            .partial_cmp(&provider_advisor_score(left))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.name.cmp(&right.name))
    });

    let mut selected = Vec::new();
    let mut seen_provider_types = HashSet::new();
    for profile in &ranked {
        if selected.len() >= max {
            return selected;
        }
        let provider_type = normalize_provider_type(profile.provider_type.as_str());
        if seen_provider_types.insert(provider_type) {
            selected.push(profile.clone());
        }
    }

    let mut selected_ids = selected
        .iter()
        .map(provider_identity)
        .collect::<HashSet<_>>();
    for profile in ranked {
        if selected.len() >= max {
            break;
        }
        if selected_ids.insert(provider_identity(&profile)) {
            selected.push(profile);
        }
    }
    selected
}

fn provider_identity(profile: &ProviderProfile) -> String {
    format!(
        "{}:{}:{}",
        normalize_provider_type(profile.provider_type.as_str()),
        profile.name.to_ascii_lowercase(),
        profile
            .model
            .clone()
            .unwrap_or_default()
            .to_ascii_lowercase()
    )
}

fn provider_advisor_score(profile: &ProviderProfile) -> f64 {
    let mut score = profile.weight;
    if profile.preferred {
        score += 0.25;
    }
    for specialty in &profile.specialties {
        let specialty = specialty.to_ascii_lowercase();
        if [
            "trading",
            "market",
            "risk",
            "reasoning",
            "analysis",
            "research",
        ]
        .iter()
        .any(|hint| specialty.contains(hint))
        {
            score += 0.08;
        }
        if specialty.contains("local") || specialty.contains("privacy") {
            score += 0.04;
        }
    }
    for role in &profile.roles {
        let role = role.to_ascii_lowercase();
        if matches!(
            role.as_str(),
            "assistant" | "researcher" | "planner" | "reviewer"
        ) {
            score += 0.03;
        }
    }
    score
}

fn provider_can_advise(profile: &ProviderProfile) -> bool {
    let provider_type = normalize_provider_type(profile.provider_type.as_str());
    if provider_type.is_empty() {
        return false;
    }
    match provider_type.as_str() {
        "openai" => has_usable_value(profile.api_key.as_deref()) || env_has("OPENAI_API_KEY"),
        "nvidia" => has_usable_value(profile.api_key.as_deref()) || env_has("NVIDIA_API_KEY"),
        "gemini" => {
            has_usable_value(profile.api_key.as_deref())
                || has_usable_value(profile.access_token.as_deref())
                || env_has("GEMINI_API_KEY")
                || env_has("GEMINI_ACCESS_TOKEN")
                || env_has("GOOGLE_ACCESS_TOKEN")
        }
        "ollama" => true,
        _ => true,
    }
}

fn has_usable_value(value: Option<&str>) -> bool {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            let lowered = value.to_ascii_lowercase();
            !matches!(
                lowered.as_str(),
                "none" | "null" | "nil" | "undefined" | "changeme"
            )
        })
        .unwrap_or(false)
}

fn env_has(name: &str) -> bool {
    env::var(name)
        .ok()
        .as_deref()
        .map(|value| has_usable_value(Some(value)))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Provider query
// ---------------------------------------------------------------------------

async fn query_provider(
    service: GailService,
    profile: ProviderProfile,
    prompt: String,
    system: String,
    timeout_secs: u64,
) -> AiAdvice {
    let request = ProviderCompletionRequest {
        provider: profile.provider_type.clone(),
        model: profile.model.clone(),
        api_key: profile.api_key.clone(),
        access_token: profile.access_token.clone(),
        base_url: profile.base_url.clone(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(prompt),
        }],
        system: Some(system),
        max_tokens: Some(512),
        temperature: Some(0.2),
        timeout_seconds: Some(timeout_secs),
        reasoning_effort: None,
        request_category: Some("trading_advisory".to_string()),
        workflow: Some("trading".to_string()),
        role: Some("assistant".to_string()),
        min_model_size_b: None,
        strict_no_downgrade: None,
    };

    let weight = profile.weight;
    let provider_name = profile.name.clone();
    let provider_type = profile.provider_type.clone();

    match service.direct_complete(request).await {
        Ok(resp) => {
            let raw = resp.text.clone();
            let parsed = parse_advisory_response(&raw);
            AiAdvice {
                provider: provider_name,
                model: Some(resp.model),
                action: parsed.action,
                confidence: parsed.confidence,
                reasoning: parsed.reasoning,
                suggested_amount_usd: parsed.suggested_amount_usd,
                risk_score: parsed.risk_score,
                risk_flags: parsed.risk_flags,
                target_symbol: parsed.target_symbol,
                raw_response: raw,
                parsed_ok: parsed.parsed_ok,
                weight,
            }
        }
        Err(err) => {
            debug!(
                "trading: advisory query to {} failed: {}",
                provider_type, err
            );
            AiAdvice {
                provider: provider_name,
                model: None,
                action: "hold".to_string(),
                confidence: 0.0,
                reasoning: format!("provider error: {err}"),
                suggested_amount_usd: None,
                risk_score: 1.0,
                risk_flags: if message_indicates_quota(&err.to_string()) {
                    vec!["provider_rate_limited".to_string()]
                } else {
                    vec!["provider_error".to_string()]
                },
                target_symbol: None,
                raw_response: String::new(),
                parsed_ok: false,
                weight,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

fn advisory_system_prompt() -> String {
    r#"You are a professional cryptocurrency trading advisor with deep expertise in technical and fundamental analysis.

Your task: analyse the provided market data and research context, then provide a concise trading recommendation.

CRITICAL: Your response MUST be valid JSON only, with NO additional text before or after. Use this exact schema:
{
  "action": "buy" | "sell" | "hold" | "strong_buy" | "strong_sell",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one or two sentences explaining your recommendation>",
  "suggested_amount_usd": <float or null>,
  "risk_score": <float 0.0 to 1.0>,
  "risk_flags": ["<short risk or uncertainty labels>"],
  "target_symbol": "<symbol from MARKET DATA or null>"
}

Rules:
- confidence 0.0 means complete uncertainty; 1.0 means absolute certainty
- If signals conflict or data is insufficient, recommend "hold" with low confidence
- Be conservative: only recommend strong_buy/strong_sell with confidence > 0.75 and risk_score < 0.45
- Micro-trade context: position sizes are small (< $25 USD), but do not force trades when evidence is weak
- Consider portfolio exposure to avoid over-concentration
- Treat unvalued or illiquid portfolio assets as risk flags, not as trade targets
- Do not suggest assets that are missing from MARKET DATA
- Only output valid JSON, no markdown fences, no commentary"#
        .to_string()
}

fn build_advisory_prompt(
    snapshots: &[MarketSnapshot],
    research: &ResearchContext,
    portfolio: &OctobotPortfolio,
) -> String {
    let market_section = if snapshots.is_empty() {
        "No market data available.".to_string()
    } else {
        let lines: Vec<String> = snapshots
            .iter()
            .take(10)
            .map(|s| {
                let trend = match s.price_change_pct_24h {
                    Some(p) if p > 3.0 => "↑↑",
                    Some(p) if p > 0.5 => "↑",
                    Some(p) if p < -3.0 => "↓↓",
                    Some(p) if p < -0.5 => "↓",
                    _ => "→",
                };
                format!(
                    "  {}/{}: price={:.4}, 24h_chg={:.2}% {}, vol24h={:.2}",
                    s.exchange,
                    s.symbol,
                    s.price,
                    s.price_change_pct_24h.unwrap_or(0.0),
                    trend,
                    s.volume_24h.unwrap_or(0.0),
                )
            })
            .collect();
        lines.join("\n")
    };

    let portfolio_section = {
        let total = portfolio
            .total_value_usd
            .map(|v| format!("${v:.2}"))
            .unwrap_or_else(|| "unknown".to_string());
        let top: Vec<String> = portfolio
            .currencies
            .iter()
            .filter(|(_, b)| b.total > 0.0)
            .take(8)
            .map(|(sym, b)| {
                format!(
                    "  {sym}: {:.6} (${:.2})",
                    b.total,
                    b.value_usd.unwrap_or(0.0)
                )
            })
            .collect();
        let unvalued = portfolio
            .currencies
            .iter()
            .filter(|(_, b)| b.total > 0.0 && b.value_usd.is_none())
            .map(|(sym, _)| sym.as_str())
            .take(12)
            .collect::<Vec<_>>();
        let unvalued_line = if unvalued.is_empty() {
            "Unvalued holdings: none".to_string()
        } else {
            format!("Unvalued holdings: {}", unvalued.join(", "))
        };
        format!(
            "Total portfolio value: {total}\n{unvalued_line}\n{}",
            top.join("\n")
        )
    };

    let research_section = if research.is_empty() {
        "No research context available.".to_string()
    } else {
        let citations = research
            .matches
            .iter()
            .take(6)
            .map(|m| {
                let src = m
                    .source
                    .as_deref()
                    .or(m.citation.as_deref())
                    .unwrap_or("unknown_source");
                format!("  - {:.2} {}", m.score, src)
            })
            .collect::<Vec<_>>();
        let citations_block = if citations.is_empty() {
            "Top citations: none".to_string()
        } else {
            format!("Top citations:\n{}", citations.join("\n"))
        };
        format!(
            "Research source: {}\n{}\nResearch context:\n{}",
            research.source,
            citations_block,
            &research.context[..research.context.len().min(1800)]
        )
    };

    format!(
        "MARKET DATA:\n{market_section}\n\nPORTFOLIO:\n{portfolio_section}\n\nRESEARCH:\n{research_section}\n\nProvide your trading recommendation as JSON."
    )
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ParsedAdvisory {
    action: String,
    confidence: f64,
    reasoning: String,
    suggested_amount_usd: Option<f64>,
    risk_score: f64,
    risk_flags: Vec<String>,
    target_symbol: Option<String>,
    parsed_ok: bool,
}

fn parse_advisory_response(raw: &str) -> ParsedAdvisory {
    let Some(value) = extract_advisory_json(raw) else {
        return ParsedAdvisory {
            action: "hold".to_string(),
            confidence: 0.0,
            reasoning: "failed to parse advisory response as JSON".to_string(),
            suggested_amount_usd: None,
            risk_score: 1.0,
            risk_flags: vec!["invalid_json".to_string()],
            target_symbol: None,
            parsed_ok: false,
        };
    };

    let Some(object) = advisory_payload_object(&value) else {
        return ParsedAdvisory {
            action: "hold".to_string(),
            confidence: 0.0,
            reasoning: "advisory response was not a JSON object".to_string(),
            suggested_amount_usd: None,
            risk_score: 1.0,
            risk_flags: vec!["invalid_shape".to_string()],
            target_symbol: None,
            parsed_ok: false,
        };
    };

    if looks_like_schema_echo(object) {
        return ParsedAdvisory {
            action: "hold".to_string(),
            confidence: 0.0,
            reasoning: "model returned a JSON schema instead of a trading advisory".to_string(),
            suggested_amount_usd: None,
            risk_score: 1.0,
            risk_flags: vec!["schema_echo".to_string()],
            target_symbol: None,
            parsed_ok: false,
        };
    }

    let mut action = string_field(object, &["action", "recommendation", "decision"])
        .map(normalise_action)
        .unwrap_or_else(|| "hold".to_string());
    let mut confidence = numeric_field(object, &["confidence", "confidence_score", "probability"])
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let risk_score = risk_score_from_value(
        object
            .get("risk_score")
            .or_else(|| object.get("risk"))
            .or_else(|| object.get("risk_level")),
    );
    if action.starts_with("strong_") && (confidence < 0.75 || risk_score >= 0.45) {
        action = match action.as_str() {
            "strong_buy" => "buy".to_string(),
            "strong_sell" => "sell".to_string(),
            _ => action,
        };
        confidence = confidence.min(0.74);
    }

    let reasoning = string_field(
        object,
        &[
            "reasoning",
            "rationale",
            "reason",
            "explanation",
            "analysis",
        ],
    )
    .unwrap_or("no reasoning provided")
    .trim()
    .chars()
    .take(600)
    .collect::<String>();
    let suggested_amount_usd = numeric_field(
        object,
        &[
            "suggested_amount_usd",
            "amount_usd",
            "trade_amount_usd",
            "size_usd",
        ],
    )
    .filter(|value| *value > 0.0);
    let risk_flags = risk_flags_from_value(
        object
            .get("risk_flags")
            .or_else(|| object.get("flags"))
            .or_else(|| object.get("uncertainties")),
    );
    let target_symbol = string_field(
        object,
        &["target_symbol", "symbol", "trading_pair", "pair", "market"],
    )
    .map(str::trim)
    .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("null"))
    .map(ToOwned::to_owned);

    ParsedAdvisory {
        action,
        confidence,
        reasoning,
        suggested_amount_usd,
        risk_score,
        risk_flags,
        target_symbol,
        parsed_ok: true,
    }
}

fn extract_advisory_json(raw: &str) -> Option<Value> {
    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    serde_json::from_str::<Value>(cleaned).ok().or_else(|| {
        let start = cleaned.find(|ch| ch == '{' || ch == '[')?;
        let end = cleaned.rfind(|ch| ch == '}' || ch == ']')?;
        if end <= start {
            return None;
        }
        serde_json::from_str::<Value>(&cleaned[start..=end]).ok()
    })
}

fn advisory_payload_object(value: &Value) -> Option<&serde_json::Map<String, Value>> {
    match value {
        Value::Object(object) => object
            .get("advice")
            .or_else(|| object.get("recommendation"))
            .or_else(|| object.get("decision"))
            .and_then(Value::as_object)
            .or(Some(object)),
        Value::Array(items) => items.iter().find_map(Value::as_object),
        _ => None,
    }
}

fn looks_like_schema_echo(object: &serde_json::Map<String, Value>) -> bool {
    let schema_keys = ["$defs", "properties", "required", "additionalProperties"];
    let schema_key_count = schema_keys
        .iter()
        .filter(|key| object.contains_key(**key))
        .count();
    schema_key_count >= 2
        && object
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|value| value.eq_ignore_ascii_case("object"))
        && !object.contains_key("action")
}

fn string_field<'a>(object: &'a serde_json::Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|key| object.get(*key)?.as_str())
}

fn numeric_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| numeric_value(object.get(*key)?))
}

fn numeric_value(value: &Value) -> Option<f64> {
    value.as_f64().or_else(|| {
        let raw = value.as_str()?.trim().trim_end_matches('%');
        let parsed = raw.parse::<f64>().ok()?;
        if value.as_str()?.trim().ends_with('%') {
            Some(parsed / 100.0)
        } else {
            Some(parsed)
        }
    })
}

fn risk_score_from_value(value: Option<&Value>) -> f64 {
    match value {
        Some(value) => numeric_value(value)
            .or_else(|| {
                let label = value.as_str()?.trim().to_ascii_lowercase();
                Some(match label.as_str() {
                    "very_low" | "very low" => 0.1,
                    "low" => 0.25,
                    "medium" | "moderate" => 0.5,
                    "high" => 0.8,
                    "very_high" | "very high" => 0.95,
                    _ => return None,
                })
            })
            .unwrap_or(0.5)
            .clamp(0.0, 1.0),
        None => 0.5,
    }
}

fn risk_flags_from_value(value: Option<&Value>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    match value {
        Value::Array(items) => items
            .iter()
            .filter_map(Value::as_str)
            .map(normalise_risk_flag)
            .filter(|value| !value.is_empty())
            .take(8)
            .collect(),
        Value::String(text) => text
            .split([',', ';'])
            .map(normalise_risk_flag)
            .filter(|value| !value.is_empty())
            .take(8)
            .collect(),
        _ => Vec::new(),
    }
}

fn normalise_risk_flag(raw: &str) -> String {
    raw.trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn normalise_action(raw: &str) -> String {
    match raw.to_ascii_lowercase().trim() {
        "strong_buy" | "strongbuy" | "strong buy" => "strong_buy",
        "strong_sell" | "strongsell" | "strong sell" => "strong_sell",
        "buy" => "buy",
        "sell" => "sell",
        _ => "hold",
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Consensus aggregation
// ---------------------------------------------------------------------------

fn action_to_signal(action: &str) -> f64 {
    match action {
        "strong_buy" => 1.0,
        "buy" => 0.5,
        "hold" => 0.0,
        "sell" => -0.5,
        "strong_sell" => -1.0,
        _ => 0.0,
    }
}

fn signal_to_action(signal: f64) -> String {
    match signal {
        s if s >= 0.65 => "strong_buy",
        s if s >= 0.2 => "buy",
        s if s <= -0.65 => "strong_sell",
        s if s <= -0.2 => "sell",
        _ => "hold",
    }
    .to_string()
}

fn aggregate_consensus(advices: Vec<AiAdvice>, failures: usize) -> AiConsensus {
    let responders = advices.iter().filter(|a| a.parsed_ok).count();
    if responders == 0 {
        return AiConsensus {
            failures,
            ..AiConsensus::uncertain()
        };
    }

    // Weighted average signal using quality-weight * confidence.
    let mut weighted_signal = 0.0_f64;
    let mut weighted_abs_signal = 0.0_f64;
    let mut total_weight = 0.0_f64;
    let mut weighted_confidence = 0.0_f64;
    let mut weighted_risk = 0.0_f64;
    let mut vote_map: HashMap<String, f64> = HashMap::new();

    for advice in &advices {
        if !advice.parsed_ok {
            continue;
        }
        let provider_weight = advice.weight.max(0.05);
        let risk_adjustment = (1.0 - (advice.risk_score.clamp(0.0, 1.0) * 0.35)).clamp(0.55, 1.0);
        let effective_weight = (provider_weight * advice.confidence * risk_adjustment).max(0.01);
        let signal = action_to_signal(&advice.action);
        weighted_signal += signal * effective_weight;
        weighted_abs_signal += signal.abs() * effective_weight;
        weighted_confidence += advice.confidence * provider_weight;
        weighted_risk += advice.risk_score.clamp(0.0, 1.0) * provider_weight;
        total_weight += effective_weight;
        *vote_map.entry(advice.action.clone()).or_insert(0.0) += effective_weight;
    }

    let signal = if total_weight > 0.0 {
        weighted_signal / total_weight
    } else {
        0.0
    };
    let action = signal_to_action(signal);

    let provider_weight_sum: f64 = advices
        .iter()
        .filter(|a| a.parsed_ok)
        .map(|a| a.weight.max(0.05))
        .sum();
    let base_confidence = if provider_weight_sum > 0.0 {
        (weighted_confidence / provider_weight_sum).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let average_risk = if provider_weight_sum > 0.0 {
        (weighted_risk / provider_weight_sum).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let agreement = if weighted_abs_signal > 0.0 {
        (weighted_signal.abs() / weighted_abs_signal).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let coverage = responders as f64 / (responders + failures).max(1) as f64;
    let confidence = (base_confidence
        * (0.55 + 0.45 * agreement)
        * (0.65 + 0.35 * coverage)
        * (1.0 - 0.25 * average_risk))
        .clamp(0.0, 1.0);

    let vote_distribution = json!({
        "votes": vote_map,
        "agreement": agreement,
        "coverage": coverage,
        "average_risk": average_risk,
    });

    AiConsensus {
        action,
        confidence,
        signal,
        vote_distribution,
        advices,
        responders,
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(name: &str, provider_type: &str, weight: f64) -> ProviderProfile {
        ProviderProfile {
            name: name.to_string(),
            provider_type: provider_type.to_string(),
            model: Some(format!("{name}-model")),
            api_key: if matches!(provider_type, "ollama") {
                None
            } else {
                Some("token".to_string())
            },
            roles: vec!["assistant".to_string()],
            specialties: vec!["trading".to_string(), "reasoning".to_string()],
            weight,
            ..ProviderProfile::default()
        }
    }

    fn advice(action: &str, confidence: f64, weight: f64, risk_score: f64) -> AiAdvice {
        AiAdvice {
            provider: "test".to_string(),
            model: Some("model".to_string()),
            action: action.to_string(),
            confidence,
            reasoning: "reason".to_string(),
            suggested_amount_usd: None,
            risk_score,
            risk_flags: Vec::new(),
            target_symbol: Some("BTC/USDT".to_string()),
            raw_response: String::new(),
            parsed_ok: true,
            weight,
        }
    }

    #[test]
    fn select_trading_profiles_prefers_provider_diversity_and_keeps_ollama() {
        let profiles = vec![
            profile("nvidia-a", "nvidia", 10.0),
            profile("nvidia-b", "nvidia", 9.0),
            profile("gemini-a", "gemini", 1.0),
            profile("ollama-a", "ollama", 0.5),
        ];
        let selected = select_trading_profiles(&profiles, 3);
        let provider_types = selected
            .iter()
            .map(|profile| normalize_provider_type(profile.provider_type.as_str()))
            .collect::<HashSet<_>>();
        assert_eq!(selected.len(), 3);
        assert!(provider_types.contains("nvidia"));
        assert!(provider_types.contains("gemini"));
        assert!(provider_types.contains("ollama"));
    }

    #[test]
    fn parse_advisory_response_rejects_schema_echo() {
        let parsed = parse_advisory_response(
            r#"{"$defs":{},"properties":{"steps":{}},"required":["steps"],"type":"object"}"#,
        );
        assert!(!parsed.parsed_ok);
        assert_eq!(parsed.action, "hold");
        assert!(parsed.risk_flags.contains(&"schema_echo".to_string()));
    }

    #[test]
    fn parse_advisory_response_accepts_nested_recommendation_and_numeric_strings() {
        let parsed = parse_advisory_response(
            r#"{"recommendation":{"action":"strong buy","confidence":"82%","reasoning":"breakout with volume","suggested_amount_usd":"6.5","risk_score":"low","risk_flags":["thin liquidity"],"target_symbol":"BTC/USDT"}}"#,
        );
        assert!(parsed.parsed_ok);
        assert_eq!(parsed.action, "strong_buy");
        assert!((parsed.confidence - 0.82).abs() < 0.001);
        assert_eq!(parsed.suggested_amount_usd, Some(6.5));
        assert_eq!(parsed.risk_score, 0.25);
        assert_eq!(parsed.risk_flags, vec!["thin_liquidity".to_string()]);
        assert_eq!(parsed.target_symbol.as_deref(), Some("BTC/USDT"));
    }

    #[test]
    fn aggregate_consensus_penalizes_disagreement_and_failures() {
        let clean = aggregate_consensus(
            vec![advice("buy", 0.8, 1.0, 0.2), advice("buy", 0.8, 1.0, 0.2)],
            0,
        );
        let conflicted = aggregate_consensus(
            vec![advice("buy", 0.8, 1.0, 0.2), advice("sell", 0.8, 1.0, 0.2)],
            2,
        );

        assert!(clean.confidence > conflicted.confidence);
        assert_eq!(conflicted.action, "hold");
        assert!(
            conflicted.vote_distribution["agreement"]
                .as_f64()
                .is_some_and(|value| value < 0.2)
        );
        assert!(
            conflicted.vote_distribution["coverage"]
                .as_f64()
                .is_some_and(|value| value <= 0.5)
        );
    }
}
