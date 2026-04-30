/// Multi-AI advisory system for trading decisions.
///
/// Fires all configured AI providers in parallel via `GailService::direct_complete()`,
/// sends each a structured prompt describing the current market state, and aggregates
/// their responses into a consensus recommendation.
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::task::JoinSet;
use tracing::{debug, warn};

use crate::{
    config::ProviderProfile,
    models::{ChatMessage, MessageContent, ProviderCompletionRequest},
    orchestration::GailService,
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
        let config = self.service.config();
        let mut profiles: Vec<ProviderProfile> = config
            .providers
            .iter()
            .filter(|p| !p.provider_type.is_empty() && p.api_key.is_some())
            .cloned()
            .collect();
        // Sort by weight descending so highest-quality providers are preferred.
        profiles.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        profiles.truncate(max);
        profiles
    }
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
    };

    let weight = profile.weight;
    let provider_name = profile.name.clone();
    let provider_type = profile.provider_type.clone();

    match service.direct_complete(request).await {
        Ok(resp) => {
            let raw = resp.text.clone();
            let (action, confidence, reasoning, suggested_usd, parsed_ok) =
                parse_advisory_response(&raw);
            AiAdvice {
                provider: provider_name,
                model: Some(resp.model),
                action,
                confidence,
                reasoning,
                suggested_amount_usd: suggested_usd,
                raw_response: raw,
                parsed_ok,
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
  "suggested_amount_usd": <float or null>
}

Rules:
- confidence 0.0 means complete uncertainty; 1.0 means absolute certainty
- If signals conflict or data is insufficient, recommend "hold" with low confidence
- Be conservative: only recommend strong_buy/strong_sell with confidence > 0.75
- Micro-trade context: position sizes are small (< $25 USD), so risk management is less critical than signal clarity
- Consider portfolio exposure to avoid over-concentration
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
        format!("Total portfolio value: {total}\n{}", top.join("\n"))
    };

    let research_section = if research.is_empty() {
        "No research context available.".to_string()
    } else {
        format!(
            "Research context:\n{}",
            &research.context[..research.context.len().min(1500)]
        )
    };

    format!(
        "MARKET DATA:\n{market_section}\n\nPORTFOLIO:\n{portfolio_section}\n\nRESEARCH:\n{research_section}\n\nProvide your trading recommendation as JSON."
    )
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

fn parse_advisory_response(raw: &str) -> (String, f64, String, Option<f64>, bool) {
    // Strip markdown fences if the model wrapped the JSON.
    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    // Find JSON object boundaries.
    let start = cleaned.find('{').unwrap_or(0);
    let end = cleaned.rfind('}').map(|i| i + 1).unwrap_or(cleaned.len());
    let json_str = &cleaned[start..end];

    match serde_json::from_str::<Value>(json_str) {
        Ok(v) => {
            let action = v
                .get("action")
                .and_then(Value::as_str)
                .map(normalise_action)
                .unwrap_or_else(|| "hold".to_string());
            let confidence = v
                .get("confidence")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let reasoning = v
                .get("reasoning")
                .and_then(Value::as_str)
                .unwrap_or("no reasoning provided")
                .to_string();
            let suggested_usd = v.get("suggested_amount_usd").and_then(Value::as_f64);
            (action, confidence, reasoning, suggested_usd, true)
        }
        Err(_) => (
            "hold".to_string(),
            0.0,
            "failed to parse advisory response".to_string(),
            None,
            false,
        ),
    }
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
            failures: failures + advices.len(),
            ..AiConsensus::uncertain()
        };
    }

    // Weighted average signal using quality-weight * confidence.
    let mut weighted_signal = 0.0_f64;
    let mut total_weight = 0.0_f64;
    let mut vote_map: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for advice in &advices {
        if !advice.parsed_ok {
            continue;
        }
        let effective_weight = (advice.weight * advice.confidence).max(0.01);
        let signal = action_to_signal(&advice.action);
        weighted_signal += signal * effective_weight;
        total_weight += effective_weight;
        *vote_map.entry(advice.action.clone()).or_insert(0.0) += effective_weight;
    }

    let signal = if total_weight > 0.0 {
        weighted_signal / total_weight
    } else {
        0.0
    };
    let action = signal_to_action(signal);

    // Confidence: mean of individual confidences, weighted by quality weight.
    let conf_num: f64 = advices
        .iter()
        .filter(|a| a.parsed_ok)
        .map(|a| a.confidence * a.weight)
        .sum();
    let conf_den: f64 = advices
        .iter()
        .filter(|a| a.parsed_ok)
        .map(|a| a.weight)
        .sum();
    let confidence = if conf_den > 0.0 {
        (conf_num / conf_den).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let vote_distribution = serde_json::to_value(&vote_map).unwrap_or(json!({}));

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
