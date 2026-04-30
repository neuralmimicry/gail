/// Decision engine: combines Type-2 fuzzy logic output with multi-AI consensus
/// to produce a final trade decision, applying risk management gates.
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::debug;

use super::advisor::AiConsensus;
use super::config::TradingConfig;
use super::fuzzy::FuzzyDecision;
use super::octobot::MarketSnapshot;
use super::state::{TradingState, TradeAction};

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Decision output
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeDecision {
    pub action: TradeAction,
    pub exchange: String,
    pub symbol: String,
    pub amount_usd: f64,
    pub confidence: f64,
    pub rationale: String,
    /// Fuzzy component signal [-1, 1].
    pub fuzzy_signal: f64,
    pub fuzzy_confidence: f64,
    /// AI consensus signal [-1, 1].
    pub ai_signal: f64,
    pub ai_confidence: f64,
    /// Blended signal [-1, 1].
    pub blended_signal: f64,
    /// Whether an operator override was applied.
    pub override_applied: bool,
}

impl TradeDecision {
    pub fn hold(reason: impl Into<String>) -> Self {
        Self {
            action: TradeAction::Hold,
            exchange: String::new(),
            symbol: String::new(),
            amount_usd: 0.0,
            confidence: 0.0,
            rationale: reason.into(),
            fuzzy_signal: 0.0,
            fuzzy_confidence: 0.0,
            ai_signal: 0.0,
            ai_confidence: 0.0,
            blended_signal: 0.0,
            override_applied: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Decision engine
// ---------------------------------------------------------------------------

pub struct DecisionEngine {
    fuzzy_weight: f64,
}

impl DecisionEngine {
    pub fn new(fuzzy_weight: f64) -> Self {
        Self { fuzzy_weight: fuzzy_weight.clamp(0.0, 1.0) }
    }

    /// Produce a final trade decision from fuzzy output, AI consensus, and current state.
    pub fn decide(
        &self,
        fuzzy: &FuzzyDecision,
        consensus: &AiConsensus,
        best_market: Option<&MarketSnapshot>,
        state: &TradingState,
        config: &TradingConfig,
    ) -> TradeDecision {
        // Check for operator override first.
        if let Some(ref ov) = state.pending_override {
            return self.apply_override(ov, config);
        }

        // Effective config (apply runtime overrides if present).
        let effective_config = EffectiveConfig::from(config, &state.config_overrides);

        // Blend fuzzy signal and AI consensus signal.
        let ai_weight = 1.0 - self.fuzzy_weight;
        let blended_signal = fuzzy.signal * self.fuzzy_weight + consensus.signal * ai_weight;
        let blended_confidence =
            fuzzy.confidence * self.fuzzy_weight + consensus.confidence * ai_weight;

        debug!(
            "trading: decision — fuzzy={:.3}/{:.3} ai={:.3}/{:.3} blended={:.3}/{:.3}",
            fuzzy.signal, fuzzy.confidence,
            consensus.signal, consensus.confidence,
            blended_signal, blended_confidence
        );

        // Confidence threshold gate.
        if blended_confidence < effective_config.fuzzy_confidence_threshold {
            return TradeDecision {
                action: TradeAction::Hold,
                rationale: format!(
                    "Confidence {:.2} below threshold {:.2}",
                    blended_confidence, effective_config.fuzzy_confidence_threshold
                ),
                fuzzy_signal: fuzzy.signal,
                fuzzy_confidence: fuzzy.confidence,
                ai_signal: consensus.signal,
                ai_confidence: consensus.confidence,
                blended_signal,
                ..TradeDecision::hold("")
            };
        }

        // Open position gate.
        let open = state.open_positions.len();
        if open >= effective_config.max_open_positions && blended_signal > 0.0 {
            return TradeDecision {
                action: TradeAction::Hold,
                rationale: format!(
                    "Max open positions reached ({}/{})",
                    open, effective_config.max_open_positions
                ),
                fuzzy_signal: fuzzy.signal,
                fuzzy_confidence: fuzzy.confidence,
                ai_signal: consensus.signal,
                ai_confidence: consensus.confidence,
                blended_signal,
                ..TradeDecision::hold("")
            };
        }

        // Cooldown gate.
        if let Some(last) = state.last_trade_at {
            let elapsed = now_ts() - last;
            if elapsed < effective_config.min_trade_interval_seconds as f64 {
                return TradeDecision {
                    action: TradeAction::Hold,
                    rationale: format!(
                        "Cooldown: {:.0}s remaining",
                        effective_config.min_trade_interval_seconds as f64 - elapsed
                    ),
                    fuzzy_signal: fuzzy.signal,
                    fuzzy_confidence: fuzzy.confidence,
                    ai_signal: consensus.signal,
                    ai_confidence: consensus.confidence,
                    blended_signal,
                    ..TradeDecision::hold("")
                };
            }
        }

        // Determine action from blended signal.
        let action = signal_to_action(blended_signal);

        // Pick best market target.
        let (exchange, symbol) = match best_market {
            Some(m) => (m.exchange.clone(), m.symbol.clone()),
            None => {
                return TradeDecision {
                    action: TradeAction::Hold,
                    rationale: "No target market available".to_string(),
                    fuzzy_signal: fuzzy.signal,
                    fuzzy_confidence: fuzzy.confidence,
                    ai_signal: consensus.signal,
                    ai_confidence: consensus.confidence,
                    blended_signal,
                    ..TradeDecision::hold("")
                };
            }
        };

        // Size the trade.
        let amount_usd = size_trade(
            blended_signal.abs(),
            blended_confidence,
            effective_config.micro_trade_min_usd,
            effective_config.micro_trade_max_usd,
        );

        // Build rationale from top AI opinions.
        let rationale = build_rationale(&action, blended_signal, blended_confidence, consensus);

        TradeDecision {
            action,
            exchange,
            symbol,
            amount_usd,
            confidence: blended_confidence,
            rationale,
            fuzzy_signal: fuzzy.signal,
            fuzzy_confidence: fuzzy.confidence,
            ai_signal: consensus.signal,
            ai_confidence: consensus.confidence,
            blended_signal,
            override_applied: false,
        }
    }

    fn apply_override(
        &self,
        ov: &super::state::TradeOverride,
        config: &TradingConfig,
    ) -> TradeDecision {
        let action = ov.action.clone();
        let exchange = ov.exchange.clone().unwrap_or_default();
        let symbol = ov.symbol.clone().unwrap_or_default();
        let amount_usd = ov
            .amount_usd
            .unwrap_or(config.micro_trade_min_usd)
            .clamp(config.micro_trade_min_usd, config.micro_trade_max_usd);
        TradeDecision {
            action,
            exchange,
            symbol,
            amount_usd,
            confidence: 1.0,
            rationale: format!(
                "Operator override by {}: {}",
                ov.issued_by,
                ov.reason.as_deref().unwrap_or("no reason given")
            ),
            fuzzy_signal: 0.0,
            fuzzy_confidence: 0.0,
            ai_signal: 0.0,
            ai_confidence: 0.0,
            blended_signal: 0.0,
            override_applied: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct EffectiveConfig {
    fuzzy_confidence_threshold: f64,
    max_open_positions: usize,
    min_trade_interval_seconds: u64,
    micro_trade_min_usd: f64,
    micro_trade_max_usd: f64,
}

impl EffectiveConfig {
    fn from(base: &TradingConfig, overrides: &Option<super::config::TradingConfigOverride>) -> Self {
        let ov = overrides.as_ref();
        Self {
            fuzzy_confidence_threshold: ov
                .and_then(|o| o.fuzzy_confidence_threshold)
                .unwrap_or(base.fuzzy_confidence_threshold),
            max_open_positions: ov
                .and_then(|o| o.max_open_positions)
                .unwrap_or(base.max_open_positions),
            min_trade_interval_seconds: base.min_trade_interval_seconds,
            micro_trade_min_usd: ov
                .and_then(|o| o.micro_trade_min_usd)
                .unwrap_or(base.micro_trade_min_usd),
            micro_trade_max_usd: ov
                .and_then(|o| o.micro_trade_max_usd)
                .unwrap_or(base.micro_trade_max_usd),
        }
    }
}

fn signal_to_action(signal: f64) -> TradeAction {
    match signal {
        s if s >= 0.65 => TradeAction::StrongBuy,
        s if s >= 0.2 => TradeAction::Buy,
        s if s <= -0.65 => TradeAction::StrongSell,
        s if s <= -0.2 => TradeAction::Sell,
        _ => TradeAction::Hold,
    }
}

fn size_trade(signal_strength: f64, confidence: f64, min_usd: f64, max_usd: f64) -> f64 {
    // Trade size scales with signal strength and confidence.
    let scale = (signal_strength * confidence).clamp(0.0, 1.0);
    let raw = min_usd + (max_usd - min_usd) * scale;
    // Round to 2 decimal places.
    (raw * 100.0).round() / 100.0
}

fn build_rationale(
    action: &TradeAction,
    signal: f64,
    confidence: f64,
    consensus: &AiConsensus,
) -> String {
    let action_str = action.to_string();
    let top_reasoning: Vec<&str> = consensus
        .advices
        .iter()
        .filter(|a| a.parsed_ok && !a.reasoning.is_empty())
        .take(2)
        .map(|a| a.reasoning.as_str())
        .collect();

    let ai_summary = if top_reasoning.is_empty() {
        format!("AI consensus: {} ({} responders)", consensus.action, consensus.responders)
    } else {
        top_reasoning.join("; ")
    };

    format!(
        "Action={action_str} signal={signal:.3} confidence={confidence:.2}. {ai_summary}"
    )
}
