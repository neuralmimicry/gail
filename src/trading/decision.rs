/// Decision engine: combines Type-2 fuzzy logic output with multi-AI consensus
/// to produce a final trade decision, applying risk management gates.
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::debug;

use super::advisor::AiConsensus;
use super::config::TradingConfig;
use super::fuzzy::FuzzyDecision;
use super::octobot::MarketSnapshot;
use super::state::{ExecutedTrade, TradeAction, TradingState};

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
    /// Whether historical ROI feedback influenced this decision.
    pub roi_feedback_applied: bool,
    /// Signed blended-signal adjustment sourced from historical ROI performance.
    pub roi_feedback_signal_adjustment: f64,
    /// Confidence multiplier sourced from historical ROI performance.
    pub roi_feedback_confidence_multiplier: f64,
    /// Number of historical directional samples used for ROI feedback.
    pub roi_feedback_samples: usize,
    /// Average directional ROI used for feedback (fractional form, e.g. 0.02 = 2%).
    pub roi_feedback_avg_directional_roi: Option<f64>,
    /// Directional win-rate used for feedback.
    pub roi_feedback_win_rate: Option<f64>,
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
            roi_feedback_applied: false,
            roi_feedback_signal_adjustment: 0.0,
            roi_feedback_confidence_multiplier: 1.0,
            roi_feedback_samples: 0,
            roi_feedback_avg_directional_roi: None,
            roi_feedback_win_rate: None,
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
        Self {
            fuzzy_weight: fuzzy_weight.clamp(0.0, 1.0),
        }
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
        let base_blended_signal = fuzzy.signal * self.fuzzy_weight + consensus.signal * ai_weight;
        let base_blended_confidence =
            fuzzy.confidence * self.fuzzy_weight + consensus.confidence * ai_weight;

        // Optional historical ROI feedback: if recent directional decisions
        // have been consistently poor, Gail dampens new signals/confidence.
        // If they have performed well, Gail allows a bounded boost.
        let roi_feedback = roi_feedback_adjustment(
            &state.recent_trades,
            best_market.map(|market| market.symbol.as_str()),
            base_blended_signal,
            &effective_config,
        );
        let mut blended_signal = base_blended_signal;
        let mut blended_confidence = base_blended_confidence;
        if let Some(ref adjustment) = roi_feedback {
            blended_signal = (blended_signal + adjustment.signal_adjustment).clamp(-1.0, 1.0);
            blended_confidence =
                (blended_confidence * adjustment.confidence_multiplier).clamp(0.0, 1.0);
        }

        debug!(
            "trading: decision — fuzzy={:.3}/{:.3} ai={:.3}/{:.3} blended_base={:.3}/{:.3} blended_adj={:.3}/{:.3} roi_applied={}",
            fuzzy.signal,
            fuzzy.confidence,
            consensus.signal,
            consensus.confidence,
            base_blended_signal,
            base_blended_confidence,
            blended_signal,
            blended_confidence,
            roi_feedback.is_some()
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

        // Build rationale from top AI opinions and ROI feedback context.
        let rationale = build_rationale(
            &action,
            blended_signal,
            blended_confidence,
            consensus,
            roi_feedback.as_ref(),
        );

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
            roi_feedback_applied: roi_feedback.is_some(),
            roi_feedback_signal_adjustment: roi_feedback
                .as_ref()
                .map(|adjustment| adjustment.signal_adjustment)
                .unwrap_or(0.0),
            roi_feedback_confidence_multiplier: roi_feedback
                .as_ref()
                .map(|adjustment| adjustment.confidence_multiplier)
                .unwrap_or(1.0),
            roi_feedback_samples: roi_feedback
                .as_ref()
                .map(|adjustment| adjustment.samples)
                .unwrap_or(0),
            roi_feedback_avg_directional_roi: roi_feedback
                .as_ref()
                .map(|adjustment| adjustment.avg_directional_roi),
            roi_feedback_win_rate: roi_feedback.as_ref().map(|adjustment| adjustment.win_rate),
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
            roi_feedback_applied: false,
            roi_feedback_signal_adjustment: 0.0,
            roi_feedback_confidence_multiplier: 1.0,
            roi_feedback_samples: 0,
            roi_feedback_avg_directional_roi: None,
            roi_feedback_win_rate: None,
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
    decision_roi_feedback_enabled: bool,
    decision_roi_feedback_lookback_trades: usize,
    decision_roi_feedback_min_samples: usize,
    decision_roi_feedback_target_roi_pct: f64,
    decision_roi_feedback_max_signal_adjustment: f64,
    decision_roi_feedback_max_confidence_penalty: f64,
    decision_roi_feedback_max_confidence_boost: f64,
}

impl EffectiveConfig {
    fn from(
        base: &TradingConfig,
        overrides: &Option<super::config::TradingConfigOverride>,
    ) -> Self {
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
            decision_roi_feedback_enabled: base.decision_roi_feedback_enabled,
            decision_roi_feedback_lookback_trades: base.decision_roi_feedback_lookback_trades,
            decision_roi_feedback_min_samples: base.decision_roi_feedback_min_samples,
            decision_roi_feedback_target_roi_pct: base.decision_roi_feedback_target_roi_pct,
            decision_roi_feedback_max_signal_adjustment: base
                .decision_roi_feedback_max_signal_adjustment,
            decision_roi_feedback_max_confidence_penalty: base
                .decision_roi_feedback_max_confidence_penalty,
            decision_roi_feedback_max_confidence_boost: base
                .decision_roi_feedback_max_confidence_boost,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirectionalAction {
    Buy,
    Sell,
}

#[derive(Clone, Copy, Debug)]
struct DirectionalRoiSummary {
    samples: usize,
    avg_directional_roi: f64,
    win_rate: f64,
}

impl DirectionalRoiSummary {
    fn empty() -> Self {
        Self {
            samples: 0,
            avg_directional_roi: 0.0,
            win_rate: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RoiFeedbackAdjustment {
    signal_adjustment: f64,
    confidence_multiplier: f64,
    samples: usize,
    avg_directional_roi: f64,
    win_rate: f64,
}

fn roi_feedback_adjustment(
    trades: &std::collections::VecDeque<ExecutedTrade>,
    preferred_symbol: Option<&str>,
    signal: f64,
    config: &EffectiveConfig,
) -> Option<RoiFeedbackAdjustment> {
    if !config.decision_roi_feedback_enabled {
        return None;
    }
    let direction = directional_action_for_signal(signal)?;
    let lookback = config.decision_roi_feedback_lookback_trades.max(2);
    let min_samples = config.decision_roi_feedback_min_samples.max(2);
    let symbol_summary = preferred_symbol.and_then(|symbol| {
        let summary = directional_roi_summary(trades, direction, Some(symbol), lookback);
        (summary.samples >= min_samples).then_some(summary)
    });
    let summary = symbol_summary
        .unwrap_or_else(|| directional_roi_summary(trades, direction, None, lookback));
    if summary.samples < min_samples {
        return None;
    }

    let target_roi = (config.decision_roi_feedback_target_roi_pct / 100.0).max(0.001);
    let normalized_roi = (summary.avg_directional_roi / target_roi).clamp(-1.0, 1.0);
    let win_bias = ((summary.win_rate - 0.5) * 2.0).clamp(-1.0, 1.0);
    let performance = (normalized_roi * 0.7 + win_bias * 0.3).clamp(-1.0, 1.0);

    if performance.abs() < 0.01 {
        return None;
    }

    let direction_sign = if direction == DirectionalAction::Buy {
        1.0
    } else {
        -1.0
    };
    let signal_adjustment =
        direction_sign * performance * config.decision_roi_feedback_max_signal_adjustment;
    let confidence_multiplier = if performance < 0.0 {
        1.0 - (-performance * config.decision_roi_feedback_max_confidence_penalty)
    } else {
        1.0 + (performance * config.decision_roi_feedback_max_confidence_boost)
    }
    .clamp(0.05, 2.0);

    Some(RoiFeedbackAdjustment {
        signal_adjustment,
        confidence_multiplier,
        samples: summary.samples,
        avg_directional_roi: summary.avg_directional_roi,
        win_rate: summary.win_rate,
    })
}

fn directional_action_for_signal(signal: f64) -> Option<DirectionalAction> {
    if signal > 0.0 {
        Some(DirectionalAction::Buy)
    } else if signal < 0.0 {
        Some(DirectionalAction::Sell)
    } else {
        None
    }
}

fn directional_roi_summary(
    trades: &std::collections::VecDeque<ExecutedTrade>,
    direction: DirectionalAction,
    symbol_filter: Option<&str>,
    lookback_trades: usize,
) -> DirectionalRoiSummary {
    // Keep only priced buy/sell records, then evaluate each decision against the
    // next priced trade on the same symbol. This approximates whether the
    // decision direction was profitable before the next tactical adjustment.
    let priced = trades
        .iter()
        .filter(|trade| {
            trade
                .price
                .is_some_and(|price| price.is_finite() && price > 0.0)
                && matches!(
                    trade.action,
                    TradeAction::Buy
                        | TradeAction::StrongBuy
                        | TradeAction::Sell
                        | TradeAction::StrongSell
                )
        })
        .collect::<Vec<_>>();
    if priced.len() < 2 {
        return DirectionalRoiSummary::empty();
    }

    let start = priced
        .len()
        .saturating_sub(lookback_trades.saturating_add(1));
    let mut directional_rois = Vec::new();
    for idx in start..priced.len().saturating_sub(1) {
        let trade = priced[idx];
        let action = match trade.action {
            TradeAction::Buy | TradeAction::StrongBuy => DirectionalAction::Buy,
            TradeAction::Sell | TradeAction::StrongSell => DirectionalAction::Sell,
            _ => continue,
        };
        if action != direction {
            continue;
        }
        if let Some(symbol) = symbol_filter
            && !trade.symbol.eq_ignore_ascii_case(symbol)
        {
            continue;
        }
        let Some(entry_price) = trade.price else {
            continue;
        };
        let Some(next_trade) = priced
            .iter()
            .skip(idx + 1)
            .find(|next| next.symbol.eq_ignore_ascii_case(&trade.symbol))
        else {
            continue;
        };
        let Some(exit_price) = next_trade.price else {
            continue;
        };

        let market_return = ((exit_price - entry_price) / entry_price).clamp(-1.0, 1.0);
        let directional_roi = if action == DirectionalAction::Buy {
            market_return
        } else {
            -market_return
        };
        directional_rois.push(directional_roi);
    }

    if directional_rois.is_empty() {
        return DirectionalRoiSummary::empty();
    }

    let samples = directional_rois.len();
    let avg_directional_roi = directional_rois.iter().sum::<f64>() / samples as f64;
    let win_rate =
        directional_rois.iter().filter(|roi| **roi > 0.0).count() as f64 / samples as f64;
    DirectionalRoiSummary {
        samples,
        avg_directional_roi,
        win_rate,
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
    roi_feedback: Option<&RoiFeedbackAdjustment>,
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
        format!(
            "AI consensus: {} ({} responders)",
            consensus.action, consensus.responders
        )
    } else {
        top_reasoning.join("; ")
    };

    let mut rationale =
        format!("Action={action_str} signal={signal:.3} confidence={confidence:.2}. {ai_summary}");
    if let Some(feedback) = roi_feedback {
        rationale.push_str(&format!(
            " Historical directional ROI feedback: avg_roi={:.2}% win_rate={:.0}% samples={} signal_adj={:+.3} confidence_x={:.2}.",
            feedback.avg_directional_roi * 100.0,
            feedback.win_rate * 100.0,
            feedback.samples,
            feedback.signal_adjustment,
            feedback.confidence_multiplier
        ));
    }
    rationale
}
