/// Decision engine: combines Type-2 fuzzy logic output with multi-AI consensus
/// to produce a final trade decision, applying risk management gates.
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::debug;

use super::advisor::AiConsensus;
use super::config::TradingConfig;
use super::economics::{TradeEconomics, estimate_trade_economics};
use super::fuzzy::FuzzyDecision;
use super::octobot::MarketSnapshot;
use super::outcomes::OutcomeLedger;
use super::state::{TradeAction, TradingState};

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
    /// Wall-clock time at which the advisory decision was materialized.
    pub created_at: f64,
    /// Timestamp of the market observation underlying the decision.
    pub market_fetched_at: Option<f64>,
    /// Price used to calculate signal economics before immediate repricing.
    pub reference_price: Option<f64>,
    /// Fee/slippage-aware expected edge used by gating and sizing.
    pub economics: TradeEconomics,
    /// Parsed provider/model identities that contributed to consensus.
    pub provider_keys: Vec<String>,
    /// Compact market regime label used by outcome calibration.
    pub market_regime: String,
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
            created_at: now_ts(),
            market_fetched_at: None,
            reference_price: None,
            economics: TradeEconomics::default(),
            provider_keys: Vec::new(),
            market_regime: "unknown".to_string(),
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
        let effective_config =
            EffectiveConfig::from(config, &state.config_overrides, self.fuzzy_weight);

        // Blend fuzzy signal and AI consensus signal.
        let fuzzy_weight = effective_config.fuzzy_weight;
        let ai_weight = 1.0 - fuzzy_weight;
        let base_blended_signal = fuzzy.signal * fuzzy_weight + consensus.signal * ai_weight;
        let base_blended_confidence =
            fuzzy.confidence * fuzzy_weight + consensus.confidence * ai_weight;
        let (target_exchange, target_symbol) = best_market
            .map(|market| (market.exchange.clone(), market.symbol.clone()))
            .unwrap_or_else(|| (String::new(), String::new()));

        // Optional historical ROI feedback: if recent directional decisions
        // have been consistently poor, Gail dampens new signals/confidence.
        // If they have performed well, Gail allows a bounded boost.
        let roi_feedback = roi_feedback_adjustment(
            &state.outcome_ledger,
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
        let adaptive_gate = adaptive_confidence_gate(&effective_config, consensus);

        debug!(
            "trading: decision — fuzzy={:.3}/{:.3} ai={:.3}/{:.3} blended_base={:.3}/{:.3} blended_adj={:.3}/{:.3} gate={:.3}/{:.3} coverage={:.2} responders={} failures={} roi_applied={}",
            fuzzy.signal,
            fuzzy.confidence,
            consensus.signal,
            consensus.confidence,
            base_blended_signal,
            base_blended_confidence,
            blended_signal,
            blended_confidence,
            adaptive_gate.threshold,
            adaptive_gate.base_threshold,
            adaptive_gate.coverage,
            adaptive_gate.responders,
            adaptive_gate.failures,
            roi_feedback.is_some()
        );

        // Confidence threshold gate.
        if blended_confidence < adaptive_gate.threshold {
            return TradeDecision {
                action: TradeAction::Hold,
                exchange: target_exchange.clone(),
                symbol: target_symbol.clone(),
                rationale: format!(
                    "Confidence {:.2} below adaptive threshold {:.2} (base {:.2}, coverage {:.0}%, responders={}, failures={})",
                    blended_confidence,
                    adaptive_gate.threshold,
                    adaptive_gate.base_threshold,
                    adaptive_gate.coverage * 100.0,
                    adaptive_gate.responders,
                    adaptive_gate.failures
                ),
                confidence: blended_confidence,
                fuzzy_signal: fuzzy.signal,
                fuzzy_confidence: fuzzy.confidence,
                ai_signal: consensus.signal,
                ai_confidence: consensus.confidence,
                blended_signal,
                ..TradeDecision::hold("")
            };
        }

        // Open position gate.
        let open = state.open_position_count();
        if open >= effective_config.max_open_positions && blended_signal > 0.0 {
            return TradeDecision {
                action: TradeAction::Hold,
                exchange: target_exchange.clone(),
                symbol: target_symbol.clone(),
                rationale: format!(
                    "Max open positions reached ({}/{})",
                    open, effective_config.max_open_positions
                ),
                confidence: blended_confidence,
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
                    exchange: target_exchange.clone(),
                    symbol: target_symbol.clone(),
                    rationale: format!(
                        "Cooldown: {:.0}s remaining",
                        effective_config.min_trade_interval_seconds as f64 - elapsed
                    ),
                    confidence: blended_confidence,
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
                    exchange: target_exchange.clone(),
                    symbol: target_symbol.clone(),
                    rationale: "No target market available".to_string(),
                    confidence: blended_confidence,
                    fuzzy_signal: fuzzy.signal,
                    fuzzy_confidence: fuzzy.confidence,
                    ai_signal: consensus.signal,
                    ai_confidence: consensus.confidence,
                    blended_signal,
                    ..TradeDecision::hold("")
                };
            }
        };

        let provider_keys = consensus
            .advices
            .iter()
            .filter(|advice| advice.parsed_ok)
            .map(|advice| {
                advice.model.as_ref().map_or_else(
                    || advice.provider.clone(),
                    |model| format!("{}/{}", advice.provider, model),
                )
            })
            .collect::<Vec<_>>();
        let market_regime = market_regime_label(best_market);
        let calibration = state.outcome_ledger.calibration_for(
            &symbol,
            &provider_keys,
            &market_regime,
            config.markout_calibration_min_samples,
        );
        let economics = estimate_trade_economics(
            blended_signal.abs(),
            blended_confidence,
            calibration.multiplier,
            config,
        );
        if !matches!(action, TradeAction::Hold) && !economics.is_worthwhile() {
            return TradeDecision {
                action: TradeAction::Hold,
                exchange,
                symbol,
                confidence: blended_confidence,
                rationale: format!(
                    "Expected net edge {:.1}bps below required {:.1}bps (gross {:.1}bps, costs {:.1}bps, calibration {:.2}x)",
                    economics.expected_net_edge_bps,
                    economics.required_net_edge_bps,
                    economics.expected_gross_edge_bps,
                    economics.estimated_round_trip_cost_bps,
                    economics.calibration_multiplier,
                ),
                fuzzy_signal: fuzzy.signal,
                fuzzy_confidence: fuzzy.confidence,
                ai_signal: consensus.signal,
                ai_confidence: consensus.confidence,
                blended_signal,
                economics,
                provider_keys,
                market_regime,
                market_fetched_at: best_market.map(|market| market.fetched_at),
                reference_price: best_market.map(|market| market.price),
                ..TradeDecision::hold("")
            };
        }

        // Size the trade.
        let amount_usd = size_trade(
            blended_signal.abs(),
            blended_confidence,
            economics.size_multiplier(),
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
            created_at: now_ts(),
            market_fetched_at: best_market.map(|market| market.fetched_at),
            reference_price: best_market.map(|market| market.price),
            economics,
            provider_keys,
            market_regime,
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
            created_at: now_ts(),
            market_fetched_at: None,
            reference_price: None,
            economics: TradeEconomics::default(),
            provider_keys: Vec::new(),
            market_regime: "operator_override".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct EffectiveConfig {
    fuzzy_weight: f64,
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

#[derive(Clone, Copy, Debug)]
struct AdaptiveConfidenceGate {
    threshold: f64,
    base_threshold: f64,
    coverage: f64,
    responders: usize,
    failures: usize,
}

fn adaptive_confidence_gate(
    config: &EffectiveConfig,
    consensus: &AiConsensus,
) -> AdaptiveConfidenceGate {
    let base_threshold = config.fuzzy_confidence_threshold.clamp(0.0, 1.0);
    let responders = consensus.responders;
    let failures = consensus.failures;
    let total = (responders + failures).max(1) as f64;
    let coverage = responders as f64 / total;
    let agreement = consensus
        .vote_distribution
        .get("agreement")
        .and_then(serde_json::Value::as_f64)
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(1.0);
    let average_risk = consensus
        .vote_distribution
        .get("average_risk")
        .and_then(serde_json::Value::as_f64)
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(0.5);

    let mut threshold = base_threshold;
    let should_tighten = responders == 0 || failures > 0 || responders < 2;
    if should_tighten {
        let responder_depth = (responders as f64 / 3.0).clamp(0.0, 1.0);
        let degradation = ((1.0 - coverage) * 0.45
            + (1.0 - responder_depth) * 0.30
            + (1.0 - agreement) * 0.15
            + average_risk * 0.10)
            .clamp(0.0, 1.0);
        let maximum_penalty = (1.0 - base_threshold).min(0.22);
        threshold = base_threshold + maximum_penalty * degradation;
        if responders == 0 {
            threshold = 1.0;
        }
    }

    AdaptiveConfidenceGate {
        threshold: threshold.clamp(base_threshold, 1.0),
        base_threshold,
        coverage,
        responders,
        failures,
    }
}

impl EffectiveConfig {
    fn from(
        base: &TradingConfig,
        overrides: &Option<super::config::TradingConfigOverride>,
        default_fuzzy_weight: f64,
    ) -> Self {
        let ov = overrides.as_ref();
        Self {
            fuzzy_weight: ov
                .and_then(|o| o.fuzzy_weight)
                .unwrap_or(default_fuzzy_weight)
                .clamp(0.0, 1.0),
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
            decision_roi_feedback_target_roi_pct: ov
                .and_then(|o| o.decision_roi_feedback_target_roi_pct)
                .unwrap_or(base.decision_roi_feedback_target_roi_pct)
                .clamp(0.1, 50.0),
            decision_roi_feedback_max_signal_adjustment: ov
                .and_then(|o| o.decision_roi_feedback_max_signal_adjustment)
                .unwrap_or(base.decision_roi_feedback_max_signal_adjustment)
                .clamp(0.0, 0.5),
            decision_roi_feedback_max_confidence_penalty: ov
                .and_then(|o| o.decision_roi_feedback_max_confidence_penalty)
                .unwrap_or(base.decision_roi_feedback_max_confidence_penalty)
                .clamp(0.0, 0.95),
            decision_roi_feedback_max_confidence_boost: ov
                .and_then(|o| o.decision_roi_feedback_max_confidence_boost)
                .unwrap_or(base.decision_roi_feedback_max_confidence_boost)
                .clamp(0.0, 0.5),
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
    outcomes: &OutcomeLedger,
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
        let summary = directional_roi_summary(outcomes, direction, Some(symbol), lookback);
        (summary.samples >= min_samples).then_some(summary)
    });
    let summary = symbol_summary
        .unwrap_or_else(|| directional_roi_summary(outcomes, direction, None, lookback));
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
    outcomes: &OutcomeLedger,
    direction: DirectionalAction,
    symbol_filter: Option<&str>,
    lookback_trades: usize,
) -> DirectionalRoiSummary {
    let summary = outcomes.directional_performance(
        symbol_filter,
        direction == DirectionalAction::Buy,
        lookback_trades,
    );
    if summary.samples == 0 {
        return DirectionalRoiSummary::empty();
    }
    DirectionalRoiSummary {
        samples: summary.samples,
        avg_directional_roi: summary.average_net_return_bps / 10_000.0,
        win_rate: summary.win_rate,
    }
}

fn size_trade(
    signal_strength: f64,
    confidence: f64,
    edge_multiplier: f64,
    min_usd: f64,
    max_usd: f64,
) -> f64 {
    // Trade size scales with signal, confidence, and fee-adjusted expected edge.
    let scale = (signal_strength * confidence * edge_multiplier).clamp(0.0, 1.0);
    let raw = min_usd + (max_usd - min_usd) * scale;
    // Round to 2 decimal places.
    (raw * 100.0).round() / 100.0
}

fn market_regime_label(snapshot: Option<&MarketSnapshot>) -> String {
    let change = snapshot
        .and_then(|market| market.price_change_pct_24h)
        .unwrap_or(0.0);
    match change {
        value if value >= 3.0 => "bull_volatile",
        value if value >= 0.5 => "bull",
        value if value <= -3.0 => "bear_volatile",
        value if value <= -0.5 => "bear",
        _ => "range",
    }
    .to_string()
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
