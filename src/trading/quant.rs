//! Persistent deterministic quant shadow evaluation and guarded migration.
//!
//! The controller deliberately separates *prediction* from *execution*:
//!
//! - while in [`QuantMode::Shadow`], Gail continues to use the LLM consensus
//!   for live decisions and records paired, non-overlapping fixed-horizon
//!   markouts for the LLM and every bounded quant parameter set;
//! - parameter sets rank the same universe at the same timestamp and retain
//!   their independently selected USDT markets under identical cost models;
//! - migration to [`QuantMode::Primary`] requires a positive quant mean, a
//!   configured net-USDT advantage over the LLM, sufficient actionable
//!   observations, bounded downside regression, and repeated confirmations;
//! - all controller state is serializable and lives in `TradingState`, so an
//!   active parameter selection or completed migration survives a restart.
//!
//! This module contains no network I/O. Its hot path is a bounded scan over
//! market snapshots and therefore completes independently of LLM availability.

use std::{cmp::Ordering, collections::VecDeque};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use super::{
    advisor::{AiAdvice, AiConsensus},
    config::TradingConfig,
    datalake::{MarketHistoricalFeatures, market_feature_key},
    octobot::{MarketSnapshot, OctobotPortfolio},
    quantitative::portfolio::{
        CrossSectionalAllocator, CrossSectionalConfig, CrossSectionalInput, PortfolioAllocation,
    },
};

const QUANT_STATE_VERSION: u32 = 1;
const BUY_SIGNAL_THRESHOLD: f64 = 0.2;
const STRONG_SIGNAL_THRESHOLD: f64 = 0.65;

/// Which decision implementation currently owns the advisory stage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantMode {
    /// LLM remains primary; quant predictions are evaluated but never execute.
    #[default]
    Shadow,
    /// The persisted migration guard passed; deterministic quant is primary.
    Primary,
}

impl QuantMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::Primary => "primary",
        }
    }
}

/// Bounded, explainable parameters for one quant candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantParameters {
    pub id: String,
    pub kind: QuantParameterKind,
    pub short_momentum_weight: f64,
    pub mid_momentum_weight: f64,
    pub long_momentum_weight: f64,
    pub live_momentum_weight: f64,
    pub volume_confirmation_weight: f64,
    pub risk_attenuation: f64,
    pub entry_threshold: f64,
}

impl Default for QuantParameters {
    fn default() -> Self {
        Self {
            id: "balanced-v1".to_string(),
            kind: QuantParameterKind::Momentum,
            short_momentum_weight: 0.45,
            mid_momentum_weight: 0.30,
            long_momentum_weight: 0.15,
            live_momentum_weight: 0.10,
            volume_confirmation_weight: 0.10,
            risk_attenuation: 0.50,
            entry_threshold: 0.24,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantParameterKind {
    #[default]
    Momentum,
    /// Explicit no-trade benchmark. Its return is always zero.
    Cash,
}

impl QuantParameters {
    fn normalize(&mut self) {
        self.id = self.id.trim().to_ascii_lowercase();
        if self.kind == QuantParameterKind::Cash {
            self.short_momentum_weight = 0.0;
            self.mid_momentum_weight = 0.0;
            self.long_momentum_weight = 0.0;
            self.live_momentum_weight = 0.0;
            self.volume_confirmation_weight = 0.0;
            self.risk_attenuation = 0.0;
            self.entry_threshold = 0.80;
            return;
        }
        self.short_momentum_weight = self.short_momentum_weight.clamp(0.0, 1.0);
        self.mid_momentum_weight = self.mid_momentum_weight.clamp(0.0, 1.0);
        self.long_momentum_weight = self.long_momentum_weight.clamp(0.0, 1.0);
        self.live_momentum_weight = self.live_momentum_weight.clamp(0.0, 1.0);
        self.volume_confirmation_weight = self.volume_confirmation_weight.clamp(0.0, 0.5);
        self.risk_attenuation = self.risk_attenuation.clamp(0.0, 0.95);
        self.entry_threshold = self.entry_threshold.clamp(0.08, 0.80);
        let total = self.short_momentum_weight
            + self.mid_momentum_weight
            + self.long_momentum_weight
            + self.live_momentum_weight;
        if total <= f64::EPSILON {
            self.short_momentum_weight = 0.45;
            self.mid_momentum_weight = 0.30;
            self.long_momentum_weight = 0.15;
            self.live_momentum_weight = 0.10;
        } else {
            self.short_momentum_weight /= total;
            self.mid_momentum_weight /= total;
            self.long_momentum_weight /= total;
            self.live_momentum_weight /= total;
        }
    }
}

fn default_parameter_sets() -> Vec<QuantParameters> {
    vec![
        QuantParameters::default(),
        QuantParameters {
            id: "fast-trend-v1".to_string(),
            kind: QuantParameterKind::Momentum,
            short_momentum_weight: 0.62,
            mid_momentum_weight: 0.20,
            long_momentum_weight: 0.08,
            live_momentum_weight: 0.10,
            volume_confirmation_weight: 0.16,
            risk_attenuation: 0.55,
            entry_threshold: 0.28,
        },
        QuantParameters {
            id: "slow-trend-v1".to_string(),
            kind: QuantParameterKind::Momentum,
            short_momentum_weight: 0.25,
            mid_momentum_weight: 0.35,
            long_momentum_weight: 0.30,
            live_momentum_weight: 0.10,
            volume_confirmation_weight: 0.06,
            risk_attenuation: 0.45,
            entry_threshold: 0.20,
        },
        QuantParameters {
            id: "risk-controlled-v1".to_string(),
            kind: QuantParameterKind::Momentum,
            short_momentum_weight: 0.40,
            mid_momentum_weight: 0.30,
            long_momentum_weight: 0.20,
            live_momentum_weight: 0.10,
            volume_confirmation_weight: 0.12,
            risk_attenuation: 0.75,
            entry_threshold: 0.34,
        },
        QuantParameters {
            id: "volume-confirmed-v1".to_string(),
            kind: QuantParameterKind::Momentum,
            short_momentum_weight: 0.50,
            mid_momentum_weight: 0.25,
            long_momentum_weight: 0.15,
            live_momentum_weight: 0.10,
            volume_confirmation_weight: 0.25,
            risk_attenuation: 0.60,
            entry_threshold: 0.26,
        },
        QuantParameters {
            id: "cash-v1".to_string(),
            kind: QuantParameterKind::Cash,
            short_momentum_weight: 0.0,
            mid_momentum_weight: 0.0,
            long_momentum_weight: 0.0,
            live_momentum_weight: 0.0,
            volume_confirmation_weight: 0.0,
            risk_attenuation: 0.0,
            entry_threshold: 0.80,
        },
    ]
    .into_iter()
    .map(|mut parameters| {
        parameters.normalize();
        parameters
    })
    .collect()
}

/// One deterministic prediction emitted by a parameter set.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantPrediction {
    pub parameter_id: String,
    /// Market selected by this complete parameter policy. Parameter arms may
    /// legitimately choose different USDT markets at the same timestamp.
    pub exchange: String,
    pub symbol: String,
    pub entry_price: f64,
    pub signal: f64,
    pub confidence: f64,
    pub risk_score: f64,
    pub actionable: bool,
}

/// Rich primary result used to create a consensus-compatible decision input.
#[derive(Clone, Debug, Default)]
pub struct QuantSignal {
    pub parameter_id: String,
    pub exchange: String,
    pub symbol: String,
    pub signal: f64,
    pub confidence: f64,
    pub risk_score: f64,
    /// Policy actionability before empirical cost and risk overlays.
    pub raw_actionable: bool,
    pub actionable: bool,
    pub selected_horizon_seconds: Option<u64>,
    pub expected_gross_edge_bps: Option<f64>,
    pub edge_lower_bound_bps: Option<f64>,
    pub estimated_round_trip_cost_bps: f64,
    pub edge_gate_reason: Option<String>,
    /// Cross-sectional target weight; cash retains any unallocated residual.
    pub target_weight: f64,
    pub rebalance_weight: f64,
    pub cross_sectional_score: f64,
    pub portfolio_allocations: Vec<PortfolioAllocation>,
    pub rationale: String,
}

impl QuantSignal {
    pub fn hold(reason: impl Into<String>) -> Self {
        Self {
            rationale: reason.into(),
            ..Self::default()
        }
    }

    pub fn action(&self) -> &'static str {
        signal_action(self.signal, self.actionable)
    }

    /// Adapt the quant signal to Gail's existing consensus consumer without
    /// pretending a network provider participated. The synthetic identity is
    /// also persisted in trade markouts for strategy-version calibration.
    pub fn as_consensus(&self) -> AiConsensus {
        let action = self.action().to_string();
        let advice = AiAdvice {
            provider: "quant".to_string(),
            model: Some(self.parameter_id.clone()),
            action: action.clone(),
            confidence: self.confidence,
            reasoning: self.rationale.clone(),
            suggested_amount_usd: None,
            risk_score: self.risk_score,
            risk_flags: quant_risk_flags(self.risk_score),
            target_symbol: (!self.symbol.is_empty()).then(|| self.symbol.clone()),
            raw_response: String::new(),
            parsed_ok: true,
            weight: 1.0,
        };
        AiConsensus {
            action,
            confidence: self.confidence,
            signal: if self.actionable { self.signal } else { 0.0 },
            vote_distribution: json!({
                "source": "deterministic_quant",
                "parameter_id": self.parameter_id,
                "agreement": 1.0,
                "coverage": 1.0,
                "average_risk": self.risk_score,
                "raw_actionable": self.raw_actionable,
                "selected_horizon_seconds": self.selected_horizon_seconds,
                "expected_gross_edge_bps": self.expected_gross_edge_bps,
                "edge_lower_bound_bps": self.edge_lower_bound_bps,
                "estimated_round_trip_cost_bps": self.estimated_round_trip_cost_bps,
                "edge_gate_reason": self.edge_gate_reason,
                "target_weight": self.target_weight,
                "rebalance_weight": self.rebalance_weight,
                "cross_sectional_score": self.cross_sectional_score,
                "portfolio_allocations": self.portfolio_allocations,
            }),
            advices: vec![advice],
            responders: 1,
            failures: 0,
        }
    }
}

fn quant_risk_flags(risk_score: f64) -> Vec<String> {
    if risk_score >= 0.72 {
        vec!["quant_high_realized_risk".to_string()]
    } else if risk_score >= 0.50 {
        vec!["quant_elevated_realized_risk".to_string()]
    } else {
        Vec::new()
    }
}

/// Pending paired policy markout at one shared decision timestamp and horizon.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantPendingEvaluation {
    pub evaluation_id: String,
    pub created_at: f64,
    pub due_at: f64,
    pub expires_at: f64,
    pub exchange: String,
    pub symbol: String,
    pub entry_price: f64,
    pub active_parameter_id: String,
    /// The LLM policy's own market, kept separate so it is never scored
    /// against a price interval chosen by quant.
    pub llm_exchange: String,
    pub llm_symbol: String,
    pub llm_entry_price: f64,
    pub llm_signal: Option<f64>,
    pub llm_actionable: bool,
    pub predictions: Vec<QuantPrediction>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantCandidateOutcome {
    pub parameter_id: String,
    pub exchange: String,
    pub symbol: String,
    pub actionable: bool,
    pub net_return_bps: f64,
}

/// Resolved evaluation retained in a bounded rolling ledger.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantResolvedEvaluation {
    pub evaluation_id: String,
    pub created_at: f64,
    pub resolved_at: f64,
    pub exchange: String,
    pub symbol: String,
    pub active_parameter_id: String,
    pub active_quant_actionable: bool,
    pub active_quant_net_return_bps: f64,
    pub llm_actionable: bool,
    pub llm_net_return_bps: Option<f64>,
    pub candidate_outcomes: Vec<QuantCandidateOutcome>,
}

/// Persisted controller state. New fields use serde defaults so older trading
/// snapshots remain compatible.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantMigrationState {
    pub version: u32,
    pub mode: QuantMode,
    pub active_parameter_id: String,
    pub parameter_sets: Vec<QuantParameters>,
    pub pending: VecDeque<QuantPendingEvaluation>,
    pub resolved: VecDeque<QuantResolvedEvaluation>,
    pub promotion_streak: usize,
    /// Consecutive rolling-window failures while quant is primary.
    #[serde(default)]
    pub demotion_streak: usize,
    pub initialized_at: Option<f64>,
    pub promoted_at: Option<f64>,
    pub last_shadow_recorded_at: Option<f64>,
    pub total_resolved: u64,
    pub last_tuned_resolved_count: u64,
    /// Most recent Gail-native validation of the currently promotable arm.
    pub native_validation_parameter_id: Option<String>,
    pub native_validation_at: Option<f64>,
}

impl Default for QuantMigrationState {
    fn default() -> Self {
        Self {
            version: QUANT_STATE_VERSION,
            mode: QuantMode::Shadow,
            active_parameter_id: QuantParameters::default().id,
            parameter_sets: default_parameter_sets(),
            pending: VecDeque::new(),
            resolved: VecDeque::new(),
            promotion_streak: 0,
            demotion_streak: 0,
            initialized_at: None,
            promoted_at: None,
            last_shadow_recorded_at: None,
            total_resolved: 0,
            last_tuned_resolved_count: 0,
            native_validation_parameter_id: None,
            native_validation_at: None,
        }
    }
}

impl QuantMigrationState {
    /// Repair legacy/partial state and clamp every persisted parameter.
    pub fn normalize(&mut self) {
        self.version = QUANT_STATE_VERSION;
        if self.parameter_sets.is_empty() {
            self.parameter_sets = default_parameter_sets();
        }
        if !self
            .parameter_sets
            .iter()
            .any(|parameters| parameters.kind == QuantParameterKind::Cash)
        {
            self.parameter_sets.push(QuantParameters {
                id: "cash-v1".to_string(),
                kind: QuantParameterKind::Cash,
                ..QuantParameters::default()
            });
        }
        for parameters in &mut self.parameter_sets {
            parameters.normalize();
        }
        self.parameter_sets.sort_by(|a, b| a.id.cmp(&b.id));
        self.parameter_sets.dedup_by(|a, b| a.id == b.id);
        if !self
            .parameter_sets
            .iter()
            .any(|parameters| parameters.id == self.active_parameter_id)
        {
            self.active_parameter_id = self.parameter_sets[0].id.clone();
        }
        self.total_resolved = self.total_resolved.max(self.resolved.len() as u64);
    }

    pub fn active_parameters(&self) -> &QuantParameters {
        self.parameter_sets
            .iter()
            .find(|parameters| parameters.id == self.active_parameter_id)
            .unwrap_or(&self.parameter_sets[0])
    }

    pub fn is_primary(&self) -> bool {
        self.mode == QuantMode::Primary
    }

    pub fn initialize(&mut self, now: f64) -> bool {
        self.normalize();
        if self.initialized_at.is_none() {
            self.initialized_at = Some(now);
            true
        } else {
            false
        }
    }

    /// Resolve every due observation for which an exact venue/symbol price is
    /// available, tune the active parameter arm, and evaluate migration.
    pub fn resolve_due(
        &mut self,
        snapshots: &[MarketSnapshot],
        config: &TradingConfig,
        now: f64,
    ) -> QuantUpdate {
        self.normalize();
        let mut update = QuantUpdate::default();
        let mut remaining = VecDeque::new();
        while let Some(pending) = self.pending.pop_front() {
            if pending.due_at > now {
                remaining.push_back(pending);
                continue;
            }
            let round_trip_cost_bps =
                2.0 * (config.estimated_fee_bps + config.estimated_slippage_bps);
            let candidate_outcomes = pending
                .predictions
                .iter()
                .filter_map(|prediction| {
                    let exchange = if prediction.exchange.is_empty() {
                        &pending.exchange
                    } else {
                        &prediction.exchange
                    };
                    let symbol = if prediction.symbol.is_empty() {
                        &pending.symbol
                    } else {
                        &prediction.symbol
                    };
                    let entry_price = if prediction.entry_price > 0.0 {
                        prediction.entry_price
                    } else {
                        pending.entry_price
                    };
                    policy_net_return_bps(
                        snapshots,
                        exchange,
                        symbol,
                        entry_price,
                        prediction.signal,
                        prediction.actionable,
                        round_trip_cost_bps,
                    )
                    .map(|net_return_bps| QuantCandidateOutcome {
                        parameter_id: prediction.parameter_id.clone(),
                        exchange: exchange.clone(),
                        symbol: symbol.clone(),
                        actionable: prediction.actionable,
                        net_return_bps,
                    })
                })
                .collect::<Vec<_>>();
            let Some(active) = candidate_outcomes
                .iter()
                .find(|outcome| outcome.parameter_id == pending.active_parameter_id)
                .cloned()
            else {
                if pending.expires_at > now {
                    remaining.push_back(pending);
                } else {
                    update.expired += 1;
                }
                continue;
            };
            let llm_net_return_bps = pending.llm_signal.and_then(|signal| {
                let exchange = if pending.llm_exchange.is_empty() {
                    &pending.exchange
                } else {
                    &pending.llm_exchange
                };
                let symbol = if pending.llm_symbol.is_empty() {
                    &pending.symbol
                } else {
                    &pending.llm_symbol
                };
                let entry_price = if pending.llm_entry_price > 0.0 {
                    pending.llm_entry_price
                } else {
                    pending.entry_price
                };
                policy_net_return_bps(
                    snapshots,
                    exchange,
                    symbol,
                    entry_price,
                    signal,
                    pending.llm_actionable,
                    round_trip_cost_bps,
                )
            });
            if pending.llm_signal.is_some() && llm_net_return_bps.is_none() {
                if pending.expires_at > now {
                    remaining.push_back(pending);
                } else {
                    update.expired += 1;
                }
                continue;
            }
            self.resolved.push_back(QuantResolvedEvaluation {
                evaluation_id: pending.evaluation_id,
                created_at: pending.created_at,
                resolved_at: now,
                exchange: active.exchange.clone(),
                symbol: active.symbol.clone(),
                active_parameter_id: pending.active_parameter_id,
                active_quant_actionable: active.actionable,
                active_quant_net_return_bps: active.net_return_bps,
                llm_actionable: pending.llm_actionable,
                llm_net_return_bps,
                candidate_outcomes,
            });
            self.total_resolved = self.total_resolved.saturating_add(1);
            update.resolved += 1;
        }
        self.pending = remaining;
        while self.resolved.len() > config.quant_shadow_ledger_size {
            self.resolved.pop_front();
        }

        if update.resolved > 0 {
            update.parameter_adjustment = self.maybe_retune(config);
            // Continue evaluating the paired LLM benchmark after promotion so
            // quant can be rolled back when its edge disappears.
            update.migration = self.evaluate_migration(config, now);
            update.performance = Some(self.controller_performance(config));
        }
        update
    }

    /// Record one non-overlapping paired observation. Sampling at least one
    /// markout horizon apart avoids counting heavily overlapping returns as
    /// independent evidence for migration.
    pub fn record_evaluation(
        &mut self,
        signal: &QuantSignal,
        context: QuantEvaluationContext<'_>,
    ) -> Option<QuantRecordSummary> {
        let QuantEvaluationContext {
            snapshots,
            historical_features,
            portfolio,
            llm_consensus,
            llm_snapshot,
            llm_evaluation_allowed,
            config,
            now,
        } = context;
        self.normalize();
        if !config.quant_shadow_enabled || signal.symbol.is_empty() {
            return None;
        }
        let snapshot = exact_market_snapshot(snapshots, &signal.exchange, &signal.symbol)?;
        if self.last_shadow_recorded_at.is_some_and(|last| {
            now - last + f64::EPSILON < config.quant_shadow_horizon_seconds as f64
        }) {
            return None;
        }
        let predictions = self
            .parameter_sets
            .iter()
            .map(|parameters| {
                let candidate = evaluate_universe_for_parameters(
                    parameters,
                    snapshots,
                    historical_features,
                    portfolio,
                    &config.quantitative.portfolio,
                );
                let entry_price =
                    exact_market_snapshot(snapshots, &candidate.exchange, &candidate.symbol)
                        .map(|market| market.price)
                        .unwrap_or(0.0);
                QuantPrediction {
                    parameter_id: parameters.id.clone(),
                    exchange: candidate.exchange,
                    symbol: candidate.symbol,
                    entry_price,
                    signal: candidate.signal,
                    confidence: candidate.confidence,
                    risk_score: candidate.risk_score,
                    actionable: candidate.actionable,
                }
            })
            .collect::<Vec<_>>();
        let llm_signal = llm_consensus.map(|consensus| consensus.signal.clamp(-1.0, 1.0));
        let llm_actionable = llm_evaluation_allowed
            && llm_snapshot.is_some()
            && llm_consensus.is_some_and(|consensus| consensus.action.as_str() != "hold");
        let horizon = config.quant_shadow_horizon_seconds as f64;
        let evaluation_id = Uuid::new_v4().to_string();
        self.pending.push_back(QuantPendingEvaluation {
            evaluation_id: evaluation_id.clone(),
            created_at: now,
            due_at: now + horizon,
            expires_at: now + horizon + config.quant_shadow_expiry_seconds as f64,
            exchange: snapshot.exchange.clone(),
            symbol: snapshot.symbol.clone(),
            entry_price: snapshot.price,
            active_parameter_id: self.active_parameter_id.clone(),
            llm_exchange: llm_snapshot
                .map(|market| market.exchange.clone())
                .unwrap_or_default(),
            llm_symbol: llm_snapshot
                .map(|market| market.symbol.clone())
                .unwrap_or_default(),
            llm_entry_price: llm_snapshot.map(|market| market.price).unwrap_or(0.0),
            llm_signal,
            llm_actionable,
            predictions,
        });
        while self.pending.len() > config.quant_shadow_ledger_size {
            self.pending.pop_front();
        }
        self.last_shadow_recorded_at = Some(now);
        Some(QuantRecordSummary {
            evaluation_id,
            mode: self.mode.clone(),
            parameter_id: signal.parameter_id.clone(),
            exchange: snapshot.exchange.clone(),
            symbol: snapshot.symbol.clone(),
            signal: signal.signal,
            confidence: signal.confidence,
            actionable: signal.actionable,
            llm_signal,
            llm_actionable,
            due_at: now + horizon,
        })
    }

    fn maybe_retune(&mut self, config: &TradingConfig) -> Option<QuantParameterAdjustment> {
        let total = self.total_resolved;
        if self.resolved.len() < config.quant_tuning_min_samples
            || total.saturating_sub(self.last_tuned_resolved_count)
                < config.quant_tuning_interval_samples as u64
        {
            return None;
        }
        let performances = self.parameter_performances(
            config.quant_migration_window_samples,
            config.quantitative.selection.confidence_z_score,
        );
        let current = performances
            .iter()
            .find(|performance| performance.parameter_id == self.active_parameter_id)?;
        let selection = &config.quantitative.selection;
        let cash = performances
            .iter()
            .find(|performance| performance.parameter_id == selection.cash_parameter_id)?;
        let best_trading_arm = performances
            .iter()
            .filter(|performance| performance.parameter_id != selection.cash_parameter_id)
            .filter(|performance| {
                performance.samples >= config.quant_tuning_min_samples
                    && performance.actionable_samples >= config.quant_tuning_min_actionable_samples
                    && performance.net_edge_lower_bound_bps
                        > selection.minimum_actionable_net_edge_bps
            })
            .max_by(|left, right| {
                left.risk_adjusted_score_bps
                    .partial_cmp(&right.risk_adjusted_score_bps)
                    .unwrap_or(Ordering::Equal)
            });
        // Cash wins whenever no trading arm establishes positive absolute edge.
        let best = best_trading_arm.unwrap_or(cash);
        self.last_tuned_resolved_count = total;
        let improvement = best.risk_adjusted_score_bps - current.risk_adjusted_score_bps;
        if best.parameter_id == self.active_parameter_id
            || improvement + f64::EPSILON < config.quant_tuning_min_outperformance_bps
        {
            return None;
        }
        let previous = self.active_parameter_id.clone();
        self.active_parameter_id = best.parameter_id.clone();
        Some(QuantParameterAdjustment {
            previous_parameter_id: previous,
            selected_parameter_id: best.parameter_id.clone(),
            risk_adjusted_improvement_bps: improvement,
            selected_performance: best.clone(),
        })
    }

    fn evaluate_migration(
        &mut self,
        config: &TradingConfig,
        now: f64,
    ) -> Option<QuantMigrationDecision> {
        let paired = self
            .resolved
            .iter()
            .rev()
            .filter(|item| {
                item.llm_net_return_bps.is_some()
                    && item.active_parameter_id == self.active_parameter_id
            })
            .take(config.quant_migration_window_samples)
            .collect::<Vec<_>>();
        let quant_values = paired
            .iter()
            .map(|item| item.active_quant_net_return_bps)
            .collect::<Vec<_>>();
        let quant_actions = paired
            .iter()
            .map(|item| item.active_quant_actionable)
            .collect::<Vec<_>>();
        let llm_values = paired
            .iter()
            .filter_map(|item| item.llm_net_return_bps)
            .collect::<Vec<_>>();
        let llm_actions = paired
            .iter()
            .map(|item| item.llm_actionable)
            .collect::<Vec<_>>();
        let actionable_samples = paired
            .iter()
            .filter(|item| item.active_quant_actionable)
            .count();
        let quant = PerformanceSummary::from_policy_values(
            &quant_values,
            &quant_actions,
            config.quantitative.selection.confidence_z_score,
        );
        let llm = PerformanceSummary::from_policy_values(
            &llm_values,
            &llm_actions,
            config.quantitative.selection.confidence_z_score,
        );
        let outperformance =
            quant.opportunity_mean_net_return_bps - llm.opportunity_mean_net_return_bps;
        let native_validation_current = !config
            .quantitative
            .selection
            .require_native_validation_for_promotion
            || (self.native_validation_parameter_id.as_deref()
                == Some(self.active_parameter_id.as_str())
                && self.native_validation_at.is_some_and(|validated_at| {
                    now - validated_at
                        <= config
                            .quantitative
                            .selection
                            .native_validation_max_age_seconds as f64
                }));
        let active_is_cash = self.active_parameters().kind == QuantParameterKind::Cash;
        let qualified = paired.len() >= config.quant_migration_min_samples
            && actionable_samples >= config.quant_migration_min_actionable_samples
            && !active_is_cash
            && native_validation_current
            && quant.net_edge_lower_bound_bps
                > config
                    .quantitative
                    .selection
                    .minimum_actionable_net_edge_bps
            && quant.opportunity_mean_net_return_bps > 0.0
            && outperformance >= config.quant_migration_min_outperformance_bps
            && quant.mean_downside_bps
                <= llm.mean_downside_bps + config.quant_migration_max_downside_regression_bps;
        if self.mode == QuantMode::Primary {
            if qualified {
                self.demotion_streak = 0;
                return None;
            }
            if paired.len() < config.quant_migration_min_samples {
                return None;
            }
            self.demotion_streak = self.demotion_streak.saturating_add(1);
            if self.demotion_streak < config.quant_migration_required_streak {
                return None;
            }
            self.mode = QuantMode::Shadow;
            self.promoted_at = None;
            self.promotion_streak = 0;
            self.demotion_streak = 0;
            return Some(QuantMigrationDecision {
                transition: QuantMigrationTransition::Demoted,
                parameter_id: self.active_parameter_id.clone(),
                samples: paired.len(),
                actionable_samples,
                quant,
                llm,
                outperformance_bps: outperformance,
                confirmation_streak: config.quant_migration_required_streak,
            });
        }

        self.demotion_streak = 0;
        if qualified {
            self.promotion_streak = self.promotion_streak.saturating_add(1);
        } else {
            self.promotion_streak = 0;
        }
        if self.promotion_streak < config.quant_migration_required_streak {
            return None;
        }
        self.mode = QuantMode::Primary;
        self.promoted_at = Some(now);
        Some(QuantMigrationDecision {
            transition: QuantMigrationTransition::Promoted,
            parameter_id: self.active_parameter_id.clone(),
            samples: paired.len(),
            actionable_samples,
            quant,
            llm,
            outperformance_bps: outperformance,
            confirmation_streak: self.promotion_streak,
        })
    }

    pub fn controller_performance(&self, config: &TradingConfig) -> QuantControllerPerformance {
        let paired = self
            .resolved
            .iter()
            .rev()
            .filter(|item| {
                item.llm_net_return_bps.is_some()
                    && item.active_parameter_id == self.active_parameter_id
            })
            .take(config.quant_migration_window_samples)
            .collect::<Vec<_>>();
        let quant_values = paired
            .iter()
            .map(|item| item.active_quant_net_return_bps)
            .collect::<Vec<_>>();
        let quant_actions = paired
            .iter()
            .map(|item| item.active_quant_actionable)
            .collect::<Vec<_>>();
        let llm_values = paired
            .iter()
            .filter_map(|item| item.llm_net_return_bps)
            .collect::<Vec<_>>();
        let llm_actions = paired
            .iter()
            .map(|item| item.llm_actionable)
            .collect::<Vec<_>>();
        QuantControllerPerformance {
            paired_samples: paired.len(),
            quant: PerformanceSummary::from_policy_values(
                &quant_values,
                &quant_actions,
                config.quantitative.selection.confidence_z_score,
            ),
            llm: PerformanceSummary::from_policy_values(
                &llm_values,
                &llm_actions,
                config.quantitative.selection.confidence_z_score,
            ),
        }
    }

    fn parameter_performances(
        &self,
        window: usize,
        confidence_z_score: f64,
    ) -> Vec<ParameterPerformance> {
        self.parameter_sets
            .iter()
            .map(|parameters| {
                let outcomes = self
                    .resolved
                    .iter()
                    .rev()
                    .filter_map(|item| {
                        item.candidate_outcomes
                            .iter()
                            .find(|outcome| outcome.parameter_id == parameters.id)
                    })
                    .take(window)
                    .collect::<Vec<_>>();
                let values = outcomes
                    .iter()
                    .map(|outcome| outcome.net_return_bps)
                    .collect::<Vec<_>>();
                let actions = outcomes
                    .iter()
                    .map(|outcome| outcome.actionable)
                    .collect::<Vec<_>>();
                let summary =
                    PerformanceSummary::from_policy_values(&values, &actions, confidence_z_score);
                let is_cash = parameters.kind == QuantParameterKind::Cash;
                ParameterPerformance {
                    parameter_id: parameters.id.clone(),
                    samples: summary.samples,
                    actionable_samples: summary.actionable_samples,
                    mean_net_return_bps: summary.mean_net_return_bps,
                    opportunity_mean_net_return_bps: summary.opportunity_mean_net_return_bps,
                    net_edge_lower_bound_bps: if is_cash {
                        0.0
                    } else {
                        summary.net_edge_lower_bound_bps
                    },
                    win_rate: summary.win_rate,
                    mean_downside_bps: summary.mean_downside_bps,
                    risk_adjusted_score_bps: if is_cash {
                        0.0
                    } else {
                        summary.net_edge_lower_bound_bps - summary.mean_downside_bps * 0.25
                    },
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PerformanceSummary {
    pub samples: usize,
    pub actionable_samples: usize,
    pub mean_net_return_bps: f64,
    pub opportunity_mean_net_return_bps: f64,
    pub net_edge_lower_bound_bps: f64,
    pub win_rate: f64,
    pub mean_downside_bps: f64,
}

impl PerformanceSummary {
    fn from_policy_values(values: &[f64], actionable: &[bool], confidence_z_score: f64) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let actionable_values = values
            .iter()
            .zip(actionable.iter().copied())
            .filter_map(|(value, is_actionable)| is_actionable.then_some(*value))
            .collect::<Vec<_>>();
        let opportunity_mean_net_return_bps = values.iter().sum::<f64>() / values.len() as f64;
        if actionable_values.is_empty() {
            return Self {
                samples: values.len(),
                opportunity_mean_net_return_bps,
                ..Self::default()
            };
        }
        let mean = actionable_values.iter().sum::<f64>() / actionable_values.len() as f64;
        let variance = if actionable_values.len() > 1 {
            actionable_values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (actionable_values.len() - 1) as f64
        } else {
            0.0
        };
        let lower_bound = mean
            - confidence_z_score.clamp(0.0, 4.0) * variance.sqrt()
                / (actionable_values.len() as f64).sqrt();
        let losses = actionable_values
            .iter()
            .filter(|value| **value < 0.0)
            .map(|value| value.abs())
            .collect::<Vec<_>>();
        Self {
            samples: values.len(),
            actionable_samples: actionable_values.len(),
            mean_net_return_bps: mean,
            opportunity_mean_net_return_bps,
            net_edge_lower_bound_bps: lower_bound,
            win_rate: actionable_values
                .iter()
                .filter(|value| **value > 0.0)
                .count() as f64
                / actionable_values.len() as f64,
            mean_downside_bps: if losses.is_empty() {
                0.0
            } else {
                losses.iter().sum::<f64>() / losses.len() as f64
            },
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ParameterPerformance {
    pub parameter_id: String,
    pub samples: usize,
    pub actionable_samples: usize,
    pub mean_net_return_bps: f64,
    pub opportunity_mean_net_return_bps: f64,
    pub net_edge_lower_bound_bps: f64,
    pub win_rate: f64,
    pub mean_downside_bps: f64,
    pub risk_adjusted_score_bps: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuantParameterAdjustment {
    pub previous_parameter_id: String,
    pub selected_parameter_id: String,
    pub risk_adjusted_improvement_bps: f64,
    pub selected_performance: ParameterPerformance,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuantMigrationDecision {
    pub transition: QuantMigrationTransition,
    pub parameter_id: String,
    pub samples: usize,
    pub actionable_samples: usize,
    pub quant: PerformanceSummary,
    pub llm: PerformanceSummary,
    pub outperformance_bps: f64,
    pub confirmation_streak: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantMigrationTransition {
    #[default]
    Promoted,
    Demoted,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct QuantControllerPerformance {
    pub paired_samples: usize,
    pub quant: PerformanceSummary,
    pub llm: PerformanceSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuantRecordSummary {
    pub evaluation_id: String,
    pub mode: QuantMode,
    pub parameter_id: String,
    pub exchange: String,
    pub symbol: String,
    pub signal: f64,
    pub confidence: f64,
    pub actionable: bool,
    pub llm_signal: Option<f64>,
    pub llm_actionable: bool,
    pub due_at: f64,
}

/// Borrowed inputs for one policy-level shadow sample. Grouping the context
/// keeps the persistence controller API stable as comparison metadata grows.
pub struct QuantEvaluationContext<'a> {
    pub snapshots: &'a [MarketSnapshot],
    pub historical_features: &'a std::collections::HashMap<String, MarketHistoricalFeatures>,
    pub portfolio: &'a OctobotPortfolio,
    pub llm_consensus: Option<&'a AiConsensus>,
    pub llm_snapshot: Option<&'a MarketSnapshot>,
    pub llm_evaluation_allowed: bool,
    pub config: &'a TradingConfig,
    pub now: f64,
}

#[derive(Clone, Debug, Default)]
pub struct QuantUpdate {
    pub resolved: usize,
    pub expired: usize,
    pub parameter_adjustment: Option<QuantParameterAdjustment>,
    pub migration: Option<QuantMigrationDecision>,
    pub performance: Option<QuantControllerPerformance>,
}

/// Select the strongest deterministic USDT-quoted opportunity under the
/// active parameter set. Ranking is directional and risk-adjusted; unlike the
/// legacy absolute-24h-move rank it does not equate a crash with a buy signal.
pub fn evaluate_universe(
    state: &QuantMigrationState,
    snapshots: &[MarketSnapshot],
    historical_features: &std::collections::HashMap<String, MarketHistoricalFeatures>,
    portfolio: &OctobotPortfolio,
    cross_sectional_config: &CrossSectionalConfig,
) -> QuantSignal {
    evaluate_universe_for_parameters(
        state.active_parameters(),
        snapshots,
        historical_features,
        portfolio,
        cross_sectional_config,
    )
}

pub(crate) fn evaluate_universe_for_parameters(
    parameters: &QuantParameters,
    snapshots: &[MarketSnapshot],
    historical_features: &std::collections::HashMap<String, MarketHistoricalFeatures>,
    portfolio: &OctobotPortfolio,
    cross_sectional_config: &CrossSectionalConfig,
) -> QuantSignal {
    if parameters.kind == QuantParameterKind::Cash {
        let benchmark = snapshots
            .iter()
            .filter(|snapshot| snapshot.price.is_finite() && snapshot.price > 0.0)
            .filter(|snapshot| is_usdt_quote(&snapshot.symbol))
            .min_by(|left, right| {
                left.exchange
                    .cmp(&right.exchange)
                    .then_with(|| left.symbol.cmp(&right.symbol))
            });
        return benchmark.map_or_else(
            || QuantSignal::hold("Cash arm: no usable USDT benchmark market"),
            |snapshot| QuantSignal {
                parameter_id: parameters.id.clone(),
                exchange: snapshot.exchange.clone(),
                symbol: snapshot.symbol.clone(),
                rationale: "Explicit cash/no-trade benchmark".to_string(),
                ..QuantSignal::default()
            },
        );
    }
    let candidates = snapshots
        .par_iter()
        .filter(|snapshot| snapshot.price.is_finite() && snapshot.price > 0.0)
        .filter(|snapshot| is_usdt_quote(&snapshot.symbol))
        .map(|snapshot| {
            let history =
                historical_features.get(&market_feature_key(&snapshot.exchange, &snapshot.symbol));
            let signal = evaluate_symbol(parameters, snapshot, history);
            let inventory_eligible =
                signal.signal >= 0.0 || portfolio_holds_symbol(portfolio, &signal.symbol);
            let volatility_pct = history
                .and_then(|features| features.volatility_pct)
                .unwrap_or_else(|| range_risk(snapshot) * 8.0)
                .abs()
                .max(0.01);
            let quote_volume_usd = snapshot
                .microstructure
                .quote_volume_24h
                .or_else(|| snapshot.volume_24h.map(|volume| volume * snapshot.price))
                .filter(|value| value.is_finite() && *value >= 0.0);
            let microstructure_signal = snapshot
                .microstructure
                .order_flow_imbalance
                .or(snapshot.microstructure.trade_flow_imbalance)
                .unwrap_or(0.0)
                .clamp(-1.0, 1.0);
            let microstructure_weight = cross_sectional_config.microstructure_signal_weight;
            let input = CrossSectionalInput {
                exchange: snapshot.exchange.clone(),
                symbol: snapshot.symbol.clone(),
                directional_signal: (signal.signal * (1.0 - microstructure_weight)
                    + microstructure_signal * microstructure_weight)
                    .clamp(-1.0, 1.0),
                confidence: signal.confidence,
                risk_score: signal.risk_score,
                volatility_pct,
                quote_volume_usd,
                spread_bps: snapshot.spread_bps(),
                depth_usd: snapshot.executable_depth_usd(),
                listing_age_days: snapshot.microstructure.listing_age_days,
                inventory_eligible,
                actionable: signal.actionable,
                correlation_cluster: correlation_cluster(&snapshot.symbol),
                current_weight: portfolio_symbol_weight(portfolio, &snapshot.symbol),
            };
            (signal, input)
        })
        .collect::<Vec<_>>();
    let recommendation = CrossSectionalAllocator::new(cross_sectional_config.clone())
        .allocate(candidates.iter().map(|(_, input)| input.clone()).collect());
    let Some(selected) = recommendation.allocations.first() else {
        return benchmark_hold_signal(
            parameters,
            snapshots,
            format!(
                "No executable cross-sectional allocation (eligible={}, excluded={})",
                recommendation.eligible_markets, recommendation.excluded_markets
            ),
        );
    };
    let Some((mut signal, _)) = candidates.into_iter().find(|(signal, _)| {
        signal.exchange.eq_ignore_ascii_case(&selected.exchange)
            && signal.symbol.eq_ignore_ascii_case(&selected.symbol)
    }) else {
        return benchmark_hold_signal(
            parameters,
            snapshots,
            "Cross-sectional allocation could not be mapped to a market".to_string(),
        );
    };
    signal.target_weight = selected.target_weight;
    signal.rebalance_weight = selected.rebalance_weight;
    signal.cross_sectional_score = selected.factor_score;
    signal.portfolio_allocations = recommendation.allocations;
    signal.rationale.push_str(&format!(
        "; cross-sectional rank=1 score={:.3} target_weight={:.3} rebalance={:.3} cash_weight={:.3}",
        signal.cross_sectional_score,
        signal.target_weight,
        signal.rebalance_weight,
        recommendation.cash_weight
    ));
    signal
}

fn benchmark_hold_signal(
    parameters: &QuantParameters,
    snapshots: &[MarketSnapshot],
    reason: String,
) -> QuantSignal {
    let benchmark = snapshots
        .iter()
        .filter(|snapshot| snapshot.price.is_finite() && snapshot.price > 0.0)
        .filter(|snapshot| is_usdt_quote(&snapshot.symbol))
        .min_by(|left, right| {
            left.exchange
                .cmp(&right.exchange)
                .then_with(|| left.symbol.cmp(&right.symbol))
        });
    benchmark.map_or_else(
        || QuantSignal::hold(reason.clone()),
        |snapshot| QuantSignal {
            parameter_id: parameters.id.clone(),
            exchange: snapshot.exchange.clone(),
            symbol: snapshot.symbol.clone(),
            rationale: reason.clone(),
            ..QuantSignal::default()
        },
    )
}

fn correlation_cluster(symbol: &str) -> String {
    let base = symbol
        .split('/')
        .next()
        .unwrap_or(symbol)
        .to_ascii_uppercase();
    if matches!(base.as_str(), "BTC" | "ETH" | "BNB" | "SOL") {
        "large_cap".to_string()
    } else {
        base
    }
}

fn portfolio_holds_symbol(portfolio: &OctobotPortfolio, symbol: &str) -> bool {
    let Some(base) = symbol.split('/').next().map(str::trim) else {
        return false;
    };
    portfolio.currencies.iter().any(|(asset, balance)| {
        asset.eq_ignore_ascii_case(base)
            && balance.total.is_finite()
            && balance.total > 0.0
            && balance
                .value_usd
                .is_some_and(|value| value.is_finite() && value > 0.01)
    })
}

fn portfolio_symbol_weight(portfolio: &OctobotPortfolio, symbol: &str) -> f64 {
    let Some(base) = symbol.split('/').next().map(str::trim) else {
        return 0.0;
    };
    let asset_value = portfolio
        .currencies
        .iter()
        .find(|(asset, _)| asset.eq_ignore_ascii_case(base))
        .and_then(|(_, balance)| balance.value_usd)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(0.0);
    let total_value = portfolio
        .total_value_usd
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or_else(|| {
            portfolio
                .currencies
                .values()
                .filter_map(|balance| balance.value_usd)
                .filter(|value| value.is_finite() && *value > 0.0)
                .sum()
        });
    if total_value <= f64::EPSILON {
        0.0
    } else {
        (asset_value / total_value).clamp(0.0, 1.0)
    }
}

pub fn evaluate_symbol(
    parameters: &QuantParameters,
    snapshot: &MarketSnapshot,
    history: Option<&MarketHistoricalFeatures>,
) -> QuantSignal {
    let prediction = prediction_for(snapshot, history, parameters);
    let history_samples = history.map(|features| features.samples).unwrap_or(0);
    QuantSignal {
        parameter_id: parameters.id.clone(),
        exchange: snapshot.exchange.clone(),
        symbol: snapshot.symbol.clone(),
        signal: prediction.signal,
        confidence: prediction.confidence,
        risk_score: prediction.risk_score,
        raw_actionable: prediction.actionable,
        actionable: prediction.actionable,
        rationale: format!(
            "deterministic quant {} signal={:.3} confidence={:.3} risk={:.3} history_samples={history_samples}",
            parameters.id, prediction.signal, prediction.confidence, prediction.risk_score
        ),
        ..QuantSignal::default()
    }
}

fn prediction_for(
    snapshot: &MarketSnapshot,
    history: Option<&MarketHistoricalFeatures>,
    parameters: &QuantParameters,
) -> QuantPrediction {
    if parameters.kind == QuantParameterKind::Cash {
        return QuantPrediction {
            parameter_id: parameters.id.clone(),
            exchange: snapshot.exchange.clone(),
            symbol: snapshot.symbol.clone(),
            entry_price: snapshot.price,
            ..QuantPrediction::default()
        };
    }
    let volatility_pct = history
        .and_then(|features| features.volatility_pct)
        .filter(|value| value.is_finite())
        .map(f64::abs)
        .unwrap_or(2.0)
        .max(0.25);
    let live = snapshot
        .price_change_pct_1h
        .or(snapshot.price_change_pct_24h)
        .map(|value| volatility_normalised_return(value, volatility_pct, 2.0))
        .unwrap_or(0.0);
    let short = history
        .and_then(|features| features.momentum_short_pct)
        .map(|value| volatility_normalised_return(value, volatility_pct, 2.0))
        .unwrap_or(live);
    let mid = history
        .and_then(|features| features.momentum_mid_pct)
        .map(|value| volatility_normalised_return(value, volatility_pct, 4.0))
        .unwrap_or(live * 0.5);
    let long = history
        .and_then(|features| features.momentum_long_pct)
        .map(|value| volatility_normalised_return(value, volatility_pct, 8.0))
        .unwrap_or(0.0);
    let volume_confirmation = history
        .and_then(|features| features.volume_ratio_short_long)
        .or_else(|| snapshot.volume_change_pct.map(|value| 1.0 + value / 100.0))
        .map(|ratio| ((ratio - 1.0) / 1.5).clamp(-1.0, 1.0))
        .unwrap_or(0.0);
    let risk_score = history
        .map(MarketHistoricalFeatures::risk_pressure)
        .unwrap_or_else(|| range_risk(snapshot));
    let momentum = short * parameters.short_momentum_weight
        + mid * parameters.mid_momentum_weight
        + long * parameters.long_momentum_weight
        + live * parameters.live_momentum_weight;
    let confirmation_direction = if momentum.abs() <= f64::EPSILON {
        0.0
    } else {
        momentum.signum()
    };
    let confirmed = momentum
        + confirmation_direction * volume_confirmation * parameters.volume_confirmation_weight;
    let signal = (confirmed * (1.0 - parameters.risk_attenuation * risk_score)).clamp(-1.0, 1.0);
    let coverage = feature_coverage(history, snapshot);
    let strength = (signal.abs() / parameters.entry_threshold.max(0.01)).clamp(0.0, 1.0);
    let confidence =
        ((0.35 + strength * 0.55 + coverage * 0.10) * (1.0 - risk_score * 0.30)).clamp(0.0, 1.0);
    let actionable = coverage >= 0.45
        && signal.abs() + f64::EPSILON >= parameters.entry_threshold
        && risk_score < 0.82;
    QuantPrediction {
        parameter_id: parameters.id.clone(),
        exchange: snapshot.exchange.clone(),
        symbol: snapshot.symbol.clone(),
        entry_price: snapshot.price,
        signal,
        confidence,
        risk_score,
        actionable,
    }
}

fn volatility_normalised_return(return_pct: f64, volatility_pct: f64, horizon_scale: f64) -> f64 {
    let denominator = (volatility_pct.abs().max(0.25) * horizon_scale.max(1.0)).max(0.25);
    (return_pct / denominator).clamp(-1.0, 1.0)
}

fn feature_coverage(history: Option<&MarketHistoricalFeatures>, snapshot: &MarketSnapshot) -> f64 {
    let Some(history) = history else {
        return if snapshot.price_change_pct_1h.is_some() && snapshot.volume_24h.is_some() {
            0.40
        } else {
            0.20
        };
    };
    let available = [
        history.momentum_short_pct,
        history.momentum_mid_pct,
        history.momentum_long_pct,
        history.volatility_pct,
        history.drawdown_pct,
        history.volume_ratio_short_long,
    ]
    .iter()
    .filter(|value| value.is_some())
    .count();
    (available as f64 / 6.0).clamp(0.0, 1.0)
}

fn range_risk(snapshot: &MarketSnapshot) -> f64 {
    match (snapshot.high_24h, snapshot.low_24h) {
        (Some(high), Some(low)) if high > 0.0 && low > 0.0 && high >= low => {
            (((high - low) / low) / 0.20).clamp(0.0, 1.0)
        }
        _ => 0.5,
    }
}

fn exact_market_snapshot<'a>(
    snapshots: &'a [MarketSnapshot],
    exchange: &str,
    symbol: &str,
) -> Option<&'a MarketSnapshot> {
    snapshots
        .iter()
        .filter(|snapshot| snapshot.price.is_finite() && snapshot.price > 0.0)
        .filter(|snapshot| snapshot.exchange.eq_ignore_ascii_case(exchange))
        .filter(|snapshot| snapshot.symbol.eq_ignore_ascii_case(symbol))
        .max_by(|left, right| {
            left.fetched_at
                .partial_cmp(&right.fetched_at)
                .unwrap_or(Ordering::Equal)
        })
}

fn directional_net_return_bps(
    signal: f64,
    actionable: bool,
    raw_return_bps: f64,
    round_trip_cost_bps: f64,
) -> f64 {
    if !actionable || signal.abs() < BUY_SIGNAL_THRESHOLD {
        return 0.0;
    }
    signal.signum() * raw_return_bps - round_trip_cost_bps.max(0.0)
}

/// Resolve one policy's independently selected market. Holds deliberately
/// need no future quote and score zero; actionable policies remain pending
/// until their exact venue/symbol quote is available or the record expires.
fn policy_net_return_bps(
    snapshots: &[MarketSnapshot],
    exchange: &str,
    symbol: &str,
    entry_price: f64,
    signal: f64,
    actionable: bool,
    round_trip_cost_bps: f64,
) -> Option<f64> {
    if !actionable || signal.abs() < BUY_SIGNAL_THRESHOLD {
        return Some(0.0);
    }
    if !entry_price.is_finite() || entry_price <= 0.0 {
        return None;
    }
    let observed = exact_market_snapshot(snapshots, exchange, symbol)?;
    let raw_return_bps = ((observed.price / entry_price) - 1.0) * 10_000.0;
    raw_return_bps.is_finite().then(|| {
        directional_net_return_bps(signal, actionable, raw_return_bps, round_trip_cost_bps)
    })
}

fn signal_action(signal: f64, actionable: bool) -> &'static str {
    if !actionable {
        "hold"
    } else if signal >= STRONG_SIGNAL_THRESHOLD {
        "strong_buy"
    } else if signal >= BUY_SIGNAL_THRESHOLD {
        "buy"
    } else if signal <= -STRONG_SIGNAL_THRESHOLD {
        "strong_sell"
    } else if signal <= -BUY_SIGNAL_THRESHOLD {
        "sell"
    } else {
        "hold"
    }
}

fn is_usdt_quote(symbol: &str) -> bool {
    symbol
        .rsplit_once('/')
        .is_some_and(|(_, quote)| quote.trim().eq_ignore_ascii_case("USDT"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TradingConfig {
        let mut config = TradingConfig {
            quant_shadow_enabled: true,
            quant_shadow_horizon_seconds: 60,
            quant_shadow_expiry_seconds: 300,
            quant_shadow_ledger_size: 100,
            quant_tuning_min_samples: 3,
            quant_tuning_min_actionable_samples: 1,
            quant_tuning_interval_samples: 1,
            quant_tuning_min_outperformance_bps: 1.0,
            quant_migration_min_samples: 3,
            quant_migration_min_actionable_samples: 2,
            quant_migration_window_samples: 20,
            quant_migration_min_outperformance_bps: 5.0,
            quant_migration_max_downside_regression_bps: 50.0,
            quant_migration_required_streak: 1,
            estimated_fee_bps: 1.0,
            estimated_slippage_bps: 1.0,
            ..TradingConfig::default()
        };
        config
            .quantitative
            .selection
            .require_native_validation_for_promotion = false;
        config
    }

    fn snapshot(price: f64, fetched_at: f64) -> MarketSnapshot {
        MarketSnapshot {
            exchange: "binance".to_string(),
            symbol: "BTC/USDT".to_string(),
            price,
            price_change_pct_1h: Some(2.0),
            price_change_pct_24h: Some(4.0),
            volume_24h: Some(10_000_000.0),
            volume_change_pct: Some(25.0),
            high_24h: Some(price * 1.03),
            low_24h: Some(price * 0.97),
            fetched_at,
            microstructure: Default::default(),
        }
    }

    fn history() -> MarketHistoricalFeatures {
        MarketHistoricalFeatures {
            samples: 100,
            momentum_short_pct: Some(3.0),
            momentum_mid_pct: Some(5.0),
            momentum_long_pct: Some(8.0),
            volatility_pct: Some(2.0),
            drawdown_pct: Some(-3.0),
            volume_ratio_short_long: Some(1.5),
            ..MarketHistoricalFeatures::default()
        }
    }

    fn history_map(
        market: &MarketSnapshot,
    ) -> std::collections::HashMap<String, MarketHistoricalFeatures> {
        std::collections::HashMap::from([(
            market_feature_key(&market.exchange, &market.symbol),
            history(),
        )])
    }

    #[test]
    fn quant_signal_is_deterministic_and_cost_independent() {
        let parameters = QuantParameters::default();
        let first = evaluate_symbol(&parameters, &snapshot(100.0, 1.0), Some(&history()));
        let second = evaluate_symbol(&parameters, &snapshot(100.0, 1.0), Some(&history()));
        assert_eq!(first.signal, second.signal);
        assert_eq!(first.confidence, second.confidence);
        assert!(first.signal > 0.0);
        assert!(first.actionable);
    }

    #[test]
    fn missing_history_fails_closed_to_hold() {
        let signal = evaluate_symbol(&QuantParameters::default(), &snapshot(100.0, 1.0), None);
        assert!(!signal.actionable);
        assert_eq!(signal.action(), "hold");
    }

    #[test]
    fn shadow_samples_are_non_overlapping_and_persistable() {
        let mut state = QuantMigrationState::default();
        let config = config();
        let market = snapshot(100.0, 1.0);
        let histories = history_map(&market);
        let portfolio = OctobotPortfolio::default();
        let signal = evaluate_symbol(state.active_parameters(), &market, Some(&history()));
        assert!(
            state
                .record_evaluation(
                    &signal,
                    QuantEvaluationContext {
                        snapshots: std::slice::from_ref(&market),
                        historical_features: &histories,
                        portfolio: &portfolio,
                        llm_consensus: None,
                        llm_snapshot: None,
                        llm_evaluation_allowed: false,
                        config: &config,
                        now: 1.0,
                    },
                )
                .is_some()
        );
        assert!(
            state
                .record_evaluation(
                    &signal,
                    QuantEvaluationContext {
                        snapshots: std::slice::from_ref(&market),
                        historical_features: &histories,
                        portfolio: &portfolio,
                        llm_consensus: None,
                        llm_snapshot: None,
                        llm_evaluation_allowed: false,
                        config: &config,
                        now: 30.0,
                    },
                )
                .is_none()
        );
        assert!(
            state
                .record_evaluation(
                    &signal,
                    QuantEvaluationContext {
                        snapshots: std::slice::from_ref(&market),
                        historical_features: &histories,
                        portfolio: &portfolio,
                        llm_consensus: None,
                        llm_snapshot: None,
                        llm_evaluation_allowed: false,
                        config: &config,
                        now: 61.0,
                    },
                )
                .is_some()
        );
        let restored: QuantMigrationState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(restored.pending.len(), 2);
        assert_eq!(restored.last_shadow_recorded_at, Some(61.0));
    }

    #[test]
    fn due_markout_subtracts_round_trip_costs() {
        let mut state = QuantMigrationState::default();
        let config = config();
        let market = snapshot(100.0, 1.0);
        let histories = history_map(&market);
        let portfolio = OctobotPortfolio::default();
        let signal = evaluate_symbol(state.active_parameters(), &market, Some(&history()));
        state.record_evaluation(
            &signal,
            QuantEvaluationContext {
                snapshots: std::slice::from_ref(&market),
                historical_features: &histories,
                portfolio: &portfolio,
                llm_consensus: Some(&signal.as_consensus()),
                llm_snapshot: Some(&market),
                llm_evaluation_allowed: true,
                config: &config,
                now: 1.0,
            },
        );
        let update = state.resolve_due(&[snapshot(101.0, 61.0)], &config, 61.0);
        assert_eq!(update.resolved, 1);
        let resolved = state.resolved.back().unwrap();
        assert!((resolved.active_quant_net_return_bps - 96.0).abs() < 1e-6);
        assert!((resolved.llm_net_return_bps.unwrap() - 96.0).abs() < 1e-6);
    }

    #[test]
    fn policy_markouts_use_each_advisors_selected_market() {
        let mut state = QuantMigrationState::default();
        let config = config();
        let active = state.active_parameter_id.clone();
        state.pending.push_back(QuantPendingEvaluation {
            evaluation_id: "paired-policy-targets".to_string(),
            due_at: 60.0,
            expires_at: 360.0,
            exchange: "binance".to_string(),
            symbol: "ETH/USDT".to_string(),
            entry_price: 100.0,
            active_parameter_id: active.clone(),
            llm_exchange: "binance".to_string(),
            llm_symbol: "BTC/USDT".to_string(),
            llm_entry_price: 100.0,
            llm_signal: Some(1.0),
            llm_actionable: true,
            predictions: vec![QuantPrediction {
                parameter_id: active,
                exchange: "binance".to_string(),
                symbol: "ETH/USDT".to_string(),
                entry_price: 100.0,
                signal: 1.0,
                actionable: true,
                ..QuantPrediction::default()
            }],
            ..QuantPendingEvaluation::default()
        });
        let markets = [
            MarketSnapshot {
                symbol: "ETH/USDT".to_string(),
                ..snapshot(99.0, 60.0)
            },
            MarketSnapshot {
                symbol: "BTC/USDT".to_string(),
                ..snapshot(101.0, 60.0)
            },
        ];

        let update = state.resolve_due(&markets, &config, 60.0);

        assert_eq!(update.resolved, 1);
        let resolved = state.resolved.back().unwrap();
        assert!((resolved.active_quant_net_return_bps + 104.0).abs() < 1e-6);
        assert!((resolved.llm_net_return_bps.unwrap() - 96.0).abs() < 1e-6);
    }

    #[test]
    fn migration_requires_guarded_quant_outperformance() {
        let mut state = QuantMigrationState::default();
        let config = config();
        let active = state.active_parameter_id.clone();
        for index in 0..3 {
            state.resolved.push_back(QuantResolvedEvaluation {
                evaluation_id: index.to_string(),
                active_parameter_id: active.clone(),
                active_quant_actionable: true,
                active_quant_net_return_bps: 25.0,
                llm_actionable: true,
                llm_net_return_bps: Some(5.0),
                candidate_outcomes: state
                    .parameter_sets
                    .iter()
                    .map(|parameters| QuantCandidateOutcome {
                        parameter_id: parameters.id.clone(),
                        actionable: true,
                        net_return_bps: if parameters.id == active { 25.0 } else { 10.0 },
                        ..QuantCandidateOutcome::default()
                    })
                    .collect(),
                ..QuantResolvedEvaluation::default()
            });
        }
        let decision = state.evaluate_migration(&config, 100.0);
        assert!(decision.is_some());
        assert_eq!(
            decision.unwrap().transition,
            QuantMigrationTransition::Promoted
        );
        assert_eq!(state.mode, QuantMode::Primary);
        assert_eq!(state.promoted_at, Some(100.0));
    }

    #[test]
    fn migration_demotes_quant_after_sustained_llm_outperformance() {
        let mut state = QuantMigrationState::default();
        let config = config();
        let active = state.active_parameter_id.clone();
        state.mode = QuantMode::Primary;
        state.promoted_at = Some(1.0);
        for index in 0..3 {
            state.resolved.push_back(QuantResolvedEvaluation {
                evaluation_id: index.to_string(),
                active_parameter_id: active.clone(),
                active_quant_actionable: true,
                active_quant_net_return_bps: -25.0,
                llm_actionable: true,
                llm_net_return_bps: Some(25.0),
                ..QuantResolvedEvaluation::default()
            });
        }
        let decision = state.evaluate_migration(&config, 100.0).unwrap();
        assert_eq!(decision.transition, QuantMigrationTransition::Demoted);
        assert_eq!(state.mode, QuantMode::Shadow);
        assert!(state.promoted_at.is_none());
    }

    #[test]
    fn parameter_tuning_selects_paired_outperformer() {
        let mut state = QuantMigrationState::default();
        let config = config();
        let active = state.active_parameter_id.clone();
        let challenger = state
            .parameter_sets
            .iter()
            .find(|parameters| parameters.id != active)
            .unwrap()
            .id
            .clone();
        for index in 0..3 {
            state.resolved.push_back(QuantResolvedEvaluation {
                evaluation_id: index.to_string(),
                candidate_outcomes: state
                    .parameter_sets
                    .iter()
                    .map(|parameters| QuantCandidateOutcome {
                        parameter_id: parameters.id.clone(),
                        actionable: true,
                        net_return_bps: if parameters.id == challenger {
                            30.0
                        } else {
                            5.0
                        },
                        ..QuantCandidateOutcome::default()
                    })
                    .collect(),
                ..QuantResolvedEvaluation::default()
            });
        }
        state.normalize();
        let adjustment = state.maybe_retune(&config).unwrap();
        assert_eq!(adjustment.previous_parameter_id, active);
        assert_eq!(adjustment.selected_parameter_id, challenger);
        assert_eq!(state.active_parameter_id, challenger);
    }

    #[test]
    fn parameter_tuning_selects_cash_when_every_trading_arm_loses() {
        let mut state = QuantMigrationState::default();
        let config = config();
        let previous = state.active_parameter_id.clone();
        for index in 0..3 {
            state.resolved.push_back(QuantResolvedEvaluation {
                evaluation_id: index.to_string(),
                candidate_outcomes: state
                    .parameter_sets
                    .iter()
                    .map(|parameters| QuantCandidateOutcome {
                        parameter_id: parameters.id.clone(),
                        actionable: parameters.kind != QuantParameterKind::Cash,
                        net_return_bps: if parameters.kind == QuantParameterKind::Cash {
                            0.0
                        } else {
                            -25.0
                        },
                        ..QuantCandidateOutcome::default()
                    })
                    .collect(),
                ..QuantResolvedEvaluation::default()
            });
        }
        state.normalize();
        let adjustment = state.maybe_retune(&config).unwrap();
        assert_eq!(adjustment.previous_parameter_id, previous);
        assert_eq!(adjustment.selected_parameter_id, "cash-v1");
        assert_eq!(state.active_parameter_id, "cash-v1");
    }

    #[test]
    fn performance_separates_actionable_expectancy_from_hold_opportunities() {
        let summary = PerformanceSummary::from_policy_values(
            &[-50.0, 0.0, 0.0, 0.0],
            &[true, false, false, false],
            0.0,
        );
        assert_eq!(summary.samples, 4);
        assert_eq!(summary.actionable_samples, 1);
        assert_eq!(summary.mean_net_return_bps, -50.0);
        assert_eq!(summary.opportunity_mean_net_return_bps, -12.5);
    }
}
