//! Diversifying alpha sleeves and the bounded LLM risk overlay.
//!
//! Momentum remains Gail's only execution-capable quantitative policy. Pairs
//! and carry are evaluated through a common interface, persisted and measured
//! in shadow until the deployment can guarantee atomic multi-leg execution.
//! This avoids presenting a statistically attractive hedge as executable when
//! one leg could fail and leave an unintended directional position.
//!
//! The LLM overlay is deliberately asymmetric: recent, sufficiently reliable
//! LLM risk evidence may veto or reduce a quant recommendation, but can never
//! originate a trade, reverse its direction or increase its size. The last
//! overlay is bounded by a time-to-live, so the primary quant path has no
//! synchronous dependency on an LLM provider.

use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::super::{advisor::AiConsensus, octobot::MarketSnapshot, quant::QuantSignal};

/// Stable key used by cost telemetry and sleeve matching.
pub fn market_key(exchange: &str, symbol: &str) -> String {
    format!(
        "{}|{}",
        exchange.trim().to_ascii_lowercase(),
        symbol.trim().to_ascii_uppercase()
    )
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaSleeveKind {
    #[default]
    Pairs,
    Carry,
}

/// Normalised output shared by every diversifying sleeve.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SleeveRecommendation {
    pub sleeve_id: String,
    pub kind: AlphaSleeveKind,
    pub generated_at: f64,
    pub exchange: String,
    pub symbol: String,
    pub hedge_exchange: Option<String>,
    pub hedge_symbol: Option<String>,
    /// Direction of the primary leg in `[-1, 1]`.
    pub signal: f64,
    pub confidence: f64,
    pub expected_gross_edge_bps: f64,
    pub estimated_round_trip_cost_bps: f64,
    pub expected_net_edge_bps: f64,
    /// Whether the research rule identifies an economically valid opportunity.
    pub actionable: bool,
    /// Whether Gail can safely execute all required legs in this deployment.
    pub executable: bool,
    pub shadow_only: bool,
    pub rationale: String,
}

/// Read-only input passed to each alpha sleeve.
pub struct SleeveContext<'a> {
    pub snapshots: &'a [MarketSnapshot],
    pub round_trip_costs_bps: &'a HashMap<String, f64>,
    pub now: f64,
}

/// Common research interface for independently testable alpha sleeves.
pub trait AlphaSleeve: Send + Sync {
    fn name(&self) -> &'static str;
    fn evaluate(&self, context: &SleeveContext<'_>) -> Vec<SleeveRecommendation>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PairDefinition {
    pub id: String,
    pub first_symbol: String,
    pub second_symbol: String,
    /// Optional venue restriction. Empty values select each symbol's freshest
    /// available venue independently.
    pub exchange: String,
}

impl Default for PairDefinition {
    fn default() -> Self {
        Self {
            id: "btc-eth".to_string(),
            first_symbol: "BTC/USDT".to_string(),
            second_symbol: "ETH/USDT".to_string(),
            exchange: String::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PairsSleeveConfig {
    pub enabled: bool,
    pub shadow_only: bool,
    pub atomic_hedge_execution_supported: bool,
    pub pairs: Vec<PairDefinition>,
    pub lookback_samples: usize,
    pub minimum_samples: usize,
    pub minimum_correlation: f64,
    pub entry_z_score: f64,
    pub exit_z_score: f64,
    pub minimum_net_edge_bps: f64,
    /// Non-overlapping shadow markout horizon used to measure realised edge.
    pub shadow_horizon_seconds: u64,
}

impl Default for PairsSleeveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            shadow_only: true,
            atomic_hedge_execution_supported: false,
            pairs: vec![PairDefinition::default()],
            lookback_samples: 240,
            minimum_samples: 60,
            minimum_correlation: 0.60,
            entry_z_score: 2.0,
            exit_z_score: 0.50,
            minimum_net_edge_bps: 12.0,
            shadow_horizon_seconds: 14_400,
        }
    }
}

impl PairsSleeveConfig {
    pub fn normalise(&mut self) {
        self.lookback_samples = self.lookback_samples.clamp(10, 10_000);
        self.minimum_samples = self.minimum_samples.clamp(5, self.lookback_samples);
        self.minimum_correlation = self.minimum_correlation.clamp(-1.0, 1.0);
        self.entry_z_score = self.entry_z_score.clamp(0.50, 8.0);
        self.exit_z_score = self.exit_z_score.clamp(0.0, self.entry_z_score);
        self.minimum_net_edge_bps = self.minimum_net_edge_bps.clamp(0.0, 2_500.0);
        self.shadow_horizon_seconds = self.shadow_horizon_seconds.clamp(300, 30 * 86_400);
        self.pairs.retain(|pair| {
            !pair.id.trim().is_empty()
                && !pair.first_symbol.trim().is_empty()
                && !pair.second_symbol.trim().is_empty()
                && !pair.first_symbol.eq_ignore_ascii_case(&pair.second_symbol)
        });
        for pair in &mut self.pairs {
            pair.id = pair.id.trim().to_ascii_lowercase();
            pair.first_symbol = pair.first_symbol.trim().to_ascii_uppercase();
            pair.second_symbol = pair.second_symbol.trim().to_ascii_uppercase();
            pair.exchange = pair.exchange.trim().to_ascii_lowercase();
        }
    }

    fn can_execute_atomically(&self) -> bool {
        !self.shadow_only && self.atomic_hedge_execution_supported
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PairObservation {
    pub observed_at: f64,
    pub first_log_price: f64,
    pub second_log_price: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PairsSleeveState {
    pub observations: HashMap<String, VecDeque<PairObservation>>,
}

impl PairsSleeveState {
    fn observe(&mut self, snapshots: &[MarketSnapshot], config: &PairsSleeveConfig, now: f64) {
        for pair in &config.pairs {
            let first = latest_snapshot(snapshots, &pair.exchange, &pair.first_symbol);
            let second = latest_snapshot(snapshots, &pair.exchange, &pair.second_symbol);
            let (Some(first), Some(second)) = (first, second) else {
                continue;
            };
            if first.price <= 0.0 || second.price <= 0.0 {
                continue;
            }
            let observed_at = first
                .fetched_at
                .min(second.fetched_at)
                .max(now.min(first.fetched_at.max(second.fetched_at)));
            let history = self.observations.entry(pair.id.clone()).or_default();
            if history
                .back()
                .is_some_and(|last| observed_at <= last.observed_at + f64::EPSILON)
            {
                continue;
            }
            history.push_back(PairObservation {
                observed_at,
                first_log_price: first.price.ln(),
                second_log_price: second.price.ln(),
            });
            while history.len() > config.lookback_samples {
                history.pop_front();
            }
        }
    }
}

struct PairsAlphaSleeve<'a> {
    config: &'a PairsSleeveConfig,
    state: &'a PairsSleeveState,
}

impl AlphaSleeve for PairsAlphaSleeve<'_> {
    fn name(&self) -> &'static str {
        "pairs"
    }

    fn evaluate(&self, context: &SleeveContext<'_>) -> Vec<SleeveRecommendation> {
        if !self.config.enabled {
            return Vec::new();
        }
        self.config
            .pairs
            .par_iter()
            .map(|pair| self.evaluate_pair(pair, context))
            .collect()
    }
}

impl PairsAlphaSleeve<'_> {
    fn evaluate_pair(
        &self,
        pair: &PairDefinition,
        context: &SleeveContext<'_>,
    ) -> SleeveRecommendation {
        let first = latest_snapshot(context.snapshots, &pair.exchange, &pair.first_symbol);
        let second = latest_snapshot(context.snapshots, &pair.exchange, &pair.second_symbol);
        let mut recommendation = SleeveRecommendation {
            sleeve_id: format!("pairs:{}", pair.id),
            kind: AlphaSleeveKind::Pairs,
            generated_at: context.now,
            exchange: first.map(|item| item.exchange.clone()).unwrap_or_default(),
            symbol: pair.first_symbol.clone(),
            hedge_exchange: second.map(|item| item.exchange.clone()),
            hedge_symbol: Some(pair.second_symbol.clone()),
            shadow_only: !self.config.can_execute_atomically(),
            ..SleeveRecommendation::default()
        };
        let Some(history) = self.state.observations.get(&pair.id) else {
            recommendation.rationale = "Awaiting the first paired observation".to_string();
            return recommendation;
        };
        if history.len() < self.config.minimum_samples {
            recommendation.rationale = format!(
                "Pairs warm-up: {} of {} observations",
                history.len(),
                self.config.minimum_samples
            );
            return recommendation;
        }
        let ratios = history
            .iter()
            .map(|row| row.first_log_price - row.second_log_price)
            .collect::<Vec<_>>();
        let mean = arithmetic_mean(&ratios);
        let standard_deviation = sample_standard_deviation(&ratios, mean);
        let correlation = price_return_correlation(history);
        let Some(current_ratio) = ratios.last().copied() else {
            return recommendation;
        };
        if standard_deviation <= f64::EPSILON || !standard_deviation.is_finite() {
            recommendation.rationale = "Pair ratio has no measurable dispersion".to_string();
            return recommendation;
        }
        let z_score = (current_ratio - mean) / standard_deviation;
        let first_cost = first
            .and_then(|snapshot| {
                context
                    .round_trip_costs_bps
                    .get(&market_key(&snapshot.exchange, &snapshot.symbol))
            })
            .copied()
            .unwrap_or(0.0);
        let second_cost = second
            .and_then(|snapshot| {
                context
                    .round_trip_costs_bps
                    .get(&market_key(&snapshot.exchange, &snapshot.symbol))
            })
            .copied()
            .unwrap_or(0.0);
        let expected_gross_edge_bps =
            ((z_score.abs() - self.config.exit_z_score).max(0.0) * standard_deviation * 10_000.0)
                .clamp(0.0, 10_000.0);
        let estimated_round_trip_cost_bps = first_cost + second_cost;
        let expected_net_edge_bps = expected_gross_edge_bps - estimated_round_trip_cost_bps;
        let statistical_entry = z_score.abs() >= self.config.entry_z_score
            && correlation >= self.config.minimum_correlation;
        let actionable = statistical_entry
            && expected_net_edge_bps + f64::EPSILON >= self.config.minimum_net_edge_bps;
        recommendation.signal = if actionable { -z_score.signum() } else { 0.0 };
        recommendation.confidence = if actionable {
            ((z_score.abs() - self.config.entry_z_score + 1.0) / 3.0).clamp(0.0, 1.0)
                * correlation.max(0.0)
        } else {
            0.0
        };
        recommendation.expected_gross_edge_bps = expected_gross_edge_bps;
        recommendation.estimated_round_trip_cost_bps = estimated_round_trip_cost_bps;
        recommendation.expected_net_edge_bps = expected_net_edge_bps;
        recommendation.actionable = actionable;
        recommendation.executable = actionable && self.config.can_execute_atomically();
        recommendation.rationale = format!(
            "log-ratio z={z_score:.2}, return correlation={correlation:.2}, gross={expected_gross_edge_bps:.1}bps, two-leg cost={estimated_round_trip_cost_bps:.1}bps{}",
            if recommendation.shadow_only {
                "; shadow-only pending atomic hedge execution"
            } else {
                ""
            }
        );
        recommendation
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CarrySleeveConfig {
    pub enabled: bool,
    pub shadow_only: bool,
    pub atomic_hedge_execution_supported: bool,
    pub funding_interval_hours: f64,
    pub expected_holding_days: f64,
    pub minimum_abs_funding_rate: f64,
    pub maximum_abs_funding_rate: f64,
    pub minimum_open_interest_usd: f64,
    pub maximum_abs_basis_bps: f64,
    pub minimum_annualised_net_edge_bps: f64,
    pub maximum_leverage: f64,
    pub minimum_liquidation_buffer_pct: f64,
    pub allow_reverse_carry: bool,
    /// Non-overlapping interval over which funding and basis changes resolve.
    pub shadow_horizon_seconds: u64,
}

impl Default for CarrySleeveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            shadow_only: true,
            atomic_hedge_execution_supported: false,
            funding_interval_hours: 8.0,
            expected_holding_days: 30.0,
            minimum_abs_funding_rate: 0.0001,
            maximum_abs_funding_rate: 0.01,
            minimum_open_interest_usd: 1_000_000.0,
            maximum_abs_basis_bps: 2_000.0,
            minimum_annualised_net_edge_bps: 150.0,
            maximum_leverage: 2.0,
            minimum_liquidation_buffer_pct: 20.0,
            allow_reverse_carry: false,
            shadow_horizon_seconds: 28_800,
        }
    }
}

impl CarrySleeveConfig {
    pub fn normalise(&mut self) {
        self.funding_interval_hours = self.funding_interval_hours.clamp(1.0, 168.0);
        self.expected_holding_days = self.expected_holding_days.clamp(1.0, 365.0);
        self.minimum_abs_funding_rate = self.minimum_abs_funding_rate.clamp(0.0, 0.10);
        self.maximum_abs_funding_rate = self
            .maximum_abs_funding_rate
            .clamp(self.minimum_abs_funding_rate.max(0.0001), 0.10);
        self.minimum_open_interest_usd = self.minimum_open_interest_usd.clamp(0.0, 1.0e15);
        self.maximum_abs_basis_bps = self.maximum_abs_basis_bps.clamp(1.0, 100_000.0);
        self.minimum_annualised_net_edge_bps =
            self.minimum_annualised_net_edge_bps.clamp(0.0, 100_000.0);
        self.maximum_leverage = self.maximum_leverage.clamp(1.0, 20.0);
        self.minimum_liquidation_buffer_pct = self.minimum_liquidation_buffer_pct.clamp(1.0, 95.0);
        self.shadow_horizon_seconds = self.shadow_horizon_seconds.clamp(300, 30 * 86_400);
    }

    fn can_execute_atomically(&self) -> bool {
        !self.shadow_only && self.atomic_hedge_execution_supported
    }
}

struct CarryAlphaSleeve<'a> {
    config: &'a CarrySleeveConfig,
}

impl AlphaSleeve for CarryAlphaSleeve<'_> {
    fn name(&self) -> &'static str {
        "carry"
    }

    fn evaluate(&self, context: &SleeveContext<'_>) -> Vec<SleeveRecommendation> {
        if !self.config.enabled {
            return Vec::new();
        }
        context
            .snapshots
            .par_iter()
            .filter(|snapshot| snapshot.microstructure.funding_rate.is_some())
            .map(|snapshot| self.evaluate_market(snapshot, context))
            .collect()
    }
}

impl CarryAlphaSleeve<'_> {
    fn evaluate_market(
        &self,
        snapshot: &MarketSnapshot,
        context: &SleeveContext<'_>,
    ) -> SleeveRecommendation {
        let funding_rate = snapshot.microstructure.funding_rate.unwrap_or(0.0);
        let basis_bps = snapshot.microstructure.futures_basis_bps.unwrap_or(0.0);
        let periods_per_year = 24.0 / self.config.funding_interval_hours * 365.0;
        let funding_annualised_bps = funding_rate * periods_per_year * 10_000.0;
        let basis_annualised_bps = basis_bps * 365.0 / self.config.expected_holding_days;
        let expected_gross_edge_bps = funding_annualised_bps + basis_annualised_bps;
        let round_trip_cost_bps = context
            .round_trip_costs_bps
            .get(&market_key(&snapshot.exchange, &snapshot.symbol))
            .copied()
            .unwrap_or(0.0)
            * 2.0;
        let annualised_cost_bps = round_trip_cost_bps * 365.0 / self.config.expected_holding_days;
        let expected_net_edge_bps = expected_gross_edge_bps - annualised_cost_bps;
        let range_pct = match (snapshot.high_24h, snapshot.low_24h) {
            (Some(high), Some(low)) if snapshot.price > 0.0 && high >= low => {
                (high - low) / snapshot.price * 100.0
            }
            _ => 0.0,
        };
        // A deliberately conservative approximation used only as a research
        // gate. Venue liquidation prices must be available before live use.
        let liquidation_buffer_pct =
            100.0 / self.config.maximum_leverage - 2.0 * range_pct - basis_bps.abs() / 100.0;
        let open_interest = snapshot.microstructure.open_interest.unwrap_or(0.0);
        let direction_supported = expected_gross_edge_bps >= 0.0 || self.config.allow_reverse_carry;
        let constraints_pass = funding_rate.abs() >= self.config.minimum_abs_funding_rate
            && funding_rate.abs() <= self.config.maximum_abs_funding_rate
            && basis_bps.abs() <= self.config.maximum_abs_basis_bps
            && open_interest >= self.config.minimum_open_interest_usd
            && liquidation_buffer_pct >= self.config.minimum_liquidation_buffer_pct
            && direction_supported;
        let actionable = constraints_pass
            && expected_net_edge_bps + f64::EPSILON >= self.config.minimum_annualised_net_edge_bps;
        let shadow_only = !self.config.can_execute_atomically();
        SleeveRecommendation {
            sleeve_id: format!(
                "carry:{}:{}",
                snapshot.exchange.to_ascii_lowercase(),
                snapshot.symbol.to_ascii_uppercase()
            ),
            kind: AlphaSleeveKind::Carry,
            generated_at: context.now,
            exchange: snapshot.exchange.clone(),
            symbol: snapshot.symbol.clone(),
            hedge_exchange: Some(snapshot.exchange.clone()),
            hedge_symbol: Some(snapshot.symbol.clone()),
            signal: if actionable {
                -expected_gross_edge_bps.signum()
            } else {
                0.0
            },
            confidence: if actionable {
                (expected_net_edge_bps / self.config.minimum_annualised_net_edge_bps.max(1.0) / 3.0)
                    .clamp(0.0, 1.0)
            } else {
                0.0
            },
            expected_gross_edge_bps,
            // Persist the one-off two-leg cost. Expected net edge above is
            // annualised separately, while realised shadow markouts subtract
            // this cost exactly once.
            estimated_round_trip_cost_bps: round_trip_cost_bps,
            expected_net_edge_bps,
            actionable,
            executable: actionable && self.config.can_execute_atomically(),
            shadow_only,
            rationale: format!(
                "annualised funding={funding_annualised_bps:.1}bps, basis={basis_annualised_bps:.1}bps, cost={annualised_cost_bps:.1}bps, open interest=${open_interest:.0}, liquidation buffer≈{liquidation_buffer_pct:.1}%{}",
                if shadow_only {
                    "; shadow-only pending atomic spot/futures hedge execution"
                } else {
                    ""
                }
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AlphaSleevesConfig {
    pub enabled: bool,
    pub recommendation_ledger_size: usize,
    pub pairs: PairsSleeveConfig,
    pub carry: CarrySleeveConfig,
}

impl Default for AlphaSleevesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            recommendation_ledger_size: 2_000,
            pairs: PairsSleeveConfig::default(),
            carry: CarrySleeveConfig::default(),
        }
    }
}

impl AlphaSleevesConfig {
    pub fn normalise(&mut self) {
        self.recommendation_ledger_size = self.recommendation_ledger_size.clamp(100, 100_000);
        self.pairs.normalise();
        self.carry.normalise();
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AlphaSleevesState {
    pub pairs: PairsSleeveState,
    pub latest_recommendations: Vec<SleeveRecommendation>,
    pub recommendation_ledger: VecDeque<SleeveRecommendation>,
    pub evaluations: u64,
    pub last_evaluated_at: Option<f64>,
    /// Non-overlapping opportunities awaiting a future two-leg markout.
    pub pending: VecDeque<SleeveShadowObservation>,
    /// Realised, cost-adjusted research outcomes used to assess sleeve ROI.
    pub resolved: VecDeque<SleeveResolvedObservation>,
    pub last_recorded_by_sleeve: HashMap<String, f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SleeveShadowObservation {
    pub observation_id: String,
    pub sleeve_id: String,
    pub kind: AlphaSleeveKind,
    pub created_at: f64,
    pub due_at: f64,
    pub expires_at: f64,
    pub exchange: String,
    pub symbol: String,
    pub entry_price: f64,
    pub hedge_exchange: String,
    pub hedge_symbol: String,
    pub hedge_entry_price: f64,
    pub signal: f64,
    pub estimated_round_trip_cost_bps: f64,
    pub entry_funding_rate: Option<f64>,
    pub entry_basis_bps: Option<f64>,
    pub funding_interval_hours: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SleeveResolvedObservation {
    pub observation_id: String,
    pub sleeve_id: String,
    pub kind: AlphaSleeveKind,
    pub created_at: f64,
    pub resolved_at: f64,
    pub gross_return_bps: f64,
    pub estimated_round_trip_cost_bps: f64,
    pub net_return_bps: f64,
}

impl AlphaSleevesState {
    fn resolve_due(&mut self, snapshots: &[MarketSnapshot], now: f64, ledger_size: usize) {
        let mut retained = VecDeque::with_capacity(self.pending.len());
        while let Some(observation) = self.pending.pop_front() {
            if observation.due_at > now {
                retained.push_back(observation);
                continue;
            }
            let first = latest_snapshot(snapshots, &observation.exchange, &observation.symbol);
            let hedge = latest_snapshot(
                snapshots,
                &observation.hedge_exchange,
                &observation.hedge_symbol,
            );
            let (Some(first), Some(hedge)) = (first, hedge) else {
                if now <= observation.expires_at {
                    retained.push_back(observation);
                }
                continue;
            };
            let gross_return_bps = match observation.kind {
                AlphaSleeveKind::Pairs => {
                    let first_return = (first.price / observation.entry_price - 1.0) * 10_000.0;
                    let hedge_return =
                        (hedge.price / observation.hedge_entry_price - 1.0) * 10_000.0;
                    observation.signal * (first_return - hedge_return)
                }
                AlphaSleeveKind::Carry => {
                    let elapsed_hours = (now - observation.created_at).max(0.0) / 3_600.0;
                    let periods = elapsed_hours / observation.funding_interval_hours.max(1.0);
                    let funding_bps =
                        observation.entry_funding_rate.unwrap_or(0.0) * periods * 10_000.0;
                    let basis_convergence_bps = observation.entry_basis_bps.unwrap_or(0.0)
                        - first.microstructure.futures_basis_bps.unwrap_or(0.0);
                    -observation.signal * (funding_bps + basis_convergence_bps)
                }
            };
            self.resolved.push_back(SleeveResolvedObservation {
                observation_id: observation.observation_id,
                sleeve_id: observation.sleeve_id,
                kind: observation.kind,
                created_at: observation.created_at,
                resolved_at: now,
                gross_return_bps,
                estimated_round_trip_cost_bps: observation.estimated_round_trip_cost_bps,
                net_return_bps: gross_return_bps - observation.estimated_round_trip_cost_bps,
            });
        }
        self.pending = retained;
        while self.resolved.len() > ledger_size {
            self.resolved.pop_front();
        }
    }

    fn record_actionable(
        &mut self,
        recommendations: &[SleeveRecommendation],
        snapshots: &[MarketSnapshot],
        config: &AlphaSleevesConfig,
        now: f64,
    ) {
        for recommendation in recommendations
            .iter()
            .filter(|recommendation| recommendation.actionable)
        {
            let horizon = match recommendation.kind {
                AlphaSleeveKind::Pairs => config.pairs.shadow_horizon_seconds,
                AlphaSleeveKind::Carry => config.carry.shadow_horizon_seconds,
            };
            if self
                .last_recorded_by_sleeve
                .get(&recommendation.sleeve_id)
                .is_some_and(|last| now - *last + f64::EPSILON < horizon as f64)
            {
                continue;
            }
            let first =
                latest_snapshot(snapshots, &recommendation.exchange, &recommendation.symbol);
            let hedge_exchange = recommendation.hedge_exchange.as_deref().unwrap_or_default();
            let hedge_symbol = recommendation.hedge_symbol.as_deref().unwrap_or_default();
            let hedge = latest_snapshot(snapshots, hedge_exchange, hedge_symbol);
            let (Some(first), Some(hedge)) = (first, hedge) else {
                continue;
            };
            self.pending.push_back(SleeveShadowObservation {
                observation_id: format!("{}-{now:.6}", recommendation.sleeve_id),
                sleeve_id: recommendation.sleeve_id.clone(),
                kind: recommendation.kind.clone(),
                created_at: now,
                due_at: now + horizon as f64,
                expires_at: now + 2.0 * horizon as f64,
                exchange: recommendation.exchange.clone(),
                symbol: recommendation.symbol.clone(),
                entry_price: first.price,
                hedge_exchange: hedge.exchange.clone(),
                hedge_symbol: hedge.symbol.clone(),
                hedge_entry_price: hedge.price,
                signal: recommendation.signal,
                estimated_round_trip_cost_bps: recommendation.estimated_round_trip_cost_bps,
                entry_funding_rate: first.microstructure.funding_rate,
                entry_basis_bps: first.microstructure.futures_basis_bps,
                funding_interval_hours: config.carry.funding_interval_hours,
            });
            self.last_recorded_by_sleeve
                .insert(recommendation.sleeve_id.clone(), now);
        }
        while self.pending.len() > config.recommendation_ledger_size {
            self.pending.pop_front();
        }
    }
}

/// Evaluate independent sleeves on a blocking worker. Rayon evaluates the
/// pair universe and carry universe concurrently inside that worker, leaving
/// Tokio's I/O threads free for exchange and provider traffic.
pub async fn evaluate_sleeves_async(
    mut state: AlphaSleevesState,
    snapshots: Vec<MarketSnapshot>,
    round_trip_costs_bps: HashMap<String, f64>,
    config: AlphaSleevesConfig,
    now: f64,
) -> Result<AlphaSleevesState, String> {
    if !config.enabled {
        return Ok(state);
    }
    tokio::task::spawn_blocking(move || {
        state.resolve_due(&snapshots, now, config.recommendation_ledger_size);
        state.pairs.observe(&snapshots, &config.pairs, now);
        let context = SleeveContext {
            snapshots: &snapshots,
            round_trip_costs_bps: &round_trip_costs_bps,
            now,
        };
        let pairs = PairsAlphaSleeve {
            config: &config.pairs,
            state: &state.pairs,
        };
        let carry = CarryAlphaSleeve {
            config: &config.carry,
        };
        // Names are part of the common interface and are retained in case a
        // future coordinator exposes per-sleeve latency metrics.
        let _sleeve_names = (pairs.name(), carry.name());
        let (mut pair_recommendations, carry_recommendations) =
            rayon::join(|| pairs.evaluate(&context), || carry.evaluate(&context));
        pair_recommendations.extend(carry_recommendations);
        pair_recommendations.sort_by(|left, right| {
            right
                .expected_net_edge_bps
                .partial_cmp(&left.expected_net_edge_bps)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.sleeve_id.cmp(&right.sleeve_id))
        });
        state.record_actionable(&pair_recommendations, &snapshots, &config, now);
        state.latest_recommendations = pair_recommendations.clone();
        for recommendation in pair_recommendations {
            state.recommendation_ledger.push_back(recommendation);
        }
        while state.recommendation_ledger.len() > config.recommendation_ledger_size {
            state.recommendation_ledger.pop_front();
        }
        state.evaluations = state.evaluations.saturating_add(1);
        state.last_evaluated_at = Some(now);
        state
    })
    .await
    .map_err(|error| format!("alpha-sleeve worker failed: {error}"))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmRiskOverlayConfig {
    pub enabled: bool,
    pub time_to_live_seconds: u64,
    pub minimum_coverage: f64,
    pub minimum_agreement: f64,
    pub dampening_risk_score: f64,
    pub veto_risk_score: f64,
    pub minimum_multiplier: f64,
}

impl Default for LlmRiskOverlayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            time_to_live_seconds: 21_600,
            minimum_coverage: 0.50,
            minimum_agreement: 0.50,
            dampening_risk_score: 0.55,
            veto_risk_score: 0.85,
            minimum_multiplier: 0.25,
        }
    }
}

impl LlmRiskOverlayConfig {
    pub fn normalise(&mut self) {
        self.time_to_live_seconds = self.time_to_live_seconds.clamp(60, 7 * 86_400);
        self.minimum_coverage = self.minimum_coverage.clamp(0.0, 1.0);
        self.minimum_agreement = self.minimum_agreement.clamp(0.0, 1.0);
        self.dampening_risk_score = self.dampening_risk_score.clamp(0.0, 0.99);
        self.veto_risk_score = self.veto_risk_score.clamp(self.dampening_risk_score, 1.0);
        self.minimum_multiplier = self.minimum_multiplier.clamp(0.0, 1.0);
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmRiskOverlayState {
    pub updated_at: Option<f64>,
    pub expires_at: Option<f64>,
    pub risk_score: f64,
    pub coverage: f64,
    pub agreement: f64,
    pub confidence_multiplier: f64,
    pub veto: bool,
    pub responders: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LlmOverlayApplication {
    pub active: bool,
    pub vetoed: bool,
    pub multiplier: f64,
    pub reason: Option<String>,
}

impl LlmRiskOverlayState {
    /// Refresh the overlay only from a sufficiently representative LLM round.
    /// Poor provider coverage leaves the previous bounded-TTL value untouched.
    pub fn update_from_consensus(
        &mut self,
        consensus: &AiConsensus,
        now: f64,
        config: &LlmRiskOverlayConfig,
    ) -> bool {
        if !config.enabled || consensus.responders == 0 {
            return false;
        }
        let total = consensus.responders.saturating_add(consensus.failures);
        let coverage = consensus
            .vote_distribution
            .get("coverage")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_else(|| {
                if total == 0 {
                    0.0
                } else {
                    consensus.responders as f64 / total as f64
                }
            })
            .clamp(0.0, 1.0);
        let agreement = consensus
            .vote_distribution
            .get("agreement")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        if coverage < config.minimum_coverage || agreement < config.minimum_agreement {
            return false;
        }
        let risk_score = consensus
            .vote_distribution
            .get("average_risk")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_else(|| {
                let weighted = consensus
                    .advices
                    .iter()
                    .filter(|advice| advice.parsed_ok)
                    .map(|advice| (advice.risk_score.clamp(0.0, 1.0), advice.weight.max(0.05)))
                    .collect::<Vec<_>>();
                let total_weight = weighted.iter().map(|(_, weight)| weight).sum::<f64>();
                if total_weight <= f64::EPSILON {
                    1.0
                } else {
                    weighted
                        .iter()
                        .map(|(risk, weight)| risk * weight)
                        .sum::<f64>()
                        / total_weight
                }
            })
            .clamp(0.0, 1.0);
        let veto = risk_score >= config.veto_risk_score;
        let confidence_multiplier = if risk_score <= config.dampening_risk_score {
            1.0
        } else {
            let range = (config.veto_risk_score - config.dampening_risk_score).max(f64::EPSILON);
            let severity = ((risk_score - config.dampening_risk_score) / range).clamp(0.0, 1.0);
            (1.0 - severity * (1.0 - config.minimum_multiplier))
                .clamp(config.minimum_multiplier, 1.0)
        };
        self.updated_at = Some(now);
        self.expires_at = Some(now + config.time_to_live_seconds as f64);
        self.risk_score = risk_score;
        self.coverage = coverage;
        self.agreement = agreement;
        self.confidence_multiplier = confidence_multiplier;
        self.veto = veto;
        self.responders = consensus.responders;
        true
    }

    /// Apply only downside controls. A hold remains a hold and signal sign is
    /// preserved even when confidence and edge are reduced.
    pub fn apply_to_quant(
        &self,
        signal: &mut QuantSignal,
        now: f64,
        config: &LlmRiskOverlayConfig,
    ) -> LlmOverlayApplication {
        if !config.enabled
            || self.expires_at.is_none_or(|expires_at| now > expires_at)
            || self.updated_at.is_none()
        {
            return LlmOverlayApplication::default();
        }
        let was_actionable = signal.actionable;
        if self.veto && was_actionable {
            signal.actionable = false;
            signal.edge_gate_reason = Some(format!(
                "LLM risk overlay veto: risk {:.2} from {} responders",
                self.risk_score, self.responders
            ));
        } else if was_actionable {
            signal.confidence *= self.confidence_multiplier;
            signal.signal *= self.confidence_multiplier;
            if let Some(edge) = signal.edge_lower_bound_bps.as_mut() {
                *edge *= self.confidence_multiplier;
            }
            signal.rationale.push_str(&format!(
                "; LLM risk overlay multiplier={:.2} risk={:.2}",
                self.confidence_multiplier, self.risk_score
            ));
            if signal.signal.abs() < 0.2 {
                signal.actionable = false;
                signal.edge_gate_reason = Some(
                    "LLM risk overlay reduced the signal below the execution threshold".to_string(),
                );
            }
        }
        LlmOverlayApplication {
            active: true,
            vetoed: self.veto && was_actionable,
            multiplier: if self.veto {
                0.0
            } else {
                self.confidence_multiplier
            },
            reason: signal.edge_gate_reason.clone().filter(|_| self.veto),
        }
    }
}

fn latest_snapshot<'a>(
    snapshots: &'a [MarketSnapshot],
    exchange: &str,
    symbol: &str,
) -> Option<&'a MarketSnapshot> {
    snapshots
        .iter()
        .filter(|snapshot| exchange.is_empty() || snapshot.exchange.eq_ignore_ascii_case(exchange))
        .filter(|snapshot| snapshot.symbol.eq_ignore_ascii_case(symbol))
        .filter(|snapshot| snapshot.price.is_finite() && snapshot.price > 0.0)
        .max_by(|left, right| {
            left.fetched_at
                .partial_cmp(&right.fetched_at)
                .unwrap_or(Ordering::Equal)
        })
}

fn arithmetic_mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn sample_standard_deviation(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}

fn price_return_correlation(history: &VecDeque<PairObservation>) -> f64 {
    if history.len() < 3 {
        return 0.0;
    }
    let first_returns = history
        .iter()
        .zip(history.iter().skip(1))
        .map(|(previous, current)| current.first_log_price - previous.first_log_price)
        .collect::<Vec<_>>();
    let second_returns = history
        .iter()
        .zip(history.iter().skip(1))
        .map(|(previous, current)| current.second_log_price - previous.second_log_price)
        .collect::<Vec<_>>();
    let first_mean = arithmetic_mean(&first_returns);
    let second_mean = arithmetic_mean(&second_returns);
    let covariance = first_returns
        .iter()
        .zip(&second_returns)
        .map(|(first, second)| (first - first_mean) * (second - second_mean))
        .sum::<f64>();
    let first_variance = first_returns
        .iter()
        .map(|value| (value - first_mean).powi(2))
        .sum::<f64>();
    let second_variance = second_returns
        .iter()
        .map(|value| (value - second_mean).powi(2))
        .sum::<f64>();
    let denominator = (first_variance * second_variance).sqrt();
    if denominator <= f64::EPSILON {
        0.0
    } else {
        (covariance / denominator).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trading::{
        advisor::{AiAdvice, AiConsensus},
        octobot::MarketMicrostructure,
    };
    use serde_json::json;

    fn snapshot(symbol: &str, price: f64, timestamp: f64) -> MarketSnapshot {
        MarketSnapshot {
            exchange: "test".to_string(),
            symbol: symbol.to_string(),
            price,
            high_24h: Some(price * 1.01),
            low_24h: Some(price * 0.99),
            fetched_at: timestamp,
            ..MarketSnapshot::default()
        }
    }

    #[test]
    fn pairs_sleeve_detects_a_cost_positive_ratio_dislocation() {
        let mut config = PairsSleeveConfig {
            minimum_samples: 6,
            lookback_samples: 20,
            minimum_correlation: -1.0,
            entry_z_score: 1.0,
            exit_z_score: 0.0,
            minimum_net_edge_bps: 1.0,
            ..PairsSleeveConfig::default()
        };
        config.normalise();
        let mut state = PairsSleeveState::default();
        for index in 0..8 {
            let first_price = if index == 7 {
                115.0
            } else {
                100.0 + index as f64
            };
            let rows = vec![
                snapshot("BTC/USDT", first_price, index as f64 + 1.0),
                snapshot("ETH/USDT", 50.0 + index as f64 * 0.5, index as f64 + 1.0),
            ];
            state.observe(&rows, &config, index as f64 + 1.0);
        }
        let rows = vec![
            snapshot("BTC/USDT", 115.0, 8.0),
            snapshot("ETH/USDT", 53.5, 8.0),
        ];
        let costs = HashMap::from([
            (market_key("test", "BTC/USDT"), 2.0),
            (market_key("test", "ETH/USDT"), 2.0),
        ]);
        let sleeve = PairsAlphaSleeve {
            config: &config,
            state: &state,
        };
        let recommendations = sleeve.evaluate(&SleeveContext {
            snapshots: &rows,
            round_trip_costs_bps: &costs,
            now: 8.0,
        });
        assert!(recommendations[0].actionable);
        assert!(recommendations[0].shadow_only);
        assert!(!recommendations[0].executable);
    }

    #[test]
    fn carry_sleeve_applies_cost_margin_and_liquidity_constraints() {
        let mut row = snapshot("BTC/USDT", 100.0, 1.0);
        row.microstructure = MarketMicrostructure {
            funding_rate: Some(0.0005),
            futures_basis_bps: Some(50.0),
            open_interest: Some(50_000_000.0),
            ..MarketMicrostructure::default()
        };
        let config = CarrySleeveConfig::default();
        let costs = HashMap::from([(market_key("test", "BTC/USDT"), 20.0)]);
        let sleeve = CarryAlphaSleeve { config: &config };
        let recommendations = sleeve.evaluate(&SleeveContext {
            snapshots: &[row],
            round_trip_costs_bps: &costs,
            now: 1.0,
        });
        assert!(recommendations[0].actionable);
        assert!(recommendations[0].shadow_only);
        assert!(!recommendations[0].executable);
    }

    #[test]
    fn pairs_shadow_markout_resolves_both_legs_after_costs() {
        let mut state = AlphaSleevesState::default();
        state.pending.push_back(SleeveShadowObservation {
            observation_id: "pairs-test-1".to_string(),
            sleeve_id: "pairs:test".to_string(),
            kind: AlphaSleeveKind::Pairs,
            created_at: 1.0,
            due_at: 2.0,
            expires_at: 10.0,
            exchange: "test".to_string(),
            symbol: "BTC/USDT".to_string(),
            entry_price: 100.0,
            hedge_exchange: "test".to_string(),
            hedge_symbol: "ETH/USDT".to_string(),
            hedge_entry_price: 50.0,
            signal: 1.0,
            estimated_round_trip_cost_bps: 20.0,
            ..SleeveShadowObservation::default()
        });
        state.resolve_due(
            &[
                snapshot("BTC/USDT", 102.0, 2.0),
                snapshot("ETH/USDT", 50.5, 2.0),
            ],
            2.0,
            100,
        );
        assert!(state.pending.is_empty());
        assert_eq!(state.resolved.len(), 1);
        let result = &state.resolved[0];
        assert!((result.gross_return_bps - 100.0).abs() < 1e-9);
        assert!((result.net_return_bps - 80.0).abs() < 1e-9);
    }

    fn consensus(risk: f64) -> AiConsensus {
        AiConsensus {
            action: "hold".to_string(),
            confidence: 0.8,
            signal: 0.0,
            vote_distribution: json!({
                "coverage": 1.0,
                "agreement": 0.9,
                "average_risk": risk,
            }),
            advices: vec![AiAdvice {
                provider: "test".to_string(),
                model: None,
                action: "hold".to_string(),
                confidence: 0.8,
                reasoning: String::new(),
                suggested_amount_usd: None,
                risk_score: risk,
                risk_flags: Vec::new(),
                target_symbol: None,
                raw_response: String::new(),
                parsed_ok: true,
                weight: 1.0,
            }],
            responders: 1,
            failures: 0,
        }
    }

    #[test]
    fn llm_overlay_can_only_reduce_or_veto_quant() {
        let config = LlmRiskOverlayConfig::default();
        let mut overlay = LlmRiskOverlayState::default();
        assert!(overlay.update_from_consensus(&consensus(0.70), 100.0, &config));
        let mut actionable = QuantSignal {
            signal: 0.8,
            confidence: 0.9,
            actionable: true,
            edge_lower_bound_bps: Some(40.0),
            ..QuantSignal::default()
        };
        overlay.apply_to_quant(&mut actionable, 101.0, &config);
        assert!((0.0..0.8).contains(&actionable.signal));
        assert!(actionable.confidence < 0.9);

        let mut hold = QuantSignal::default();
        overlay.apply_to_quant(&mut hold, 101.0, &config);
        assert!(!hold.actionable);
        assert_eq!(hold.signal, 0.0);

        assert!(overlay.update_from_consensus(&consensus(0.95), 200.0, &config));
        overlay.apply_to_quant(&mut actionable, 201.0, &config);
        assert!(!actionable.actionable);
    }

    #[test]
    fn expired_overlay_does_not_change_quant() {
        let config = LlmRiskOverlayConfig {
            time_to_live_seconds: 60,
            ..LlmRiskOverlayConfig::default()
        };
        let mut overlay = LlmRiskOverlayState::default();
        overlay.update_from_consensus(&consensus(0.95), 100.0, &config);
        let mut signal = QuantSignal {
            signal: 0.8,
            confidence: 0.9,
            actionable: true,
            ..QuantSignal::default()
        };
        let application = overlay.apply_to_quant(&mut signal, 161.0, &config);
        assert!(!application.active);
        assert!(signal.actionable);
        assert_eq!(signal.signal, 0.8);
    }
}
