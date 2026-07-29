//! Multi-horizon, cost-aware calibration for quant signals.
//!
//! A signal is not a profit estimate.  This module records non-overlapping
//! forward observations at several holding horizons, estimates a conservative
//! conditional gross return and permits execution only when its lower bound
//! clears executable costs and the configured safety margin.

use std::{cmp::Ordering, collections::HashMap, collections::VecDeque};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::super::{octobot::MarketSnapshot, quant::QuantSignal};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MultiHorizonConfig {
    pub enabled: bool,
    /// Forward labels evaluated independently. Defaults cover intraday through
    /// daily holding periods without pretending a 60-day factor is a 15-minute
    /// forecast.
    pub horizons_seconds: Vec<u64>,
    pub expiry_grace_seconds: u64,
    pub ledger_size: usize,
    pub minimum_samples: usize,
    pub confidence_z_score: f64,
    pub minimum_net_edge_bps: f64,
    /// Fail closed when no statistically usable calibration is available.
    pub require_calibrated_edge: bool,
}

impl Default for MultiHorizonConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            horizons_seconds: vec![900, 3_600, 14_400, 86_400],
            expiry_grace_seconds: 3_600,
            ledger_size: 20_000,
            minimum_samples: 24,
            confidence_z_score: 1.645,
            minimum_net_edge_bps: 10.0,
            require_calibrated_edge: true,
        }
    }
}

impl MultiHorizonConfig {
    pub fn normalise(&mut self) {
        self.horizons_seconds = self
            .horizons_seconds
            .iter()
            .copied()
            .map(|horizon| horizon.clamp(60, 30 * 86_400))
            .collect();
        self.horizons_seconds.sort_unstable();
        self.horizons_seconds.dedup();
        if self.horizons_seconds.is_empty() {
            self.horizons_seconds = vec![900, 3_600, 14_400, 86_400];
        }
        self.expiry_grace_seconds = self.expiry_grace_seconds.clamp(60, 7 * 86_400);
        self.ledger_size = self.ledger_size.clamp(100, 1_000_000);
        self.minimum_samples = self.minimum_samples.clamp(3, 100_000);
        self.confidence_z_score = self.confidence_z_score.clamp(0.0, 4.0);
        self.minimum_net_edge_bps = self.minimum_net_edge_bps.clamp(0.0, 2_500.0);
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PendingHorizonObservation {
    pub observation_id: String,
    pub created_at: f64,
    pub due_at: f64,
    pub expires_at: f64,
    pub horizon_seconds: u64,
    pub exchange: String,
    pub symbol: String,
    pub entry_price: f64,
    pub signal: f64,
    pub confidence: f64,
    pub regime: String,
    pub estimated_round_trip_cost_bps: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ResolvedHorizonObservation {
    pub observation_id: String,
    pub created_at: f64,
    pub resolved_at: f64,
    pub horizon_seconds: u64,
    pub exchange: String,
    pub symbol: String,
    pub signal: f64,
    pub confidence: f64,
    pub regime: String,
    pub gross_return_bps: f64,
    pub estimated_round_trip_cost_bps: f64,
    pub net_return_bps: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EdgeEstimate {
    pub horizon_seconds: u64,
    pub samples: usize,
    pub conditioning_level: String,
    pub mean_gross_return_bps: f64,
    pub standard_deviation_bps: f64,
    pub gross_lower_bound_bps: f64,
    pub estimated_round_trip_cost_bps: f64,
    pub net_lower_bound_bps: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HorizonResolveSummary {
    pub resolved: usize,
    pub expired: usize,
}

/// Persistent calibration ledger.  Older trading-state snapshots restore
/// safely through `serde(default)`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MultiHorizonCalibrationState {
    pub pending: VecDeque<PendingHorizonObservation>,
    pub resolved: VecDeque<ResolvedHorizonObservation>,
    pub last_recorded_by_horizon: HashMap<u64, f64>,
}

impl MultiHorizonCalibrationState {
    /// Record raw policy intent before the edge gate is applied.  This avoids a
    /// self-defeating cold start in which a fail-closed gate never gathers the
    /// observations required to calibrate itself.
    pub fn record_signal(
        &mut self,
        signal: &QuantSignal,
        snapshot: &MarketSnapshot,
        regime: &str,
        estimated_round_trip_cost_bps: f64,
        config: &MultiHorizonConfig,
        now: f64,
    ) -> usize {
        if !config.enabled
            || !signal.raw_actionable
            || signal.signal.abs() < 0.2
            || !snapshot.price.is_finite()
            || snapshot.price <= 0.0
        {
            return 0;
        }
        let mut recorded = 0usize;
        for horizon in &config.horizons_seconds {
            if self
                .last_recorded_by_horizon
                .get(horizon)
                .is_some_and(|last| now - *last + f64::EPSILON < *horizon as f64)
            {
                continue;
            }
            let due_at = now + *horizon as f64;
            self.pending.push_back(PendingHorizonObservation {
                observation_id: format!("{}-{horizon}", Uuid::new_v4()),
                created_at: now,
                due_at,
                expires_at: due_at + config.expiry_grace_seconds as f64,
                horizon_seconds: *horizon,
                exchange: snapshot.exchange.clone(),
                symbol: snapshot.symbol.clone(),
                entry_price: snapshot.price,
                signal: signal.signal,
                confidence: signal.confidence,
                regime: regime.to_string(),
                estimated_round_trip_cost_bps: estimated_round_trip_cost_bps.max(0.0),
            });
            self.last_recorded_by_horizon.insert(*horizon, now);
            recorded += 1;
        }
        while self.pending.len() > config.ledger_size {
            self.pending.pop_front();
        }
        recorded
    }

    pub fn resolve_due(
        &mut self,
        snapshots: &[MarketSnapshot],
        config: &MultiHorizonConfig,
        now: f64,
    ) -> HorizonResolveSummary {
        let mut summary = HorizonResolveSummary::default();
        let mut remaining = VecDeque::new();
        while let Some(pending) = self.pending.pop_front() {
            if pending.due_at > now {
                remaining.push_back(pending);
                continue;
            }
            let observed = exact_market_snapshot(snapshots, &pending.exchange, &pending.symbol);
            let Some(observed) = observed else {
                if pending.expires_at > now {
                    remaining.push_back(pending);
                } else {
                    summary.expired += 1;
                }
                continue;
            };
            let raw_return_bps = ((observed.price / pending.entry_price) - 1.0) * 10_000.0;
            let gross_return_bps = pending.signal.signum() * raw_return_bps;
            if gross_return_bps.is_finite() {
                self.resolved.push_back(ResolvedHorizonObservation {
                    observation_id: pending.observation_id,
                    created_at: pending.created_at,
                    resolved_at: now,
                    horizon_seconds: pending.horizon_seconds,
                    exchange: pending.exchange,
                    symbol: pending.symbol,
                    signal: pending.signal,
                    confidence: pending.confidence,
                    regime: pending.regime,
                    gross_return_bps,
                    estimated_round_trip_cost_bps: pending.estimated_round_trip_cost_bps,
                    net_return_bps: gross_return_bps - pending.estimated_round_trip_cost_bps,
                });
                summary.resolved += 1;
            }
        }
        self.pending = remaining;
        while self.resolved.len() > config.ledger_size {
            self.resolved.pop_front();
        }
        summary
    }

    /// Return the strongest horizon whose conservative expected gross move is
    /// supported by enough past observations.
    pub fn best_edge(
        &self,
        signal: &QuantSignal,
        regime: &str,
        estimated_round_trip_cost_bps: f64,
        config: &MultiHorizonConfig,
    ) -> Option<EdgeEstimate> {
        config
            .horizons_seconds
            .iter()
            .filter_map(|horizon| {
                self.estimate(
                    &signal.symbol,
                    regime,
                    signal.signal.signum(),
                    *horizon,
                    estimated_round_trip_cost_bps,
                    config,
                )
            })
            .max_by(|left, right| {
                left.net_lower_bound_bps
                    .partial_cmp(&right.net_lower_bound_bps)
                    .unwrap_or(Ordering::Equal)
            })
    }

    fn estimate(
        &self,
        symbol: &str,
        regime: &str,
        direction: f64,
        horizon_seconds: u64,
        estimated_round_trip_cost_bps: f64,
        config: &MultiHorizonConfig,
    ) -> Option<EdgeEstimate> {
        let base = self
            .resolved
            .iter()
            .filter(|item| item.horizon_seconds == horizon_seconds)
            .filter(|item| item.signal.signum() == direction.signum())
            .collect::<Vec<_>>();
        let candidates = [
            (
                "symbol_regime",
                base.iter()
                    .copied()
                    .filter(|item| item.symbol.eq_ignore_ascii_case(symbol))
                    .filter(|item| item.regime.eq_ignore_ascii_case(regime))
                    .collect::<Vec<_>>(),
            ),
            (
                "symbol",
                base.iter()
                    .copied()
                    .filter(|item| item.symbol.eq_ignore_ascii_case(symbol))
                    .collect::<Vec<_>>(),
            ),
            ("horizon", base),
        ];
        let (conditioning_level, observations) = candidates
            .into_iter()
            .find(|(_, observations)| observations.len() >= config.minimum_samples)?;
        let values = observations
            .iter()
            .map(|item| item.gross_return_bps)
            .collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = if values.len() > 1 {
            values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (values.len() - 1) as f64
        } else {
            0.0
        };
        let standard_deviation = variance.sqrt();
        let gross_lower_bound =
            mean - config.confidence_z_score * standard_deviation / (values.len() as f64).sqrt();
        Some(EdgeEstimate {
            horizon_seconds,
            samples: values.len(),
            conditioning_level: conditioning_level.to_string(),
            mean_gross_return_bps: mean,
            standard_deviation_bps: standard_deviation,
            gross_lower_bound_bps: gross_lower_bound,
            estimated_round_trip_cost_bps,
            net_lower_bound_bps: gross_lower_bound - estimated_round_trip_cost_bps,
        })
    }

    /// Apply the empirical execution gate while preserving raw policy intent
    /// for shadow observation collection.
    pub fn gate_signal(
        &self,
        signal: &mut QuantSignal,
        regime: &str,
        estimated_round_trip_cost_bps: f64,
        config: &MultiHorizonConfig,
    ) {
        signal.estimated_round_trip_cost_bps = estimated_round_trip_cost_bps;
        if !config.enabled || !signal.raw_actionable {
            return;
        }
        let estimate = self.best_edge(signal, regime, estimated_round_trip_cost_bps, config);
        let Some(estimate) = estimate else {
            if config.require_calibrated_edge {
                signal.actionable = false;
                signal.edge_gate_reason = Some(format!(
                    "cost-aware hold: fewer than {} comparable multi-horizon observations",
                    config.minimum_samples
                ));
                append_gate_rationale(signal);
            }
            return;
        };
        signal.selected_horizon_seconds = Some(estimate.horizon_seconds);
        signal.expected_gross_edge_bps = Some(estimate.mean_gross_return_bps);
        signal.edge_lower_bound_bps = Some(estimate.net_lower_bound_bps);
        if estimate.net_lower_bound_bps + f64::EPSILON < config.minimum_net_edge_bps {
            signal.actionable = false;
            signal.edge_gate_reason = Some(format!(
                "cost-aware hold: {}s net lower bound {:.2}bps below {:.2}bps",
                estimate.horizon_seconds, estimate.net_lower_bound_bps, config.minimum_net_edge_bps
            ));
        } else {
            signal.actionable = true;
            signal.edge_gate_reason = None;
        }
        append_gate_rationale(signal);
    }
}

fn append_gate_rationale(signal: &mut QuantSignal) {
    if let Some(reason) = signal.edge_gate_reason.as_deref()
        && !signal.rationale.contains(reason)
    {
        signal.rationale.push_str("; ");
        signal.rationale.push_str(reason);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(price: f64, timestamp: f64) -> MarketSnapshot {
        MarketSnapshot {
            exchange: "test".to_string(),
            symbol: "BTC/USDT".to_string(),
            price,
            fetched_at: timestamp,
            ..MarketSnapshot::default()
        }
    }

    fn raw_signal() -> QuantSignal {
        QuantSignal {
            parameter_id: "test".to_string(),
            exchange: "test".to_string(),
            symbol: "BTC/USDT".to_string(),
            signal: 0.8,
            confidence: 0.8,
            raw_actionable: true,
            actionable: true,
            ..QuantSignal::default()
        }
    }

    #[test]
    fn horizons_are_recorded_independently_and_without_overlap() {
        let config = MultiHorizonConfig {
            horizons_seconds: vec![60, 120],
            minimum_samples: 3,
            ..MultiHorizonConfig::default()
        };
        let mut state = MultiHorizonCalibrationState::default();
        assert_eq!(
            state.record_signal(
                &raw_signal(),
                &snapshot(100.0, 0.0),
                "trend",
                10.0,
                &config,
                0.0
            ),
            2
        );
        assert_eq!(
            state.record_signal(
                &raw_signal(),
                &snapshot(100.0, 30.0),
                "trend",
                10.0,
                &config,
                30.0
            ),
            0
        );
        let summary = state.resolve_due(&[snapshot(101.0, 60.0)], &config, 60.0);
        assert_eq!(summary.resolved, 1);
        assert_eq!(state.pending.len(), 1);
        assert!((state.resolved[0].net_return_bps - 90.0).abs() < 1e-9);
    }

    #[test]
    fn cost_gate_selects_the_profitable_supported_horizon() {
        let config = MultiHorizonConfig {
            horizons_seconds: vec![60, 120],
            minimum_samples: 3,
            confidence_z_score: 0.0,
            minimum_net_edge_bps: 5.0,
            ..MultiHorizonConfig::default()
        };
        let mut state = MultiHorizonCalibrationState::default();
        for index in 0..3 {
            for (horizon, gross) in [(60, 15.0), (120, 80.0)] {
                state.resolved.push_back(ResolvedHorizonObservation {
                    observation_id: format!("{index}-{horizon}"),
                    horizon_seconds: horizon,
                    symbol: "BTC/USDT".to_string(),
                    signal: 0.8,
                    regime: "trend".to_string(),
                    gross_return_bps: gross,
                    ..ResolvedHorizonObservation::default()
                });
            }
        }
        let mut signal = raw_signal();
        state.gate_signal(&mut signal, "trend", 50.0, &config);
        assert!(signal.actionable);
        assert_eq!(signal.selected_horizon_seconds, Some(120));
        assert_eq!(signal.edge_lower_bound_bps, Some(30.0));
    }

    #[test]
    fn cost_gate_fails_closed_during_calibration_cold_start() {
        let config = MultiHorizonConfig {
            horizons_seconds: vec![60],
            minimum_samples: 3,
            ..MultiHorizonConfig::default()
        };
        let state = MultiHorizonCalibrationState::default();
        let mut signal = raw_signal();
        state.gate_signal(&mut signal, "trend", 50.0, &config);
        assert!(!signal.actionable);
        assert!(signal.edge_gate_reason.is_some());
        assert!(signal.raw_actionable);
    }
}
