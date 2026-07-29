//! Fill telemetry and executable transaction-cost estimation.
//!
//! Modelled slippage is retained only as a cold-start fallback. Once enough
//! venue/symbol/side fills exist, their robust median is compared with current
//! spread/depth impact and the more conservative estimate gates the trade.

use std::{cmp::Ordering, collections::VecDeque};

use serde::{Deserialize, Serialize};

use super::super::octobot::MarketSnapshot;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecutionTelemetryConfig {
    pub enabled: bool,
    pub ledger_size: usize,
    pub minimum_empirical_samples: usize,
    pub maximum_cost_bps: f64,
}

impl Default for ExecutionTelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ledger_size: 10_000,
            minimum_empirical_samples: 8,
            maximum_cost_bps: 2_500.0,
        }
    }
}

impl ExecutionTelemetryConfig {
    pub fn normalise(&mut self) {
        self.ledger_size = self.ledger_size.clamp(100, 1_000_000);
        self.minimum_empirical_samples = self.minimum_empirical_samples.clamp(3, 10_000);
        self.maximum_cost_bps = self.maximum_cost_bps.clamp(10.0, 10_000.0);
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecutionCostObservation {
    pub observed_at: f64,
    pub exchange: String,
    pub symbol: String,
    pub side: String,
    pub order_id: String,
    pub notional_usd: f64,
    pub reference_price: f64,
    pub fill_price: f64,
    pub observed_slippage_bps: f64,
    pub estimated_fee_bps: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecutableCostEstimate {
    pub round_trip_cost_bps: f64,
    pub projected_market_cost_bps: f64,
    pub empirical_market_cost_bps: Option<f64>,
    pub empirical_samples: usize,
    pub used_fallback: bool,
}

/// Result of applying immediate price and transaction-cost changes to the
/// decision-time net edge.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RepricedEdge {
    pub cost_regression_bps: f64,
    pub net_edge_bps: f64,
}

/// Recalculate net edge without rewarding a last-moment cost improvement.
/// Improvements remain a conservative execution buffer; regressions must be
/// paid in full before an order may be submitted.
pub fn reprice_net_edge(
    decision_net_edge_bps: f64,
    decision_round_trip_cost_bps: f64,
    fresh_round_trip_cost_bps: f64,
    adverse_drift_bps: f64,
) -> RepricedEdge {
    let cost_regression_bps = (fresh_round_trip_cost_bps - decision_round_trip_cost_bps).max(0.0);
    RepricedEdge {
        cost_regression_bps,
        net_edge_bps: decision_net_edge_bps - adverse_drift_bps.max(0.0) - cost_regression_bps,
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecutionTelemetryState {
    pub observations: VecDeque<ExecutionCostObservation>,
}

impl ExecutionTelemetryState {
    #[allow(clippy::too_many_arguments)]
    pub fn record_fill(
        &mut self,
        observed_at: f64,
        exchange: &str,
        symbol: &str,
        side: &str,
        order_id: &str,
        notional_usd: f64,
        reference_price: f64,
        fill_price: f64,
        estimated_fee_bps: f64,
        config: &ExecutionTelemetryConfig,
    ) -> bool {
        if !config.enabled
            || reference_price <= 0.0
            || fill_price <= 0.0
            || !reference_price.is_finite()
            || !fill_price.is_finite()
        {
            return false;
        }
        let signed_move_bps = (fill_price / reference_price - 1.0) * 10_000.0;
        let observed_slippage_bps = if side.eq_ignore_ascii_case("sell") {
            -signed_move_bps
        } else {
            signed_move_bps
        }
        .max(0.0);
        self.observations.push_back(ExecutionCostObservation {
            observed_at,
            exchange: exchange.to_ascii_lowercase(),
            symbol: symbol.to_ascii_uppercase(),
            side: side.to_ascii_lowercase(),
            order_id: order_id.to_string(),
            notional_usd: notional_usd.max(0.0),
            reference_price,
            fill_price,
            observed_slippage_bps,
            estimated_fee_bps: estimated_fee_bps.max(0.0),
        });
        while self.observations.len() > config.ledger_size {
            self.observations.pop_front();
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn estimate_round_trip(
        &self,
        snapshot: &MarketSnapshot,
        entry_side: &str,
        notional_usd: f64,
        one_way_fee_bps: f64,
        fallback_one_way_slippage_bps: f64,
        config: &ExecutionTelemetryConfig,
    ) -> ExecutableCostEstimate {
        if !config.enabled {
            return ExecutableCostEstimate {
                round_trip_cost_bps: (2.0
                    * (one_way_fee_bps.max(0.0) + fallback_one_way_slippage_bps.max(0.0)))
                .clamp(0.0, config.maximum_cost_bps),
                projected_market_cost_bps: 2.0 * fallback_one_way_slippage_bps.max(0.0),
                empirical_market_cost_bps: None,
                empirical_samples: 0,
                used_fallback: true,
            };
        }
        let exit_side = if entry_side.eq_ignore_ascii_case("sell") {
            "buy"
        } else {
            "sell"
        };
        let entry_projection = snapshot.projected_one_way_slippage_bps(
            entry_side,
            notional_usd,
            fallback_one_way_slippage_bps,
        );
        let exit_projection = snapshot.projected_one_way_slippage_bps(
            exit_side,
            notional_usd,
            fallback_one_way_slippage_bps,
        );
        let projected_market_cost_bps = entry_projection + exit_projection;
        let entry_empirical = self.empirical_slippage(
            &snapshot.exchange,
            &snapshot.symbol,
            entry_side,
            config.minimum_empirical_samples,
        );
        let exit_empirical = self.empirical_slippage(
            &snapshot.exchange,
            &snapshot.symbol,
            exit_side,
            config.minimum_empirical_samples,
        );
        let empirical_samples = entry_empirical
            .as_ref()
            .map(|(_, samples)| *samples)
            .unwrap_or(0)
            + exit_empirical
                .as_ref()
                .map(|(_, samples)| *samples)
                .unwrap_or(0);
        let empirical_market_cost_bps = match (entry_empirical, exit_empirical) {
            (Some((entry, _)), Some((exit, _))) => Some(entry + exit),
            _ => None,
        };
        let market_cost = empirical_market_cost_bps
            .map(|empirical| empirical.max(projected_market_cost_bps))
            .unwrap_or(projected_market_cost_bps);
        ExecutableCostEstimate {
            round_trip_cost_bps: (2.0 * one_way_fee_bps.max(0.0) + market_cost)
                .clamp(0.0, config.maximum_cost_bps),
            projected_market_cost_bps,
            empirical_market_cost_bps,
            empirical_samples,
            used_fallback: snapshot.spread_bps().is_none()
                || snapshot.executable_depth_usd().is_none(),
        }
    }

    fn empirical_slippage(
        &self,
        exchange: &str,
        symbol: &str,
        side: &str,
        minimum_samples: usize,
    ) -> Option<(f64, usize)> {
        let mut values = self
            .observations
            .iter()
            .rev()
            .filter(|observation| observation.exchange.eq_ignore_ascii_case(exchange))
            .filter(|observation| observation.symbol.eq_ignore_ascii_case(symbol))
            .filter(|observation| observation.side.eq_ignore_ascii_case(side))
            .take(500)
            .map(|observation| observation.observed_slippage_bps)
            .collect::<Vec<_>>();
        if values.len() < minimum_samples {
            return None;
        }
        values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        let middle = values.len() / 2;
        let median = if values.len() % 2 == 0 {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        };
        Some((median, values.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trading::octobot::MarketMicrostructure;

    #[test]
    fn projected_cost_uses_spread_depth_and_fees() {
        let snapshot = MarketSnapshot {
            exchange: "test".to_string(),
            symbol: "BTC/USDT".to_string(),
            price: 100.0,
            microstructure: MarketMicrostructure {
                best_bid: Some(99.9),
                best_ask: Some(100.1),
                bid_depth_usd: Some(10_000.0),
                ask_depth_usd: Some(10_000.0),
                ..MarketMicrostructure::default()
            },
            ..MarketSnapshot::default()
        };
        let estimate = ExecutionTelemetryState::default().estimate_round_trip(
            &snapshot,
            "buy",
            100.0,
            10.0,
            15.0,
            &ExecutionTelemetryConfig::default(),
        );
        assert!(estimate.round_trip_cost_bps > 20.0);
        assert!(!estimate.used_fallback);
    }

    #[test]
    fn fill_history_supplies_empirical_cost_after_minimum_samples() {
        let config = ExecutionTelemetryConfig {
            minimum_empirical_samples: 3,
            ..ExecutionTelemetryConfig::default()
        };
        let mut state = ExecutionTelemetryState::default();
        for index in 0..3 {
            state.record_fill(
                index as f64,
                "test",
                "BTC/USDT",
                "buy",
                &index.to_string(),
                25.0,
                100.0,
                100.2,
                10.0,
                &config,
            );
            state.record_fill(
                index as f64,
                "test",
                "BTC/USDT",
                "sell",
                &format!("s{index}"),
                25.0,
                100.0,
                99.8,
                10.0,
                &config,
            );
        }
        let snapshot = MarketSnapshot {
            exchange: "test".to_string(),
            symbol: "BTC/USDT".to_string(),
            price: 100.0,
            ..MarketSnapshot::default()
        };
        let estimate = state.estimate_round_trip(&snapshot, "buy", 25.0, 10.0, 1.0, &config);
        assert_eq!(estimate.empirical_samples, 6);
        assert!(estimate.empirical_market_cost_bps.unwrap() > 39.0);
    }

    #[test]
    fn immediate_reprice_charges_drift_and_only_adverse_cost_change() {
        let adverse = reprice_net_edge(50.0, 20.0, 30.0, 5.0);
        assert_eq!(adverse.cost_regression_bps, 10.0);
        assert_eq!(adverse.net_edge_bps, 35.0);

        let improved = reprice_net_edge(50.0, 30.0, 20.0, 5.0);
        assert_eq!(improved.cost_regression_bps, 0.0);
        assert_eq!(improved.net_edge_bps, 45.0);
    }
}
