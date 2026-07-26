//! Deterministic pre-trade economics.
//!
//! LLM confidence is not a profit estimate. This module converts the bounded
//! signal/confidence pair into an intentionally conservative expected move,
//! applies outcome calibration, and subtracts round-trip fee/slippage costs.
//! Keeping this calculation pure makes the live gate, sizing, logs, and tests
//! use exactly the same assumptions.

use serde::{Deserialize, Serialize};

use super::config::TradingConfig;

/// Auditable economics attached to every decision.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TradeEconomics {
    pub expected_gross_edge_bps: f64,
    pub estimated_round_trip_cost_bps: f64,
    pub expected_net_edge_bps: f64,
    pub required_net_edge_bps: f64,
    pub calibration_multiplier: f64,
}

impl TradeEconomics {
    pub fn is_worthwhile(&self) -> bool {
        self.expected_net_edge_bps + f64::EPSILON >= self.required_net_edge_bps
    }

    /// Scale used for position sizing after the minimum profitable edge has
    /// been cleared. A marginal trade remains small; a strongly positive trade
    /// can use the configured maximum without exceeding it.
    pub fn size_multiplier(&self) -> f64 {
        if !self.is_worthwhile() {
            return 0.0;
        }
        let surplus = self.expected_net_edge_bps - self.required_net_edge_bps;
        let useful_range = (self.expected_gross_edge_bps - self.required_net_edge_bps)
            .max(self.required_net_edge_bps.max(1.0));
        (0.20 + 0.80 * (surplus / useful_range).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    }
}

pub fn estimate_trade_economics(
    signal_strength: f64,
    confidence: f64,
    calibration_multiplier: f64,
    config: &TradingConfig,
) -> TradeEconomics {
    let calibration_multiplier = calibration_multiplier.clamp(0.25, 1.50);
    let expected_gross_edge_bps = config.expected_move_bps
        * signal_strength.clamp(0.0, 1.0)
        * confidence.clamp(0.0, 1.0)
        * calibration_multiplier;
    // A markout must cover entering now and eventually unwinding the position.
    let estimated_round_trip_cost_bps =
        2.0 * (config.estimated_fee_bps + config.estimated_slippage_bps);
    TradeEconomics {
        expected_gross_edge_bps,
        estimated_round_trip_cost_bps,
        expected_net_edge_bps: expected_gross_edge_bps - estimated_round_trip_cost_bps,
        required_net_edge_bps: config.minimum_net_edge_bps,
        calibration_multiplier,
    }
}

/// Price movement that hurts the proposed trade between decision and submit.
pub fn adverse_reprice_drift_bps(side: &str, reference_price: f64, current_price: f64) -> f64 {
    if !reference_price.is_finite()
        || reference_price <= 0.0
        || !current_price.is_finite()
        || current_price <= 0.0
    {
        return f64::INFINITY;
    }
    let change_bps = (current_price - reference_price) / reference_price * 10_000.0;
    if side.eq_ignore_ascii_case("buy") {
        change_bps.max(0.0)
    } else if side.eq_ignore_ascii_case("sell") {
        (-change_bps).max(0.0)
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn economics_rejects_fee_negative_signal() {
        let config = TradingConfig::default();
        let economics = estimate_trade_economics(0.2, 0.5, 1.0, &config);
        assert!(!economics.is_worthwhile());
        assert_eq!(economics.size_multiplier(), 0.0);
    }

    #[test]
    fn adverse_drift_is_directional() {
        assert_eq!(adverse_reprice_drift_bps("buy", 100.0, 101.0), 100.0);
        assert_eq!(adverse_reprice_drift_bps("buy", 100.0, 99.0), 0.0);
        assert_eq!(adverse_reprice_drift_bps("sell", 100.0, 99.0), 100.0);
    }
}
