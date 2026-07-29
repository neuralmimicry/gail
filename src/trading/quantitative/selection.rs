//! Conservative strategy selection and promotion configuration.
//!
//! A no-trade policy is an economically meaningful alternative, not a missing
//! observation.  The controller therefore compares actionable expectancy with
//! an explicit cash arm and requires a confidence-adjusted positive edge before
//! a trading arm may replace it.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct StrategySelectionConfig {
    pub cash_parameter_id: String,
    pub minimum_actionable_net_edge_bps: f64,
    pub confidence_z_score: f64,
    pub require_native_validation_for_promotion: bool,
    pub native_validation_max_age_seconds: u64,
}

impl Default for StrategySelectionConfig {
    fn default() -> Self {
        Self {
            cash_parameter_id: "cash-v1".to_string(),
            minimum_actionable_net_edge_bps: 0.0,
            confidence_z_score: 1.645,
            require_native_validation_for_promotion: true,
            native_validation_max_age_seconds: 24 * 3_600,
        }
    }
}

impl StrategySelectionConfig {
    pub fn normalise(&mut self) {
        self.cash_parameter_id = self.cash_parameter_id.trim().to_ascii_lowercase();
        if self.cash_parameter_id.is_empty() {
            self.cash_parameter_id = "cash-v1".to_string();
        }
        self.minimum_actionable_net_edge_bps =
            self.minimum_actionable_net_edge_bps.clamp(0.0, 2_500.0);
        self.confidence_z_score = self.confidence_z_score.clamp(0.0, 4.0);
        self.native_validation_max_age_seconds = self
            .native_validation_max_age_seconds
            .clamp(300, 30 * 86_400);
    }
}
