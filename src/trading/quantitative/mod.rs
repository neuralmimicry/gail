//! Second-generation quantitative research and execution components.
//!
//! The legacy [`super::quant`] module remains the durable shadow-to-primary
//! controller.  This module supplies reusable, independently testable building
//! blocks for native replay, multi-horizon calibration, cross-sectional
//! allocation, execution telemetry, alpha sleeves and LLM risk overlays.  The
//! separation is deliberate: research workloads may use parallel CPU work,
//! whereas the live bridge must remain asynchronous and bounded.

pub mod backtest;
pub mod calibration;
pub mod portfolio;
pub mod selection;
pub mod sleeves;
pub mod telemetry;

use serde::{Deserialize, Serialize};

use backtest::NativeBacktestConfig;
use calibration::MultiHorizonConfig;
use portfolio::CrossSectionalConfig;
use selection::StrategySelectionConfig;
use sleeves::{AlphaSleevesConfig, LlmRiskOverlayConfig};
use telemetry::ExecutionTelemetryConfig;

/// Configuration for the modular quantitative stack.
///
/// New settings are nested beneath `trading.quantitative`, which keeps the
/// legacy configuration stable while allowing individual research components
/// to be enabled independently.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantitativeConfig {
    pub enabled: bool,
    pub native_backtest_enabled: bool,
    pub native_backtest: NativeBacktestConfig,
    pub calibration: MultiHorizonConfig,
    pub selection: StrategySelectionConfig,
    pub portfolio: CrossSectionalConfig,
    pub execution_telemetry: ExecutionTelemetryConfig,
    pub alpha_sleeves: AlphaSleevesConfig,
    pub llm_risk_overlay: LlmRiskOverlayConfig,
}

impl Default for QuantitativeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            native_backtest_enabled: true,
            native_backtest: NativeBacktestConfig::default(),
            calibration: MultiHorizonConfig::default(),
            selection: StrategySelectionConfig::default(),
            portfolio: CrossSectionalConfig::default(),
            execution_telemetry: ExecutionTelemetryConfig::default(),
            alpha_sleeves: AlphaSleevesConfig::default(),
            llm_risk_overlay: LlmRiskOverlayConfig::default(),
        }
    }
}

impl QuantitativeConfig {
    pub fn normalise(&mut self) {
        self.native_backtest.normalise();
        self.calibration.normalise();
        self.selection.normalise();
        self.portfolio.normalise();
        self.execution_telemetry.normalise();
        self.alpha_sleeves.normalise();
        self.llm_risk_overlay.normalise();
    }
}
