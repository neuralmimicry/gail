/// Gail trading bridge — backtesting integration.
///
/// Runs periodic backtests against OctoBot's backtesting API to assess whether
/// the current trading approach is producing positive returns.  The assessment
/// is used as a safety gate: if the approach is deemed "unprofitable", the
/// bridge will log a warning and can optionally pause live trading.
///
/// OctoBot backtesting endpoints used:
///   POST /backtesting?action_type=start_backtesting  — start a run
///   GET  /backtesting?update_type=backtesting_report — poll for results
///   GET  /backtesting?update_type=backtesting_data_files — list data files
///   GET  /backtesting_run_id                         — latest run ID
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::config::TradingConfig;
use super::octobot::{BacktestRunReport, BacktestStartRequest, OctobotClient};

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Assessment
// ---------------------------------------------------------------------------

/// Qualitative assessment of the backtested trading approach.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApproachAssessment {
    /// Profitability exceeds the configured threshold — approach is viable.
    Viable,
    /// Slightly below threshold (within 5 percentage points) — approach is marginal.
    Marginal,
    /// Clearly unprofitable — approach should be reviewed.
    Unprofitable,
    /// Backtest could not run or results could not be parsed.
    Incomplete,
}

impl std::fmt::Display for ApproachAssessment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Viable => write!(f, "viable"),
            Self::Marginal => write!(f, "marginal"),
            Self::Unprofitable => write!(f, "unprofitable"),
            Self::Incomplete => write!(f, "incomplete"),
        }
    }
}

// ---------------------------------------------------------------------------
// BacktestSummary — stored in TradingState
// ---------------------------------------------------------------------------

/// Result of a completed or attempted backtesting run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BacktestSummary {
    /// When this backtest was executed (Unix timestamp).
    pub run_at: f64,
    /// Qualitative verdict.
    pub assessment: ApproachAssessment,
    /// Overall profitability % (positive = profit).  `None` if not available.
    pub profitability_pct: Option<f64>,
    /// Market buy-and-hold profitability % for the same period.
    pub market_avg_pct: Option<f64>,
    /// Whether the approach beat the market (alpha > 0).
    pub beats_market: Option<bool>,
    /// Number of trades executed.
    pub total_trades: usize,
    /// Number of errors encountered during the backtest.
    pub errors_count: usize,
    /// Symbols covered.
    pub symbols: Vec<String>,
    /// Human-readable notes (e.g. reason for failure).
    pub notes: String,
    /// OctoBot run ID, if available.
    pub run_id: Option<u64>,
}

impl BacktestSummary {
    /// Construct a summary representing a failed/incomplete run.
    pub fn incomplete(reason: impl Into<String>) -> Self {
        Self {
            run_at: now_ts(),
            assessment: ApproachAssessment::Incomplete,
            profitability_pct: None,
            market_avg_pct: None,
            beats_market: None,
            total_trades: 0,
            errors_count: 0,
            symbols: Vec::new(),
            notes: reason.into(),
            run_id: None,
        }
    }

    /// Build a summary from a fully parsed OctoBot report.
    pub fn from_report(report: &BacktestRunReport, threshold: f64, run_id: Option<u64>) -> Self {
        let profitability_pct = report.best_profitability();
        let market_avg_pct = report.best_market_avg();

        let beats_market = match (profitability_pct, market_avg_pct) {
            (Some(p), Some(m)) => Some(p > m),
            _ => None,
        };

        let assessment = assess(profitability_pct, threshold);

        let notes = match &assessment {
            ApproachAssessment::Viable => format!(
                "Profitable: {:.2}% return vs {:.2}% market avg",
                profitability_pct.unwrap_or(0.0),
                market_avg_pct.unwrap_or(0.0)
            ),
            ApproachAssessment::Marginal => format!(
                "Marginal: {:.2}% return (threshold {threshold:.2}%)",
                profitability_pct.unwrap_or(0.0)
            ),
            ApproachAssessment::Unprofitable => format!(
                "Unprofitable: {:.2}% return",
                profitability_pct.unwrap_or(0.0)
            ),
            ApproachAssessment::Incomplete => "No profitability data".to_string(),
        };

        Self {
            run_at: now_ts(),
            assessment,
            profitability_pct,
            market_avg_pct,
            beats_market,
            total_trades: report.total_trades,
            errors_count: report.errors_count,
            symbols: report.symbols.clone(),
            notes,
            run_id,
        }
    }
}

fn assess(profitability_pct: Option<f64>, threshold: f64) -> ApproachAssessment {
    match profitability_pct {
        None => ApproachAssessment::Incomplete,
        Some(p) if p >= threshold => ApproachAssessment::Viable,
        Some(p) if p >= threshold - 5.0 => ApproachAssessment::Marginal,
        _ => ApproachAssessment::Unprofitable,
    }
}

// ---------------------------------------------------------------------------
// BacktestEngine
// ---------------------------------------------------------------------------

/// Orchestrates a full backtesting run: start → poll → parse → assess.
pub struct BacktestEngine {
    octobot: OctobotClient,
    /// Minimum profitability % to consider the approach "viable".
    profitability_threshold: f64,
    /// How long to wait between polling attempts.
    poll_interval: Duration,
    /// Maximum polling attempts before timing out.
    max_polls: usize,
}

impl BacktestEngine {
    /// Create a new engine.  Uses sensible poll defaults (5 s interval, 60 polls = 5 min timeout).
    pub fn new(octobot: OctobotClient, profitability_threshold: f64) -> Self {
        Self {
            octobot,
            profitability_threshold,
            poll_interval: Duration::from_secs(5),
            max_polls: 60,
        }
    }

    /// Create with custom poll parameters (useful in tests to avoid long waits).
    pub fn with_poll_params(
        octobot: OctobotClient,
        profitability_threshold: f64,
        poll_interval: Duration,
        max_polls: usize,
    ) -> Self {
        Self {
            octobot,
            profitability_threshold,
            poll_interval,
            max_polls,
        }
    }

    /// Run a complete backtesting cycle using explicit parameters.
    pub async fn run(&self, request: &BacktestStartRequest) -> BacktestSummary {
        debug!(
            "trading: starting OctoBot backtest (files={:?})",
            request.files
        );

        // 1. Start the backtest.
        if let Err(err) = self.octobot.start_backtest(request).await {
            warn!("trading: could not start backtest: {}", err);
            return BacktestSummary::incomplete(format!("start failed: {err}"));
        }
        info!("trading: OctoBot backtest started");

        // 2. Get the run ID (best-effort).
        let run_id = self.octobot.get_backtest_run_id().await.ok().flatten();

        // 3. Poll for results.
        for attempt in 0..self.max_polls {
            tokio::time::sleep(self.poll_interval).await;
            match self.octobot.get_backtest_report().await {
                Ok(Some(report)) => {
                    let summary =
                        BacktestSummary::from_report(&report, self.profitability_threshold, run_id);
                    info!(
                        "trading: backtest complete — assessment={} profit={:?}%",
                        summary.assessment, summary.profitability_pct
                    );
                    return summary;
                }
                Ok(None) => {
                    debug!("trading: backtest still running (poll {})", attempt + 1);
                }
                Err(err) => {
                    warn!("trading: backtest report poll failed: {}", err);
                    return BacktestSummary::incomplete(format!("poll failed: {err}"));
                }
            }
        }

        warn!("trading: backtest timed out after {} polls", self.max_polls);
        BacktestSummary::incomplete(format!("timed out after {} polls", self.max_polls))
    }

    /// Run a backtest using defaults derived from the trading config.
    ///
    /// - Uses `config.backtest_data_files` if specified, otherwise lets OctoBot
    ///   choose automatically (empty files list triggers OctoBot's default selection).
    /// - Time window: last `config.backtest_lookback_days` days.
    pub async fn run_with_config(&self, config: &TradingConfig) -> BacktestSummary {
        let now_ms = (now_ts() * 1000.0) as i64;
        let lookback_ms = (config.backtest_lookback_days as i64) * 86_400_000;
        let start_ms = now_ms - lookback_ms;

        let request = BacktestStartRequest {
            files: config.backtest_data_files.clone(),
            start_timestamp: Some(start_ms),
            end_timestamp: Some(now_ms),
            enable_logs: false,
        };
        self.run(&request).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_report(profit: f64, market: f64, trades: usize) -> BacktestRunReport {
        let mut profitability = std::collections::HashMap::new();
        profitability.insert("binance".to_string(), profit);
        let mut market_avg = std::collections::HashMap::new();
        market_avg.insert("binance".to_string(), market);
        BacktestRunReport {
            profitability,
            market_average_profitability: market_avg,
            reference_market: "USDT".to_string(),
            trading_mode: "TestMode".to_string(),
            starting_portfolio: std::collections::HashMap::new(),
            end_portfolio: std::collections::HashMap::new(),
            total_trades: trades,
            symbols: vec!["BTC/USDT".to_string()],
            errors_count: 0,
            raw: serde_json::Value::Null,
        }
    }

    #[test]
    fn assess_viable_above_threshold() {
        let report = make_report(5.0, 2.0, 10);
        let summary = BacktestSummary::from_report(&report, 0.0, None);
        assert_eq!(summary.assessment, ApproachAssessment::Viable);
        assert_eq!(summary.profitability_pct, Some(5.0));
        assert_eq!(summary.beats_market, Some(true));
    }

    #[test]
    fn assess_marginal_just_below_threshold() {
        let report = make_report(-3.0, 2.0, 5);
        let summary = BacktestSummary::from_report(&report, 0.0, None);
        assert_eq!(summary.assessment, ApproachAssessment::Marginal);
        assert_eq!(summary.beats_market, Some(false));
    }

    #[test]
    fn assess_unprofitable_far_below_threshold() {
        let report = make_report(-20.0, 1.0, 3);
        let summary = BacktestSummary::from_report(&report, 0.0, None);
        assert_eq!(summary.assessment, ApproachAssessment::Unprofitable);
    }

    #[test]
    fn assess_incomplete_when_no_profitability() {
        let summary = BacktestSummary::incomplete("connection refused");
        assert_eq!(summary.assessment, ApproachAssessment::Incomplete);
        assert!(summary.profitability_pct.is_none());
        assert_eq!(summary.notes, "connection refused");
    }

    #[test]
    fn beats_market_is_correct() {
        let report_beat = make_report(10.0, 5.0, 8);
        let s = BacktestSummary::from_report(&report_beat, 0.0, None);
        assert_eq!(s.beats_market, Some(true));

        let report_lag = make_report(3.0, 7.0, 8);
        let s2 = BacktestSummary::from_report(&report_lag, 0.0, None);
        assert_eq!(s2.beats_market, Some(false));
    }

    #[test]
    fn approach_assessment_display() {
        assert_eq!(ApproachAssessment::Viable.to_string(), "viable");
        assert_eq!(ApproachAssessment::Marginal.to_string(), "marginal");
        assert_eq!(ApproachAssessment::Unprofitable.to_string(), "unprofitable");
        assert_eq!(ApproachAssessment::Incomplete.to_string(), "incomplete");
    }

    #[test]
    fn summary_notes_non_empty() {
        let report = make_report(8.0, 3.0, 12);
        let s = BacktestSummary::from_report(&report, 0.0, Some(42));
        assert!(!s.notes.is_empty(), "notes should describe the result");
        assert_eq!(s.run_id, Some(42));
        assert_eq!(s.total_trades, 12);
    }
}
