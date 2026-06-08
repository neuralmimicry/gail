use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tokio::{fs, sync::Mutex};
use tracing::{debug, warn};

use super::backtest::BacktestSummary;
use super::config::TradingConfigOverride;
use super::octobot::{OctobotExchange, OctobotOrder, OctobotPortfolio};
use crate::adaptive_schema::AdaptiveApiSchema;

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Log entry
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TradingLogEntry {
    pub ts: f64,
    pub level: String,
    pub category: String,
    pub message: String,
    #[serde(default)]
    pub context: serde_json::Value,
}

impl Default for TradingLogEntry {
    fn default() -> Self {
        Self {
            ts: 0.0,
            level: String::new(),
            category: String::new(),
            message: String::new(),
            context: serde_json::Value::Null,
        }
    }
}

impl TradingLogEntry {
    pub fn new(
        level: impl Into<String>,
        category: impl Into<String>,
        message: impl Into<String>,
        context: serde_json::Value,
    ) -> Self {
        Self {
            ts: now_ts(),
            level: level.into(),
            category: category.into(),
            message: message.into(),
            context,
        }
    }
}

// ---------------------------------------------------------------------------
// Trade action
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TradeAction {
    Buy,
    Sell,
    Hold,
    StrongBuy,
    StrongSell,
    Cancel,
}

impl std::fmt::Display for TradeAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buy => write!(f, "buy"),
            Self::Sell => write!(f, "sell"),
            Self::Hold => write!(f, "hold"),
            Self::StrongBuy => write!(f, "strong_buy"),
            Self::StrongSell => write!(f, "strong_sell"),
            Self::Cancel => write!(f, "cancel"),
        }
    }
}

// ---------------------------------------------------------------------------
// Executed trade record
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutedTrade {
    pub ts: f64,
    pub exchange: String,
    pub symbol: String,
    pub action: TradeAction,
    pub amount_usd: f64,
    pub price: Option<f64>,
    pub order_id: Option<String>,
    pub confidence: f64,
    pub rationale: String,
    pub ai_votes: serde_json::Value,
    pub fuzzy_confidence: f64,
    pub ai_confidence: f64,
}

// ---------------------------------------------------------------------------
// Override request
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeOverride {
    pub action: TradeAction,
    pub exchange: Option<String>,
    pub symbol: Option<String>,
    pub amount_usd: Option<f64>,
    pub reason: Option<String>,
    pub issued_at: f64,
    pub issued_by: String,
}

// ---------------------------------------------------------------------------
// Backtest auto-tuning trial state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BacktestAutoTuneTrial {
    pub started_at: f64,
    pub history_len_at_start: usize,
    pub baseline_mean_profit_pct: f64,
    pub baseline_median_profit_pct: f64,
    pub baseline_samples: usize,
    pub previous_overrides: Option<TradingConfigOverride>,
    pub candidate_overrides: TradingConfigOverride,
    pub trigger_assessment: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct BacktestAutoTuneState {
    pub active_trial: Option<BacktestAutoTuneTrial>,
    pub cooldown_until: Option<f64>,
    pub last_action: Option<String>,
    pub last_action_at: Option<f64>,
}

// ---------------------------------------------------------------------------
// Runtime status snapshot (lightweight, for status endpoint)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradingStatusSnapshot {
    pub enabled: bool,
    pub paused: bool,
    pub last_evaluation_at: Option<f64>,
    pub last_trade_at: Option<f64>,
    pub evaluation_count: u64,
    pub trade_count: u64,
    pub open_positions: usize,
    pub last_error: Option<String>,
    pub has_pending_override: bool,
    pub config_overrides_active: bool,
    pub last_backtest_assessment: Option<String>,
    pub last_backtest_at: Option<f64>,
    pub api_schema_version: u64,
    pub api_schema_hints: usize,
    pub recent_api_adjustments: usize,
}

// ---------------------------------------------------------------------------
// TradingState — shared across bridge loop and HTTP handlers
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct TradingState {
    pub paused: bool,
    pub last_evaluation_at: Option<f64>,
    pub last_trade_at: Option<f64>,
    pub evaluation_count: u64,
    pub trade_count: u64,
    pub pending_override: Option<TradeOverride>,
    pub current_portfolio: Option<OctobotPortfolio>,
    pub open_positions: Vec<OctobotOrder>,
    pub available_exchanges: Vec<OctobotExchange>,
    pub recent_trades: VecDeque<ExecutedTrade>,
    pub activity_log: VecDeque<TradingLogEntry>,
    pub last_error: Option<String>,
    pub config_overrides: Option<TradingConfigOverride>,
    pub log_ring_size: usize,
    pub trade_ring_size: usize,
    /// Most recent backtesting run result.
    pub last_backtest: Option<BacktestSummary>,
    /// Ring buffer of historical backtest summaries (most recent last).
    pub backtest_history: VecDeque<BacktestSummary>,
    /// Runtime state for guarded backtest-driven strategy tuning.
    #[serde(default)]
    pub backtest_auto_tune: BacktestAutoTuneState,
    /// Adaptive reference of OctoBot API endpoint shapes and semantic hints.
    #[serde(default)]
    pub api_schema: AdaptiveApiSchema,
    /// Fingerprints for external OctoBot log rows already copied into Gail's log.
    #[serde(default)]
    pub observed_external_log_fingerprints: VecDeque<String>,
}

impl TradingState {
    pub fn new(log_ring_size: usize, trade_ring_size: usize) -> Self {
        Self {
            paused: false,
            last_evaluation_at: None,
            last_trade_at: None,
            evaluation_count: 0,
            trade_count: 0,
            pending_override: None,
            current_portfolio: None,
            open_positions: Vec::new(),
            available_exchanges: Vec::new(),
            recent_trades: VecDeque::with_capacity(trade_ring_size),
            activity_log: VecDeque::with_capacity(log_ring_size),
            last_error: None,
            config_overrides: None,
            log_ring_size,
            trade_ring_size,
            last_backtest: None,
            backtest_history: VecDeque::with_capacity(20),
            backtest_auto_tune: BacktestAutoTuneState::default(),
            api_schema: AdaptiveApiSchema::default(),
            observed_external_log_fingerprints: VecDeque::with_capacity(500),
        }
    }

    pub fn log(
        &mut self,
        level: impl Into<String>,
        category: impl Into<String>,
        message: impl Into<String>,
        context: serde_json::Value,
    ) {
        if self.activity_log.len() >= self.log_ring_size {
            self.activity_log.pop_front();
        }
        self.activity_log
            .push_back(TradingLogEntry::new(level, category, message, context));
    }

    pub fn log_info(&mut self, category: impl Into<String>, message: impl Into<String>) {
        self.log("info", category, message, serde_json::Value::Null);
    }

    pub fn log_warn(&mut self, category: impl Into<String>, message: impl Into<String>) {
        self.log("warn", category, message, serde_json::Value::Null);
    }

    pub fn log_error(&mut self, category: impl Into<String>, message: impl Into<String>) {
        let msg: String = message.into();
        self.last_error = Some(msg.clone());
        self.log("error", category, msg, serde_json::Value::Null);
    }

    pub fn record_trade(&mut self, trade: ExecutedTrade) {
        if self.recent_trades.len() >= self.trade_ring_size {
            self.recent_trades.pop_front();
        }
        self.last_trade_at = Some(trade.ts);
        self.trade_count += 1;
        self.recent_trades.push_back(trade);
    }

    pub fn status_snapshot(&self, enabled: bool) -> TradingStatusSnapshot {
        TradingStatusSnapshot {
            enabled,
            paused: self.paused,
            last_evaluation_at: self.last_evaluation_at,
            last_trade_at: self.last_trade_at,
            evaluation_count: self.evaluation_count,
            trade_count: self.trade_count,
            open_positions: self.open_positions.len(),
            last_error: self.last_error.clone(),
            has_pending_override: self.pending_override.is_some(),
            config_overrides_active: self.config_overrides.is_some(),
            last_backtest_assessment: self
                .last_backtest
                .as_ref()
                .map(|b| b.assessment.to_string()),
            last_backtest_at: self.last_backtest.as_ref().map(|b| b.run_at),
            api_schema_version: self.api_schema.version,
            api_schema_hints: self.api_schema.semantic_hints.len(),
            recent_api_adjustments: self.api_schema.recent_adjustments.len(),
        }
    }

    /// Store a backtest result.  Maintains a ring buffer of up to 20 historical runs.
    pub fn record_backtest(&mut self, summary: BacktestSummary) {
        if self.backtest_history.len() >= 20 {
            self.backtest_history.pop_front();
        }
        self.backtest_history.push_back(summary.clone());
        self.last_backtest = Some(summary);
    }

    /// Take any pending override and clear it from the state.
    pub fn take_override(&mut self) -> Option<TradeOverride> {
        self.pending_override.take()
    }

    pub fn remember_external_log(&mut self, fingerprint: String) -> bool {
        if self
            .observed_external_log_fingerprints
            .iter()
            .any(|existing| existing == &fingerprint)
        {
            return false;
        }
        if self.observed_external_log_fingerprints.len() >= 500 {
            self.observed_external_log_fingerprints.pop_front();
        }
        self.observed_external_log_fingerprints
            .push_back(fingerprint);
        true
    }
}

// ---------------------------------------------------------------------------
// Shared handle
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SharedTradingState(pub Arc<Mutex<TradingState>>);

impl SharedTradingState {
    pub fn new(log_ring_size: usize, trade_ring_size: usize) -> Self {
        Self(Arc::new(Mutex::new(TradingState::new(
            log_ring_size,
            trade_ring_size,
        ))))
    }

    pub async fn log(
        &self,
        level: impl Into<String>,
        category: impl Into<String>,
        message: impl Into<String>,
        context: serde_json::Value,
    ) {
        let mut state = self.0.lock().await;
        state.log(level, category, message, context);
    }

    pub async fn log_info(&self, category: impl Into<String>, message: impl Into<String>) {
        let mut state = self.0.lock().await;
        state.log_info(category, message);
    }

    pub async fn log_warn(&self, category: impl Into<String>, message: impl Into<String>) {
        let mut state = self.0.lock().await;
        state.log_warn(category, message);
    }

    pub async fn log_error(&self, category: impl Into<String>, message: impl Into<String>) {
        let mut state = self.0.lock().await;
        state.log_error(category, message);
    }

    /// Persist state snapshot to disk asynchronously (best-effort).
    pub async fn persist(&self, path: &PathBuf) {
        let snapshot = {
            let state = self.0.lock().await;
            match serde_json::to_string_pretty(&*state) {
                Ok(json) => json,
                Err(err) => {
                    warn!(
                        "trading: failed to serialise state for persistence: {}",
                        err
                    );
                    return;
                }
            }
        };
        if let Some(parent) = path.parent()
            && let Err(err) = fs::create_dir_all(parent).await
        {
            warn!(
                "trading: failed to create state dir {}: {}",
                parent.display(),
                err
            );
            return;
        }
        if let Err(err) = fs::write(path, snapshot).await {
            warn!(
                "trading: failed to write state to {}: {}",
                path.display(),
                err
            );
        } else {
            debug!("trading: state persisted to {}", path.display());
        }
    }

    /// Restore state from disk if the file exists (best-effort, partial restore).
    pub async fn restore(&self, path: &PathBuf) {
        match fs::read_to_string(path).await {
            Ok(raw) => match parse_persisted_state(&raw) {
                Ok((restored, repaired_snapshot)) => {
                    let mut state = self.0.lock().await;
                    // Restore counters and history but not ephemeral fields
                    state.evaluation_count = restored.evaluation_count;
                    state.trade_count = restored.trade_count;
                    state.last_evaluation_at = restored.last_evaluation_at;
                    state.last_trade_at = restored.last_trade_at;
                    state.recent_trades = restored.recent_trades;
                    state.activity_log = restored.activity_log;
                    state.config_overrides = restored.config_overrides;
                    state.last_backtest = restored.last_backtest;
                    state.backtest_history = restored.backtest_history;
                    state.backtest_auto_tune = restored.backtest_auto_tune;
                    state.api_schema = restored.api_schema;
                    state.observed_external_log_fingerprints =
                        restored.observed_external_log_fingerprints;
                    state.log_info("startup", "Restored trading state from disk");
                    drop(state);
                    if let Some(snapshot) = repaired_snapshot {
                        warn!(
                            "trading: repaired legacy persisted state format in {}",
                            path.display()
                        );
                        if let Err(err) = fs::write(path, snapshot).await {
                            warn!(
                                "trading: failed to write repaired state to {}: {}",
                                path.display(),
                                err
                            );
                        }
                    }
                }
                Err(err) => {
                    warn!(
                        "trading: failed to parse persisted state from {}: {}",
                        path.display(),
                        err
                    );
                }
            },
            Err(_) => {
                debug!("trading: no persisted state found at {}", path.display());
            }
        }
    }
}

fn parse_persisted_state(raw: &str) -> Result<(TradingState, Option<String>), serde_json::Error> {
    let mut payload = serde_json::from_str::<serde_json::Value>(raw)?;
    let repaired = repair_legacy_state_payload(&mut payload);
    let state = serde_json::from_value::<TradingState>(payload.clone())?;
    let repaired_snapshot = if repaired {
        Some(serde_json::to_string_pretty(&payload)?)
    } else {
        None
    };
    Ok((state, repaired_snapshot))
}

fn repair_legacy_state_payload(payload: &mut serde_json::Value) -> bool {
    let Some(root) = payload.as_object_mut() else {
        return false;
    };
    let mut repaired = false;

    if let Some(activity_log) = root
        .get_mut("activity_log")
        .and_then(serde_json::Value::as_array_mut)
    {
        for entry in activity_log {
            let Some(entry) = entry.as_object_mut() else {
                continue;
            };
            if !entry.contains_key("context") {
                entry.insert("context".to_string(), serde_json::Value::Null);
                repaired = true;
            }
        }
    }

    if !root.contains_key("api_schema") {
        root.insert("api_schema".to_string(), serde_json::json!({}));
        repaired = true;
    }
    if !root.contains_key("observed_external_log_fingerprints") {
        root.insert(
            "observed_external_log_fingerprints".to_string(),
            serde_json::json!([]),
        );
        repaired = true;
    }
    if !root.contains_key("backtest_auto_tune") {
        root.insert("backtest_auto_tune".to_string(), serde_json::json!({}));
        repaired = true;
    }

    repaired
}
