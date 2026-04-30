use serde::{Deserialize, Serialize};

/// Configuration for the Gail crypto-trading bridge.
///
/// All string fields support `${ENV_VAR}` interpolation (applied by GailConfig::load).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TradingConfig {
    /// Whether the trading bridge is active.
    pub enabled: bool,

    /// Base URL of the OctoBot web service (e.g. `http://octobot.octobot.svc.cluster.local:5001`).
    pub octobot_base_url: String,

    /// OctoBot login password used for session-based API access.
    pub octobot_password: Option<String>,

    /// Base URL for the Refiner RAG service.
    pub refiner_base_url: String,

    /// Bearer token for Refiner API requests.
    pub refiner_api_token: Option<String>,

    /// client_ids that are treated as admins for trading write operations.
    pub admin_client_ids: Vec<String>,

    /// How often Gail runs a full market evaluation (seconds).
    pub evaluation_interval_seconds: u64,

    /// Maximum number of AI providers to consult in parallel per evaluation.
    pub max_parallel_advisors: usize,

    /// Maximum USD value per micro-trade.
    pub micro_trade_max_usd: f64,

    /// Minimum USD value per micro-trade.
    pub micro_trade_min_usd: f64,

    /// Maximum number of simultaneously open positions.
    pub max_open_positions: usize,

    /// Minimum seconds to wait between consecutive trades on the same symbol.
    pub min_trade_interval_seconds: u64,

    /// Restrict trading to these exchanges (empty = all available).
    pub target_exchanges: Vec<String>,

    /// Restrict trading to these currency symbols (empty = all available).
    pub target_currencies: Vec<String>,

    /// Combined fuzzy+AI confidence required before placing a trade (0.0–1.0).
    pub fuzzy_confidence_threshold: f64,

    /// Template for Refiner research queries.
    /// Supports `{currency}`, `{exchange}`, `{date}` placeholders.
    pub research_query_template: String,

    /// How many top RAG results to request from Refiner.
    pub research_top_k: usize,

    /// Maximum entries kept in the in-memory activity log ring buffer.
    pub log_ring_size: usize,

    /// Maximum entries kept in the recent trades ring buffer.
    pub trade_ring_size: usize,

    /// Path to persist trading state snapshot (JSON).
    pub data_path: String,

    /// Weight given to the fuzzy engine output when blending with AI consensus (0.0–1.0).
    /// AI consensus weight = 1.0 - fuzzy_weight.
    pub fuzzy_weight: f64,

    /// Timeout for OctoBot API calls (seconds).
    pub octobot_timeout_seconds: f64,

    /// Timeout for Refiner research calls (seconds).
    pub refiner_timeout_seconds: f64,

    /// Timeout for each AI advisor call (seconds).
    pub advisor_timeout_seconds: f64,

    // -----------------------------------------------------------------------
    // Backtesting
    // -----------------------------------------------------------------------

    /// Whether to run periodic backtests as a safety check on the approach.
    pub backtesting_enabled: bool,

    /// How often Gail triggers an automatic backtest (seconds).  Default: 86400 (daily).
    pub backtest_interval_seconds: u64,

    /// Minimum profitability % required for the approach to be assessed as "viable".
    /// 0.0 means any positive return qualifies.  Set higher (e.g. 2.0) for stricter gating.
    pub backtest_profitability_threshold: f64,

    /// Explicit list of OctoBot `.data` file paths to use for backtesting.
    /// When empty, the bridge asks OctoBot for its available data files and uses
    /// all that match `backtest_symbols`.
    pub backtest_data_files: Vec<String>,

    /// Symbols to include in automatic data-file selection (empty = all available).
    pub backtest_symbols: Vec<String>,

    /// How many days of historical data to include in each backtest window.
    pub backtest_lookback_days: u32,

    /// If `true` and the most recent backtest assessment is `Unprofitable`,
    /// the live trading loop will be paused automatically until the approach is reviewed.
    pub backtest_pause_on_failure: bool,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            octobot_base_url: String::new(),
            octobot_password: None,
            refiner_base_url: String::new(),
            refiner_api_token: None,
            admin_client_ids: vec!["pbisaacs".to_string()],
            evaluation_interval_seconds: 60,
            max_parallel_advisors: 5,
            micro_trade_max_usd: 25.0,
            micro_trade_min_usd: 1.0,
            max_open_positions: 5,
            min_trade_interval_seconds: 120,
            target_exchanges: Vec::new(),
            target_currencies: Vec::new(),
            fuzzy_confidence_threshold: 0.65,
            research_query_template:
                "cryptocurrency market sentiment {currency} {exchange} {date}".to_string(),
            research_top_k: 5,
            log_ring_size: 1000,
            trade_ring_size: 200,
            data_path: "./data/trading_state.json".to_string(),
            fuzzy_weight: 0.4,
            octobot_timeout_seconds: 10.0,
            refiner_timeout_seconds: 15.0,
            advisor_timeout_seconds: 30.0,
            backtesting_enabled: true,
            backtest_interval_seconds: 86_400,
            backtest_profitability_threshold: 0.0,
            backtest_data_files: Vec::new(),
            backtest_symbols: vec!["BTC/USDT".to_string()],
            backtest_lookback_days: 30,
            backtest_pause_on_failure: false,
        }
    }
}

impl TradingConfig {
    /// Returns true if the minimum viable configuration is present to start the bridge.
    pub fn is_viable(&self) -> bool {
        self.enabled
            && !self.octobot_base_url.trim().is_empty()
    }

    /// Clamp and sanitise values after deserialisation.
    pub fn normalize(&mut self) {
        self.micro_trade_max_usd = self.micro_trade_max_usd.max(0.01);
        self.micro_trade_min_usd = self.micro_trade_min_usd.max(0.01)
            .min(self.micro_trade_max_usd);
        self.fuzzy_confidence_threshold = self.fuzzy_confidence_threshold.clamp(0.0, 1.0);
        self.fuzzy_weight = self.fuzzy_weight.clamp(0.0, 1.0);
        self.evaluation_interval_seconds = self.evaluation_interval_seconds.max(10);
        self.max_parallel_advisors = self.max_parallel_advisors.clamp(1, 20);
        self.max_open_positions = self.max_open_positions.clamp(1, 50);
        self.log_ring_size = self.log_ring_size.clamp(10, 10_000);
        self.trade_ring_size = self.trade_ring_size.clamp(10, 5_000);
        self.octobot_timeout_seconds = self.octobot_timeout_seconds.max(1.0);
        self.refiner_timeout_seconds = self.refiner_timeout_seconds.max(1.0);
        self.advisor_timeout_seconds = self.advisor_timeout_seconds.max(5.0);
        if self.research_query_template.trim().is_empty() {
            self.research_query_template =
                "cryptocurrency market sentiment {currency} {exchange} {date}".to_string();
        }
        if self.data_path.trim().is_empty() {
            self.data_path = "./data/trading_state.json".to_string();
        }
        self.backtest_profitability_threshold = self.backtest_profitability_threshold.clamp(-100.0, 100.0);
        self.backtest_interval_seconds = self.backtest_interval_seconds.max(300); // at least 5 min
        self.backtest_lookback_days = self.backtest_lookback_days.clamp(1, 365);
    }
}

/// Runtime-mutable overrides for TradingConfig, settable via the API without restarting Gail.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TradingConfigOverride {
    pub evaluation_interval_seconds: Option<u64>,
    pub micro_trade_max_usd: Option<f64>,
    pub micro_trade_min_usd: Option<f64>,
    pub max_open_positions: Option<usize>,
    pub fuzzy_confidence_threshold: Option<f64>,
    pub target_exchanges: Option<Vec<String>>,
    pub target_currencies: Option<Vec<String>>,
}
