use std::collections::HashSet;

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

    /// Optional OctoBot native web-auth password.
    /// Leave unset for shared-auth deployments where `/api/ping` is reachable.
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

    /// Enable periodic discovery of non-portfolio symbols from the wider market universe.
    pub token_discovery_enabled: bool,

    /// How often Gail runs non-portfolio discovery/scoring (seconds).
    pub token_discovery_interval_seconds: u64,

    /// Max symbols requested from OctoBot per discovery/pruning scan.
    pub token_discovery_snapshot_limit: usize,

    /// Number of top market-ranked non-portfolio symbols to score with AI+Fuzzy.
    pub token_discovery_candidate_pool_size: usize,

    /// Minimum composite score required before auto-buying a discovered symbol.
    pub token_discovery_min_composite_score: f64,

    /// Enable periodic pruning review of currently held non-stable assets.
    pub portfolio_pruning_enabled: bool,

    /// How often Gail reviews held symbols for potential selloff (seconds).
    pub portfolio_pruning_interval_seconds: u64,

    /// Minimum holding USD value required for a symbol to be considered in pruning.
    pub portfolio_pruning_min_holding_usd: f64,

    /// Number of held symbols to score with AI+Fuzzy in each pruning cycle.
    pub portfolio_pruning_candidate_pool_size: usize,

    /// Minimum bearish composite score required before auto-selling a held symbol.
    pub portfolio_pruning_min_composite_score: f64,

    /// Combined fuzzy+AI confidence required before placing a trade (0.0–1.0).
    pub fuzzy_confidence_threshold: f64,

    /// Template for Refiner research queries.
    /// Supports `{currency}`, `{exchange}`, `{date}` placeholders.
    pub research_query_template: String,

    /// Refiner RAG index name used for trading research lookups.
    pub research_index_name: String,

    /// Source-domain hints used to bias research queries (e.g. `bloomberg.com`).
    /// Gail fans out `site:<domain>` query variants in parallel.
    pub research_site_hints: Vec<String>,

    /// Maximum number of Refiner research queries to run in parallel per cycle.
    /// Includes the base query and any `site:<domain>` variants.
    pub research_max_parallel_queries: usize,

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

    // -----------------------------------------------------------------------
    // Historical ROI feedback
    // -----------------------------------------------------------------------
    /// Whether Gail should adapt live decisions based on historical buy/sell ROI outcomes.
    pub decision_roi_feedback_enabled: bool,

    /// Number of most recent executed trades to inspect for directional ROI feedback.
    pub decision_roi_feedback_lookback_trades: usize,

    /// Minimum directional samples required before ROI feedback is applied.
    pub decision_roi_feedback_min_samples: usize,

    /// Per-decision ROI target used for normalization, expressed as a percentage.
    pub decision_roi_feedback_target_roi_pct: f64,

    /// Maximum absolute signal adjustment applied by ROI feedback.
    pub decision_roi_feedback_max_signal_adjustment: f64,

    /// Maximum confidence penalty applied when historical ROI performance is poor.
    pub decision_roi_feedback_max_confidence_penalty: f64,

    /// Maximum confidence boost applied when historical ROI performance is strong.
    pub decision_roi_feedback_max_confidence_boost: f64,

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

    /// Optional path to persist discovered OctoBot backtest `.data` files (JSON).
    /// Empty means "derive from `data_path` parent directory".
    pub backtest_data_catalog_path: String,

    /// Whether Gail may trigger OctoBot's historical data collector when
    /// no matching backtest `.data` files are available.
    pub backtest_data_collection_enabled: bool,

    /// Exchange used for OctoBot historical data-collector requests.
    pub backtest_data_collection_exchange: String,

    /// Time frames requested when Gail triggers OctoBot historical collection.
    pub backtest_data_collection_time_frames: Vec<String>,

    /// Cooldown between automatic OctoBot collector requests (seconds).
    pub backtest_data_collection_cooldown_seconds: u64,

    /// If `true` and the most recent backtest assessment is `Unprofitable`,
    /// the live trading loop will be paused automatically until the approach is reviewed.
    pub backtest_pause_on_failure: bool,

    /// Whether Gail is allowed to call OctoBot order-placement paths.
    ///
    /// Gail enables live execution by default for the trading bridge rollout.
    /// The OctoBot client still reports a clear execution error if the deployed
    /// OctoBot surface does not expose a supported live order-placement path.
    pub live_execution_enabled: bool,
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
            token_discovery_enabled: true,
            token_discovery_interval_seconds: 1_800,
            token_discovery_snapshot_limit: 250,
            token_discovery_candidate_pool_size: 12,
            token_discovery_min_composite_score: 0.55,
            portfolio_pruning_enabled: true,
            portfolio_pruning_interval_seconds: 1_800,
            portfolio_pruning_min_holding_usd: 20.0,
            portfolio_pruning_candidate_pool_size: 12,
            portfolio_pruning_min_composite_score: 0.55,
            fuzzy_confidence_threshold: 0.65,
            research_query_template: "cryptocurrency market sentiment {currency} {exchange} {date}"
                .to_string(),
            research_index_name: "crypto".to_string(),
            research_site_hints: vec!["bloomberg.com".to_string()],
            research_max_parallel_queries: 3,
            research_top_k: 5,
            log_ring_size: 1000,
            trade_ring_size: 200,
            data_path: "./data/trading_state.json".to_string(),
            fuzzy_weight: 0.4,
            decision_roi_feedback_enabled: true,
            decision_roi_feedback_lookback_trades: 120,
            decision_roi_feedback_min_samples: 8,
            decision_roi_feedback_target_roi_pct: 1.0,
            decision_roi_feedback_max_signal_adjustment: 0.2,
            decision_roi_feedback_max_confidence_penalty: 0.35,
            decision_roi_feedback_max_confidence_boost: 0.1,
            octobot_timeout_seconds: 10.0,
            refiner_timeout_seconds: 15.0,
            advisor_timeout_seconds: 30.0,
            backtesting_enabled: true,
            backtest_interval_seconds: 86_400,
            backtest_profitability_threshold: 0.0,
            backtest_data_files: Vec::new(),
            backtest_symbols: vec!["BTC/USDT".to_string()],
            backtest_lookback_days: 30,
            backtest_data_catalog_path: String::new(),
            backtest_data_collection_enabled: true,
            backtest_data_collection_exchange: "binance".to_string(),
            backtest_data_collection_time_frames: vec!["1h".to_string(), "1d".to_string()],
            backtest_data_collection_cooldown_seconds: 3_600,
            backtest_pause_on_failure: false,
            live_execution_enabled: true,
        }
    }
}

impl TradingConfig {
    /// Returns true if the minimum viable configuration is present to start the bridge.
    pub fn is_viable(&self) -> bool {
        self.enabled && !self.octobot_base_url.trim().is_empty()
    }

    /// Clamp and sanitise values after deserialisation.
    pub fn normalize(&mut self) {
        self.micro_trade_max_usd = self.micro_trade_max_usd.max(0.01);
        self.micro_trade_min_usd = self
            .micro_trade_min_usd
            .max(0.01)
            .min(self.micro_trade_max_usd);
        self.fuzzy_confidence_threshold = self.fuzzy_confidence_threshold.clamp(0.0, 1.0);
        self.fuzzy_weight = self.fuzzy_weight.clamp(0.0, 1.0);
        self.decision_roi_feedback_lookback_trades =
            self.decision_roi_feedback_lookback_trades.clamp(10, 5_000);
        self.decision_roi_feedback_min_samples = self
            .decision_roi_feedback_min_samples
            .clamp(2, self.decision_roi_feedback_lookback_trades.max(2));
        self.decision_roi_feedback_target_roi_pct =
            self.decision_roi_feedback_target_roi_pct.clamp(0.1, 50.0);
        self.decision_roi_feedback_max_signal_adjustment = self
            .decision_roi_feedback_max_signal_adjustment
            .clamp(0.0, 0.5);
        self.decision_roi_feedback_max_confidence_penalty = self
            .decision_roi_feedback_max_confidence_penalty
            .clamp(0.0, 0.95);
        self.decision_roi_feedback_max_confidence_boost = self
            .decision_roi_feedback_max_confidence_boost
            .clamp(0.0, 0.5);
        self.evaluation_interval_seconds = self.evaluation_interval_seconds.max(10);
        self.token_discovery_interval_seconds = self.token_discovery_interval_seconds.max(60);
        self.token_discovery_snapshot_limit = self.token_discovery_snapshot_limit.clamp(10, 2_000);
        self.token_discovery_candidate_pool_size =
            self.token_discovery_candidate_pool_size.clamp(1, 64);
        self.token_discovery_min_composite_score =
            self.token_discovery_min_composite_score.clamp(0.0, 2.0);
        self.portfolio_pruning_interval_seconds = self.portfolio_pruning_interval_seconds.max(60);
        self.portfolio_pruning_min_holding_usd = self.portfolio_pruning_min_holding_usd.max(0.0);
        self.portfolio_pruning_candidate_pool_size =
            self.portfolio_pruning_candidate_pool_size.clamp(1, 64);
        self.portfolio_pruning_min_composite_score =
            self.portfolio_pruning_min_composite_score.clamp(0.0, 2.0);
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
        if self.research_index_name.trim().is_empty() {
            self.research_index_name = "crypto".to_string();
        }
        self.research_max_parallel_queries = self.research_max_parallel_queries.clamp(1, 8);
        self.research_top_k = self.research_top_k.clamp(1, 25);
        let mut seen_site_hints: HashSet<String> = HashSet::new();
        self.research_site_hints = self
            .research_site_hints
            .iter()
            .map(|hint| hint.trim().to_string())
            .filter(|hint| !hint.is_empty())
            .filter(|hint| seen_site_hints.insert(hint.to_ascii_lowercase()))
            .collect();
        if self.data_path.trim().is_empty() {
            self.data_path = "./data/trading_state.json".to_string();
        }
        self.backtest_profitability_threshold =
            self.backtest_profitability_threshold.clamp(-100.0, 100.0);
        self.backtest_interval_seconds = self.backtest_interval_seconds.max(300); // at least 5 min
        self.backtest_lookback_days = self.backtest_lookback_days.clamp(1, 365);
        self.backtest_data_catalog_path = self.backtest_data_catalog_path.trim().to_string();
        self.backtest_data_collection_exchange =
            self.backtest_data_collection_exchange.trim().to_string();
        if self.backtest_data_collection_exchange.is_empty() {
            self.backtest_data_collection_exchange = "binance".to_string();
        }
        let mut seen_backtest_time_frames: HashSet<String> = HashSet::new();
        self.backtest_data_collection_time_frames = self
            .backtest_data_collection_time_frames
            .iter()
            .map(|time_frame| time_frame.trim().to_string())
            .filter(|time_frame| !time_frame.is_empty())
            .filter(|time_frame| seen_backtest_time_frames.insert(time_frame.to_ascii_lowercase()))
            .collect();
        if self.backtest_data_collection_time_frames.is_empty() {
            self.backtest_data_collection_time_frames = vec!["1h".to_string(), "1d".to_string()];
        }
        self.backtest_data_collection_cooldown_seconds =
            self.backtest_data_collection_cooldown_seconds.max(60);
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
