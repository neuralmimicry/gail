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
///   POST /data_collector?action_type=start_collector — collect missing data
///   GET  /backtesting_run_id                         — latest run ID
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::{debug, info, warn};

use super::config::TradingConfig;
use super::octobot::{BacktestRunReport, BacktestStartRequest, OctobotClient};

const DEFAULT_BACKTEST_CATALOG_FILENAME: &str = "backtest_data_catalog.json";
const DEFAULT_BACKTEST_COLLECTION_SYMBOL: &str = "BTC/USDT";
const BACKTEST_MAX_FILES_PER_SYMBOL_TIMEFRAME: usize = 3;
const BACKTEST_MAX_UNKNOWN_TIMEFRAME_FILES_PER_SYMBOL: usize = 2;
const BACKTEST_MAX_GENERIC_COLLECTOR_FILES: usize = 8;
const BACKTEST_MAX_SUBSET_RUNS_PER_CYCLE: usize = 6;
const BACKTEST_MIN_SUBSET_RUNS_PER_CYCLE: usize = 2;
const BACKTEST_TARGET_SUCCESSFUL_SUBSET_RUNS_PER_CYCLE: usize = 1;
const BACKTEST_MAX_SINGLE_UNKNOWN_FILE_SUBSETS: usize = 4;

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct BacktestDataCatalog {
    #[serde(default)]
    files: Vec<String>,
    #[serde(default)]
    updated_at: f64,
    #[serde(default)]
    last_collection_requested_at: Option<f64>,
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

#[derive(Clone, Debug)]
enum DataCollectionRequestOutcome {
    Started,
    AlreadyRunning,
    CoolingDown { retry_after_seconds: u64 },
}

#[derive(Clone, Debug)]
struct BacktestFileSubset {
    label: String,
    files: Vec<String>,
}

#[derive(Clone, Debug)]
struct BacktestSubsetOutcome {
    label: String,
    file_count: usize,
    summary: BacktestSummary,
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
        let mut started_new_run = true;
        if let Err(err) = self.octobot.start_backtest(request).await {
            if is_already_running_backtest_error(&err) {
                started_new_run = false;
                info!(
                    "trading: OctoBot reports a backtest is already running; polling existing run"
                );
            } else {
                warn!("trading: could not start backtest: {}", err);
                return BacktestSummary::incomplete(format!("start failed: {err}"));
            }
        }
        if started_new_run {
            info!("trading: OctoBot backtest started");
        }

        // 2. Poll for results.
        // Retrieve run_id only once a report is available to reduce noisy
        // transient run-id endpoint failures while OctoBot is still starting.
        let mut run_id = None;
        for attempt in 0..self.max_polls {
            tokio::time::sleep(self.poll_interval).await;
            match self.octobot.get_backtest_report().await {
                Ok(Some(report)) => {
                    if run_id.is_none() {
                        run_id = self.octobot.get_backtest_run_id().await.ok().flatten();
                    }
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
    /// - Uses `config.backtest_data_files` if specified, otherwise asks OctoBot
    ///   for its available `.data` files and selects files matching
    ///   `config.backtest_symbols`.
    /// - Time window: last `config.backtest_lookback_days` days.
    pub async fn run_with_config(&self, config: &TradingConfig) -> BacktestSummary {
        let now_ms = (now_ts() * 1000.0) as i64;
        let lookback_ms = (config.backtest_lookback_days as i64) * 86_400_000;
        let start_ms = now_ms - lookback_ms;
        let files = match self.resolve_backtest_files(config).await {
            Ok(files) => files,
            Err(reason) => {
                warn!("trading: skipping OctoBot backtest: {}", reason);
                return BacktestSummary::incomplete(reason);
            }
        };

        let subsets = build_backtest_file_subsets(files);
        if subsets.is_empty() {
            return BacktestSummary::incomplete(
                "no backtest file subsets available after selection".to_string(),
            );
        }

        let minimum_runs = subsets.len().min(BACKTEST_MIN_SUBSET_RUNS_PER_CYCLE);
        let mut successful_runs = 0usize;
        let mut outcomes = Vec::new();
        for (index, subset) in subsets.iter().enumerate() {
            info!(
                "trading: running backtest subset {}/{} label={} files={}",
                index + 1,
                subsets.len(),
                subset.label,
                subset.files.len()
            );
            let request = BacktestStartRequest {
                files: subset.files.clone(),
                start_timestamp: Some(start_ms),
                end_timestamp: Some(now_ms),
                enable_logs: false,
            };
            let summary = self.run(&request).await;
            if summary.profitability_pct.is_some() {
                successful_runs += 1;
            }
            info!(
                "trading: backtest subset complete label={} assessment={} profit={:?}% files={}",
                subset.label,
                summary.assessment,
                summary.profitability_pct,
                subset.files.len()
            );
            outcomes.push(BacktestSubsetOutcome {
                label: subset.label.clone(),
                file_count: subset.files.len(),
                summary,
            });
            let attempted_runs = outcomes.len();
            if attempted_runs >= minimum_runs
                && successful_runs >= BACKTEST_TARGET_SUCCESSFUL_SUBSET_RUNS_PER_CYCLE
            {
                break;
            }
        }

        aggregate_backtest_subset_outcomes(outcomes, self.profitability_threshold)
    }

    async fn resolve_backtest_files(&self, config: &TradingConfig) -> Result<Vec<String>, String> {
        if !config.backtest_data_files.is_empty() {
            return Ok(config.backtest_data_files.clone());
        }

        let catalog_path = resolve_backtest_catalog_path(config);
        let mut catalog = load_backtest_data_catalog(&catalog_path).await;
        let now = now_ts();
        let catalog_refresh_seconds = config.backtest_data_catalog_refresh_seconds as f64;
        let catalog_is_stale = catalog.files.is_empty()
            || catalog.updated_at <= 0.0
            || now - catalog.updated_at >= catalog_refresh_seconds;

        if catalog_is_stale {
            match self
                .discover_and_cache_backtest_files(
                    &catalog_path,
                    &mut catalog,
                    &config.backtest_symbols,
                )
                .await
            {
                Ok(selected) if !selected.is_empty() => return Ok(selected),
                Ok(_) => {}
                Err(err) => {
                    warn!(
                        "trading: OctoBot backtest data-file discovery failed: {}",
                        err
                    );
                }
            }
        } else {
            debug!(
                "trading: using fresh backtest data-file catalog path={} age={:.0}s (refresh={}s)",
                catalog_path.display(),
                now - catalog.updated_at,
                config.backtest_data_catalog_refresh_seconds
            );
        }

        let cached_selected =
            select_backtest_data_files(catalog.files.clone(), &config.backtest_symbols);
        if !cached_selected.is_empty() {
            if config.backtest_data_collection_enabled {
                let stale_after_seconds = config.backtest_data_collection_cooldown_seconds as f64;
                let selected_data_age =
                    selected_backtest_data_age_seconds(&cached_selected, catalog.updated_at, now);
                let data_is_stale = selected_data_age
                    .map(|age| age >= stale_after_seconds)
                    .unwrap_or(true);
                if data_is_stale {
                    match self
                        .maybe_request_data_collection(config, &catalog_path, &mut catalog, now)
                        .await
                    {
                        Ok(DataCollectionRequestOutcome::Started) => {
                            let symbols = configured_collection_symbols(config);
                            info!(
                                "trading: backtest data appears stale (age={:?}s), started OctoBot historical data collection for {} on {} [{}] while continuing with {} backtest files",
                                selected_data_age.map(|age| age.round() as u64),
                                symbols.join(", "),
                                config.backtest_data_collection_exchange,
                                config.backtest_data_collection_time_frames.join(", "),
                                cached_selected.len(),
                            );
                        }
                        Ok(DataCollectionRequestOutcome::AlreadyRunning) => {
                            info!(
                                "trading: backtest data appears stale (age={:?}s), OctoBot data collector already running; continuing with {} backtest files",
                                selected_data_age.map(|age| age.round() as u64),
                                cached_selected.len(),
                            );
                        }
                        Ok(DataCollectionRequestOutcome::CoolingDown {
                            retry_after_seconds,
                        }) => {
                            debug!(
                                "trading: backtest data appears stale (age={:?}s) but collector refresh is on cooldown (~{}s remaining); continuing with {} backtest files",
                                selected_data_age.map(|age| age.round() as u64),
                                retry_after_seconds,
                                cached_selected.len(),
                            );
                        }
                        Err(err) => {
                            warn!(
                                "trading: failed to refresh backtest data collection for stale files: {}; continuing with {} cached backtest files",
                                err,
                                cached_selected.len(),
                            );
                        }
                    }
                }
            }
            if catalog_is_stale {
                warn!(
                    "trading: using stale cached backtest data-file catalog path={} files={} age={:.0}s",
                    catalog_path.display(),
                    cached_selected.len(),
                    now - catalog.updated_at
                );
            } else {
                info!(
                    "trading: using cached backtest data-file catalog path={} files={}",
                    catalog_path.display(),
                    cached_selected.len()
                );
            }
            return Ok(cached_selected);
        }

        if !catalog_is_stale {
            match self
                .discover_and_cache_backtest_files(
                    &catalog_path,
                    &mut catalog,
                    &config.backtest_symbols,
                )
                .await
            {
                Ok(selected) if !selected.is_empty() => return Ok(selected),
                Ok(_) => {}
                Err(err) => {
                    warn!(
                        "trading: OctoBot backtest data-file discovery failed: {}",
                        err
                    );
                }
            }
        }

        if config.backtest_data_collection_enabled {
            let symbols = configured_collection_symbols(config);
            match self
                .maybe_request_data_collection(config, &catalog_path, &mut catalog, now)
                .await
            {
                Ok(DataCollectionRequestOutcome::Started) => {
                    return Err(format!(
                        "no OctoBot backtesting .data files matched Gail's configured backtest symbols; started OctoBot historical data collection for {} on {} [{}]",
                        symbols.join(", "),
                        config.backtest_data_collection_exchange,
                        config.backtest_data_collection_time_frames.join(", ")
                    ));
                }
                Ok(DataCollectionRequestOutcome::AlreadyRunning) => {
                    return Err(
                        "no OctoBot backtesting .data files matched Gail's configured backtest symbols; OctoBot data collector is already running"
                            .to_string(),
                    );
                }
                Ok(DataCollectionRequestOutcome::CoolingDown {
                    retry_after_seconds,
                }) => {
                    return Err(format!(
                        "no OctoBot backtesting .data files matched Gail's configured backtest symbols; waiting for OctoBot collector output (retry in ~{retry_after_seconds}s)"
                    ));
                }
                Err(err) => {
                    return Err(format!(
                        "no OctoBot backtesting .data files matched Gail's configured backtest symbols; failed to start OctoBot data collector: {err}"
                    ));
                }
            }
        }

        Err("no OctoBot backtesting .data files matched Gail's configured backtest symbols; collect or configure backtest data files before enabling Gail backtesting".to_string())
    }

    async fn maybe_request_data_collection(
        &self,
        config: &TradingConfig,
        catalog_path: &PathBuf,
        catalog: &mut BacktestDataCatalog,
        now: f64,
    ) -> Result<DataCollectionRequestOutcome, String> {
        let cooldown_seconds = config.backtest_data_collection_cooldown_seconds as f64;
        if let Some(last_request_at) = catalog.last_collection_requested_at
            && cooldown_seconds > 0.0
            && now - last_request_at < cooldown_seconds
        {
            let retry_after_seconds = (cooldown_seconds - (now - last_request_at)).ceil() as u64;
            return Ok(DataCollectionRequestOutcome::CoolingDown {
                retry_after_seconds,
            });
        }

        let collector_symbols = configured_collection_symbols(config);
        let now_ms = (now * 1000.0) as i64;
        let start_ms = now_ms - (config.backtest_lookback_days as i64) * 86_400_000;
        match self
            .octobot
            .start_data_collector(
                &config.backtest_data_collection_exchange,
                &collector_symbols,
                &config.backtest_data_collection_time_frames,
                Some(start_ms),
                Some(now_ms),
            )
            .await
        {
            Ok(()) => {
                catalog.last_collection_requested_at = Some(now);
                persist_backtest_data_catalog(catalog_path, catalog).await;
                Ok(DataCollectionRequestOutcome::Started)
            }
            Err(err) => {
                if err.to_ascii_lowercase().contains("already running") {
                    catalog.last_collection_requested_at = Some(now);
                    persist_backtest_data_catalog(catalog_path, catalog).await;
                    Ok(DataCollectionRequestOutcome::AlreadyRunning)
                } else {
                    Err(err)
                }
            }
        }
    }

    async fn discover_and_cache_backtest_files(
        &self,
        catalog_path: &PathBuf,
        catalog: &mut BacktestDataCatalog,
        symbols: &[String],
    ) -> Result<Vec<String>, String> {
        let available = self.octobot.list_backtest_data_files().await?;
        let available = dedupe_backtest_data_files(available);
        let selected = select_backtest_data_files(available.clone(), symbols);
        if !available.is_empty() {
            catalog.files = available;
            catalog.updated_at = now_ts();
            catalog.last_collection_requested_at = None;
            persist_backtest_data_catalog(catalog_path, catalog).await;
        }
        Ok(selected)
    }
}

fn select_backtest_data_files(available: Vec<String>, symbols: &[String]) -> Vec<String> {
    let available = dedupe_backtest_data_files(available);
    let mut candidates = available
        .into_iter()
        .map(BacktestDataFileCandidate::from_path)
        .collect::<Vec<_>>();

    let wanted = symbols
        .iter()
        .map(|symbol| normalize_symbol_for_data_file(symbol))
        .filter(|symbol| !symbol.is_empty())
        .collect::<Vec<_>>();
    if wanted.is_empty() {
        sort_backtest_data_file_candidates(&mut candidates);
        return candidates
            .into_iter()
            .map(|candidate| candidate.path)
            .collect();
    }

    let mut selected = Vec::new();
    for symbol in wanted {
        let mut symbol_candidates = candidates
            .iter()
            .filter(|candidate| candidate.normalized_path.contains(symbol.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if symbol_candidates.is_empty() {
            continue;
        }

        // Keep several best files per timeframe to accumulate richer history
        // while avoiding unbounded growth.
        sort_backtest_data_file_candidates(&mut symbol_candidates);
        let mut timeframe_counts: HashMap<u64, usize> = HashMap::new();
        let mut unknown_timeframe_count = 0usize;
        for candidate in symbol_candidates {
            if let Some(timeframe_seconds) = candidate.timeframe_seconds {
                let count = timeframe_counts.entry(timeframe_seconds).or_insert(0);
                if *count < BACKTEST_MAX_FILES_PER_SYMBOL_TIMEFRAME {
                    selected.push(candidate.path);
                    *count += 1;
                }
            } else if unknown_timeframe_count < BACKTEST_MAX_UNKNOWN_TIMEFRAME_FILES_PER_SYMBOL {
                selected.push(candidate.path);
                unknown_timeframe_count += 1;
            }
        }
    }

    let selected = dedupe_backtest_data_files(selected);
    if !selected.is_empty() {
        return selected;
    }

    if candidates
        .iter()
        .all(|candidate| is_generic_octobot_collector_file(&candidate.path))
    {
        sort_backtest_data_file_candidates(&mut candidates);
        return candidates
            .into_iter()
            .take(BACKTEST_MAX_GENERIC_COLLECTOR_FILES)
            .map(|candidate| candidate.path)
            .collect();
    }

    Vec::new()
}

fn build_backtest_file_subsets(selected_files: Vec<String>) -> Vec<BacktestFileSubset> {
    let selected_files = dedupe_backtest_data_files(selected_files);
    if selected_files.is_empty() {
        return Vec::new();
    }

    let mut candidates = selected_files
        .into_iter()
        .map(BacktestDataFileCandidate::from_path)
        .collect::<Vec<_>>();
    sort_backtest_data_file_candidates(&mut candidates);

    let mut known_by_timeframe: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    let mut unknown_timeframe_files = Vec::new();
    for candidate in &candidates {
        if let Some(timeframe_seconds) = candidate.timeframe_seconds {
            known_by_timeframe
                .entry(timeframe_seconds)
                .or_default()
                .push(candidate.path.clone());
        } else {
            unknown_timeframe_files.push(candidate.path.clone());
        }
    }

    let mut subsets = Vec::new();
    let mut seen = HashSet::new();
    let one_hour_files = known_by_timeframe.get(&3_600).cloned().unwrap_or_default();
    if !one_hour_files.is_empty() {
        push_backtest_file_subset(
            &mut subsets,
            &mut seen,
            "primary_1h".to_string(),
            one_hour_files.clone(),
        );
    }

    for (timeframe_seconds, timeframe_files) in known_by_timeframe
        .iter()
        .filter(|(timeframe_seconds, _)| **timeframe_seconds >= 86_400)
    {
        if one_hour_files.is_empty() {
            break;
        }
        let mut combined = one_hour_files.clone();
        combined.extend(timeframe_files.clone());
        push_backtest_file_subset(
            &mut subsets,
            &mut seen,
            format!(
                "1h_plus_{}",
                timeframe_label_from_seconds(*timeframe_seconds)
            ),
            combined,
        );
    }

    for (index, file) in unknown_timeframe_files
        .into_iter()
        .take(BACKTEST_MAX_SINGLE_UNKNOWN_FILE_SUBSETS)
        .enumerate()
    {
        push_backtest_file_subset(
            &mut subsets,
            &mut seen,
            format!("single_unknown_{}", index + 1),
            vec![file],
        );
    }

    for (timeframe_seconds, timeframe_files) in known_by_timeframe.iter().rev() {
        push_backtest_file_subset(
            &mut subsets,
            &mut seen,
            format!("{}_only", timeframe_label_from_seconds(*timeframe_seconds)),
            timeframe_files.clone(),
        );
    }

    let all_files = candidates
        .iter()
        .map(|candidate| candidate.path.clone())
        .collect::<Vec<_>>();
    push_backtest_file_subset(
        &mut subsets,
        &mut seen,
        "combined_all".to_string(),
        all_files,
    );

    subsets.truncate(BACKTEST_MAX_SUBSET_RUNS_PER_CYCLE);
    subsets
}

fn push_backtest_file_subset(
    subsets: &mut Vec<BacktestFileSubset>,
    seen: &mut HashSet<String>,
    label: String,
    files: Vec<String>,
) {
    let files = dedupe_backtest_data_files(files);
    if files.is_empty() {
        return;
    }
    let mut fingerprint_files = files.clone();
    fingerprint_files.sort();
    let fingerprint = fingerprint_files.join("\n");
    if seen.insert(fingerprint) {
        subsets.push(BacktestFileSubset { label, files });
    }
}

fn timeframe_label_from_seconds(timeframe_seconds: u64) -> String {
    if timeframe_seconds.is_multiple_of(2_592_000) {
        return format!("{}M", timeframe_seconds / 2_592_000);
    }
    if timeframe_seconds.is_multiple_of(604_800) {
        return format!("{}w", timeframe_seconds / 604_800);
    }
    if timeframe_seconds.is_multiple_of(86_400) {
        return format!("{}d", timeframe_seconds / 86_400);
    }
    if timeframe_seconds.is_multiple_of(3_600) {
        return format!("{}h", timeframe_seconds / 3_600);
    }
    if timeframe_seconds.is_multiple_of(60) {
        return format!("{}m", timeframe_seconds / 60);
    }
    format!("{}s", timeframe_seconds)
}

fn aggregate_backtest_subset_outcomes(
    outcomes: Vec<BacktestSubsetOutcome>,
    profitability_threshold: f64,
) -> BacktestSummary {
    if outcomes.is_empty() {
        return BacktestSummary::incomplete(
            "no backtest subsets were executed; skipping backtest cycle".to_string(),
        );
    }

    let successful = outcomes
        .iter()
        .filter(|outcome| outcome.summary.profitability_pct.is_some())
        .collect::<Vec<_>>();
    if successful.is_empty() {
        let mut summary = outcomes[0].summary.clone();
        summary.notes = format!(
            "{}; subset outcomes: {}",
            summary.notes,
            summarize_subset_outcomes(&outcomes)
        );
        return summary;
    }

    let profitability_values = successful
        .iter()
        .filter_map(|outcome| outcome.summary.profitability_pct)
        .collect::<Vec<_>>();
    let market_avg_values = successful
        .iter()
        .filter_map(|outcome| outcome.summary.market_avg_pct)
        .collect::<Vec<_>>();
    let profitability_pct = if profitability_values.is_empty() {
        None
    } else {
        Some(profitability_values.iter().sum::<f64>() / profitability_values.len() as f64)
    };
    let market_avg_pct = if market_avg_values.is_empty() {
        None
    } else {
        Some(market_avg_values.iter().sum::<f64>() / market_avg_values.len() as f64)
    };

    let beats_market = match (profitability_pct, market_avg_pct) {
        (Some(profitability), Some(market_avg)) => Some(profitability > market_avg),
        _ => None,
    };

    let total_trades = successful
        .iter()
        .map(|outcome| outcome.summary.total_trades)
        .sum::<usize>();
    let errors_count = outcomes
        .iter()
        .map(|outcome| outcome.summary.errors_count)
        .sum::<usize>();

    let mut symbols = successful
        .iter()
        .flat_map(|outcome| outcome.summary.symbols.clone())
        .collect::<Vec<_>>();
    symbols.sort();
    symbols.dedup();

    let run_id = successful
        .iter()
        .rev()
        .find_map(|outcome| outcome.summary.run_id);
    let assessment = assess(profitability_pct, profitability_threshold);
    BacktestSummary {
        run_at: now_ts(),
        assessment,
        profitability_pct,
        market_avg_pct,
        beats_market,
        total_trades,
        errors_count,
        symbols,
        notes: format!(
            "multi-run backtest: successful_subsets={}/{}; outcomes={}",
            successful.len(),
            outcomes.len(),
            summarize_subset_outcomes(&outcomes)
        ),
        run_id,
    }
}

fn summarize_subset_outcomes(outcomes: &[BacktestSubsetOutcome]) -> String {
    outcomes
        .iter()
        .map(|outcome| {
            format!(
                "{}(files={},assessment={},profit={:?})",
                outcome.label,
                outcome.file_count,
                outcome.summary.assessment,
                outcome.summary.profitability_pct
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

#[derive(Clone, Debug)]
struct BacktestDataFileCandidate {
    path: String,
    normalized_path: String,
    timeframe_seconds: Option<u64>,
    freshness_ts: Option<f64>,
    coverage_seconds: Option<u64>,
}

impl BacktestDataFileCandidate {
    fn from_path(path: String) -> Self {
        let timestamps = parse_file_epoch_timestamps_seconds(path.as_str());
        let freshness_ts = generic_collector_file_timestamp(path.as_str()).or_else(|| {
            timestamps
                .iter()
                .copied()
                .max_by(|left, right| left.total_cmp(right))
        });

        let coverage_seconds = if timestamps.len() >= 2 {
            let min_ts = timestamps
                .iter()
                .copied()
                .min_by(|left, right| left.total_cmp(right));
            let max_ts = timestamps
                .iter()
                .copied()
                .max_by(|left, right| left.total_cmp(right));
            match (min_ts, max_ts) {
                (Some(min_ts), Some(max_ts)) if max_ts > min_ts => {
                    Some((max_ts - min_ts).round() as u64)
                }
                _ => None,
            }
        } else {
            None
        };

        Self {
            normalized_path: normalize_symbol_for_data_file(path.as_str()),
            timeframe_seconds: parse_file_timeframe_seconds(path.as_str()),
            freshness_ts,
            coverage_seconds,
            path,
        }
    }
}

fn sort_backtest_data_file_candidates(candidates: &mut [BacktestDataFileCandidate]) {
    candidates.sort_by(compare_backtest_data_file_candidates);
}

fn compare_backtest_data_file_candidates(
    left: &BacktestDataFileCandidate,
    right: &BacktestDataFileCandidate,
) -> std::cmp::Ordering {
    compare_optional_u64_desc(left.timeframe_seconds, right.timeframe_seconds)
        .then_with(|| compare_optional_u64_desc(left.coverage_seconds, right.coverage_seconds))
        .then_with(|| compare_optional_f64_desc(left.freshness_ts, right.freshness_ts))
        .then_with(|| right.path.cmp(&left.path))
}

fn compare_optional_u64_desc(left: Option<u64>, right: Option<u64>) -> std::cmp::Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right.cmp(&left),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn compare_optional_f64_desc(left: Option<f64>, right: Option<f64>) -> std::cmp::Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right
            .partial_cmp(&left)
            .unwrap_or(std::cmp::Ordering::Equal),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn parse_file_timeframe_seconds(path: &str) -> Option<u64> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }

    let filename = trimmed.rsplit('/').next().unwrap_or(trimmed);
    filename
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(parse_timeframe_token_seconds)
        .max()
}

fn parse_timeframe_token_seconds(token: &str) -> Option<u64> {
    let digit_end = token
        .chars()
        .position(|ch| !ch.is_ascii_digit())
        .unwrap_or(token.len());
    if digit_end == 0 || digit_end == token.len() {
        return None;
    }

    let value = token[..digit_end].parse::<u64>().ok()?;
    let unit = &token[digit_end..];
    let multiplier = match unit {
        "m" | "min" | "MIN" => 60,
        "h" | "H" => 3_600,
        "d" | "D" => 86_400,
        "w" | "W" => 604_800,
        "M" | "mo" | "Mo" | "mO" | "MO" => 2_592_000,
        _ => return None,
    };
    value.checked_mul(multiplier)
}

fn parse_file_epoch_timestamps_seconds(path: &str) -> Vec<f64> {
    path.split(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .filter_map(parse_epoch_seconds_token)
        .collect()
}

fn parse_epoch_seconds_token(token: &str) -> Option<f64> {
    if token.is_empty() {
        return None;
    }
    let value = token.parse::<f64>().ok()?;
    if !value.is_finite() {
        return None;
    }
    if (1_000_000_000_000.0..=9_999_999_999_999.0).contains(&value) {
        return Some(value / 1000.0);
    }
    if (1_000_000_000.0..=9_999_999_999.0).contains(&value) {
        return Some(value);
    }
    None
}

fn normalize_symbol_for_data_file(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn is_already_running_backtest_error(err: &str) -> bool {
    let normalized = err.to_ascii_lowercase();
    normalized.contains("already running")
        || normalized.contains("already started")
        || normalized.contains("backtesting is running")
}

fn is_generic_octobot_collector_file(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return false;
    }
    let filename = trimmed.rsplit('/').next().unwrap_or(trimmed);
    let normalized = filename.to_ascii_lowercase();
    normalized.starts_with("exchangehistorydatacollector_") && normalized.ends_with(".data")
}

fn generic_collector_file_timestamp(value: &str) -> Option<f64> {
    if !is_generic_octobot_collector_file(value) {
        return None;
    }
    let trimmed = value.trim();
    let filename = trimmed.rsplit('/').next().unwrap_or(trimmed);
    let prefix = "exchangehistorydatacollector_";
    let suffix = ".data";
    let lowercase_filename = filename.to_ascii_lowercase();
    let timestamp = filename
        .strip_prefix("ExchangeHistoryDataCollector_")
        .and_then(|rest| rest.strip_suffix(".data"))
        .or_else(|| {
            lowercase_filename
                .strip_prefix(prefix)
                .and_then(|rest| rest.strip_suffix(suffix))
        })?;
    timestamp.parse::<f64>().ok()
}

fn newest_selected_backtest_data_ts(
    selected_files: &[String],
    catalog_updated_at: f64,
) -> Option<f64> {
    let from_files = selected_files
        .iter()
        .filter_map(|file| BacktestDataFileCandidate::from_path(file.to_string()).freshness_ts)
        .max_by(|left, right| left.total_cmp(right));
    from_files.or_else(|| (catalog_updated_at > 0.0).then_some(catalog_updated_at))
}

fn selected_backtest_data_age_seconds(
    selected_files: &[String],
    catalog_updated_at: f64,
    now: f64,
) -> Option<f64> {
    newest_selected_backtest_data_ts(selected_files, catalog_updated_at)
        .map(|timestamp| (now - timestamp).max(0.0))
}

fn configured_collection_symbols(config: &TradingConfig) -> Vec<String> {
    if config.backtest_symbols.is_empty() {
        vec![DEFAULT_BACKTEST_COLLECTION_SYMBOL.to_string()]
    } else {
        config.backtest_symbols.clone()
    }
}

fn resolve_backtest_catalog_path(config: &TradingConfig) -> PathBuf {
    if !config.backtest_data_catalog_path.trim().is_empty() {
        return PathBuf::from(config.backtest_data_catalog_path.trim());
    }

    let data_path = PathBuf::from(config.data_path.trim());
    if let Some(parent) = data_path.parent() {
        return parent.join(DEFAULT_BACKTEST_CATALOG_FILENAME);
    }
    PathBuf::from(format!("./data/{DEFAULT_BACKTEST_CATALOG_FILENAME}"))
}

async fn load_backtest_data_catalog(path: &PathBuf) -> BacktestDataCatalog {
    match fs::read_to_string(path).await {
        Ok(raw) => match serde_json::from_str::<BacktestDataCatalog>(&raw) {
            Ok(mut catalog) => {
                catalog.files = dedupe_backtest_data_files(catalog.files);
                catalog
            }
            Err(err) => {
                warn!(
                    "trading: failed to parse backtest data catalog from {}: {}",
                    path.display(),
                    err
                );
                BacktestDataCatalog::default()
            }
        },
        Err(_) => BacktestDataCatalog::default(),
    }
}

async fn persist_backtest_data_catalog(path: &PathBuf, catalog: &BacktestDataCatalog) {
    let payload = match serde_json::to_string_pretty(catalog) {
        Ok(payload) => payload,
        Err(err) => {
            warn!(
                "trading: failed to serialize backtest data catalog for {}: {}",
                path.display(),
                err
            );
            return;
        }
    };

    if let Some(parent) = path.parent()
        && let Err(err) = fs::create_dir_all(parent).await
    {
        warn!(
            "trading: failed to create backtest catalog directory {}: {}",
            parent.display(),
            err
        );
        return;
    }

    if let Err(err) = fs::write(path, payload).await {
        warn!(
            "trading: failed to write backtest data catalog to {}: {}",
            path.display(),
            err
        );
    } else {
        debug!(
            "trading: persisted backtest data catalog to {}",
            path.display()
        );
    }
}

fn dedupe_backtest_data_files(files: Vec<String>) -> Vec<String> {
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();

    for file in files {
        let trimmed = file.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_string()) {
            deduped.push(trimmed.to_string());
        }
    }

    deduped
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

    #[test]
    fn backtest_file_selection_matches_symbols() {
        let available = vec![
            "user/backtesting/collector/binance_BTC_USDT_1h.data".to_string(),
            "user/backtesting/collector/binance_ETH_USDT_1h.data".to_string(),
        ];
        let selected = select_backtest_data_files(available, &["BTC/USDT".to_string()]);
        assert_eq!(
            selected,
            vec!["user/backtesting/collector/binance_BTC_USDT_1h.data"]
        );
    }

    #[test]
    fn backtest_file_selection_uses_all_when_symbols_empty() {
        let available = vec![
            "user/backtesting/collector/binance_BTC_USDT_1h.data".to_string(),
            "user/backtesting/collector/binance_ETH_USDT_1h.data".to_string(),
        ];
        let selected = select_backtest_data_files(available.clone(), &[]);
        assert_eq!(selected.len(), 2);
        for file in available {
            assert!(selected.contains(&file));
        }
    }

    #[test]
    fn backtest_file_selection_prefers_longer_timeframes_and_newer_ranges() {
        let available = vec![
            "user/backtesting/collector/binance_BTC_USDT_1h_1710000000_1710100000.data".to_string(),
            "user/backtesting/collector/binance_BTC_USDT_4h_1710000000_1710100000.data".to_string(),
            "user/backtesting/collector/binance_BTC_USDT_4h_1710200000_1710600000.data".to_string(),
            "user/backtesting/collector/binance_BTC_USDT_1d_1690000000_1690100000.data".to_string(),
        ];
        let selected = select_backtest_data_files(available, &["BTC/USDT".to_string()]);
        assert_eq!(
            selected,
            vec![
                "user/backtesting/collector/binance_BTC_USDT_1d_1690000000_1690100000.data"
                    .to_string(),
                "user/backtesting/collector/binance_BTC_USDT_4h_1710200000_1710600000.data"
                    .to_string(),
                "user/backtesting/collector/binance_BTC_USDT_4h_1710000000_1710100000.data"
                    .to_string(),
                "user/backtesting/collector/binance_BTC_USDT_1h_1710000000_1710100000.data"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn backtest_file_selection_accepts_generic_octobot_collector_files() {
        let available = vec![
            "ExchangeHistoryDataCollector_1779194033.964632.data".to_string(),
            "ExchangeHistoryDataCollector_1779194074.4955919.data".to_string(),
        ];
        let selected = select_backtest_data_files(available.clone(), &["BTC/USDT".to_string()]);
        assert_eq!(
            selected,
            vec![
                "ExchangeHistoryDataCollector_1779194074.4955919.data".to_string(),
                "ExchangeHistoryDataCollector_1779194033.964632.data".to_string(),
            ]
        );
    }

    #[test]
    fn backtest_file_selection_caps_files_per_timeframe() {
        let available = vec![
            "user/backtesting/collector/binance_BTC_USDT_1h_1710800000_1710900000.data".to_string(),
            "user/backtesting/collector/binance_BTC_USDT_1h_1710700000_1710800000.data".to_string(),
            "user/backtesting/collector/binance_BTC_USDT_1h_1710600000_1710700000.data".to_string(),
            "user/backtesting/collector/binance_BTC_USDT_1h_1710500000_1710600000.data".to_string(),
        ];
        let selected = select_backtest_data_files(available, &["BTC/USDT".to_string()]);
        assert_eq!(selected.len(), BACKTEST_MAX_FILES_PER_SYMBOL_TIMEFRAME);
        assert!(
            selected.contains(
                &"user/backtesting/collector/binance_BTC_USDT_1h_1710800000_1710900000.data"
                    .to_string()
            )
        );
        assert!(
            selected.contains(
                &"user/backtesting/collector/binance_BTC_USDT_1h_1710700000_1710800000.data"
                    .to_string()
            )
        );
        assert!(
            selected.contains(
                &"user/backtesting/collector/binance_BTC_USDT_1h_1710600000_1710700000.data"
                    .to_string()
            )
        );
        assert!(
            !selected.contains(
                &"user/backtesting/collector/binance_BTC_USDT_1h_1710500000_1710600000.data"
                    .to_string()
            )
        );
    }

    #[test]
    fn selected_backtest_data_age_uses_newest_file_timestamp() {
        let now = 1_800_000_000.0;
        let selected = vec![
            "ExchangeHistoryDataCollector_1779194033.964632.data".to_string(),
            "ExchangeHistoryDataCollector_1779194074.4955919.data".to_string(),
        ];
        let age = selected_backtest_data_age_seconds(&selected, 1_700_000_000.0, now)
            .expect("age should be calculated");
        let expected_age = now - 1_779_194_074.4955919;
        assert!(
            (age - expected_age).abs() < 1e-6,
            "age={age} expected={expected_age}"
        );
    }

    #[test]
    fn backtest_file_selection_rejects_non_matching_non_collector_files() {
        let available = vec!["other_provider_snapshot.data".to_string()];
        let selected = select_backtest_data_files(available, &["BTC/USDT".to_string()]);
        assert!(selected.is_empty());
    }

    #[test]
    fn catalog_path_defaults_to_data_path_parent() {
        let config = TradingConfig {
            data_path: "/app/data/trading_state.json".to_string(),
            ..TradingConfig::default()
        };
        let path = resolve_backtest_catalog_path(&config);
        assert_eq!(
            path.to_string_lossy(),
            "/app/data/backtest_data_catalog.json"
        );
    }

    #[test]
    fn dedupe_backtest_files_removes_empty_and_duplicates() {
        let deduped = dedupe_backtest_data_files(vec![
            "".to_string(),
            "user/backtesting/collector/binance_BTC_USDT_1h.data".to_string(),
            "user/backtesting/collector/binance_BTC_USDT_1h.data".to_string(),
            " user/backtesting/collector/binance_ETH_USDT_1h.data ".to_string(),
        ]);
        assert_eq!(
            deduped,
            vec![
                "user/backtesting/collector/binance_BTC_USDT_1h.data".to_string(),
                "user/backtesting/collector/binance_ETH_USDT_1h.data".to_string(),
            ]
        );
    }

    #[test]
    fn backtest_file_subsets_prioritize_1h_and_include_unknown_singletons() {
        let selected = vec![
            "user/backtesting/collector/binance_BTC_USDT_1h.data".to_string(),
            "user/backtesting/collector/binance_BTC_USDT_1d.data".to_string(),
            "ExchangeHistoryDataCollector_1779194074.4955919.data".to_string(),
        ];
        let subsets = build_backtest_file_subsets(selected);
        assert!(!subsets.is_empty(), "subsets should be generated");
        assert_eq!(subsets[0].label, "primary_1h");
        assert_eq!(
            subsets[0].files,
            vec!["user/backtesting/collector/binance_BTC_USDT_1h.data".to_string()]
        );
        assert!(
            subsets.iter().any(|subset| subset.label == "1h_plus_1d"
                && subset
                    .files
                    .contains(&"user/backtesting/collector/binance_BTC_USDT_1h.data".to_string())
                && subset
                    .files
                    .contains(&"user/backtesting/collector/binance_BTC_USDT_1d.data".to_string())),
            "subsets should include 1h + 1d combination"
        );
        assert!(
            subsets
                .iter()
                .any(|subset| subset.label.starts_with("single_unknown_")),
            "subsets should include unknown-timeframe single-file attempts"
        );
    }

    #[test]
    fn aggregate_backtest_subset_outcomes_uses_successful_subset_average() {
        let outcomes = vec![
            BacktestSubsetOutcome {
                label: "subset_1".to_string(),
                file_count: 1,
                summary: BacktestSummary {
                    run_at: now_ts(),
                    assessment: ApproachAssessment::Viable,
                    profitability_pct: Some(4.0),
                    market_avg_pct: Some(2.0),
                    beats_market: Some(true),
                    total_trades: 5,
                    errors_count: 0,
                    symbols: vec!["BTC/USDT".to_string()],
                    notes: "ok".to_string(),
                    run_id: Some(11),
                },
            },
            BacktestSubsetOutcome {
                label: "subset_2".to_string(),
                file_count: 1,
                summary: BacktestSummary {
                    run_at: now_ts(),
                    assessment: ApproachAssessment::Marginal,
                    profitability_pct: Some(2.0),
                    market_avg_pct: Some(1.0),
                    beats_market: Some(true),
                    total_trades: 4,
                    errors_count: 0,
                    symbols: vec!["BTC/USDT".to_string()],
                    notes: "ok".to_string(),
                    run_id: Some(12),
                },
            },
            BacktestSubsetOutcome {
                label: "subset_3".to_string(),
                file_count: 1,
                summary: BacktestSummary::incomplete("start failed"),
            },
        ];

        let summary = aggregate_backtest_subset_outcomes(outcomes, 1.0);
        assert_eq!(summary.assessment, ApproachAssessment::Viable);
        assert_eq!(summary.profitability_pct, Some(3.0));
        assert_eq!(summary.market_avg_pct, Some(1.5));
        assert_eq!(summary.beats_market, Some(true));
        assert_eq!(summary.run_id, Some(12));
        assert!(
            summary.notes.contains("successful_subsets=2/3"),
            "notes should include multi-run summary: {}",
            summary.notes
        );
    }
}
