//! Incremental market datalake for trading decisions.
//!
//! Gail ingests live OctoBot snapshots each cycle, stores them durably
//! (JSONL + optional Postgres), and derives historical features without
//! re-pulling full historical datasets on every evaluation.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    env,
    future::Future,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::{Mutex, mpsc},
};
use tokio_postgres::NoTls;
use tracing::{debug, warn};

use super::{
    config::TradingConfig,
    octobot::{MarketMicrostructure, MarketSnapshot},
    quantitative::backtest::{NativeBacktestConfig, QuantBacktestFrame},
    state::SharedTradingState,
};

const POSTGRES_PRUNE_INTERVAL_SECONDS: f64 = 3_600.0;
const MARKET_DATALAKE_SCHEMA_VERSION: u32 = 2;
// Postgres is an optional mirror for the append-only file lake. Database
// startup, reads and writes all run outside evaluation and decision tasks.
const MARKET_DATALAKE_POSTGRES_OPERATION_TIMEOUT_SECONDS: u64 = 30;
const MARKET_DATALAKE_POSTGRES_MIRROR_QUEUE_CAPACITY: usize = 32;
const MARKET_DATALAKE_POSTGRES_MIRROR_BATCH_SIZE: usize = 500;
const MARKET_DATALAKE_POSTGRES_MIRROR_RETRIES: usize = 2;
const MARKET_DATALAKE_POSTGRES_MIRROR_RETRY_DELAY_SECONDS: u64 = 1;

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MarketSample {
    pub exchange: String,
    pub symbol: String,
    pub captured_ts: f64,
    pub captured_bucket: i64,
    pub price: f64,
    pub price_change_pct_24h: Option<f64>,
    pub volume_24h: Option<f64>,
    pub high_24h: Option<f64>,
    pub low_24h: Option<f64>,
    /// Point-in-time executable-market and derivatives telemetry.  Persisting
    /// this alongside price prevents native replay from silently reverting to
    /// static spread and slippage assumptions.
    pub microstructure: MarketMicrostructure,
}

impl Default for MarketSample {
    fn default() -> Self {
        Self {
            exchange: String::new(),
            symbol: String::new(),
            captured_ts: 0.0,
            captured_bucket: 0,
            price: 0.0,
            price_change_pct_24h: None,
            volume_24h: None,
            high_24h: None,
            low_24h: None,
            microstructure: MarketMicrostructure::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MarketHistoricalFeatures {
    pub samples: usize,
    pub first_ts: Option<f64>,
    pub last_ts: Option<f64>,
    pub freshness_seconds: Option<f64>,
    pub momentum_short_pct: Option<f64>,
    pub momentum_mid_pct: Option<f64>,
    pub momentum_long_pct: Option<f64>,
    pub volatility_pct: Option<f64>,
    pub drawdown_pct: Option<f64>,
    pub volume_ratio_short_long: Option<f64>,
}

impl MarketHistoricalFeatures {
    pub fn momentum_signal(&self) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        for (weight, value) in [
            (0.55, self.momentum_short_pct),
            (0.30, self.momentum_mid_pct),
            (0.15, self.momentum_long_pct),
        ] {
            if let Some(value) = value {
                weighted_sum += weight * value;
                total_weight += weight;
            }
        }
        if total_weight <= f64::EPSILON {
            0.0
        } else {
            ((weighted_sum / total_weight) / 8.0).clamp(-1.0, 1.0)
        }
    }

    pub fn volume_regime_ratio(&self) -> f64 {
        self.volume_ratio_short_long.unwrap_or(1.0).clamp(0.0, 3.0)
    }

    pub fn risk_pressure(&self) -> f64 {
        let volatility_component = (self.volatility_pct.unwrap_or(0.0).abs() / 8.0).clamp(0.0, 1.0);
        let drawdown_component = (self.drawdown_pct.unwrap_or(0.0).abs() / 15.0).clamp(0.0, 1.0);
        (volatility_component * 0.65 + drawdown_component * 0.35).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MarketIngestSummary {
    pub received: usize,
    pub persisted: usize,
    pub deduplicated: usize,
    pub file_error: Option<String>,
    pub postgres_error: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MarketDataLakeBootstrapReport {
    pub reason: String,
    pub symbols_attempted: usize,
    pub symbols_with_history: usize,
    pub time_frames: Vec<String>,
    pub snapshots_received: usize,
    pub snapshots_persisted: usize,
    pub snapshots_deduplicated: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct MarketDataLakeMetadata {
    pub schema_version: u32,
    pub build_id: String,
    pub last_bootstrap_started_at: Option<f64>,
    pub last_bootstrap_completed_at: Option<f64>,
    pub last_bootstrap_status: Option<String>,
    pub last_bootstrap_reason: Option<String>,
    pub last_bootstrap_error: Option<String>,
    pub bootstrap_symbols_attempted: usize,
    pub bootstrap_symbols_with_history: usize,
    pub bootstrap_snapshots_persisted: usize,
}

#[derive(Clone)]
pub struct MarketDataLake {
    config: Arc<MarketDataLakeConfig>,
    state: Arc<Mutex<MarketDataLakeState>>,
    postgres_mirror_tx: Option<mpsc::Sender<PostgresMirrorTask>>,
    log_state: SharedTradingState,
}

enum PostgresMirrorTask {
    Persist(Vec<MarketSample>),
    PersistMetadata(MarketDataLakeMetadata),
    LoadSymbols {
        symbols: Vec<(String, String)>,
        since_ts: f64,
    },
}

#[derive(Clone)]
struct MarketDataLakeConfig {
    file_path: PathBuf,
    metadata_path: PathBuf,
    postgres_dsn: Option<String>,
    retention_days: u32,
    bucket_seconds: u64,
    short_window_seconds: u64,
    mid_window_seconds: u64,
    long_window_seconds: u64,
    min_samples: usize,
}

impl MarketDataLakeConfig {
    fn from_trading_config(trading_config: &TradingConfig, postgres_dsn: Option<String>) -> Self {
        Self {
            file_path: PathBuf::from(trading_config.market_datalake_file_path.clone()),
            metadata_path: metadata_path_for(Path::new(&trading_config.market_datalake_file_path)),
            postgres_dsn,
            retention_days: trading_config.market_datalake_retention_days,
            bucket_seconds: trading_config.market_datalake_bucket_seconds.max(1),
            short_window_seconds: trading_config.market_datalake_short_window_minutes * 60,
            mid_window_seconds: trading_config.market_datalake_mid_window_hours * 3_600,
            long_window_seconds: trading_config.market_datalake_long_window_days * 86_400,
            min_samples: trading_config.market_datalake_min_samples.max(2),
        }
    }

    fn retention_seconds(&self) -> f64 {
        self.retention_days as f64 * 86_400.0
    }

    fn cache_horizon_seconds(&self) -> f64 {
        let long = self.long_window_seconds as f64;
        let retention = self.retention_seconds();
        (long * 2.0).min(retention).max(long)
    }
}

#[derive(Default)]
struct MarketDataLakeState {
    samples_by_symbol: HashMap<String, VecDeque<MarketSample>>,
    loaded_symbols: HashSet<String>,
    last_postgres_prune_at: f64,
    metadata: MarketDataLakeMetadata,
    current_build_id: String,
    bootstrap_required: bool,
    bootstrap_reason: Option<String>,
}

pub fn market_feature_key(exchange: &str, symbol: &str) -> String {
    format!(
        "{}|{}",
        normalize_exchange(exchange),
        normalize_symbol(symbol)
    )
}

impl MarketDataLake {
    pub async fn new(
        trading_config: &TradingConfig,
        postgres_dsn: Option<String>,
        log_state: SharedTradingState,
    ) -> Self {
        let config = Arc::new(MarketDataLakeConfig::from_trading_config(
            trading_config,
            postgres_dsn,
        ));
        let current_build_id = current_build_id();
        let metadata = load_metadata_from_file(&config).await.unwrap_or_default();
        let bootstrap_reason = compute_bootstrap_reason(&metadata, &current_build_id);
        let state = Arc::new(Mutex::new(MarketDataLakeState {
            samples_by_symbol: HashMap::new(),
            loaded_symbols: HashSet::new(),
            last_postgres_prune_at: 0.0,
            metadata,
            current_build_id,
            bootstrap_required: bootstrap_reason.is_some(),
            bootstrap_reason,
        }));
        let (postgres_mirror_tx, postgres_mirror_rx) = if config.postgres_dsn.is_some() {
            let (sender, receiver) = mpsc::channel(MARKET_DATALAKE_POSTGRES_MIRROR_QUEUE_CAPACITY);
            (Some(sender), Some(receiver))
        } else {
            (None, None)
        };
        let lake = Self {
            config,
            state,
            postgres_mirror_tx,
            log_state,
        };
        if let Some(receiver) = postgres_mirror_rx {
            let worker = Self {
                config: lake.config.clone(),
                state: lake.state.clone(),
                postgres_mirror_tx: None,
                log_state: lake.log_state.clone(),
            };
            tokio::spawn(async move {
                worker.run_postgres_mirror_worker(receiver).await;
            });
        }
        lake
    }

    pub async fn bootstrap_required_reason(&self) -> Option<String> {
        let state = self.state.lock().await;
        state.bootstrap_reason.clone()
    }

    pub async fn mark_bootstrap_completed(&self, report: &MarketDataLakeBootstrapReport) {
        let mut state = self.state.lock().await;
        state.bootstrap_required = false;
        state.bootstrap_reason = None;
        state.metadata.schema_version = MARKET_DATALAKE_SCHEMA_VERSION;
        state.metadata.build_id = state.current_build_id.clone();
        state.metadata.last_bootstrap_completed_at = Some(now_ts());
        state.metadata.last_bootstrap_status = Some("ok".to_string());
        state.metadata.last_bootstrap_error = None;
        state.metadata.last_bootstrap_reason = Some(report.reason.clone());
        state.metadata.bootstrap_symbols_attempted = report.symbols_attempted;
        state.metadata.bootstrap_symbols_with_history = report.symbols_with_history;
        state.metadata.bootstrap_snapshots_persisted = report.snapshots_persisted;
        let metadata = state.metadata.clone();
        drop(state);
        self.persist_metadata(&metadata).await;
    }

    pub async fn mark_bootstrap_started(&self, reason: &str) {
        let mut state = self.state.lock().await;
        state.metadata.schema_version = MARKET_DATALAKE_SCHEMA_VERSION;
        state.metadata.last_bootstrap_started_at = Some(now_ts());
        state.metadata.last_bootstrap_status = Some("running".to_string());
        state.metadata.last_bootstrap_error = None;
        state.metadata.last_bootstrap_reason = Some(reason.to_string());
        let metadata = state.metadata.clone();
        drop(state);
        self.persist_metadata(&metadata).await;
    }

    pub async fn mark_bootstrap_failed(&self, reason: &str, error: &str) {
        let mut state = self.state.lock().await;
        state.bootstrap_required = true;
        state.bootstrap_reason = Some(reason.to_string());
        state.metadata.schema_version = MARKET_DATALAKE_SCHEMA_VERSION;
        state.metadata.last_bootstrap_started_at = Some(now_ts());
        state.metadata.last_bootstrap_status = Some("failed".to_string());
        state.metadata.last_bootstrap_error = Some(truncate(error, 900));
        state.metadata.last_bootstrap_reason = Some(reason.to_string());
        let metadata = state.metadata.clone();
        drop(state);
        self.persist_metadata(&metadata).await;
    }

    async fn persist_metadata(&self, metadata: &MarketDataLakeMetadata) {
        if let Err(error) = persist_metadata_file(&self.config.metadata_path, metadata).await {
            warn!(
                error = %error,
                "trading: failed to persist authoritative market datalake metadata file"
            );
            self.log_state
                .log(
                    "warn",
                    "market_datalake",
                    "Failed to persist authoritative market datalake metadata file",
                    serde_json::json!({ "error": truncate(&error, 700) }),
                )
                .await;
            return;
        }
        if let Some(sender) = &self.postgres_mirror_tx
            && let Err(error) =
                sender.try_send(PostgresMirrorTask::PersistMetadata(metadata.clone()))
        {
            let reason = match error {
                mpsc::error::TrySendError::Full(_) => "queue is full",
                mpsc::error::TrySendError::Closed(_) => "worker is unavailable",
            };
            warn!(
                reason,
                "trading: Postgres metadata mirror was deferred; file metadata remains authoritative"
            );
            self.log_state
                .log(
                    "warn",
                    "market_datalake",
                    "Postgres metadata mirror was deferred; file metadata remains authoritative",
                    serde_json::json!({ "reason": reason }),
                )
                .await;
        }
    }

    pub async fn ingest_snapshots(&self, snapshots: &[MarketSnapshot]) -> MarketIngestSummary {
        if snapshots.is_empty() {
            return MarketIngestSummary::default();
        }

        let mut dedupe: HashMap<String, MarketSample> = HashMap::new();
        let mut received = 0usize;
        for snapshot in snapshots {
            received += 1;
            let Some(sample) = sample_from_snapshot(snapshot, self.config.bucket_seconds) else {
                continue;
            };
            let dedupe_key = format!(
                "{}#{}",
                market_feature_key(&sample.exchange, &sample.symbol),
                sample.captured_bucket
            );
            match dedupe.get(&dedupe_key) {
                Some(previous) if previous.captured_ts >= sample.captured_ts => {}
                _ => {
                    dedupe.insert(dedupe_key, sample);
                }
            }
        }

        let mut samples = dedupe.into_values().collect::<Vec<_>>();
        samples.sort_by(|left, right| {
            market_feature_key(&left.exchange, &left.symbol)
                .cmp(&market_feature_key(&right.exchange, &right.symbol))
                .then_with(|| left.captured_bucket.cmp(&right.captured_bucket))
        });

        let mut summary = MarketIngestSummary {
            received,
            persisted: 0,
            deduplicated: received.saturating_sub(samples.len()),
            file_error: None,
            postgres_error: None,
        };

        if samples.is_empty() {
            return summary;
        }

        self.merge_samples_into_cache(&samples).await;

        let file_persisted = match append_samples_to_file(&self.config.file_path, &samples).await {
            Ok(()) => {
                summary.persisted = samples.len();
                true
            }
            Err(error) => {
                warn!(
                    error = %error,
                    path = %self.config.file_path.display(),
                    "trading: failed to append market datalake file records"
                );
                summary.file_error = Some(error);
                false
            }
        };

        if file_persisted && let Some(sender) = &self.postgres_mirror_tx {
            let batches = samples
                .chunks(MARKET_DATALAKE_POSTGRES_MIRROR_BATCH_SIZE)
                .collect::<Vec<_>>();
            for (index, batch) in batches.iter().enumerate() {
                if let Err(error) = sender.try_send(PostgresMirrorTask::Persist(batch.to_vec())) {
                    let reason = match error {
                        mpsc::error::TrySendError::Full(_) => "queue is full",
                        mpsc::error::TrySendError::Closed(_) => "worker is unavailable",
                    };
                    summary.postgres_error = Some(format!(
                        "market datalake Postgres mirror {reason}; file lake remains authoritative"
                    ));
                    self.log_state
                        .log(
                            "warn",
                            "market_datalake",
                            "Postgres mirror queue rejected a batch; file lake remains authoritative",
                            serde_json::json!({
                                "reason": reason,
                                "samples": batch.len(),
                                "unmirrored_samples": batches[index..].iter().map(|items| items.len()).sum::<usize>(),
                                "queue_capacity": MARKET_DATALAKE_POSTGRES_MIRROR_QUEUE_CAPACITY,
                            }),
                        )
                        .await;
                    break;
                }
            }
        }

        summary
    }

    pub async fn features_for_snapshots(
        &self,
        snapshots: &[MarketSnapshot],
    ) -> HashMap<String, MarketHistoricalFeatures> {
        let symbols = snapshots
            .iter()
            .map(|snapshot| (snapshot.exchange.clone(), snapshot.symbol.clone()))
            .collect::<Vec<_>>();
        self.ensure_symbols_loaded(&symbols).await;

        let mut out = HashMap::new();
        for snapshot in snapshots {
            let key = market_feature_key(&snapshot.exchange, &snapshot.symbol);
            let samples = {
                let state = self.state.lock().await;
                state
                    .samples_by_symbol
                    .get(&key)
                    .map(|items| items.iter().cloned().collect::<Vec<_>>())
            };
            if let Some(samples) = samples
                && let Some(features) = compute_features(&samples, now_ts(), &self.config)
            {
                out.insert(key, features);
            }
        }
        out
    }

    /// Materialise look-ahead-safe frames for Gail's native quant backtester.
    ///
    /// The in-memory sample cache is copied while holding the asynchronous
    /// mutex; feature construction is then moved to a blocking worker and
    /// parallelised by symbol.  The live ingestion path is therefore never
    /// held up by historical replay calculations.
    pub async fn native_backtest_frames(
        &self,
        backtest_config: &NativeBacktestConfig,
    ) -> Result<Vec<QuantBacktestFrame>, String> {
        let samples_by_symbol = {
            let state = self.state.lock().await;
            state
                .samples_by_symbol
                .iter()
                .map(|(key, samples)| (key.clone(), samples.iter().cloned().collect::<Vec<_>>()))
                .collect::<HashMap<_, _>>()
        };
        let lake_config = (*self.config).clone();
        let mut replay_config = backtest_config.clone();
        replay_config.normalise();
        tokio::task::spawn_blocking(move || {
            build_native_backtest_frames(samples_by_symbol, &lake_config, &replay_config)
        })
        .await
        .map_err(|error| format!("native backtest frame worker failed: {error}"))
    }

    pub async fn features_for_symbol(
        &self,
        exchange: &str,
        symbol: &str,
    ) -> Option<MarketHistoricalFeatures> {
        self.ensure_symbols_loaded(&[(exchange.to_string(), symbol.to_string())])
            .await;
        let key = market_feature_key(exchange, symbol);
        let samples = {
            let state = self.state.lock().await;
            state
                .samples_by_symbol
                .get(&key)
                .map(|items| items.iter().cloned().collect::<Vec<_>>())
        }?;
        compute_features(&samples, now_ts(), &self.config)
    }

    async fn ensure_symbols_loaded(&self, symbols: &[(String, String)]) {
        let mut targets = HashMap::<String, (String, String)>::new();
        {
            let mut state = self.state.lock().await;
            for (exchange, symbol) in symbols {
                let exchange = normalize_exchange(exchange);
                let symbol = normalize_symbol(symbol);
                let key = market_feature_key(&exchange, &symbol);
                if state.loaded_symbols.insert(key.clone()) {
                    targets.insert(key, (exchange, symbol));
                }
            }
        }
        if targets.is_empty() {
            return;
        }

        let since_ts = now_ts() - self.config.retention_seconds();
        let target_keys = targets.keys().cloned().collect::<HashSet<_>>();
        match load_samples_from_file(&self.config.file_path, &target_keys, since_ts).await {
            Ok(samples) => {
                if !samples.is_empty() {
                    self.merge_samples_into_cache(&samples).await;
                }
            }
            Err(error) => {
                warn!(
                    error = %error,
                    path = %self.config.file_path.display(),
                    symbols = target_keys.len(),
                    "trading: failed to load market history from file"
                );
            }
        }

        // Postgres is a best-effort history mirror. File-backed samples are
        // immediately available to this decision cycle; the worker can add
        // older mirrored samples to the shared cache for subsequent cycles.
        if let Some(sender) = &self.postgres_mirror_tx {
            let symbols = targets.into_values().collect::<Vec<_>>();
            if let Err(error) = sender.try_send(PostgresMirrorTask::LoadSymbols {
                symbols: symbols.clone(),
                since_ts,
            }) {
                let reason = match error {
                    mpsc::error::TrySendError::Full(_) => "queue is full",
                    mpsc::error::TrySendError::Closed(_) => "worker is unavailable",
                };
                warn!(
                    reason,
                    symbols = symbols.len(),
                    "trading: Postgres history read was deferred; using file lake"
                );
                self.log_state
                    .log(
                        "warn",
                        "market_datalake",
                        "Postgres history read was deferred; using file lake",
                        serde_json::json!({
                            "reason": reason,
                            "symbols": symbols.len(),
                            "since_ts": since_ts,
                        }),
                    )
                    .await;
            }
        }
    }

    async fn run_postgres_mirror_worker(&self, mut receiver: mpsc::Receiver<PostgresMirrorTask>) {
        let mut schema_ready = false;
        while let Some(task) = receiver.recv().await {
            if !schema_ready {
                let Some(dsn) = self.config.postgres_dsn.as_deref() else {
                    break;
                };
                match retry_market_datalake_postgres(|| initialize_postgres_schema(dsn)).await {
                    Ok(()) => schema_ready = true,
                    Err(error) => {
                        warn!(error = %error, "trading: Postgres schema initialisation failed in background mirror worker; file lake remains authoritative");
                        self.log_state
                            .log(
                                "warn",
                                "market_datalake",
                                "Postgres schema initialisation failed in background mirror worker",
                                serde_json::json!({
                                    "error": truncate(&error, 700),
                                    "file_lake_authoritative": true,
                                }),
                            )
                            .await;
                        continue;
                    }
                }
            }
            match task {
                PostgresMirrorTask::Persist(samples) => {
                    let count = samples.len();
                    let dsn = self.config.postgres_dsn.as_deref().unwrap_or_default();
                    let result =
                        retry_market_datalake_postgres(|| persist_samples_postgres(dsn, &samples))
                            .await;
                    match result {
                        Ok(()) => {
                            debug!(
                                samples = count,
                                "trading: mirrored market samples to Postgres"
                            );
                            self.maybe_prune_postgres().await;
                        }
                        Err(error) => {
                            warn!(
                                error = %error,
                                samples = count,
                                retries = MARKET_DATALAKE_POSTGRES_MIRROR_RETRIES,
                                "trading: market sample mirror failed; file lake remains authoritative"
                            );
                            self.log_state
                                .log(
                                    "warn",
                                    "market_datalake",
                                    "Market sample Postgres mirror failed; file lake remains authoritative",
                                    serde_json::json!({
                                        "error": truncate(&error, 700),
                                        "samples": count,
                                        "retries": MARKET_DATALAKE_POSTGRES_MIRROR_RETRIES,
                                    }),
                                )
                                .await;
                        }
                    }
                }
                PostgresMirrorTask::PersistMetadata(metadata) => {
                    let dsn = self.config.postgres_dsn.as_deref().unwrap_or_default();
                    if let Err(error) =
                        retry_market_datalake_postgres(|| persist_metadata_postgres(dsn, &metadata))
                            .await
                    {
                        warn!(error = %error, "trading: market datalake metadata mirror failed; file metadata remains authoritative");
                        self.log_state
                            .log(
                                "warn",
                                "market_datalake",
                                "Postgres metadata mirror failed; file metadata remains authoritative",
                                serde_json::json!({ "error": truncate(&error, 700) }),
                            )
                            .await;
                    }
                }
                PostgresMirrorTask::LoadSymbols { symbols, since_ts } => {
                    let Some(dsn) = self.config.postgres_dsn.as_deref() else {
                        continue;
                    };
                    let mut loaded = Vec::new();
                    for (exchange, symbol) in &symbols {
                        let result = retry_market_datalake_postgres(|| {
                            load_samples_from_postgres(dsn, exchange, symbol, since_ts)
                        })
                        .await;
                        match result {
                            Ok(samples) => loaded.extend(samples),
                            Err(error) => {
                                warn!(
                                    error = %error,
                                    exchange,
                                    symbol,
                                    "trading: failed to load mirrored market history; file lake remains available"
                                );
                                self.log_state
                                    .log(
                                        "warn",
                                        "market_datalake",
                                        "Postgres market-history read failed; file lake remains available",
                                        serde_json::json!({
                                            "error": truncate(&error, 700),
                                            "exchange": exchange,
                                            "symbol": symbol,
                                        }),
                                    )
                                    .await;
                            }
                        }
                    }
                    let loaded = dedupe_samples_by_symbol(loaded);
                    let count = loaded.len();
                    self.merge_mirror_samples_into_cache(&loaded).await;
                    if count > 0 {
                        debug!(
                            symbols = symbols.len(),
                            samples = count,
                            "trading: loaded mirrored market history in background"
                        );
                    }
                }
            }
        }
    }

    async fn merge_samples_into_cache(&self, samples: &[MarketSample]) {
        self.merge_samples_with_precedence(samples, true).await;
    }

    async fn merge_mirror_samples_into_cache(&self, samples: &[MarketSample]) {
        self.merge_samples_with_precedence(samples, false).await;
    }

    async fn merge_samples_with_precedence(
        &self,
        samples: &[MarketSample],
        replace_existing_bucket: bool,
    ) {
        if samples.is_empty() {
            return;
        }
        let cache_horizon_seconds = self.config.cache_horizon_seconds();
        let mut state = self.state.lock().await;
        for sample in samples {
            let key = market_feature_key(&sample.exchange, &sample.symbol);
            let deque = state
                .samples_by_symbol
                .entry(key)
                .or_insert_with(VecDeque::new);

            if let Some(existing) = deque
                .iter_mut()
                .find(|item| item.captured_bucket == sample.captured_bucket)
            {
                if replace_existing_bucket && sample.captured_ts >= existing.captured_ts {
                    *existing = sample.clone();
                }
            } else if deque
                .back()
                .is_none_or(|last| sample.captured_bucket > last.captured_bucket)
            {
                deque.push_back(sample.clone());
            } else {
                let insert_idx = deque
                    .iter()
                    .position(|item| item.captured_bucket > sample.captured_bucket)
                    .unwrap_or(deque.len());
                deque.insert(insert_idx, sample.clone());
            }

            let cutoff = sample.captured_ts - cache_horizon_seconds;
            while deque
                .front()
                .is_some_and(|front| front.captured_ts + f64::EPSILON < cutoff)
            {
                deque.pop_front();
            }
        }
    }

    async fn maybe_prune_postgres(&self) {
        let Some(dsn) = self.config.postgres_dsn.as_deref() else {
            return;
        };
        let now = now_ts();
        let should_prune = {
            let mut state = self.state.lock().await;
            if now - state.last_postgres_prune_at < POSTGRES_PRUNE_INTERVAL_SECONDS {
                false
            } else {
                state.last_postgres_prune_at = now;
                true
            }
        };
        if !should_prune {
            return;
        }
        match retry_market_datalake_postgres(|| {
            prune_postgres_retention(dsn, self.config.retention_days)
        })
        .await
        {
            Ok(_) => {}
            Err(error) => {
                warn!(
                    error = %error,
                    "trading: failed to prune old market datalake rows from Postgres"
                );
                self.log_state
                    .log(
                        "warn",
                        "market_datalake",
                        "Postgres market-data retention pruning failed; file lake remains authoritative",
                        serde_json::json!({
                            "error": truncate(&error, 700),
                            "retention_days": self.config.retention_days,
                        }),
                    )
                    .await;
            }
        }
    }
}

fn build_native_backtest_frames(
    samples_by_symbol: HashMap<String, Vec<MarketSample>>,
    lake_config: &MarketDataLakeConfig,
    backtest_config: &NativeBacktestConfig,
) -> Vec<QuantBacktestFrame> {
    let mut ranked_symbols = samples_by_symbol.into_iter().collect::<Vec<_>>();
    ranked_symbols.sort_by(|left, right| {
        let liquidity = |samples: &[MarketSample]| {
            samples
                .last()
                .and_then(|sample| sample.volume_24h.map(|volume| volume * sample.price))
                .filter(|value| value.is_finite())
                .unwrap_or(0.0)
        };
        liquidity(&right.1)
            .partial_cmp(&liquidity(&left.1))
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked_symbols.truncate(backtest_config.symbol_limit);

    let interval = backtest_config.frame_interval_seconds.max(1) as i64;
    let per_symbol = ranked_symbols
        .par_iter()
        .map(|(_, samples)| {
            let mut rows = Vec::new();
            let mut previous_interval = None;
            for (index, sample) in samples.iter().enumerate() {
                let interval_bucket = (sample.captured_ts / interval as f64).floor() as i64;
                if previous_interval == Some(interval_bucket) {
                    continue;
                }
                previous_interval = Some(interval_bucket);
                let Some(features) =
                    compute_features(&samples[..=index], sample.captured_ts, lake_config)
                else {
                    continue;
                };
                rows.push((
                    interval_bucket,
                    MarketSnapshot {
                        exchange: sample.exchange.clone(),
                        symbol: sample.symbol.clone(),
                        price: sample.price,
                        price_change_pct_1h: features.momentum_short_pct,
                        price_change_pct_24h: sample.price_change_pct_24h,
                        volume_24h: sample.volume_24h,
                        volume_change_pct: features
                            .volume_ratio_short_long
                            .map(|ratio| (ratio - 1.0) * 100.0),
                        high_24h: sample.high_24h,
                        low_24h: sample.low_24h,
                        fetched_at: sample.captured_ts,
                        microstructure: sample.microstructure.clone(),
                    },
                    features,
                ));
            }
            rows
        })
        .collect::<Vec<_>>();

    let mut frames: BTreeMap<i64, QuantBacktestFrame> = BTreeMap::new();
    for rows in per_symbol {
        for (interval_bucket, snapshot, features) in rows {
            let frame = frames
                .entry(interval_bucket)
                .or_insert_with(|| QuantBacktestFrame {
                    timestamp: interval_bucket as f64 * interval as f64,
                    ..QuantBacktestFrame::default()
                });
            frame.historical_features.insert(
                market_feature_key(&snapshot.exchange, &snapshot.symbol),
                features,
            );
            frame.snapshots.push(snapshot);
        }
    }
    let mut frames = frames.into_values().collect::<Vec<_>>();
    if frames.len() > backtest_config.maximum_frames {
        frames.drain(..frames.len() - backtest_config.maximum_frames);
    }
    frames
}

fn sample_from_snapshot(snapshot: &MarketSnapshot, bucket_seconds: u64) -> Option<MarketSample> {
    if !snapshot.price.is_finite() || snapshot.price <= 0.0 {
        return None;
    }
    let captured_ts = if snapshot.fetched_at.is_finite() && snapshot.fetched_at > 0.0 {
        snapshot.fetched_at
    } else {
        now_ts()
    };
    let captured_bucket = (captured_ts / bucket_seconds as f64).floor() as i64;
    Some(MarketSample {
        exchange: normalize_exchange(&snapshot.exchange),
        symbol: normalize_symbol(&snapshot.symbol),
        captured_ts,
        captured_bucket,
        price: snapshot.price,
        price_change_pct_24h: snapshot.price_change_pct_24h,
        volume_24h: snapshot.volume_24h,
        high_24h: snapshot.high_24h,
        low_24h: snapshot.low_24h,
        microstructure: snapshot.microstructure.clone(),
    })
}

fn compute_features(
    samples: &[MarketSample],
    now: f64,
    config: &MarketDataLakeConfig,
) -> Option<MarketHistoricalFeatures> {
    if samples.len() < config.min_samples {
        return None;
    }
    let latest = samples.last()?;
    let anchor_ts = latest.captured_ts;
    let short_window = window_slice(samples, anchor_ts - config.short_window_seconds as f64);
    let mid_window = window_slice(samples, anchor_ts - config.mid_window_seconds as f64);
    let long_window = window_slice(samples, anchor_ts - config.long_window_seconds as f64);
    if long_window.len() < config.min_samples {
        return None;
    }

    let short_momentum = window_change_pct(short_window);
    let mid_momentum = window_change_pct(mid_window);
    let long_momentum = window_change_pct(long_window);
    let volatility_pct = realized_volatility_pct(long_window);
    let drawdown_pct = drawdown_from_peak_pct(long_window);
    let short_volume_avg = average_volume(short_window);
    let long_volume_avg = average_volume(long_window);
    let volume_ratio_short_long = match (short_volume_avg, long_volume_avg) {
        (Some(short), Some(long)) if long > f64::EPSILON => Some(short / long),
        _ => None,
    };

    Some(MarketHistoricalFeatures {
        samples: long_window.len(),
        first_ts: long_window.first().map(|sample| sample.captured_ts),
        last_ts: Some(latest.captured_ts),
        freshness_seconds: Some((now - latest.captured_ts).max(0.0)),
        momentum_short_pct: short_momentum,
        momentum_mid_pct: mid_momentum,
        momentum_long_pct: long_momentum,
        volatility_pct,
        drawdown_pct,
        volume_ratio_short_long,
    })
}

fn window_slice(samples: &[MarketSample], since_ts: f64) -> &[MarketSample] {
    if samples.is_empty() {
        return samples;
    }
    let idx = samples
        .iter()
        .position(|sample| sample.captured_ts + f64::EPSILON >= since_ts)
        .unwrap_or(samples.len().saturating_sub(1));
    &samples[idx..]
}

fn window_change_pct(samples: &[MarketSample]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let first = samples.first()?.price;
    let last = samples.last()?.price;
    if first.abs() <= f64::EPSILON {
        None
    } else {
        Some(((last - first) / first) * 100.0)
    }
}

fn realized_volatility_pct(samples: &[MarketSample]) -> Option<f64> {
    if samples.len() < 3 {
        return None;
    }
    let mut returns = Vec::new();
    for window in samples.windows(2) {
        let prev = window[0].price;
        let next = window[1].price;
        if prev > 0.0 && next > 0.0 {
            returns.push((next / prev).ln());
        }
    }
    if returns.len() < 2 {
        return None;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / returns.len() as f64;
    Some(variance.sqrt() * 100.0)
}

fn drawdown_from_peak_pct(samples: &[MarketSample]) -> Option<f64> {
    let latest = samples.last()?;
    let peak = samples
        .iter()
        .map(|sample| sample.high_24h.unwrap_or(sample.price))
        .fold(0.0_f64, f64::max);
    if peak <= f64::EPSILON {
        None
    } else {
        Some(((latest.price / peak) - 1.0) * 100.0)
    }
}

fn average_volume(samples: &[MarketSample]) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0usize;
    for sample in samples {
        if let Some(volume) = sample.volume_24h
            && volume.is_finite()
            && volume >= 0.0
        {
            total += volume;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

async fn append_samples_to_file(path: &Path, samples: &[MarketSample]) -> Result<(), String> {
    if samples.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.map_err(|error| {
            format!(
                "failed to create market datalake directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
        .map_err(|error| {
            format!(
                "failed to open market datalake file {}: {error}",
                path.display()
            )
        })?;
    for sample in samples {
        let line = serde_json::to_string(sample)
            .map_err(|error| format!("failed to serialize market sample: {error}"))?;
        file.write_all(line.as_bytes()).await.map_err(|error| {
            format!(
                "failed to write market datalake file {}: {error}",
                path.display()
            )
        })?;
        file.write_all(b"\n").await.map_err(|error| {
            format!(
                "failed to write market datalake newline {}: {error}",
                path.display()
            )
        })?;
    }
    file.flush().await.map_err(|error| {
        format!(
            "failed to flush market datalake file {}: {error}",
            path.display()
        )
    })?;
    Ok(())
}

async fn retry_market_datalake_postgres<T, F, Fut>(mut operation: F) -> Result<T, String>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, String>>,
{
    let mut last_error = None;
    for attempt in 0..=MARKET_DATALAKE_POSTGRES_MIRROR_RETRIES {
        let result = tokio::time::timeout(
            Duration::from_secs(MARKET_DATALAKE_POSTGRES_OPERATION_TIMEOUT_SECONDS),
            operation(),
        )
        .await;
        match result {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(error)) => last_error = Some(error),
            Err(_) => {
                last_error = Some(format!(
                    "operation exceeded {} second deadline",
                    MARKET_DATALAKE_POSTGRES_OPERATION_TIMEOUT_SECONDS
                ));
            }
        }
        if attempt < MARKET_DATALAKE_POSTGRES_MIRROR_RETRIES {
            tokio::time::sleep(Duration::from_secs(
                MARKET_DATALAKE_POSTGRES_MIRROR_RETRY_DELAY_SECONDS
                    .saturating_mul(1_u64 << attempt),
            ))
            .await;
        }
    }
    Err(last_error.unwrap_or_else(|| "operation failed without an error".to_string()))
}

async fn load_samples_from_file(
    path: &Path,
    target_keys: &HashSet<String>,
    since_ts: f64,
) -> Result<Vec<MarketSample>, String> {
    let file = match fs::File::open(path).await {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(format!(
                "failed to open market datalake file {}: {error}",
                path.display()
            ));
        }
    };
    let mut reader = BufReader::new(file).lines();
    let mut loaded = Vec::new();
    while let Some(line) = reader.next_line().await.map_err(|error| {
        format!(
            "failed reading market datalake file {}: {error}",
            path.display()
        )
    })? {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(mut sample) = serde_json::from_str::<MarketSample>(&line) else {
            continue;
        };
        sample.exchange = normalize_exchange(&sample.exchange);
        sample.symbol = normalize_symbol(&sample.symbol);
        if sample.captured_ts + f64::EPSILON < since_ts {
            continue;
        }
        if target_keys.contains(&market_feature_key(&sample.exchange, &sample.symbol)) {
            loaded.push(sample);
        }
    }
    Ok(dedupe_samples(loaded))
}

fn dedupe_samples(samples: Vec<MarketSample>) -> Vec<MarketSample> {
    if samples.is_empty() {
        return samples;
    }
    let mut by_bucket: HashMap<i64, MarketSample> = HashMap::new();
    for sample in samples {
        match by_bucket.get(&sample.captured_bucket) {
            Some(previous) if previous.captured_ts >= sample.captured_ts => {}
            _ => {
                by_bucket.insert(sample.captured_bucket, sample);
            }
        }
    }
    let mut deduped = by_bucket.into_values().collect::<Vec<_>>();
    deduped.sort_by(|left, right| {
        left.captured_bucket
            .cmp(&right.captured_bucket)
            .then_with(|| {
                left.captured_ts
                    .partial_cmp(&right.captured_ts)
                    .unwrap_or(Ordering::Equal)
            })
    });
    deduped
}

fn dedupe_samples_by_symbol(samples: Vec<MarketSample>) -> Vec<MarketSample> {
    if samples.is_empty() {
        return samples;
    }
    let mut by_symbol_bucket: HashMap<(String, i64), MarketSample> = HashMap::new();
    for sample in samples {
        let key = (
            market_feature_key(&sample.exchange, &sample.symbol),
            sample.captured_bucket,
        );
        match by_symbol_bucket.get(&key) {
            Some(previous) if previous.captured_ts >= sample.captured_ts => {}
            _ => {
                by_symbol_bucket.insert(key, sample);
            }
        }
    }
    let mut deduped = by_symbol_bucket.into_values().collect::<Vec<_>>();
    deduped.sort_by(|left, right| {
        market_feature_key(&left.exchange, &left.symbol)
            .cmp(&market_feature_key(&right.exchange, &right.symbol))
            .then_with(|| left.captured_bucket.cmp(&right.captured_bucket))
            .then_with(|| {
                left.captured_ts
                    .partial_cmp(&right.captured_ts)
                    .unwrap_or(Ordering::Equal)
            })
    });
    deduped
}

async fn connect_postgres_client(
    dsn: &str,
) -> Result<tokio_postgres::Client, tokio_postgres::Error> {
    let (client, connection) = tokio_postgres::connect(dsn, NoTls).await?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            warn!(
                error = %error,
                "trading: market datalake Postgres connection terminated"
            );
        }
    });
    Ok(client)
}

async fn initialize_postgres_schema(dsn: &str) -> Result<(), String> {
    let client = connect_postgres_client(dsn).await.map_err(|error| {
        format!("failed to connect to Postgres for market datalake schema: {error}")
    })?;
    client
        .batch_execute(
            r#"
            SET client_min_messages TO WARNING;
            CREATE TABLE IF NOT EXISTS gail_market_snapshots (
                id BIGSERIAL PRIMARY KEY,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                captured_bucket BIGINT NOT NULL,
                captured_ts DOUBLE PRECISION NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                price_change_pct_24h DOUBLE PRECISION,
                volume_24h DOUBLE PRECISION,
                high_24h DOUBLE PRECISION,
                low_24h DOUBLE PRECISION,
                microstructure JSONB NOT NULL DEFAULT '{}'::jsonb,
                inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (exchange, symbol, captured_bucket)
            );
            CREATE INDEX IF NOT EXISTS idx_gail_market_snapshots_symbol_ts
                ON gail_market_snapshots (exchange, symbol, captured_ts DESC);
            CREATE INDEX IF NOT EXISTS idx_gail_market_snapshots_ts
                ON gail_market_snapshots (captured_ts DESC);
            ALTER TABLE gail_market_snapshots
                ADD COLUMN IF NOT EXISTS microstructure JSONB NOT NULL DEFAULT '{}'::jsonb;
            CREATE TABLE IF NOT EXISTS gail_market_snapshots_meta (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            "#,
        )
        .await
        .map_err(|error| format!("failed to initialise market datalake schema: {error}"))
}

async fn persist_samples_postgres(dsn: &str, samples: &[MarketSample]) -> Result<(), String> {
    if samples.is_empty() {
        return Ok(());
    }
    let client = connect_postgres_client(dsn).await.map_err(|error| {
        format!("failed to connect to Postgres for market snapshot persist: {error}")
    })?;
    let statement = client
        .prepare(
            r#"
            INSERT INTO gail_market_snapshots (
                exchange,
                symbol,
                captured_bucket,
                captured_ts,
                price,
                price_change_pct_24h,
                volume_24h,
                high_24h,
                low_24h,
                microstructure
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            ON CONFLICT (exchange, symbol, captured_bucket)
            DO UPDATE SET
                captured_ts = EXCLUDED.captured_ts,
                price = EXCLUDED.price,
                price_change_pct_24h = EXCLUDED.price_change_pct_24h,
                volume_24h = EXCLUDED.volume_24h,
                high_24h = EXCLUDED.high_24h,
                low_24h = EXCLUDED.low_24h,
                microstructure = EXCLUDED.microstructure
            "#,
        )
        .await
        .map_err(|error| format!("failed to prepare market datalake upsert statement: {error}"))?;
    for sample in samples {
        let microstructure = serde_json::to_value(&sample.microstructure)
            .map_err(|error| format!("failed to encode market microstructure: {error}"))?;
        client
            .execute(
                &statement,
                &[
                    &sample.exchange,
                    &sample.symbol,
                    &sample.captured_bucket,
                    &sample.captured_ts,
                    &sample.price,
                    &sample.price_change_pct_24h,
                    &sample.volume_24h,
                    &sample.high_24h,
                    &sample.low_24h,
                    &microstructure,
                ],
            )
            .await
            .map_err(|error| format!("failed to upsert market datalake sample: {error}"))?;
    }
    Ok(())
}

async fn load_samples_from_postgres(
    dsn: &str,
    exchange: &str,
    symbol: &str,
    since_ts: f64,
) -> Result<Vec<MarketSample>, String> {
    let client = connect_postgres_client(dsn).await.map_err(|error| {
        format!("failed to connect to Postgres for market snapshot load: {error}")
    })?;
    let rows = client
        .query(
            r#"
            SELECT
                exchange,
                symbol,
                captured_bucket,
                captured_ts,
                price,
                price_change_pct_24h,
                volume_24h,
                high_24h,
                low_24h,
                microstructure
            FROM gail_market_snapshots
            WHERE symbol = $1
              AND captured_ts >= $2
            ORDER BY captured_bucket ASC
            "#,
            &[&normalize_symbol(symbol), &since_ts],
        )
        .await
        .map_err(|error| format!("failed to query market datalake samples: {error}"))?;
    let target_exchange = normalize_exchange(exchange);
    Ok(rows
        .into_iter()
        .map(|row| MarketSample {
            exchange: normalize_exchange(row.get::<_, String>("exchange").as_str()),
            symbol: normalize_symbol(row.get::<_, String>("symbol").as_str()),
            captured_bucket: row.get("captured_bucket"),
            captured_ts: row.get("captured_ts"),
            price: row.get("price"),
            price_change_pct_24h: row.get("price_change_pct_24h"),
            volume_24h: row.get("volume_24h"),
            high_24h: row.get("high_24h"),
            low_24h: row.get("low_24h"),
            microstructure: serde_json::from_value(row.get("microstructure")).unwrap_or_default(),
        })
        .filter(|sample| sample.exchange == target_exchange)
        .collect::<Vec<_>>())
}

async fn prune_postgres_retention(dsn: &str, retention_days: u32) -> Result<u64, String> {
    let client = connect_postgres_client(dsn)
        .await
        .map_err(|error| format!("failed to connect to Postgres for retention prune: {error}"))?;
    let cutoff = now_ts() - (retention_days as f64 * 86_400.0);
    client
        .execute(
            "DELETE FROM gail_market_snapshots WHERE captured_ts < $1",
            &[&cutoff],
        )
        .await
        .map_err(|error| format!("failed to prune market datalake rows: {error}"))
}

fn metadata_path_for(data_file_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.meta.json", data_file_path.display()))
}

fn current_build_id() -> String {
    [
        "GAIL_BUILD_ID",
        "GAIL_CONTAINER_BUILD_ID",
        "GAIL_IMAGE_DIGEST",
        "CONTAINER_IMAGE_DIGEST",
    ]
    .iter()
    .find_map(|name| {
        env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
    .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

fn compute_bootstrap_reason(
    metadata: &MarketDataLakeMetadata,
    current_build_id: &str,
) -> Option<String> {
    if metadata.last_bootstrap_completed_at.is_none() {
        return Some("initial_startup".to_string());
    }
    if metadata.schema_version != MARKET_DATALAKE_SCHEMA_VERSION {
        return Some(format!(
            "schema_change:{}->{}",
            metadata.schema_version, MARKET_DATALAKE_SCHEMA_VERSION
        ));
    }
    if metadata.build_id.trim().is_empty() || metadata.build_id != current_build_id {
        return Some(format!(
            "build_change:{}->{}",
            metadata.build_id, current_build_id
        ));
    }
    if metadata.last_bootstrap_status.as_deref() == Some("failed") {
        return Some("retry_failed_bootstrap".to_string());
    }
    // A process restart can interrupt a bootstrap after it has persisted the
    // `running` marker.  There is no live task associated with that marker
    // after restart, so treat it as retryable instead of leaving the lake
    // permanently stale.
    if metadata.last_bootstrap_status.as_deref() == Some("running") {
        return Some("retry_interrupted_bootstrap".to_string());
    }
    None
}

async fn load_metadata_from_file(config: &MarketDataLakeConfig) -> Option<MarketDataLakeMetadata> {
    match fs::read_to_string(&config.metadata_path).await {
        Ok(raw) => serde_json::from_str::<MarketDataLakeMetadata>(&raw).ok(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            warn!(
                error = %error,
                path = %config.metadata_path.display(),
                "trading: failed to read market datalake metadata file"
            );
            None
        }
    }
}

async fn persist_metadata_file(
    path: &Path,
    metadata: &MarketDataLakeMetadata,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await.map_err(|error| {
            format!(
                "failed to create market datalake metadata directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let payload = serde_json::to_string_pretty(metadata)
        .map_err(|error| format!("failed to serialize market datalake metadata: {error}"))?;
    fs::write(path, payload).await.map_err(|error| {
        format!(
            "failed to write market datalake metadata file {}: {error}",
            path.display()
        )
    })
}

async fn persist_metadata_postgres(
    dsn: &str,
    metadata: &MarketDataLakeMetadata,
) -> Result<(), String> {
    let client = connect_postgres_client(dsn)
        .await
        .map_err(|error| format!("failed to connect to Postgres for metadata persist: {error}"))?;
    let value = serde_json::to_value(metadata)
        .map_err(|error| format!("failed to encode market datalake metadata: {error}"))?;
    client
        .execute(
            r#"
            INSERT INTO gail_market_snapshots_meta (key, value, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (key)
            DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = now()
            "#,
            &[&"bootstrap", &value],
        )
        .await
        .map_err(|error| format!("failed to persist market datalake metadata: {error}"))?;
    Ok(())
}

fn truncate(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn normalize_exchange(exchange: &str) -> String {
    let mut normalized = exchange.trim().to_ascii_lowercase();
    if let Some((head, _)) = normalized.split_once(':') {
        normalized = head.trim().to_string();
    }
    while normalized.ends_with(')') {
        let Some(start) = normalized.rfind(" (") else {
            break;
        };
        normalized.truncate(start);
        normalized = normalized.trim().to_string();
    }
    let compact = normalized
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    match compact.as_str() {
        "ku coin" => "kucoin".to_string(),
        "bit get" => "bitget".to_string(),
        "x t" => "xt".to_string(),
        _ => compact,
    }
}

fn normalize_symbol(symbol: &str) -> String {
    symbol.trim().to_ascii_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn feature_key_is_case_normalized() {
        assert_eq!(
            market_feature_key("Binance", "bnb/usdt"),
            "binance|BNB/USDT"
        );
        assert_eq!(
            market_feature_key("Bitget : NEUTRAL (Indexing 6 coins)", "doge/usdt"),
            "bitget|DOGE/USDT"
        );
    }

    #[tokio::test]
    async fn ingest_deduplicates_bucketed_samples() {
        let temp = tempdir().expect("temp dir");
        let mut config = TradingConfig {
            market_datalake_file_path: temp.path().join("lake.jsonl").to_string_lossy().to_string(),
            market_datalake_bucket_seconds: 60,
            ..TradingConfig::default()
        };
        config.normalize();
        let lake = MarketDataLake::new(&config, None, SharedTradingState::new(100, 100)).await;
        let base_ts = 1_717_430_000.0;
        let snapshots = vec![
            MarketSnapshot {
                exchange: "binance".to_string(),
                symbol: "BNB/USDT".to_string(),
                price: 620.0,
                price_change_pct_1h: None,
                price_change_pct_24h: Some(2.0),
                volume_24h: Some(1_000_000.0),
                volume_change_pct: None,
                high_24h: Some(625.0),
                low_24h: Some(610.0),
                fetched_at: base_ts,
                microstructure: Default::default(),
            },
            MarketSnapshot {
                exchange: "Binance".to_string(),
                symbol: "bnb/usdt".to_string(),
                price: 621.5,
                price_change_pct_1h: None,
                price_change_pct_24h: Some(2.3),
                volume_24h: Some(1_010_000.0),
                volume_change_pct: None,
                high_24h: Some(626.0),
                low_24h: Some(611.0),
                fetched_at: base_ts + 10.0,
                microstructure: Default::default(),
            },
        ];

        let summary = lake.ingest_snapshots(&snapshots).await;
        assert_eq!(summary.received, 2);
        assert_eq!(summary.persisted, 1);
        assert_eq!(summary.deduplicated, 1);
        assert!(summary.file_error.is_none());
    }

    #[tokio::test]
    async fn computes_history_features_from_incremental_samples() {
        let temp = tempdir().expect("temp dir");
        let mut config = TradingConfig {
            market_datalake_file_path: temp.path().join("lake.jsonl").to_string_lossy().to_string(),
            market_datalake_bucket_seconds: 60,
            market_datalake_short_window_minutes: 30,
            market_datalake_mid_window_hours: 2,
            market_datalake_long_window_days: 2,
            market_datalake_min_samples: 4,
            ..TradingConfig::default()
        };
        config.normalize();
        let lake = MarketDataLake::new(&config, None, SharedTradingState::new(100, 100)).await;

        let base_ts = 1_717_430_000.0;
        let mut snapshots = Vec::new();
        for idx in 0..12 {
            let price = 600.0 + idx as f64 * 2.0;
            snapshots.push(MarketSnapshot {
                exchange: "binance".to_string(),
                symbol: "BNB/USDT".to_string(),
                price,
                price_change_pct_1h: None,
                price_change_pct_24h: Some((price - 600.0) / 600.0 * 100.0),
                volume_24h: Some(1_000_000.0 + idx as f64 * 50_000.0),
                volume_change_pct: None,
                high_24h: Some(price + 3.0),
                low_24h: Some(price - 3.0),
                fetched_at: base_ts + idx as f64 * 300.0,
                microstructure: Default::default(),
            });
        }
        let summary = lake.ingest_snapshots(&snapshots).await;
        assert_eq!(summary.persisted, 12);

        let features = lake.features_for_symbol("binance", "BNB/USDT").await;
        let features = features.expect("features");
        assert!(features.samples >= 4);
        assert!(features.momentum_short_pct.unwrap_or(0.0) > 0.0);
        assert!(features.momentum_mid_pct.unwrap_or(0.0) > 0.0);
        assert!(features.volume_ratio_short_long.unwrap_or(1.0) > 0.0);
    }

    #[tokio::test]
    async fn bootstrap_is_required_when_metadata_is_missing() {
        let temp = tempdir().expect("temp dir");
        let mut config = TradingConfig {
            market_datalake_file_path: temp.path().join("lake.jsonl").to_string_lossy().to_string(),
            ..TradingConfig::default()
        };
        config.normalize();
        let lake = MarketDataLake::new(&config, None, SharedTradingState::new(100, 100)).await;
        let reason = lake.bootstrap_required_reason().await;
        assert_eq!(reason.as_deref(), Some("initial_startup"));
    }

    #[tokio::test]
    async fn restart_recovers_market_history_from_the_file_lake() {
        let temp = tempdir().expect("temp dir");
        let mut config = TradingConfig {
            market_datalake_file_path: temp.path().join("lake.jsonl").to_string_lossy().to_string(),
            market_datalake_bucket_seconds: 60,
            market_datalake_short_window_minutes: 30,
            market_datalake_mid_window_hours: 2,
            market_datalake_long_window_days: 1,
            market_datalake_min_samples: 4,
            ..TradingConfig::default()
        };
        config.normalize();
        let first = MarketDataLake::new(&config, None, SharedTradingState::new(100, 100)).await;
        let base_ts = now_ts() - 3_600.0;
        let snapshots = (0..12)
            .map(|idx| {
                let price = 600.0 + idx as f64 * 2.0;
                MarketSnapshot {
                    exchange: "binance".to_string(),
                    symbol: "BNB/USDT".to_string(),
                    price,
                    price_change_pct_1h: None,
                    price_change_pct_24h: Some((price - 600.0) / 600.0 * 100.0),
                    volume_24h: Some(1_000_000.0),
                    volume_change_pct: None,
                    high_24h: Some(price + 1.0),
                    low_24h: Some(price - 1.0),
                    fetched_at: base_ts + idx as f64 * 300.0,
                    microstructure: Default::default(),
                }
            })
            .collect::<Vec<_>>();
        let summary = first.ingest_snapshots(&snapshots).await;
        assert_eq!(summary.persisted, snapshots.len());
        drop(first);

        let restarted = MarketDataLake::new(&config, None, SharedTradingState::new(100, 100)).await;
        let features = restarted
            .features_for_symbol("binance", "BNB/USDT")
            .await
            .expect("file-backed history should be recovered after restart");
        assert!(features.samples >= 4);
        assert!(features.momentum_mid_pct.unwrap_or_default() > 0.0);
    }

    #[tokio::test]
    async fn full_postgres_queue_does_not_block_file_persistence() {
        let temp = tempdir().expect("temp dir");
        let mut trading_config = TradingConfig {
            market_datalake_file_path: temp.path().join("lake.jsonl").to_string_lossy().to_string(),
            market_datalake_bucket_seconds: 60,
            ..TradingConfig::default()
        };
        trading_config.normalize();
        let config = Arc::new(MarketDataLakeConfig::from_trading_config(
            &trading_config,
            Some("postgres://unavailable.invalid/gail".to_string()),
        ));
        let (sender, _receiver) = mpsc::channel(MARKET_DATALAKE_POSTGRES_MIRROR_QUEUE_CAPACITY);
        for _ in 0..MARKET_DATALAKE_POSTGRES_MIRROR_QUEUE_CAPACITY {
            sender
                .try_send(PostgresMirrorTask::Persist(Vec::new()))
                .expect("fill bounded worker queue");
        }
        let lake = MarketDataLake {
            config,
            state: Arc::new(Mutex::new(MarketDataLakeState::default())),
            postgres_mirror_tx: Some(sender),
            log_state: SharedTradingState::new(100, 100),
        };
        let snapshot = MarketSnapshot {
            exchange: "binance".to_string(),
            symbol: "ETH/USDT".to_string(),
            price: 3_000.0,
            price_change_pct_1h: None,
            price_change_pct_24h: Some(1.0),
            volume_24h: Some(1_000.0),
            volume_change_pct: None,
            high_24h: Some(3_100.0),
            low_24h: Some(2_900.0),
            fetched_at: now_ts(),
            microstructure: Default::default(),
        };

        let summary =
            tokio::time::timeout(Duration::from_secs(2), lake.ingest_snapshots(&[snapshot]))
                .await
                .expect("ingestion must not wait for the full Postgres queue");
        assert_eq!(summary.persisted, 1);
        assert!(summary.file_error.is_none());
        assert!(
            summary
                .postgres_error
                .as_deref()
                .is_some_and(|error| error.contains("queue is full"))
        );
        let contents = fs::read_to_string(&lake.config.file_path)
            .await
            .expect("durable file lake");
        assert_eq!(contents.lines().count(), 1);
    }

    #[tokio::test]
    async fn postgres_history_never_replaces_a_file_lake_bucket() {
        let temp = tempdir().expect("temp dir");
        let trading_config = TradingConfig {
            market_datalake_file_path: temp.path().join("lake.jsonl").to_string_lossy().to_string(),
            ..TradingConfig::default()
        };
        let lake = MarketDataLake {
            config: Arc::new(MarketDataLakeConfig::from_trading_config(
                &trading_config,
                None,
            )),
            state: Arc::new(Mutex::new(MarketDataLakeState::default())),
            postgres_mirror_tx: None,
            log_state: SharedTradingState::new(100, 100),
        };
        let file_sample = MarketSample {
            exchange: "binance".to_string(),
            symbol: "BTC/USDT".to_string(),
            captured_ts: 100.0,
            captured_bucket: 1,
            price: 60_000.0,
            ..MarketSample::default()
        };
        let stale_mirror_copy = MarketSample {
            captured_ts: 101.0,
            price: 61_000.0,
            ..file_sample.clone()
        };
        lake.merge_samples_into_cache(&[file_sample]).await;
        lake.merge_mirror_samples_into_cache(&[stale_mirror_copy])
            .await;

        let state = lake.state.lock().await;
        let cached = state
            .samples_by_symbol
            .get("binance|BTC/USDT")
            .and_then(|items| items.front())
            .expect("cached file sample");
        assert_eq!(cached.price, 60_000.0);
    }
}
