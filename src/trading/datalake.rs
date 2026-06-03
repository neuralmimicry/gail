//! Incremental market datalake for trading decisions.
//!
//! Gail ingests live OctoBot snapshots each cycle, stores them durably
//! (JSONL + optional Postgres), and derives historical features without
//! re-pulling full historical datasets on every evaluation.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tokio::{
    fs::{self, OpenOptions},
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::Mutex,
};
use tokio_postgres::NoTls;
use tracing::{debug, warn};

use super::{config::TradingConfig, octobot::MarketSnapshot};

const POSTGRES_PRUNE_INTERVAL_SECONDS: f64 = 3_600.0;
const MARKET_DATALAKE_SCHEMA_VERSION: u32 = 1;

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
    pub async fn new(trading_config: &TradingConfig, postgres_dsn: Option<String>) -> Self {
        let config = Arc::new(MarketDataLakeConfig::from_trading_config(
            trading_config,
            postgres_dsn,
        ));
        if let Some(dsn) = config.postgres_dsn.as_deref()
            && let Err(error) = initialize_postgres_schema(dsn).await
        {
            warn!(
                error = %error,
                "trading: market datalake Postgres schema init failed; continuing with file persistence"
            );
        }
        let current_build_id = current_build_id();
        let metadata = load_metadata(&config).await.unwrap_or_default();
        let bootstrap_reason = compute_bootstrap_reason(&metadata, &current_build_id);
        Self {
            config,
            state: Arc::new(Mutex::new(MarketDataLakeState {
                samples_by_symbol: HashMap::new(),
                loaded_symbols: HashSet::new(),
                last_postgres_prune_at: 0.0,
                metadata,
                current_build_id,
                bootstrap_required: bootstrap_reason.is_some(),
                bootstrap_reason,
            })),
        }
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
        if let Err(error) = persist_metadata(&self.config, &metadata).await {
            warn!(
                error = %error,
                "trading: failed to persist market datalake bootstrap completion metadata"
            );
        }
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
        if let Err(error) = persist_metadata(&self.config, &metadata).await {
            warn!(
                error = %error,
                "trading: failed to persist market datalake bootstrap-start metadata"
            );
        }
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
        if let Err(error) = persist_metadata(&self.config, &metadata).await {
            warn!(
                error = %error,
                "trading: failed to persist market datalake bootstrap failure metadata"
            );
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
            persisted: samples.len(),
            deduplicated: received.saturating_sub(samples.len()),
            file_error: None,
            postgres_error: None,
        };

        if samples.is_empty() {
            return summary;
        }

        self.merge_samples_into_cache(&samples).await;

        if let Err(error) = append_samples_to_file(&self.config.file_path, &samples).await {
            warn!(
                error = %error,
                path = %self.config.file_path.display(),
                "trading: failed to append market datalake file records"
            );
            summary.file_error = Some(error);
        }

        if let Some(dsn) = self.config.postgres_dsn.as_deref() {
            if let Err(error) = persist_samples_postgres(dsn, &samples).await {
                warn!(
                    error = %error,
                    "trading: failed to persist market datalake records to Postgres"
                );
                summary.postgres_error = Some(error);
            } else {
                self.maybe_prune_postgres().await;
            }
        }

        summary
    }

    pub async fn features_for_snapshots(
        &self,
        snapshots: &[MarketSnapshot],
    ) -> HashMap<String, MarketHistoricalFeatures> {
        let mut out = HashMap::new();
        for snapshot in snapshots {
            if let Some(features) = self
                .features_for_symbol(&snapshot.exchange, &snapshot.symbol)
                .await
            {
                out.insert(
                    market_feature_key(&snapshot.exchange, &snapshot.symbol),
                    features,
                );
            }
        }
        out
    }

    pub async fn features_for_symbol(
        &self,
        exchange: &str,
        symbol: &str,
    ) -> Option<MarketHistoricalFeatures> {
        self.ensure_symbol_loaded(exchange, symbol).await;
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

    async fn ensure_symbol_loaded(&self, exchange: &str, symbol: &str) {
        let key = market_feature_key(exchange, symbol);
        let should_load = {
            let mut state = self.state.lock().await;
            state.loaded_symbols.insert(key.clone())
        };
        if !should_load {
            return;
        }

        let since_ts = now_ts() - self.config.retention_seconds();
        let mut loaded = Vec::new();

        if let Some(dsn) = self.config.postgres_dsn.as_deref() {
            match load_samples_from_postgres(dsn, exchange, symbol, since_ts).await {
                Ok(samples) => loaded.extend(samples),
                Err(error) => {
                    warn!(
                        error = %error,
                        exchange = %exchange,
                        symbol = %symbol,
                        "trading: failed to load market history from Postgres"
                    );
                }
            }
        }

        match load_samples_from_file(&self.config.file_path, exchange, symbol, since_ts).await {
            Ok(samples) => loaded.extend(samples),
            Err(error) => {
                warn!(
                    error = %error,
                    path = %self.config.file_path.display(),
                    exchange = %exchange,
                    symbol = %symbol,
                    "trading: failed to load market history from file"
                );
            }
        }

        let loaded = dedupe_samples(loaded);
        if !loaded.is_empty() {
            self.merge_samples_into_cache(&loaded).await;
            debug!(
                exchange = %exchange,
                symbol = %symbol,
                samples = loaded.len(),
                "trading: loaded market datalake history for symbol"
            );
        }
    }

    async fn merge_samples_into_cache(&self, samples: &[MarketSample]) {
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
                if sample.captured_ts >= existing.captured_ts {
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
        if let Err(error) = prune_postgres_retention(dsn, self.config.retention_days).await {
            warn!(
                error = %error,
                "trading: failed to prune old market datalake rows from Postgres"
            );
        }
    }
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

async fn load_samples_from_file(
    path: &Path,
    exchange: &str,
    symbol: &str,
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
    let target_key = market_feature_key(exchange, symbol);
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
        let Ok(sample) = serde_json::from_str::<MarketSample>(&line) else {
            continue;
        };
        if sample.captured_ts + f64::EPSILON < since_ts {
            continue;
        }
        if market_feature_key(&sample.exchange, &sample.symbol) == target_key {
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
                inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (exchange, symbol, captured_bucket)
            );
            CREATE INDEX IF NOT EXISTS idx_gail_market_snapshots_symbol_ts
                ON gail_market_snapshots (exchange, symbol, captured_ts DESC);
            CREATE INDEX IF NOT EXISTS idx_gail_market_snapshots_ts
                ON gail_market_snapshots (captured_ts DESC);
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
                low_24h
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            ON CONFLICT (exchange, symbol, captured_bucket)
            DO UPDATE SET
                captured_ts = EXCLUDED.captured_ts,
                price = EXCLUDED.price,
                price_change_pct_24h = EXCLUDED.price_change_pct_24h,
                volume_24h = EXCLUDED.volume_24h,
                high_24h = EXCLUDED.high_24h,
                low_24h = EXCLUDED.low_24h
            "#,
        )
        .await
        .map_err(|error| format!("failed to prepare market datalake upsert statement: {error}"))?;
    for sample in samples {
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
                low_24h
            FROM gail_market_snapshots
            WHERE exchange = $1
              AND symbol = $2
              AND captured_ts >= $3
            ORDER BY captured_bucket ASC
            "#,
            &[
                &normalize_exchange(exchange),
                &normalize_symbol(symbol),
                &since_ts,
            ],
        )
        .await
        .map_err(|error| format!("failed to query market datalake samples: {error}"))?;
    Ok(rows
        .into_iter()
        .map(|row| MarketSample {
            exchange: row.get("exchange"),
            symbol: row.get("symbol"),
            captured_bucket: row.get("captured_bucket"),
            captured_ts: row.get("captured_ts"),
            price: row.get("price"),
            price_change_pct_24h: row.get("price_change_pct_24h"),
            volume_24h: row.get("volume_24h"),
            high_24h: row.get("high_24h"),
            low_24h: row.get("low_24h"),
        })
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
    None
}

async fn load_metadata(config: &MarketDataLakeConfig) -> Option<MarketDataLakeMetadata> {
    if let Some(dsn) = config.postgres_dsn.as_deref() {
        match load_metadata_from_postgres(dsn).await {
            Ok(Some(metadata)) => return Some(metadata),
            Ok(None) => {}
            Err(error) => {
                warn!(
                    error = %error,
                    "trading: failed to load market datalake metadata from Postgres"
                );
            }
        }
    }

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

async fn persist_metadata(
    config: &MarketDataLakeConfig,
    metadata: &MarketDataLakeMetadata,
) -> Result<(), String> {
    persist_metadata_file(&config.metadata_path, metadata).await?;
    if let Some(dsn) = config.postgres_dsn.as_deref() {
        persist_metadata_postgres(dsn, metadata).await?;
    }
    Ok(())
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

async fn load_metadata_from_postgres(dsn: &str) -> Result<Option<MarketDataLakeMetadata>, String> {
    let client = connect_postgres_client(dsn)
        .await
        .map_err(|error| format!("failed to connect to Postgres for metadata load: {error}"))?;
    let row = client
        .query_opt(
            "SELECT value FROM gail_market_snapshots_meta WHERE key = $1",
            &[&"bootstrap"],
        )
        .await
        .map_err(|error| format!("failed to query market datalake metadata: {error}"))?;
    let Some(row) = row else {
        return Ok(None);
    };
    let value: serde_json::Value = row.get("value");
    Ok(serde_json::from_value::<MarketDataLakeMetadata>(value).ok())
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
    exchange.trim().to_ascii_lowercase()
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
        let lake = MarketDataLake::new(&config, None).await;
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
        let lake = MarketDataLake::new(&config, None).await;

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
        let lake = MarketDataLake::new(&config, None).await;
        let reason = lake.bootstrap_required_reason().await;
        assert_eq!(reason.as_deref(), Some("initial_startup"));
    }
}
