/// Gail Crypto Trading Bridge — main module.
///
/// Provides `TradingBridge`, a non-blocking background service that:
///  1. Fetches live market data from OctoBot
///  2. Gathers research context from Refiner
///  3. Races bounded AI providers to a round deadline/quorum (TradingAdvisor)
///  4. Applies Type-2 fuzzy logic (FuzzyEngine)
///  5. Blends signals and applies fixed-horizon outcome calibration (DecisionEngine)
///  6. Gates fee/slippage economics and immediately reprices the exact venue
///  7. Persists an idempotency lease, then executes through OctoBot
///  8. Resolves fee-adjusted markouts for future calibration
///  9. Logs activity and atomically persists state (SharedTradingState)
///
/// The bridge is entirely non-blocking and runs in its own tokio task.
/// All HTTP handlers access state through `SharedTradingState` (Arc<Mutex<>>).
pub mod advisor;
pub mod backtest;
pub mod config;
pub mod datalake;
pub mod decision;
pub mod economics;
pub mod fuzzy;
pub mod octobot;
pub mod outcomes;
pub mod qualification;
pub mod quant;
pub mod quantitative;
pub mod refiner;
pub mod state;

#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use futures::{StreamExt, stream};
use serde::Serialize;
use serde_json::json;
use tokio::sync::oneshot;
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::{
    adaptive_schema::{self, AdaptiveApiRegistry},
    orchestration::GailService,
};
use advisor::TradingAdvisor;
use backtest::BacktestEngine;
use config::{TradingConfig, TradingConfigOverride};
use datalake::{
    MarketDataLake, MarketDataLakeBootstrapReport, MarketHistoricalFeatures, market_feature_key,
};
use decision::{DecisionEngine, TradeDecision};
use economics::adverse_reprice_drift_bps;
use fuzzy::{FuzzyEngine, FuzzyInputs};
use octobot::{
    MarketSnapshot, OCTOBOT_MARKET_SNAPSHOT_HARD_LIMIT, OctobotClient, OctobotExchange,
    OctobotLogEntry, OctobotPortfolio,
};
use outcomes::TradeMarkout;
use qualification::PaperQualificationPolicy;
use quant::{QuantMode, evaluate_symbol as evaluate_quant_symbol, evaluate_universe};
use quantitative::backtest::{NativeBacktestReport, NativeQuantBacktester};
use quantitative::sleeves::{evaluate_sleeves_async, market_key as sleeve_market_key};
use quantitative::telemetry::reprice_net_edge;
use refiner::RefinerClient;
use state::{ExecutedTrade, ExecutionIntentClaim, SharedTradingState, TradeAction, TradingState};

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn filter_fresh_market_snapshots(
    snapshots: Vec<MarketSnapshot>,
    ttl_seconds: f64,
) -> Vec<MarketSnapshot> {
    let now = now_ts();
    let ttl = ttl_seconds.max(1.0);
    let mut latest = HashMap::<String, MarketSnapshot>::new();
    for snapshot in snapshots {
        if !snapshot.price.is_finite()
            || snapshot.price <= 0.0
            || !snapshot.fetched_at.is_finite()
            || snapshot.fetched_at <= 0.0
            || snapshot.fetched_at > now + 5.0
            || now - snapshot.fetched_at > ttl
        {
            continue;
        }
        let key = format!(
            "{}|{}",
            snapshot.exchange.to_ascii_lowercase(),
            snapshot.symbol.to_ascii_uppercase()
        );
        if latest
            .get(&key)
            .is_none_or(|previous| snapshot.fetched_at > previous.fetched_at)
        {
            latest.insert(key, snapshot);
        }
    }
    latest.into_values().collect()
}

const BACKTEST_AUTOTUNE_BASELINE_RUNS: usize = 4;
const BACKTEST_AUTOTUNE_VALIDATION_RUNS: usize = 2;
const BACKTEST_AUTOTUNE_MIN_MEAN_IMPROVEMENT_PCT: f64 = 0.35;
const BACKTEST_AUTOTUNE_MAX_MEDIAN_REGRESSION_PCT: f64 = 0.25;
const BACKTEST_AUTOTUNE_COOLDOWN_SECONDS: f64 = 3_600.0;
const MAX_EXECUTIONS_PER_EVALUATION: usize = 8;

// ---------------------------------------------------------------------------
// Handle for controlling the background task
// ---------------------------------------------------------------------------

struct TradingBridgeRuntime {
    _shutdown_tx: oneshot::Sender<()>,
}

pub struct TradingBridgeHandle {
    _runtime: Arc<TradingBridgeRuntime>,
}

// ---------------------------------------------------------------------------
// TradingBridge — the main entry point shared between the background loop
// and the HTTP route handlers.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TradingBridge {
    pub state: SharedTradingState,
    pub config: Arc<TradingConfig>,
    _runtime: Arc<TradingBridgeRuntime>,
}

impl TradingBridge {
    /// Create a new bridge and immediately start the background evaluation loop.
    /// Returns the bridge handle (for HTTP route access) and a control handle
    /// that stops the loop when dropped.
    pub async fn start(config: TradingConfig, service: GailService) -> (Self, TradingBridgeHandle) {
        let state = SharedTradingState::new(config.log_ring_size, config.trade_ring_size);

        // Restore persisted state if available.
        let data_path = PathBuf::from(&config.data_path);
        state.restore(&data_path).await;

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let runtime = Arc::new(TradingBridgeRuntime {
            _shutdown_tx: shutdown_tx,
        });
        let config = Arc::new(config);
        let bridge = Self {
            state: state.clone(),
            config: config.clone(),
            _runtime: runtime.clone(),
        };
        let loop_config = config.clone();
        let loop_state = state.clone();
        let loop_service = service.clone();
        tokio::spawn(async move {
            run_evaluation_loop(loop_config, loop_state, loop_service, shutdown_rx).await;
        });

        (bridge, TradingBridgeHandle { _runtime: runtime })
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

// ---------------------------------------------------------------------------
// Background evaluation loop
// ---------------------------------------------------------------------------

async fn run_evaluation_loop(
    config: Arc<TradingConfig>,
    state: SharedTradingState,
    service: GailService,
    mut shutdown: oneshot::Receiver<()>,
) {
    let mut restored_api_schema = {
        let state = state.0.lock().await;
        state.api_schema.clone()
    };
    let global_octobot_schema = adaptive_schema::api_snapshot("octobot").await;
    if adaptive_schema_has_observations(&global_octobot_schema) {
        restored_api_schema.merge(global_octobot_schema);
    }
    {
        let mut state = state.0.lock().await;
        state.api_schema = restored_api_schema.clone();
    }
    let mut restored_registry = AdaptiveApiRegistry::default();
    restored_registry
        .apis
        .insert("octobot".to_string(), restored_api_schema.clone());
    adaptive_schema::merge_snapshot(restored_registry).await;
    let octobot = OctobotClient::new_with_schema(
        &config.octobot_base_url,
        config.octobot_password.as_deref(),
        config.octobot_timeout_seconds,
        restored_api_schema,
    );
    let restored_exchange_circuit = {
        let current = state.0.lock().await;
        current.exchange_circuit.clone()
    };
    octobot
        .restore_exchange_circuit(restored_exchange_circuit)
        .await;
    let refiner = RefinerClient::new(
        &config.refiner_base_url,
        config.refiner_api_token.as_deref(),
        config.refiner_timeout_seconds,
    );
    let fuzzy_engine = FuzzyEngine::new();
    let postgres_dsn = service.config().storage.postgres_dsn.clone();
    let advisor = TradingAdvisor::new(
        service,
        config.advisor_timeout_seconds,
        config.advisor_round_timeout_seconds,
        config.advisor_early_quorum,
        config.advisory_candidate_limit,
    );
    let decision_engine = DecisionEngine::new(config.fuzzy_weight);
    let data_path = PathBuf::from(&config.data_path);
    let market_data_lake = if config.market_datalake_enabled {
        Some(MarketDataLake::new(&config, postgres_dsn).await)
    } else {
        None
    };
    let mut pending_datalake_bootstrap_reason = if config.market_datalake_bootstrap_enabled {
        if let Some(lake) = market_data_lake.as_ref() {
            lake.bootstrap_required_reason().await
        } else {
            None
        }
    } else {
        None
    };
    let mut last_datalake_bootstrap_attempt_ts: f64 = 0.0;

    // Initialise or restore the persistent shadow-to-quant controller before
    // any advisory request can be made. The explicit markers are intentionally
    // stable so operators and log automation can identify the active mode.
    if config.quant_shadow_enabled {
        let (marker, mode, parameter_id, pending, resolved) = {
            let mut current = state.0.lock().await;
            let newly_initialized = current.quant_migration.initialize(now_ts());
            let marker = match (&current.quant_migration.mode, newly_initialized) {
                (QuantMode::Primary, _) => "QUANT_PRIMARY_RESTORED",
                (QuantMode::Shadow, true) => "QUANT_SHADOW_INITIATED",
                (QuantMode::Shadow, false) => "QUANT_SHADOW_RESTORED",
            };
            let context = json!({
                "mode": current.quant_migration.mode.as_str(),
                "active_parameter_id": current.quant_migration.active_parameter_id,
                "pending_evaluations": current.quant_migration.pending.len(),
                "resolved_evaluations": current.quant_migration.resolved.len(),
                "migration_min_samples": config.quant_migration_min_samples,
                "migration_min_actionable_samples": config.quant_migration_min_actionable_samples,
                "migration_min_outperformance_bps": config.quant_migration_min_outperformance_bps,
                "migration_required_streak": config.quant_migration_required_streak,
            });
            current.log("info", "quant_migration", marker, context);
            (
                marker,
                current.quant_migration.mode.as_str(),
                current.quant_migration.active_parameter_id.clone(),
                current.quant_migration.pending.len(),
                current.quant_migration.resolved.len(),
            )
        };
        info!(mode, parameter_id, pending, resolved, "{marker}");
        state.persist(&data_path).await;
    }

    // Initial OctoBot login.
    if let Err(err) = octobot.login().await {
        warn!("trading: OctoBot login failed at startup: {}", err);
        state
            .log_warn("startup", format!("OctoBot login failed: {err}"))
            .await;
    } else {
        state
            .log(
                "info",
                "startup",
                "Trading bridge started with cost-aware single-authority execution",
                json!({
                    "execution_authority": config.execution_authority,
                    "strict_exchange_selection": config.strict_exchange_selection,
                    "advisor_round_timeout_seconds": config.advisor_round_timeout_seconds,
                    "advisor_early_quorum": config.advisor_early_quorum,
                    "market_snapshot_ttl_seconds": config.market_snapshot_ttl_seconds,
                    "advisory_ttl_seconds": config.advisory_ttl_seconds,
                    "round_trip_cost_bps": 2.0 * (config.estimated_fee_bps + config.estimated_slippage_bps),
                    "minimum_net_edge_bps": config.minimum_net_edge_bps,
                    "markout_horizon_seconds": config.markout_horizon_seconds,
                }),
            )
            .await;
    }

    if let Some(reason) = pending_datalake_bootstrap_reason.clone()
        && let Some(lake) = market_data_lake.as_ref()
    {
        let bootstrap_ok =
            run_market_datalake_bootstrap(&config, &state, &octobot, lake, &reason).await;
        last_datalake_bootstrap_attempt_ts = now_ts();
        if bootstrap_ok {
            pending_datalake_bootstrap_reason = None;
        }
    }

    let eval_interval = Duration::from_secs(config.evaluation_interval_seconds);
    let mut tick = interval(eval_interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Backtest scheduling: track when we last ran a backtest.
    let backtest_engine = if config.backtesting_enabled {
        Some(BacktestEngine::new(
            OctobotClient::new(
                &config.octobot_base_url,
                config.octobot_password.as_deref(),
                config.octobot_timeout_seconds,
            ),
            config.backtest_profitability_threshold,
        ))
    } else {
        None
    };
    let mut last_backtest_ts: f64 = 0.0;
    let mut last_discovery_ts: f64 = 0.0;
    let mut last_pruning_ts: f64 = 0.0;
    info!(
        interval_seconds = config.evaluation_interval_seconds,
        backtesting_enabled = config.backtesting_enabled,
        "trading: evaluation loop started"
    );

    loop {
        tokio::select! {
            _ = tick.tick() => {
                let paused = {
                    let s = state.0.lock().await;
                    s.paused
                };
                if paused {
                    debug!("trading: evaluation skipped — bridge is paused");
                    continue;
                }
                if let Some(reason) = pending_datalake_bootstrap_reason.clone() {
                    let due = now_ts() - last_datalake_bootstrap_attempt_ts
                        >= config.market_datalake_bootstrap_retry_seconds as f64;
                    if due {
                        if let Some(lake) = market_data_lake.as_ref() {
                            let bootstrap_ok = run_market_datalake_bootstrap(
                                &config, &state, &octobot, lake, &reason,
                            ).await;
                            last_datalake_bootstrap_attempt_ts = now_ts();
                            if bootstrap_ok {
                                pending_datalake_bootstrap_reason = None;
                            }
                        } else {
                            pending_datalake_bootstrap_reason = None;
                        }
                    }
                }
                run_single_evaluation(
                    &config,
                    &state,
                    EvaluationServices {
                        octobot: &octobot,
                        refiner: &refiner,
                        fuzzy_engine: &fuzzy_engine,
                        advisor: &advisor,
                        decision_engine: &decision_engine,
                        market_data_lake: market_data_lake.as_ref(),
                        data_path: &data_path,
                    },
                ).await;
                {
                    let mut current = state.0.lock().await;
                    current.exchange_circuit = octobot.exchange_circuit_snapshot().await;
                }
                // The evaluation snapshot is durable before slower maintenance
                // cycles begin. Individual filled orders also persist eagerly.
                state.persist(&data_path).await;

                if config.token_discovery_enabled {
                    let due = now_ts() - last_discovery_ts >= config.token_discovery_interval_seconds as f64;
                    if due {
                        run_non_portfolio_discovery_cycle(
                            &config,
                            &state,
                            &octobot,
                            &refiner,
                            &fuzzy_engine,
                            &advisor,
                            &decision_engine,
                            market_data_lake.as_ref(),
                        ).await;
                        last_discovery_ts = now_ts();
                    }
                }

                if config.portfolio_pruning_enabled {
                    let due = now_ts() - last_pruning_ts >= config.portfolio_pruning_interval_seconds as f64;
                    if due {
                        run_portfolio_pruning_cycle(
                            &config,
                            &state,
                            &octobot,
                            &refiner,
                            &fuzzy_engine,
                            &advisor,
                            &decision_engine,
                            market_data_lake.as_ref(),
                        ).await;
                        last_pruning_ts = now_ts();
                    }
                }

                // --- Periodic backtest ---
                if config.backtesting_enabled {
                    let due = now_ts() - last_backtest_ts >= config.backtest_interval_seconds as f64;
                    let already_running = {
                        let mut current = state.0.lock().await;
                        if due && !current.backtest_in_progress {
                            current.backtest_in_progress = true;
                            false
                        } else {
                            current.backtest_in_progress
                        }
                    };
                    if due && !already_running {
                        info!("trading: running periodic backtest");
                        state.log_info("backtest", "Starting periodic backtesting run").await;
                        let native_enabled = config.quantitative.enabled
                            && config.quantitative.native_backtest_enabled;
                        if native_enabled {
                            if let Some(lake) = market_data_lake.as_ref() {
                                let mut native_config = config.quantitative.native_backtest.clone();
                                // Live fee assumptions are the single source of truth for
                                // both execution gating and scheduled replay.
                                native_config.fee_bps = config.estimated_fee_bps;
                                native_config.slippage_bps = config.estimated_slippage_bps;
                                let frames = lake.native_backtest_frames(&native_config).await;
                                match frames {
                                    Ok(frames) => {
                                        let parameter_sets = {
                                            let current = state.0.lock().await;
                                            current.quant_migration.parameter_sets.clone()
                                        };
                                        match NativeQuantBacktester::new(native_config.clone())
                                            .with_portfolio_config(config.quantitative.portfolio.clone())
                                            .run_async(frames, parameter_sets)
                                            .await
                                        {
                                            Ok(report) => {
                                                let summary = native_backtest_summary(
                                                    &report,
                                                    &native_config,
                                                );
                                                let should_pause = {
                                                    let mut current = state.0.lock().await;
                                                    let quant_primary = current.quant_migration.is_primary();
                                                    if report.promotion_qualified {
                                                        current.quant_migration.native_validation_parameter_id =
                                                            report.selected_parameter_id.clone();
                                                        current.quant_migration.native_validation_at = Some(now_ts());
                                                    } else {
                                                        current.quant_migration.native_validation_parameter_id = None;
                                                        current.quant_migration.native_validation_at = None;
                                                    }
                                                    current.last_native_quant_backtest = Some(report.clone());
                                                    current.record_backtest(summary.clone());
                                                    current.log(
                                                        if report.promotion_qualified { "info" } else { "warn" },
                                                        "native_backtest",
                                                        "GAIL_NATIVE_QUANT_BACKTEST_COMPLETE",
                                                        json!({
                                                            "promotion_qualified": report.promotion_qualified,
                                                            "selected_parameter_id": report.selected_parameter_id,
                                                            "out_of_sample": report.out_of_sample_statistics,
                                                            "probability_backtest_overfit": report.probability_backtest_overfit,
                                                            "rejection_reasons": report.rejection_reasons,
                                                        }),
                                                    );
                                                    let pause = config.backtest_pause_on_failure
                                                        && quant_primary
                                                        && !report.promotion_qualified;
                                                    if pause {
                                                        current.paused = true;
                                                        current.log_warn(
                                                            "native_backtest",
                                                            "Trading paused: primary quant failed native validation",
                                                        );
                                                    }
                                                    pause
                                                };
                                                if should_pause {
                                                    warn!("trading: primary quant paused after failed native validation");
                                                } else {
                                                    info!(
                                                        qualified = report.promotion_qualified,
                                                        pbo = report.probability_backtest_overfit,
                                                        "trading: Gail-native quant backtest complete"
                                                    );
                                                }
                                            }
                                            Err(error) => {
                                                state.log_error("native_backtest", error).await;
                                            }
                                        }
                                    }
                                    Err(error) => {
                                        state.log_error("native_backtest", error).await;
                                    }
                                }
                            } else {
                                state.log_warn(
                                    "native_backtest",
                                    "Native replay skipped because the market datalake is disabled",
                                ).await;
                            }
                        } else if let Some(ref engine) = backtest_engine {
                            // Compatibility fallback for deployments that explicitly disable
                            // Gail-native replay. OctoBot reports are never used to promote quant.
                            let summary = engine.run_with_config(&config).await;
                            let assessment = summary.assessment.to_string();
                            {
                                let mut current = state.0.lock().await;
                                current.record_backtest(summary.clone());
                                apply_backtest_auto_tuning(&config, &mut current, &summary);
                            }
                            info!("trading: OctoBot compatibility backtest complete — assessment={}", assessment);
                        }
                        // A failed/incomplete replay must not permanently lock
                        // the scheduler. Successful runs clear this in
                        // `record_backtest`; error paths clear it here.
                        state.0.lock().await.backtest_in_progress = false;
                        last_backtest_ts = now_ts();
                    }
                }
                state.persist(&data_path).await;
            }
            shutdown_result = &mut shutdown => {
                match shutdown_result {
                    Ok(()) => {
                        info!("trading: evaluation loop shutting down by request");
                        state.log_info("shutdown", "Trading bridge evaluation loop stopped").await;
                    }
                    Err(_) => {
                        warn!("trading: evaluation loop shutting down because the runtime handle was dropped");
                        state.log_warn("shutdown", "Trading bridge evaluation loop stopped after runtime handle drop").await;
                    }
                }
                state.persist(&data_path).await;
                break;
            }
        }
    }
}

fn native_backtest_summary(
    report: &NativeBacktestReport,
    config: &quantitative::backtest::NativeBacktestConfig,
) -> backtest::BacktestSummary {
    let assessment = if report.walk_forward_folds.is_empty() {
        backtest::ApproachAssessment::Incomplete
    } else if report.promotion_qualified {
        backtest::ApproachAssessment::Viable
    } else {
        backtest::ApproachAssessment::Unprofitable
    };
    let profitability_pct = (config.initial_equity_usd > 0.0).then_some(
        report.out_of_sample_statistics.cumulative_pnl_usd / config.initial_equity_usd * 100.0,
    );
    backtest::BacktestSummary {
        run_at: now_ts(),
        assessment,
        profitability_pct,
        market_avg_pct: None,
        beats_market: None,
        total_trades: report.out_of_sample_statistics.samples,
        errors_count: 0,
        symbols: Vec::new(),
        notes: format!(
            "Gail-native walk-forward replay; selected={:?}; pbo={:.3}; qualified={}; reasons={}",
            report.selected_parameter_id,
            report.probability_backtest_overfit,
            report.promotion_qualified,
            report.rejection_reasons.join(" | ")
        ),
        run_id: None,
    }
}

// ---------------------------------------------------------------------------
// Single evaluation cycle
// ---------------------------------------------------------------------------

struct EvaluationServices<'a> {
    octobot: &'a OctobotClient,
    refiner: &'a RefinerClient,
    fuzzy_engine: &'a FuzzyEngine,
    advisor: &'a TradingAdvisor,
    decision_engine: &'a DecisionEngine,
    market_data_lake: Option<&'a MarketDataLake>,
    data_path: &'a PathBuf,
}

async fn run_single_evaluation(
    config: &TradingConfig,
    state: &SharedTradingState,
    services: EvaluationServices<'_>,
) {
    let EvaluationServices {
        octobot,
        refiner,
        fuzzy_engine,
        advisor,
        decision_engine,
        market_data_lake,
        data_path,
    } = services;
    let eval_start = now_ts();
    debug!("trading: starting evaluation cycle");
    state.log_info("eval", "Starting evaluation cycle").await;

    // --- 1. Fetch market data from OctoBot ---
    let (target_exchanges, target_currencies) = resolve_target_market_filters(config, state).await;

    let evaluation_snapshot_limit = OCTOBOT_MARKET_SNAPSHOT_HARD_LIMIT;
    let fetched_market_snapshots = octobot
        .get_all_market_snapshots(
            &target_exchanges,
            &target_currencies,
            evaluation_snapshot_limit,
        )
        .await;
    let market_snapshots = filter_fresh_market_snapshots(
        fetched_market_snapshots.clone(),
        config.market_snapshot_ttl_seconds,
    );
    if market_snapshots.len() != fetched_market_snapshots.len() {
        state
            .log_warn(
                "market_data",
                format!(
                    "Discarded {} stale, future, invalid, or out-of-order market snapshots",
                    fetched_market_snapshots
                        .len()
                        .saturating_sub(market_snapshots.len())
                ),
            )
            .await;
    }
    if market_snapshots.is_empty() {
        state
            .log_error(
                "market_data",
                "No fresh market data available; skipping advisory and execution cycle",
            )
            .await;
        return;
    }
    resolve_trade_markouts(state, &market_snapshots, config).await;
    let historical_features = if let Some(lake) = market_data_lake {
        let ingest_summary = lake.ingest_snapshots(&market_snapshots).await;
        if ingest_summary.file_error.is_some() || ingest_summary.postgres_error.is_some() {
            state
                .log(
                    "warn",
                    "market_datalake",
                    "Incremental market snapshot persistence encountered an error",
                    json!({
                        "received": ingest_summary.received,
                        "persisted": ingest_summary.persisted,
                        "deduplicated": ingest_summary.deduplicated,
                        "file_error": ingest_summary.file_error,
                        "postgres_error": ingest_summary.postgres_error,
                    }),
                )
                .await;
        }
        lake.features_for_snapshots(&market_snapshots).await
    } else {
        HashMap::new()
    };
    resolve_quant_evaluations(state, &market_snapshots, config, data_path).await;
    resolve_quant_edge_calibration(state, &market_snapshots, config).await;
    let quant_primary = {
        let current = state.0.lock().await;
        config.quant_shadow_enabled && current.quant_migration.is_primary()
    };
    let market_regime = compute_market_regime_contagion(&market_snapshots);

    // --- 2. Build research query ---
    // Keep pre-consensus market ranking for research context only.
    let research_snapshot = select_best_market_candidate(&market_snapshots);
    let research_query = build_research_query(config, research_snapshot.as_ref());

    // Run remaining service calls in parallel so one slow dependency
    // does not serialize the whole evaluation cycle.
    let research_future = async {
        if quant_primary {
            refiner::ResearchContext::empty(research_query.clone())
        } else {
            refiner
                .research_with_site_hints_best_effort(
                    &config.research_index_name,
                    &research_query,
                    &config.research_site_hints,
                    config.research_top_k,
                    config.research_max_parallel_queries,
                )
                .await
        }
    };
    let (portfolio_result, open_orders_result, exchange_info_result, log_feedback_result, research) = tokio::join!(
        octobot.get_portfolio(),
        octobot.get_open_orders(),
        octobot.get_exchange_info(),
        octobot.get_recent_logs(25),
        research_future,
    );

    // Portfolio.
    let portfolio = match portfolio_result {
        Ok(p) => {
            let mut s = state.0.lock().await;
            s.current_portfolio = Some(p.clone());
            p
        }
        Err(err) => {
            warn!("trading: portfolio fetch failed: {}", err);
            state
                .log_warn("eval", format!("Portfolio fetch failed: {err}"))
                .await;
            OctobotPortfolio::default()
        }
    };

    // Open orders.
    match open_orders_result {
        Ok(orders) => {
            let mut s = state.0.lock().await;
            s.open_positions = orders;
        }
        Err(err) => {
            warn!("trading: open orders fetch failed: {}", err);
        }
    }

    // Exchange info for the dashboard.
    match exchange_info_result {
        Ok(exchanges) => {
            let mut s = state.0.lock().await;
            s.available_exchanges = exchanges;
        }
        Err(err) => {
            debug!("trading: exchange info fetch failed: {}", err);
        }
    }

    let logs = match log_feedback_result {
        Ok(logs) => logs,
        Err(err) => {
            debug!("trading: OctoBot log feedback fetch failed: {}", err);
            Vec::new()
        }
    };
    process_octobot_feedback(config, state, octobot, logs).await;

    // --- 3. Select the primary advisory implementation. ---
    // Shadow mode preserves existing LLM execution semantics. Once the
    // persisted guard promotes quant, the synchronous network call disappears
    // from the critical path and the same downstream risk/execution controls
    // consume a deterministic consensus-compatible signal.
    let regime_label = quantitative_regime_label(&market_regime);
    let mut quant_signal = {
        let current = state.0.lock().await;
        let portfolio_config = if config.quantitative.enabled {
            config.quantitative.portfolio.clone()
        } else {
            let mut disabled = config.quantitative.portfolio.clone();
            disabled.enabled = false;
            disabled
        };
        let mut signal = evaluate_universe(
            &current.quant_migration,
            &market_snapshots,
            &historical_features,
            &portfolio,
            &portfolio_config,
        );
        let entry_side = if signal.signal < 0.0 { "sell" } else { "buy" };
        let round_trip_cost_bps = market_snapshots
            .iter()
            .find(|snapshot| {
                snapshot.exchange.eq_ignore_ascii_case(&signal.exchange)
                    && snapshot.symbol.eq_ignore_ascii_case(&signal.symbol)
            })
            .map(|snapshot| {
                current
                    .execution_telemetry
                    .estimate_round_trip(
                        snapshot,
                        entry_side,
                        config.micro_trade_max_usd,
                        config.estimated_fee_bps,
                        config.estimated_slippage_bps,
                        &config.quantitative.execution_telemetry,
                    )
                    .round_trip_cost_bps
            })
            .unwrap_or_else(|| 2.0 * (config.estimated_fee_bps + config.estimated_slippage_bps));
        current.quant_edge_calibration.gate_signal(
            &mut signal,
            regime_label,
            round_trip_cost_bps,
            &config.quantitative.calibration,
        );
        signal
    };
    // Record the undampened policy intent. The LLM overlay is a live risk
    // control and must not censor the observations needed to calibrate the
    // underlying quantitative forecast.
    record_quant_edge_observations(
        state,
        &quant_signal,
        &market_snapshots,
        regime_label,
        config,
    )
    .await;
    let (alpha_sleeves_state, sleeve_costs) = {
        let current = state.0.lock().await;
        let costs = market_snapshots
            .iter()
            .map(|snapshot| {
                let buy = current.execution_telemetry.estimate_round_trip(
                    snapshot,
                    "buy",
                    config.micro_trade_max_usd,
                    config.estimated_fee_bps,
                    config.estimated_slippage_bps,
                    &config.quantitative.execution_telemetry,
                );
                let sell = current.execution_telemetry.estimate_round_trip(
                    snapshot,
                    "sell",
                    config.micro_trade_max_usd,
                    config.estimated_fee_bps,
                    config.estimated_slippage_bps,
                    &config.quantitative.execution_telemetry,
                );
                (
                    sleeve_market_key(&snapshot.exchange, &snapshot.symbol),
                    buy.round_trip_cost_bps.max(sell.round_trip_cost_bps),
                )
            })
            .collect::<HashMap<_, _>>();
        (current.alpha_sleeves.clone(), costs)
    };
    let alpha_sleeves_config = if config.quantitative.enabled {
        config.quantitative.alpha_sleeves.clone()
    } else {
        let mut disabled = config.quantitative.alpha_sleeves.clone();
        disabled.enabled = false;
        disabled
    };
    let sleeve_future = evaluate_sleeves_async(
        alpha_sleeves_state,
        market_snapshots.clone(),
        sleeve_costs,
        alpha_sleeves_config,
        now_ts(),
    );
    let llm_future = async {
        // Keep the LLM path live after quant promotion. It remains the
        // paired benchmark, risk overlay, and rollback signal while quant is
        // the preferred execution method.
        Some(
            advisor
                .consult_all(
                    &market_snapshots,
                    &historical_features,
                    &research,
                    &portfolio,
                    config.max_parallel_advisors,
                )
                .await,
        )
    };
    let (sleeve_result, llm_consensus) = tokio::join!(sleeve_future, llm_future);
    match sleeve_result {
        Ok(next_state) => {
            let actionable = next_state
                .latest_recommendations
                .iter()
                .filter(|recommendation| recommendation.actionable)
                .count();
            let executable = next_state
                .latest_recommendations
                .iter()
                .filter(|recommendation| recommendation.executable)
                .count();
            let mut current = state.0.lock().await;
            current.alpha_sleeves = next_state;
            let recommendations = current.alpha_sleeves.latest_recommendations.clone();
            current.log(
                "info",
                "quant_sleeves",
                "QUANT_ALPHA_SLEEVES_EVALUATED",
                json!({
                    "recommendations": recommendations.len(),
                    "actionable": actionable,
                    "executable": executable,
                    "latest": recommendations,
                }),
            );
        }
        Err(error) => {
            warn!(%error, "trading: alpha-sleeve evaluation failed");
            state
                .log_warn(
                    "quant_sleeves",
                    format!("Alpha-sleeve evaluation failed: {error}"),
                )
                .await;
        }
    }
    let overlay_application = {
        let mut current = state.0.lock().await;
        let overlay_enabled = config.quantitative.enabled;
        let refreshed = overlay_enabled
            && llm_consensus.as_ref().is_some_and(|consensus| {
                current.llm_risk_overlay.update_from_consensus(
                    consensus,
                    now_ts(),
                    &config.quantitative.llm_risk_overlay,
                )
            });
        let application = if overlay_enabled {
            current.llm_risk_overlay.apply_to_quant(
                &mut quant_signal,
                now_ts(),
                &config.quantitative.llm_risk_overlay,
            )
        } else {
            Default::default()
        };
        if refreshed || application.active {
            let overlay = current.llm_risk_overlay.clone();
            current.log(
                "info",
                "quant_overlay",
                "QUANT_LLM_RISK_OVERLAY_APPLIED",
                json!({
                    "refreshed": refreshed,
                    "application": application,
                    "overlay": overlay,
                }),
            );
        }
        application
    };
    let consensus = if quant_primary {
        quant_signal.as_consensus()
    } else {
        llm_consensus
            .clone()
            .unwrap_or_else(|| quant_signal.as_consensus())
    };

    debug!(
        "trading: AI consensus = action={} signal={:.3} confidence={:.2} responders={}",
        consensus.action, consensus.signal, consensus.confidence, consensus.responders
    );

    let execution_gate_reason = degraded_live_execution_reason(&consensus, config);
    if let Some(reason) = execution_gate_reason.as_deref() {
        warn!(
            "trading: live execution gated by AI quality checks: {}",
            reason
        );
        state
            .log(
                "warn",
                "decision",
                format!("Execution gated: {reason}"),
                json!({
                    "responders": consensus.responders,
                    "failures": consensus.failures,
                    "coverage": consensus_coverage(&consensus),
                    "average_risk": consensus_average_risk(&consensus),
                    "agreement": consensus_agreement(&consensus),
                }),
            )
            .await;
    }

    let pending_override = {
        let s = state.0.lock().await;
        s.pending_override.is_some()
    };
    let effective_trade_floor = effective_micro_trade_floor_usd(state, config).await;
    let target_selection = if quant_primary {
        select_quant_primary_market(
            &market_snapshots,
            &quant_signal,
            &portfolio,
            effective_trade_floor,
        )
    } else {
        choose_decision_market_candidate_with_regime(
            &market_snapshots,
            &consensus,
            research_snapshot.as_ref(),
            &portfolio,
            effective_trade_floor,
            &market_regime,
        )
    };
    record_quant_evaluation(
        state,
        &quant_signal,
        quant::QuantEvaluationContext {
            snapshots: &market_snapshots,
            historical_features: &historical_features,
            portfolio: &portfolio,
            llm_consensus: llm_consensus.as_ref(),
            llm_snapshot: target_selection.snapshot.as_ref(),
            llm_evaluation_allowed: execution_gate_reason.is_none()
                && target_selection.override_reason.is_none(),
            config,
            now: now_ts(),
        },
        data_path,
    )
    .await;
    // One market-level consensus must produce at most one economic action.
    // Applying the same advice to every exchange row leaked rationales across
    // symbols and amplified one recommendation into many correlated orders.
    let evaluation_targets = vec![target_selection.snapshot.as_ref()];
    let mut evaluated_decisions = Vec::new();

    for snapshot in evaluation_targets {
        let snapshot_history = snapshot.and_then(|target| {
            historical_features
                .get(&market_feature_key(&target.exchange, &target.symbol))
                .cloned()
        });
        let fuzzy_inputs = compute_fuzzy_inputs(
            snapshot,
            snapshot_history.as_ref(),
            &consensus,
            &research,
            &portfolio,
            Some(&market_regime),
            config,
        );
        let fuzzy_out = fuzzy_engine.evaluate(&fuzzy_inputs);

        let mut decision = {
            let s = state.0.lock().await;
            decision_engine.decide(&fuzzy_out, &consensus, snapshot, &s, config)
        };

        if !decision.override_applied
            && decision_is_actionable(&decision.action)
            && let Some(reason) = execution_gate_reason.as_deref()
        {
            decision = TradeDecision {
                action: TradeAction::Hold,
                amount_usd: 0.0,
                rationale: format!("Execution gated: {reason}"),
                ..decision
            };
        }
        if !decision.override_applied
            && decision_is_actionable(&decision.action)
            && let Some(reason) = target_selection.override_reason.as_deref()
        {
            decision = TradeDecision {
                action: TradeAction::Hold,
                amount_usd: 0.0,
                rationale: format!("Target validation failed: {reason}"),
                ..decision
            };
        }

        if decision.exchange.trim().is_empty()
            && let Some(target) = snapshot
        {
            decision.exchange = target.exchange.clone();
        }
        if decision.symbol.trim().is_empty()
            && let Some(target) = snapshot
        {
            decision.symbol = target.symbol.clone();
        }

        let logged_exchange = if decision.exchange.trim().is_empty() {
            "n/a"
        } else {
            decision.exchange.as_str()
        };
        let logged_symbol = if decision.symbol.trim().is_empty() {
            "n/a"
        } else {
            decision.symbol.as_str()
        };
        let composite_score = snapshot
            .map(|target| composite_symbol_score(target, &decision))
            .unwrap_or(0.0);
        info!(
            "trading: decision = {:?} exchange={} symbol={} amount=${:.2} confidence={:.2}",
            decision.action,
            logged_exchange,
            logged_symbol,
            decision.amount_usd,
            decision.confidence
        );

        evaluated_decisions.push(DecisionCandidate {
            decision,
            market_history: snapshot_history,
            composite_score,
        });

        // Operator overrides should only be evaluated and executed once.
        if pending_override {
            break;
        }
    }

    let mut actionable_decisions = evaluated_decisions
        .iter()
        .filter(|candidate| decision_is_actionable(&candidate.decision.action))
        .cloned()
        .collect::<Vec<_>>();
    sort_decision_candidates_for_execution(&mut actionable_decisions);

    let mut deduped_actionables = Vec::new();
    let mut seen_intents = HashSet::new();
    let mut duplicate_intents_dropped = 0usize;
    for candidate in actionable_decisions {
        if let Some(key) = decision_batch_intent_key(&candidate.decision)
            && !seen_intents.insert(key)
        {
            duplicate_intents_dropped += 1;
            continue;
        }
        deduped_actionables.push(candidate);
    }

    let ranked_actionables = deduped_actionables
        .iter()
        .enumerate()
        .map(|(idx, candidate)| {
            json!({
                "rank": idx + 1,
                "exchange": candidate.decision.exchange,
                "symbol": candidate.decision.symbol,
                "action": candidate.decision.action,
                "confidence": candidate.decision.confidence,
                "blended_signal": candidate.decision.blended_signal,
                "composite_score": candidate.composite_score,
                "amount_usd": candidate.decision.amount_usd,
            })
        })
        .collect::<Vec<_>>();
    let decision_rollup = evaluated_decisions
        .iter()
        .map(|candidate| {
            json!({
                "exchange": candidate.decision.exchange,
                "symbol": candidate.decision.symbol,
                "action": candidate.decision.action,
                "confidence": candidate.decision.confidence,
                "blended_signal": candidate.decision.blended_signal,
                "composite_score": candidate.composite_score,
                "fuzzy_signal": candidate.decision.fuzzy_signal,
                "fuzzy_confidence": candidate.decision.fuzzy_confidence,
                "rationale": truncate_message(&candidate.decision.rationale, 240),
                "market_history": candidate.market_history,
            })
        })
        .collect::<Vec<_>>();
    state
        .log(
            "info",
            "decision",
            format!(
                "Evaluated {} markets; actionable trades={}",
                evaluated_decisions.len(),
                deduped_actionables.len()
            ),
            json!({
                "ai_signal": consensus.signal,
                "ai_confidence": consensus.confidence,
                "ai_action": consensus.action,
                "ai_responders": consensus.responders,
                "ai_failures": consensus.failures,
                "ai_vote_distribution": consensus.vote_distribution,
                "advisory_implementation": if quant_primary { "quant_primary" } else { "llm_primary_quant_shadow" },
                "quant_parameter_id": quant_signal.parameter_id,
                "quant_signal": quant_signal.signal,
                "quant_confidence": quant_signal.confidence,
                "quant_risk_score": quant_signal.risk_score,
                "quant_actionable": quant_signal.actionable,
                "llm_risk_overlay": overlay_application,
                "quant_target": if quant_signal.symbol.is_empty() { None } else { Some(json!({
                    "exchange": quant_signal.exchange,
                    "symbol": quant_signal.symbol,
                })) },
                "market_regime_signal": market_regime.signal,
                "market_regime_confidence": market_regime.confidence,
                "market_regime_leaders": market_regime.leaders,
                "research_market": research_snapshot.as_ref().map(|snapshot| {
                    json!({
                        "exchange": snapshot.exchange,
                        "symbol": snapshot.symbol,
                    })
                }),
                "execution_gate_reason": execution_gate_reason,
                "target_selection": {
                    "note": target_selection.note,
                    "used_target_signal": target_selection.used_target_signal,
                    "target_support": target_selection.target_support,
                    "high_confidence_target": target_selection.high_confidence_target,
                    "override_reason": target_selection.override_reason,
                },
                "duplicate_intents_dropped": duplicate_intents_dropped,
                "decisions": decision_rollup,
                "ranked_actionables": ranked_actionables,
            }),
        )
        .await;

    // --- 6. Execute ranked actionable decisions ---
    for candidate in deduped_actionables
        .iter()
        .take(MAX_EXECUTIONS_PER_EVALUATION)
    {
        execute_if_warranted(octobot, &candidate.decision, state, config).await;
    }

    // Increment evaluation counter.
    {
        let mut s = state.0.lock().await;
        s.evaluation_count += 1;
        s.last_evaluation_at = Some(eval_start);
        s.last_error = None; // Clear previous error on successful cycle.
    }

    debug!(
        "trading: evaluation cycle complete in {:.1}ms",
        (now_ts() - eval_start) * 1000.0
    );
}

/// Resolve due shadow observations before selecting this cycle's primary
/// implementation. A successful migration marker is written into state and
/// atomically persisted before quant is allowed to replace the LLM.
async fn resolve_quant_evaluations(
    state: &SharedTradingState,
    snapshots: &[MarketSnapshot],
    config: &TradingConfig,
    data_path: &PathBuf,
) {
    if !config.quant_shadow_enabled {
        return;
    }
    let (previous_controller, update) = {
        let mut current = state.0.lock().await;
        let previous = current.quant_migration.clone();
        let update = current
            .quant_migration
            .resolve_due(snapshots, config, now_ts());
        (previous, update)
    };
    if update.resolved == 0 && update.expired == 0 {
        return;
    }

    {
        let mut current = state.0.lock().await;
        let pending = current.quant_migration.pending.len();
        let mode = current.quant_migration.mode.as_str();
        current.log(
            "info",
            "quant_migration",
            "QUANT_SHADOW_MARKOUTS_RESOLVED",
            json!({
                "resolved": update.resolved,
                "expired": update.expired,
                "pending": pending,
                "mode": mode,
                "performance": update.performance,
            }),
        );
        if let Some(adjustment) = update.parameter_adjustment.as_ref() {
            current.log(
                "info",
                "quant_migration",
                "QUANT_PARAMETERS_ADJUSTED",
                json!(adjustment),
            );
        }
        if let Some(migration) = update.migration.as_ref() {
            let marker = if matches!(
                migration.transition,
                quant::QuantMigrationTransition::Demoted
            ) {
                "QUANT_DEMOTED_LLM"
            } else {
                "QUANT_PROMOTED_PRIMARY"
            };
            current.log("warn", "quant_migration", marker, json!(migration));
        }
    }

    if let Err(error) = state.persist_checked(data_path).await {
        warn!(%error, "trading: failed to persist quant shadow evaluation state");
        let mut current = state.0.lock().await;
        current.quant_migration = previous_controller;
        if update.migration.is_some() {
            current.log(
                "error",
                "quant_migration",
                "QUANT_MIGRATION_ABORTED_PERSISTENCE",
                json!({ "error": error }),
            );
        } else {
            current.log(
                "error",
                "quant_migration",
                "QUANT_SHADOW_PERSISTENCE_FAILED",
                json!({ "error": error, "controller_update_rolled_back": true }),
            );
        }
        drop(current);
        state.persist(data_path).await;
        return;
    }

    if let Some(adjustment) = update.parameter_adjustment {
        info!(
            previous_parameter_id = adjustment.previous_parameter_id,
            selected_parameter_id = adjustment.selected_parameter_id,
            improvement_bps = adjustment.risk_adjusted_improvement_bps,
            "QUANT_PARAMETERS_ADJUSTED"
        );
    }
    if let Some(migration) = update.migration {
        let marker = if matches!(
            migration.transition,
            quant::QuantMigrationTransition::Demoted
        ) {
            "QUANT_DEMOTED_LLM"
        } else {
            "QUANT_PROMOTED_PRIMARY"
        };
        warn!(
            parameter_id = migration.parameter_id,
            samples = migration.samples,
            actionable_samples = migration.actionable_samples,
            outperformance_bps = migration.outperformance_bps,
            confirmation_streak = migration.confirmation_streak,
            transition = ?migration.transition,
            "{marker}"
        );
    }
}

/// Resolve due multi-horizon observations before the current signal is gated.
/// This ordering allows the newest non-overlapping evidence to influence the
/// next decision without introducing look-ahead.
async fn resolve_quant_edge_calibration(
    state: &SharedTradingState,
    snapshots: &[MarketSnapshot],
    config: &TradingConfig,
) {
    if !config.quantitative.enabled || !config.quantitative.calibration.enabled {
        return;
    }
    let mut current = state.0.lock().await;
    let summary = current.quant_edge_calibration.resolve_due(
        snapshots,
        &config.quantitative.calibration,
        now_ts(),
    );
    if summary.resolved > 0 || summary.expired > 0 {
        let pending = current.quant_edge_calibration.pending.len();
        let resolved_total = current.quant_edge_calibration.resolved.len();
        current.log(
            "info",
            "quant_calibration",
            "QUANT_MULTI_HORIZON_OBSERVATIONS_RESOLVED",
            json!({
                "resolved": summary.resolved,
                "expired": summary.expired,
                "pending": pending,
                "resolved_total": resolved_total,
            }),
        );
    }
}

async fn record_quant_edge_observations(
    state: &SharedTradingState,
    signal: &quant::QuantSignal,
    snapshots: &[MarketSnapshot],
    regime: &str,
    config: &TradingConfig,
) {
    if !config.quantitative.enabled || !config.quantitative.calibration.enabled {
        return;
    }
    let snapshot = snapshots
        .iter()
        .filter(|snapshot| snapshot.exchange.eq_ignore_ascii_case(&signal.exchange))
        .filter(|snapshot| snapshot.symbol.eq_ignore_ascii_case(&signal.symbol))
        .filter(|snapshot| snapshot.price.is_finite() && snapshot.price > 0.0)
        .max_by(|left, right| {
            left.fetched_at
                .partial_cmp(&right.fetched_at)
                .unwrap_or(Ordering::Equal)
        });
    let Some(snapshot) = snapshot else {
        return;
    };
    let mut current = state.0.lock().await;
    let entry_side = if signal.signal < 0.0 { "sell" } else { "buy" };
    let round_trip_cost_bps = current
        .execution_telemetry
        .estimate_round_trip(
            snapshot,
            entry_side,
            config.micro_trade_max_usd,
            config.estimated_fee_bps,
            config.estimated_slippage_bps,
            &config.quantitative.execution_telemetry,
        )
        .round_trip_cost_bps;
    let recorded = current.quant_edge_calibration.record_signal(
        signal,
        snapshot,
        regime,
        round_trip_cost_bps,
        &config.quantitative.calibration,
        now_ts(),
    );
    if recorded > 0 {
        current.log(
            "info",
            "quant_calibration",
            "QUANT_MULTI_HORIZON_OBSERVATIONS_RECORDED",
            json!({
                "exchange": signal.exchange,
                "symbol": signal.symbol,
                "raw_signal": signal.signal,
                "regime": regime,
                "horizons_recorded": recorded,
                "estimated_round_trip_cost_bps": round_trip_cost_bps,
            }),
        );
    }
}

/// Add a paired shadow observation to the durable ledger. The LLM side is
/// absent after migration, while quant candidate outcomes continue to tune the
/// active deterministic parameters.
async fn record_quant_evaluation(
    state: &SharedTradingState,
    quant_signal: &quant::QuantSignal,
    context: quant::QuantEvaluationContext<'_>,
    data_path: &PathBuf,
) {
    if !context.config.quant_shadow_enabled || quant_signal.symbol.is_empty() {
        return;
    }
    let record = {
        let mut current = state.0.lock().await;
        let record = current
            .quant_migration
            .record_evaluation(quant_signal, context);
        if let Some(record) = record.as_ref() {
            let marker = if record.mode == QuantMode::Primary {
                "QUANT_PRIMARY_EVALUATION_RECORDED"
            } else {
                "QUANT_SHADOW_EVALUATION_RECORDED"
            };
            current.log("info", "quant_migration", marker, json!(record));
        }
        record
    };
    if record.is_none() {
        return;
    }
    if let Err(error) = state.persist_checked(data_path).await {
        warn!(%error, "trading: failed to persist pending quant shadow evaluation");
        state
            .log(
                "warn",
                "quant_migration",
                "QUANT_SHADOW_PERSISTENCE_FAILED",
                json!({ "error": error }),
            )
            .await;
    }
}

async fn run_market_datalake_bootstrap(
    config: &TradingConfig,
    state: &SharedTradingState,
    octobot: &OctobotClient,
    market_data_lake: &MarketDataLake,
    reason: &str,
) -> bool {
    market_data_lake.mark_bootstrap_started(reason).await;
    state
        .log_info(
            "market_datalake",
            format!("Starting one-time market datalake bootstrap: {reason}"),
        )
        .await;

    let (target_exchanges, target_currencies) = resolve_target_market_filters(config, state).await;
    let seed_snapshots = octobot
        .get_all_market_snapshots(
            &target_exchanges,
            &target_currencies,
            config.market_datalake_bootstrap_symbol_limit,
        )
        .await;
    if seed_snapshots.is_empty() {
        let error = "No bootstrap symbols available from OctoBot";
        market_data_lake.mark_bootstrap_failed(reason, error).await;
        state.log_warn("market_datalake", error).await;
        return false;
    }

    let mut historical_snapshots = Vec::new();
    let mut symbols_with_history = 0usize;
    for snapshot in &seed_snapshots {
        let mut symbol_has_history = false;
        for time_frame in &config.market_datalake_bootstrap_time_frames {
            match octobot
                .get_market_snapshot_history(&snapshot.exchange, &snapshot.symbol, time_frame)
                .await
            {
                Ok(history) => {
                    if !history.is_empty() {
                        symbol_has_history = true;
                        historical_snapshots.extend(history);
                    }
                }
                Err(error) => {
                    debug!(
                        exchange = %snapshot.exchange,
                        symbol = %snapshot.symbol,
                        time_frame = %time_frame,
                        error = %error,
                        "trading: bootstrap history request failed for symbol"
                    );
                }
            }
        }
        if symbol_has_history {
            symbols_with_history += 1;
        }
    }

    if historical_snapshots.is_empty() {
        let error = "Bootstrap fetched no historical candle snapshots";
        market_data_lake.mark_bootstrap_failed(reason, error).await;
        state
            .log_warn("market_datalake", format!("{error}; will retry later"))
            .await;
        return false;
    }

    let ingest = market_data_lake
        .ingest_snapshots(&historical_snapshots)
        .await;
    if let Some(error) = ingest.file_error.as_deref() {
        market_data_lake.mark_bootstrap_failed(reason, error).await;
        state
            .log_warn(
                "market_datalake",
                format!("Bootstrap file persistence failed: {error}"),
            )
            .await;
        return false;
    }

    let report = MarketDataLakeBootstrapReport {
        reason: reason.to_string(),
        symbols_attempted: seed_snapshots.len(),
        symbols_with_history,
        time_frames: config.market_datalake_bootstrap_time_frames.clone(),
        snapshots_received: historical_snapshots.len(),
        snapshots_persisted: ingest.persisted,
        snapshots_deduplicated: ingest.deduplicated,
    };
    market_data_lake.mark_bootstrap_completed(&report).await;
    state
        .log(
            "info",
            "market_datalake",
            format!(
                "Market datalake bootstrap complete: symbols={} with_history={} snapshots={} persisted={} deduped={}",
                report.symbols_attempted,
                report.symbols_with_history,
                report.snapshots_received,
                report.snapshots_persisted,
                report.snapshots_deduplicated,
            ),
            json!(report),
        )
        .await;
    true
}

async fn resolve_target_market_filters(
    config: &TradingConfig,
    state: &SharedTradingState,
) -> (Vec<String>, Vec<String>) {
    let s = state.0.lock().await;
    let ov = s.config_overrides.as_ref();
    let exchanges = ov
        .and_then(|o| o.target_exchanges.clone())
        .unwrap_or_else(|| config.target_exchanges.clone());
    let currencies = ov
        .and_then(|o| o.target_currencies.clone())
        .unwrap_or_else(|| config.target_currencies.clone());
    (exchanges, currencies)
}

#[derive(Clone, Debug, Serialize)]
struct SymbolScorecard {
    at: f64,
    exchange: String,
    symbol: String,
    in_portfolio: bool,
    market_score: f64,
    price_change_pct_24h: Option<f64>,
    volume_24h: Option<f64>,
    history_momentum_short_pct: Option<f64>,
    history_momentum_mid_pct: Option<f64>,
    history_momentum_long_pct: Option<f64>,
    history_volatility_pct: Option<f64>,
    history_drawdown_pct: Option<f64>,
    history_volume_ratio_short_long: Option<f64>,
    ai_signal: f64,
    ai_confidence: f64,
    fuzzy_signal: f64,
    fuzzy_confidence: f64,
    blended_signal: f64,
    blended_confidence: f64,
    action: String,
    composite_score: f64,
    amount_usd: f64,
    rationale: String,
}

#[derive(Clone, Debug)]
struct EvaluatedSymbol {
    decision: TradeDecision,
    scorecard: SymbolScorecard,
}

#[derive(Clone, Debug)]
struct DecisionCandidate {
    decision: TradeDecision,
    market_history: Option<MarketHistoricalFeatures>,
    composite_score: f64,
}

fn decision_is_actionable(action: &TradeAction) -> bool {
    matches!(
        action,
        TradeAction::Buy | TradeAction::StrongBuy | TradeAction::Sell | TradeAction::StrongSell
    )
}

/// Batch identity is based on economic effect, not the source market row.
/// Buys for the same pair are one intent because balance-aware routing can move
/// all source exchanges onto the same funded account. Sells retain exchange
/// scope so genuinely separate holdings can still be reduced independently.
fn decision_batch_intent_key(decision: &TradeDecision) -> Option<String> {
    let side = trade_action_side(&decision.action)?;
    execution_intent_key(side, &decision.exchange, &decision.symbol)
}

fn trade_action_side(action: &TradeAction) -> Option<&'static str> {
    match action {
        TradeAction::Buy | TradeAction::StrongBuy => Some("buy"),
        TradeAction::Sell | TradeAction::StrongSell => Some("sell"),
        _ => None,
    }
}

fn execution_intent_key(side: &str, exchange: &str, symbol: &str) -> Option<String> {
    let symbol = symbol.trim().to_ascii_uppercase();
    if symbol.is_empty() {
        return None;
    }
    if side.eq_ignore_ascii_case("buy") {
        Some(format!("BUY|{symbol}"))
    } else if side.eq_ignore_ascii_case("sell") {
        let exchange = exchange.trim().to_ascii_uppercase();
        (!exchange.is_empty()).then(|| format!("SELL|{exchange}|{symbol}"))
    } else {
        None
    }
}

fn sort_decision_candidates_for_execution(candidates: &mut [DecisionCandidate]) {
    candidates.sort_by(|left, right| {
        right
            .decision
            .confidence
            .partial_cmp(&left.decision.confidence)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                right
                    .decision
                    .blended_signal
                    .abs()
                    .partial_cmp(&left.decision.blended_signal.abs())
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                right
                    .composite_score
                    .abs()
                    .partial_cmp(&left.composite_score.abs())
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                right
                    .decision
                    .amount_usd
                    .partial_cmp(&left.decision.amount_usd)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                left.decision
                    .exchange
                    .to_ascii_uppercase()
                    .cmp(&right.decision.exchange.to_ascii_uppercase())
            })
            .then_with(|| {
                left.decision
                    .symbol
                    .to_ascii_uppercase()
                    .cmp(&right.decision.symbol.to_ascii_uppercase())
            })
    });
}

async fn run_non_portfolio_discovery_cycle(
    config: &TradingConfig,
    state: &SharedTradingState,
    octobot: &OctobotClient,
    refiner: &RefinerClient,
    fuzzy_engine: &FuzzyEngine,
    advisor: &TradingAdvisor,
    decision_engine: &DecisionEngine,
    market_data_lake: Option<&MarketDataLake>,
) {
    if {
        let s = state.0.lock().await;
        s.pending_override.is_some()
    } {
        debug!("trading: discovery review skipped due to pending operator override");
        return;
    }

    let Some(portfolio) = load_current_portfolio_snapshot(state, octobot).await else {
        warn!("trading: discovery review skipped because portfolio is unavailable");
        state
            .log_warn(
                "discovery",
                "Portfolio unavailable; discovery review skipped",
            )
            .await;
        return;
    };

    let (target_exchanges, target_currencies) = resolve_target_market_filters(config, state).await;
    let snapshots = octobot
        .get_all_market_snapshots(
            &target_exchanges,
            &target_currencies,
            config.token_discovery_snapshot_limit,
        )
        .await;
    resolve_trade_markouts(state, &snapshots, config).await;
    let historical_features = if let Some(lake) = market_data_lake {
        let ingest_summary = lake.ingest_snapshots(&snapshots).await;
        if ingest_summary.file_error.is_some() || ingest_summary.postgres_error.is_some() {
            state
                .log(
                    "warn",
                    "market_datalake",
                    "Discovery cycle market snapshot persistence encountered an error",
                    json!({
                        "received": ingest_summary.received,
                        "persisted": ingest_summary.persisted,
                        "deduplicated": ingest_summary.deduplicated,
                        "file_error": ingest_summary.file_error,
                        "postgres_error": ingest_summary.postgres_error,
                    }),
                )
                .await;
        }
        lake.features_for_snapshots(&snapshots).await
    } else {
        HashMap::new()
    };
    if snapshots.is_empty() {
        state
            .log_warn(
                "discovery",
                "No market snapshots available for discovery review",
            )
            .await;
        return;
    }
    let market_regime = compute_market_regime_contagion(&snapshots);

    let candidates = select_non_portfolio_candidates(
        &snapshots,
        &portfolio,
        config.token_discovery_candidate_pool_size,
    );
    if candidates.is_empty() {
        state
            .log_info(
                "discovery",
                "No non-portfolio candidates available for discovery review",
            )
            .await;
        return;
    }

    let mut evaluated = evaluate_symbol_candidates_parallel(
        config,
        state,
        refiner,
        fuzzy_engine,
        advisor,
        decision_engine,
        &portfolio,
        &candidates,
        false,
        &historical_features,
        &market_regime,
    )
    .await;
    evaluated.sort_by(|left, right| {
        left.scorecard
            .composite_score
            .partial_cmp(&right.scorecard.composite_score)
            .unwrap_or(Ordering::Equal)
            .reverse()
    });

    let scorecards = evaluated
        .iter()
        .map(|entry| entry.scorecard.clone())
        .collect::<Vec<_>>();
    state
        .log(
            "info",
            "discovery",
            format!(
                "Scored {} non-portfolio symbols; top composite={:.3}",
                scorecards.len(),
                scorecards
                    .first()
                    .map(|entry| entry.composite_score)
                    .unwrap_or(0.0)
            ),
            json!({ "scorecards": scorecards }),
        )
        .await;

    let selected = evaluated
        .iter()
        .filter(|entry| {
            matches!(
                entry.decision.action,
                TradeAction::Buy | TradeAction::StrongBuy
            )
        })
        .max_by(|left, right| {
            left.scorecard
                .composite_score
                .partial_cmp(&right.scorecard.composite_score)
                .unwrap_or(Ordering::Equal)
        })
        .map(|entry| (entry.decision.clone(), entry.scorecard.clone()));

    let Some((decision, scorecard)) = selected else {
        state
            .log_info(
                "discovery",
                "No buy-qualified non-portfolio candidate from discovery review",
            )
            .await;
        return;
    };

    if scorecard.composite_score < config.token_discovery_min_composite_score {
        state
            .log_info(
                "discovery",
                format!(
                    "Discovery top candidate {} below threshold ({:.3} < {:.3})",
                    scorecard.symbol,
                    scorecard.composite_score,
                    config.token_discovery_min_composite_score
                ),
            )
            .await;
        return;
    }

    info!(
        "trading: discovery selected {} (score {:.3}) for automatic entry",
        scorecard.symbol, scorecard.composite_score
    );
    state
        .log(
            "info",
            "discovery",
            format!(
                "Selected discovered symbol {} for auto-buy (score {:.3})",
                scorecard.symbol, scorecard.composite_score
            ),
            json!({ "scorecard": scorecard }),
        )
        .await;
    execute_if_warranted(octobot, &decision, state, config).await;
}

async fn run_portfolio_pruning_cycle(
    config: &TradingConfig,
    state: &SharedTradingState,
    octobot: &OctobotClient,
    refiner: &RefinerClient,
    fuzzy_engine: &FuzzyEngine,
    advisor: &TradingAdvisor,
    decision_engine: &DecisionEngine,
    market_data_lake: Option<&MarketDataLake>,
) {
    if {
        let s = state.0.lock().await;
        s.pending_override.is_some()
    } {
        debug!("trading: pruning review skipped due to pending operator override");
        return;
    }

    let Some(portfolio) = load_current_portfolio_snapshot(state, octobot).await else {
        warn!("trading: pruning review skipped because portfolio is unavailable");
        state
            .log_warn("pruning", "Portfolio unavailable; pruning review skipped")
            .await;
        return;
    };

    let (target_exchanges, target_currencies) = resolve_target_market_filters(config, state).await;
    let snapshots = octobot
        .get_all_market_snapshots(
            &target_exchanges,
            &target_currencies,
            config.token_discovery_snapshot_limit,
        )
        .await;
    resolve_trade_markouts(state, &snapshots, config).await;
    let historical_features = if let Some(lake) = market_data_lake {
        let ingest_summary = lake.ingest_snapshots(&snapshots).await;
        if ingest_summary.file_error.is_some() || ingest_summary.postgres_error.is_some() {
            state
                .log(
                    "warn",
                    "market_datalake",
                    "Pruning cycle market snapshot persistence encountered an error",
                    json!({
                        "received": ingest_summary.received,
                        "persisted": ingest_summary.persisted,
                        "deduplicated": ingest_summary.deduplicated,
                        "file_error": ingest_summary.file_error,
                        "postgres_error": ingest_summary.postgres_error,
                    }),
                )
                .await;
        }
        lake.features_for_snapshots(&snapshots).await
    } else {
        HashMap::new()
    };
    if snapshots.is_empty() {
        state
            .log_warn(
                "pruning",
                "No market snapshots available for pruning review",
            )
            .await;
        return;
    }
    let market_regime = compute_market_regime_contagion(&snapshots);

    let candidates = select_portfolio_pruning_candidates(
        &snapshots,
        &portfolio,
        config.portfolio_pruning_min_holding_usd,
        config.portfolio_pruning_candidate_pool_size,
    );
    if candidates.is_empty() {
        state
            .log_info("pruning", "No held symbols eligible for pruning review")
            .await;
        return;
    }

    let mut evaluated = evaluate_symbol_candidates_parallel(
        config,
        state,
        refiner,
        fuzzy_engine,
        advisor,
        decision_engine,
        &portfolio,
        &candidates,
        true,
        &historical_features,
        &market_regime,
    )
    .await;
    evaluated.sort_by(|left, right| {
        left.scorecard
            .composite_score
            .partial_cmp(&right.scorecard.composite_score)
            .unwrap_or(Ordering::Equal)
    });

    let scorecards = evaluated
        .iter()
        .map(|entry| entry.scorecard.clone())
        .collect::<Vec<_>>();
    state
        .log(
            "info",
            "pruning",
            format!(
                "Scored {} held symbols for pruning; strongest bearish composite={:.3}",
                scorecards.len(),
                scorecards
                    .first()
                    .map(|entry| (-entry.composite_score).max(0.0))
                    .unwrap_or(0.0)
            ),
            json!({ "scorecards": scorecards }),
        )
        .await;

    let selected = evaluated
        .iter()
        .filter(|entry| {
            matches!(
                entry.decision.action,
                TradeAction::Sell | TradeAction::StrongSell
            )
        })
        .max_by(|left, right| {
            let left_score = (-left.scorecard.composite_score).max(0.0);
            let right_score = (-right.scorecard.composite_score).max(0.0);
            left_score
                .partial_cmp(&right_score)
                .unwrap_or(Ordering::Equal)
        })
        .map(|entry| (entry.decision.clone(), entry.scorecard.clone()));

    let Some((mut decision, scorecard)) = selected else {
        state
            .log_info(
                "pruning",
                "No sell-qualified held symbol from pruning review",
            )
            .await;
        return;
    };

    let bearish_score = (-scorecard.composite_score).max(0.0);
    if bearish_score < config.portfolio_pruning_min_composite_score {
        state
            .log_info(
                "pruning",
                format!(
                    "Pruning top candidate {} below threshold ({:.3} < {:.3})",
                    scorecard.symbol, bearish_score, config.portfolio_pruning_min_composite_score
                ),
            )
            .await;
        return;
    }

    if let Some(holding_usd) = holding_value_usd_for_symbol(&portfolio, &decision.symbol)
        .filter(|value| value.is_finite() && *value > 0.0)
    {
        decision.amount_usd = holding_usd.max(0.01);
    }

    info!(
        "trading: pruning selected {} (bearish score {:.3}) for automatic selloff",
        scorecard.symbol, bearish_score
    );
    state
        .log(
            "info",
            "pruning",
            format!(
                "Selected held symbol {} for auto-selloff (bearish score {:.3})",
                scorecard.symbol, bearish_score
            ),
            json!({ "scorecard": scorecard, "amount_usd": decision.amount_usd }),
        )
        .await;
    execute_if_warranted(octobot, &decision, state, config).await;
}

async fn load_current_portfolio_snapshot(
    state: &SharedTradingState,
    octobot: &OctobotClient,
) -> Option<OctobotPortfolio> {
    if let Some(cached) = {
        let s = state.0.lock().await;
        s.current_portfolio.clone()
    } {
        return Some(cached);
    }
    match octobot.get_portfolio().await {
        Ok(portfolio) => {
            let mut s = state.0.lock().await;
            s.current_portfolio = Some(portfolio.clone());
            Some(portfolio)
        }
        Err(err) => {
            warn!(
                "trading: failed to refresh portfolio for discovery/pruning review: {}",
                err
            );
            None
        }
    }
}

async fn evaluate_symbol_candidate(
    config: &TradingConfig,
    state: &SharedTradingState,
    refiner: &RefinerClient,
    fuzzy_engine: &FuzzyEngine,
    advisor: &TradingAdvisor,
    decision_engine: &DecisionEngine,
    portfolio: &OctobotPortfolio,
    snapshot: &MarketSnapshot,
    in_portfolio: bool,
    historical_features: Option<&MarketHistoricalFeatures>,
    market_regime: Option<&MarketRegimeContagion>,
) -> EvaluatedSymbol {
    let quant_parameters = {
        let current = state.0.lock().await;
        (config.quant_shadow_enabled && current.quant_migration.is_primary())
            .then(|| current.quant_migration.active_parameters().clone())
    };
    let research_query = build_research_query(config, Some(snapshot));
    let research = if quant_parameters.is_some() {
        refiner::ResearchContext::empty(research_query)
    } else {
        refiner
            .research_with_site_hints_best_effort(
                &config.research_index_name,
                &research_query,
                &config.research_site_hints,
                config.research_top_k,
                config.research_max_parallel_queries,
            )
            .await
    };
    let consensus = if let Some(parameters) = quant_parameters.as_ref() {
        evaluate_quant_symbol(parameters, snapshot, historical_features).as_consensus()
    } else {
        advisor
            .consult_all(
                std::slice::from_ref(snapshot),
                &historical_features_map(snapshot, historical_features),
                &research,
                portfolio,
                config.max_parallel_advisors,
            )
            .await
    };
    let fuzzy_inputs = compute_fuzzy_inputs(
        Some(snapshot),
        historical_features,
        &consensus,
        &research,
        portfolio,
        market_regime,
        config,
    );
    let fuzzy = fuzzy_engine.evaluate(&fuzzy_inputs);
    let decision = {
        let s = state.0.lock().await;
        decision_engine.decide(&fuzzy, &consensus, Some(snapshot), &s, config)
    };
    let composite_score = composite_symbol_score(snapshot, &decision);
    let scorecard = SymbolScorecard {
        at: now_ts(),
        exchange: snapshot.exchange.clone(),
        symbol: snapshot.symbol.clone(),
        in_portfolio,
        market_score: market_score(snapshot),
        price_change_pct_24h: snapshot.price_change_pct_24h,
        volume_24h: snapshot.volume_24h,
        history_momentum_short_pct: historical_features
            .and_then(|feature| feature.momentum_short_pct),
        history_momentum_mid_pct: historical_features.and_then(|feature| feature.momentum_mid_pct),
        history_momentum_long_pct: historical_features
            .and_then(|feature| feature.momentum_long_pct),
        history_volatility_pct: historical_features.and_then(|feature| feature.volatility_pct),
        history_drawdown_pct: historical_features.and_then(|feature| feature.drawdown_pct),
        history_volume_ratio_short_long: historical_features
            .and_then(|feature| feature.volume_ratio_short_long),
        ai_signal: consensus.signal,
        ai_confidence: consensus.confidence,
        fuzzy_signal: fuzzy.signal,
        fuzzy_confidence: fuzzy.confidence,
        blended_signal: decision.blended_signal,
        blended_confidence: decision.confidence,
        action: decision.action.to_string(),
        composite_score,
        amount_usd: decision.amount_usd,
        rationale: truncate_message(&decision.rationale, 220),
    };
    EvaluatedSymbol {
        decision,
        scorecard,
    }
}

/// Evaluate discovery/pruning candidates with bounded outer parallelism.
///
/// A symbol evaluation performs independent Refiner and advisor I/O, making it
/// safe to overlap. `buffer_unordered` avoids head-of-line blocking while the
/// configured bound prevents `symbols × advisors` from becoming an unbounded
/// request burst. Ordering is restored later by explicit score sorting.
#[allow(clippy::too_many_arguments)]
async fn evaluate_symbol_candidates_parallel(
    config: &TradingConfig,
    state: &SharedTradingState,
    refiner: &RefinerClient,
    fuzzy_engine: &FuzzyEngine,
    advisor: &TradingAdvisor,
    decision_engine: &DecisionEngine,
    portfolio: &OctobotPortfolio,
    candidates: &[MarketSnapshot],
    in_portfolio: bool,
    historical_features: &HashMap<String, MarketHistoricalFeatures>,
    market_regime: &MarketRegimeContagion,
) -> Vec<EvaluatedSymbol> {
    stream::iter(candidates.iter().cloned())
        .map(|snapshot| {
            let historical = historical_features
                .get(&market_feature_key(&snapshot.exchange, &snapshot.symbol))
                .cloned();
            async move {
                evaluate_symbol_candidate(
                    config,
                    state,
                    refiner,
                    fuzzy_engine,
                    advisor,
                    decision_engine,
                    portfolio,
                    &snapshot,
                    in_portfolio,
                    historical.as_ref(),
                    Some(market_regime),
                )
                .await
            }
        })
        .buffer_unordered(config.max_parallel_symbol_evaluations.max(1))
        .collect()
        .await
}

fn historical_features_map(
    snapshot: &MarketSnapshot,
    features: Option<&MarketHistoricalFeatures>,
) -> HashMap<String, MarketHistoricalFeatures> {
    let mut map = HashMap::new();
    if let Some(features) = features {
        map.insert(
            market_feature_key(&snapshot.exchange, &snapshot.symbol),
            features.clone(),
        );
    }
    map
}

fn select_non_portfolio_candidates(
    snapshots: &[MarketSnapshot],
    portfolio: &OctobotPortfolio,
    pool_size: usize,
) -> Vec<MarketSnapshot> {
    let mut ranked = snapshots
        .iter()
        .filter(|snapshot| snapshot.price.is_finite() && snapshot.price > 0.0)
        .filter(|snapshot| snapshot_has_stable_quote(snapshot))
        .filter(|snapshot| !snapshot_in_portfolio(snapshot, portfolio))
        .cloned()
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        market_score(right)
            .partial_cmp(&market_score(left))
            .unwrap_or(Ordering::Equal)
    });

    let mut seen_assets = HashSet::new();
    ranked
        .into_iter()
        .filter(|snapshot| {
            symbol_base_asset(&snapshot.symbol)
                .is_some_and(|asset| seen_assets.insert(asset.trim().to_ascii_uppercase()))
        })
        .take(pool_size.max(1))
        .collect()
}

fn select_portfolio_pruning_candidates(
    snapshots: &[MarketSnapshot],
    portfolio: &OctobotPortfolio,
    min_holding_usd: f64,
    pool_size: usize,
) -> Vec<MarketSnapshot> {
    let held_assets = portfolio
        .currencies
        .iter()
        .filter(|(asset, balance)| {
            !is_stablecoin(asset)
                && (balance.free > 0.0 || balance.total > 0.0)
                && balance
                    .value_usd
                    .is_some_and(|value| value.is_finite() && value >= min_holding_usd)
        })
        .map(|(asset, _)| asset.to_ascii_uppercase())
        .collect::<HashSet<_>>();

    let mut selected = held_assets
        .into_iter()
        .filter_map(|asset| preferred_snapshot_for_asset(snapshots, &asset))
        .collect::<Vec<_>>();
    selected.sort_by(|left, right| {
        market_score(right)
            .partial_cmp(&market_score(left))
            .unwrap_or(Ordering::Equal)
    });
    selected.truncate(pool_size.max(1));
    selected
}

fn preferred_snapshot_for_asset(
    snapshots: &[MarketSnapshot],
    asset: &str,
) -> Option<MarketSnapshot> {
    preferred_snapshot_for_asset_ref(snapshots, asset).cloned()
}

fn preferred_snapshot_for_asset_ref<'a>(
    snapshots: &'a [MarketSnapshot],
    asset: &str,
) -> Option<&'a MarketSnapshot> {
    snapshots
        .iter()
        .filter(|snapshot| snapshot.price.is_finite() && snapshot.price > 0.0)
        .filter(|snapshot| snapshot_has_stable_quote(snapshot))
        .filter(|snapshot| {
            symbol_base_asset(&snapshot.symbol).is_some_and(|base| base.eq_ignore_ascii_case(asset))
        })
        .max_by(|left, right| {
            let left_quote = quote_priority(symbol_quote_asset(&left.symbol).unwrap_or_default());
            let right_quote = quote_priority(symbol_quote_asset(&right.symbol).unwrap_or_default());
            left_quote.cmp(&right_quote).then_with(|| {
                market_score(left)
                    .partial_cmp(&market_score(right))
                    .unwrap_or(Ordering::Equal)
            })
        })
}

fn snapshot_has_stable_quote(snapshot: &MarketSnapshot) -> bool {
    symbol_quote_asset(&snapshot.symbol).is_some_and(is_stablecoin)
}

fn snapshot_in_portfolio(snapshot: &MarketSnapshot, portfolio: &OctobotPortfolio) -> bool {
    let Some(base_asset) = symbol_base_asset(&snapshot.symbol) else {
        return false;
    };
    portfolio.currencies.get(base_asset).is_some_and(|balance| {
        (balance.free > 0.0 || balance.total > 0.0)
            && balance
                .value_usd
                .is_some_and(|value| value.is_finite() && value > 0.01)
    })
}

fn holding_value_usd_for_symbol(portfolio: &OctobotPortfolio, symbol: &str) -> Option<f64> {
    let base_asset = symbol_base_asset(symbol)?;
    let balance = portfolio.currencies.get(base_asset)?;
    sellable_value_usd(balance.free, balance.total, balance.value_usd, None)
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn sellable_value_usd_for_snapshot(
    portfolio: &OctobotPortfolio,
    snapshot: &MarketSnapshot,
) -> Option<f64> {
    let base_asset = symbol_base_asset(&snapshot.symbol)?;
    let balance = portfolio.currencies.get(base_asset)?;
    sellable_value_usd(
        balance.free,
        balance.total,
        balance.value_usd,
        Some(snapshot.price),
    )
    .filter(|value| value.is_finite() && *value > 0.0)
}

fn snapshot_sellable_above_floor(
    snapshot: &MarketSnapshot,
    portfolio: &OctobotPortfolio,
    min_sellable_usd: f64,
) -> bool {
    let floor = min_sellable_usd.max(0.01);
    sellable_value_usd_for_snapshot(portfolio, snapshot)
        .is_some_and(|value| value + f64::EPSILON >= floor)
}

fn symbol_base_asset(symbol: &str) -> Option<&str> {
    symbol
        .split('/')
        .next()
        .map(str::trim)
        .filter(|asset| !asset.is_empty())
}

fn symbol_quote_asset(symbol: &str) -> Option<&str> {
    symbol
        .split('/')
        .nth(1)
        .map(str::trim)
        .filter(|asset| !asset.is_empty())
}

fn quote_priority(quote: &str) -> usize {
    if quote.eq_ignore_ascii_case("USDT") {
        5
    } else if quote.eq_ignore_ascii_case("USDC") {
        4
    } else if quote.eq_ignore_ascii_case("BUSD") {
        3
    } else if quote.eq_ignore_ascii_case("DAI") {
        2
    } else if quote.eq_ignore_ascii_case("USD") || quote.eq_ignore_ascii_case("EUR") {
        1
    } else {
        0
    }
}

fn action_direction_multiplier(action: &TradeAction) -> f64 {
    match action {
        TradeAction::StrongBuy => 1.15,
        TradeAction::Buy => 1.0,
        TradeAction::Hold | TradeAction::Cancel => 0.0,
        TradeAction::Sell => -1.0,
        TradeAction::StrongSell => -1.15,
    }
}

fn composite_symbol_score(snapshot: &MarketSnapshot, decision: &TradeDecision) -> f64 {
    let direction = action_direction_multiplier(&decision.action);
    if direction.abs() < f64::EPSILON {
        return 0.0;
    }
    let trend_strength = snapshot.price_change_pct_24h.unwrap_or(0.0).abs().min(25.0) / 25.0;
    let liquidity_strength =
        ((snapshot.volume_24h.unwrap_or(0.0) + 1.0).ln() / 18.0).clamp(0.0, 1.0);
    let market_quality = (trend_strength * 0.6 + liquidity_strength * 0.4).clamp(0.0, 1.0);
    direction
        * decision.confidence.clamp(0.0, 1.0)
        * decision.blended_signal.abs().clamp(0.0, 1.0)
        * (0.7 + 0.3 * market_quality)
}

fn truncate_message(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

async fn process_octobot_feedback(
    config: &TradingConfig,
    state: &SharedTradingState,
    octobot: &OctobotClient,
    logs: Vec<OctobotLogEntry>,
) {
    let schema = octobot.api_schema_snapshot().await;
    let recommended_min = schema.numeric_hints.get("micro_trade_min_usd").copied();

    let mut s = state.0.lock().await;
    for entry in logs.iter().filter(|entry| significant_octobot_log(entry)) {
        let fingerprint = octobot_log_fingerprint(entry);
        if s.remember_external_log(fingerprint) {
            s.log(
                log_level_for_external_entry(entry),
                "octobot_log",
                format!(
                    "OctoBot {} {}: {}",
                    entry.level, entry.source, entry.message
                ),
                json!(entry),
            );
        }
    }

    if let Some(min_usd) = recommended_min {
        apply_trade_floor_feedback(config, &mut s, min_usd);
    }
    s.api_schema = schema;
}

fn significant_octobot_log(entry: &OctobotLogEntry) -> bool {
    let level = entry.level.to_ascii_lowercase();
    level.contains("error")
        || level.contains("warn")
        || entry.message.contains("MissingMinimalExchangeTradeVolume")
        || entry.message.contains("ManagerToolCall")
        || entry.message.contains("nvidia upstream error")
}

fn log_level_for_external_entry(entry: &OctobotLogEntry) -> &'static str {
    if entry.level.eq_ignore_ascii_case("ERROR") {
        "error"
    } else if entry.level.to_ascii_lowercase().contains("warn") {
        "warn"
    } else {
        "info"
    }
}

fn octobot_log_fingerprint(entry: &OctobotLogEntry) -> String {
    format!(
        "{}|{}|{}|{}",
        entry.time.as_deref().unwrap_or_default(),
        entry.level,
        entry.source,
        entry.message.chars().take(300).collect::<String>()
    )
}

fn apply_trade_floor_feedback(
    config: &TradingConfig,
    state: &mut TradingState,
    recommended_min_usd: f64,
) {
    if !recommended_min_usd.is_finite() || recommended_min_usd <= 0.0 {
        return;
    }
    let target = recommended_min_usd
        .max(config.micro_trade_min_usd)
        .min(1_000_000.0);
    let current_min = state
        .config_overrides
        .as_ref()
        .and_then(|overrides| overrides.micro_trade_min_usd)
        .unwrap_or(config.micro_trade_min_usd);
    let current_max = state
        .config_overrides
        .as_ref()
        .and_then(|overrides| overrides.micro_trade_max_usd)
        .unwrap_or(config.micro_trade_max_usd);
    let changed_min = target > current_min + f64::EPSILON;
    let changed_max = current_max + f64::EPSILON < target;
    if changed_min || changed_max {
        let overrides: &mut TradingConfigOverride =
            state.config_overrides.get_or_insert_with(Default::default);
        if changed_min {
            overrides.micro_trade_min_usd = Some(target);
        }
        if changed_max {
            overrides.micro_trade_max_usd = Some(target);
        }
    }
    if changed_min || changed_max {
        state.log(
            "warn",
            "adaptive_schema",
            format!("Adjusted micro-trade sizing from OctoBot exchange minimum: min=${target:.2}"),
            json!({
                "recommended_micro_trade_min_usd": target,
                "micro_trade_min_usd_changed": changed_min,
                "micro_trade_max_usd_changed": changed_max,
            }),
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct BacktestProfitStats {
    mean: f64,
    median: f64,
    min: f64,
}

fn collect_successful_backtest_profits(
    history: &std::collections::VecDeque<backtest::BacktestSummary>,
) -> Vec<f64> {
    history
        .iter()
        .filter_map(|summary| summary.profitability_pct)
        .filter(|profit| profit.is_finite())
        .collect()
}

fn calculate_backtest_profit_stats(values: &[f64]) -> Option<BacktestProfitStats> {
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let middle = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    };
    let min = sorted[0];
    Some(BacktestProfitStats { mean, median, min })
}

fn round_tuning_value(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn propose_backtest_tuning_candidate(
    config: &TradingConfig,
    state: &TradingState,
    latest_profit_pct: f64,
    assessment: &backtest::ApproachAssessment,
) -> Option<(TradingConfigOverride, String)> {
    let mut candidate = state.config_overrides.clone().unwrap_or_default();
    let current_threshold = candidate
        .fuzzy_confidence_threshold
        .unwrap_or(config.fuzzy_confidence_threshold);
    let current_max_positions = candidate
        .max_open_positions
        .unwrap_or(config.max_open_positions)
        .max(1);
    let current_min_usd = candidate
        .micro_trade_min_usd
        .unwrap_or(config.micro_trade_min_usd)
        .max(0.01);
    let current_max_usd = candidate
        .micro_trade_max_usd
        .unwrap_or(config.micro_trade_max_usd)
        .max(current_min_usd);
    let current_fuzzy_weight = candidate.fuzzy_weight.unwrap_or(config.fuzzy_weight);
    let current_roi_penalty = candidate
        .decision_roi_feedback_max_confidence_penalty
        .unwrap_or(config.decision_roi_feedback_max_confidence_penalty);
    let current_roi_signal_adjustment = candidate
        .decision_roi_feedback_max_signal_adjustment
        .unwrap_or(config.decision_roi_feedback_max_signal_adjustment);

    let (threshold_step, size_scale, weight_step, reduce_positions) = match assessment {
        backtest::ApproachAssessment::Marginal => {
            let scale = if latest_profit_pct < 0.0 { 0.95 } else { 0.98 };
            (0.02, scale, -0.03, latest_profit_pct < 0.0)
        }
        backtest::ApproachAssessment::Unprofitable => (0.05, 0.88, -0.06, true),
        _ => return None,
    };

    let mut reasons = Vec::new();

    let new_threshold = (current_threshold + threshold_step).clamp(0.42, 0.92);
    if new_threshold > current_threshold + f64::EPSILON {
        candidate.fuzzy_confidence_threshold = Some(round_tuning_value(new_threshold));
        reasons.push(format!(
            "raise confidence gate {:.2}→{:.2}",
            current_threshold, new_threshold
        ));
    }

    let new_weight = (current_fuzzy_weight + weight_step).clamp(0.15, 0.85);
    if (new_weight - current_fuzzy_weight).abs() > f64::EPSILON {
        candidate.fuzzy_weight = Some(round_tuning_value(new_weight));
        reasons.push(format!(
            "shift fuzzy blend {:.2}→{:.2}",
            current_fuzzy_weight, new_weight
        ));
    }

    let new_max_usd = (current_max_usd * size_scale)
        .max(current_min_usd)
        .max(0.01);
    if new_max_usd + f64::EPSILON < current_max_usd {
        candidate.micro_trade_max_usd = Some(round_tuning_value(new_max_usd));
        reasons.push(format!(
            "cap max trade ${:.2}→${:.2}",
            current_max_usd, new_max_usd
        ));
    }

    if reduce_positions && current_max_positions > 1 {
        let new_max_positions = current_max_positions.saturating_sub(1).max(1);
        if new_max_positions < current_max_positions {
            candidate.max_open_positions = Some(new_max_positions);
            reasons.push(format!(
                "reduce max positions {}→{}",
                current_max_positions, new_max_positions
            ));
        }
    }

    if latest_profit_pct < 0.0 {
        let penalty_step = if matches!(assessment, backtest::ApproachAssessment::Unprofitable) {
            0.07
        } else {
            0.03
        };
        let new_penalty = (current_roi_penalty + penalty_step).clamp(0.0, 0.95);
        if new_penalty > current_roi_penalty + f64::EPSILON {
            candidate.decision_roi_feedback_max_confidence_penalty =
                Some(round_tuning_value(new_penalty));
            reasons.push(format!(
                "increase ROI confidence penalty {:.2}→{:.2}",
                current_roi_penalty, new_penalty
            ));
        }

        let signal_step = if matches!(assessment, backtest::ApproachAssessment::Unprofitable) {
            0.05
        } else {
            0.02
        };
        let new_signal_adjustment = (current_roi_signal_adjustment + signal_step).clamp(0.0, 0.5);
        if new_signal_adjustment > current_roi_signal_adjustment + f64::EPSILON {
            candidate.decision_roi_feedback_max_signal_adjustment =
                Some(round_tuning_value(new_signal_adjustment));
            reasons.push(format!(
                "increase ROI signal adaptation {:.2}→{:.2}",
                current_roi_signal_adjustment, new_signal_adjustment
            ));
        }
    }

    let candidate_max = candidate
        .micro_trade_max_usd
        .unwrap_or(config.micro_trade_max_usd)
        .max(0.01);
    let candidate_min = candidate
        .micro_trade_min_usd
        .unwrap_or(config.micro_trade_min_usd)
        .max(0.01);
    if candidate_min > candidate_max {
        candidate.micro_trade_min_usd = Some(round_tuning_value(candidate_max));
    }

    if reasons.is_empty() {
        None
    } else {
        Some((candidate, reasons.join("; ")))
    }
}

fn apply_backtest_auto_tuning(
    config: &TradingConfig,
    state: &mut TradingState,
    latest_summary: &backtest::BacktestSummary,
) {
    let Some(latest_profit_pct) = latest_summary.profitability_pct.filter(|p| p.is_finite()) else {
        return;
    };
    let now = now_ts();

    if let Some(trial) = state.backtest_auto_tune.active_trial.clone() {
        let mut validation_profits = state
            .backtest_history
            .iter()
            .skip(trial.history_len_at_start)
            .filter_map(|summary| summary.profitability_pct)
            .filter(|profit| profit.is_finite())
            .collect::<Vec<_>>();
        if validation_profits.len() >= BACKTEST_AUTOTUNE_VALIDATION_RUNS {
            if validation_profits.len() > BACKTEST_AUTOTUNE_VALIDATION_RUNS {
                let skip = validation_profits.len() - BACKTEST_AUTOTUNE_VALIDATION_RUNS;
                validation_profits = validation_profits.into_iter().skip(skip).collect();
            }
            if let Some(validation_stats) = calculate_backtest_profit_stats(&validation_profits) {
                let mean_improvement = validation_stats.mean - trial.baseline_mean_profit_pct;
                let median_regression = trial.baseline_median_profit_pct - validation_stats.median;
                let acceptable = mean_improvement >= BACKTEST_AUTOTUNE_MIN_MEAN_IMPROVEMENT_PCT
                    && median_regression <= BACKTEST_AUTOTUNE_MAX_MEDIAN_REGRESSION_PCT;

                if acceptable {
                    let action = format!(
                        "Kept adaptive tuning profile (validation mean {:+.2}% vs baseline {:+.2}%, median {:+.2}% vs {:+.2}%)",
                        validation_stats.mean,
                        trial.baseline_mean_profit_pct,
                        validation_stats.median,
                        trial.baseline_median_profit_pct
                    );
                    state.backtest_auto_tune.active_trial = None;
                    state.backtest_auto_tune.cooldown_until = None;
                    state.backtest_auto_tune.last_action = Some(action.clone());
                    state.backtest_auto_tune.last_action_at = Some(now);
                    state.log(
                        "info",
                        "backtest_tuning",
                        action,
                        json!({
                            "validation_runs": validation_profits.len(),
                            "validation_mean_profit_pct": validation_stats.mean,
                            "validation_median_profit_pct": validation_stats.median,
                            "validation_min_profit_pct": validation_stats.min,
                            "baseline_mean_profit_pct": trial.baseline_mean_profit_pct,
                            "baseline_median_profit_pct": trial.baseline_median_profit_pct,
                            "trigger_assessment": trial.trigger_assessment,
                        }),
                    );
                } else {
                    let action = format!(
                        "Reverted adaptive tuning profile (validation mean {:+.2}% vs baseline {:+.2}%)",
                        validation_stats.mean, trial.baseline_mean_profit_pct
                    );
                    state.config_overrides = trial.previous_overrides.clone();
                    state.backtest_auto_tune.active_trial = None;
                    state.backtest_auto_tune.cooldown_until =
                        Some(now + BACKTEST_AUTOTUNE_COOLDOWN_SECONDS);
                    state.backtest_auto_tune.last_action = Some(action.clone());
                    state.backtest_auto_tune.last_action_at = Some(now);
                    state.log(
                        "warn",
                        "backtest_tuning",
                        action,
                        json!({
                            "validation_runs": validation_profits.len(),
                            "validation_mean_profit_pct": validation_stats.mean,
                            "validation_median_profit_pct": validation_stats.median,
                            "validation_min_profit_pct": validation_stats.min,
                            "baseline_mean_profit_pct": trial.baseline_mean_profit_pct,
                            "baseline_median_profit_pct": trial.baseline_median_profit_pct,
                            "cooldown_seconds": BACKTEST_AUTOTUNE_COOLDOWN_SECONDS,
                            "trigger_assessment": trial.trigger_assessment,
                        }),
                    );
                }
            }
        }
        return;
    }

    if let Some(cooldown_until) = state.backtest_auto_tune.cooldown_until
        && now < cooldown_until
    {
        return;
    }

    if !matches!(
        latest_summary.assessment,
        backtest::ApproachAssessment::Marginal | backtest::ApproachAssessment::Unprofitable
    ) {
        return;
    }

    let successful_profits = collect_successful_backtest_profits(&state.backtest_history);
    if successful_profits.len() < BACKTEST_AUTOTUNE_BASELINE_RUNS {
        return;
    }
    let baseline_slice = &successful_profits[successful_profits
        .len()
        .saturating_sub(BACKTEST_AUTOTUNE_BASELINE_RUNS)..];
    let Some(baseline_stats) = calculate_backtest_profit_stats(baseline_slice) else {
        return;
    };

    let Some((candidate_overrides, rationale)) = propose_backtest_tuning_candidate(
        config,
        state,
        latest_profit_pct,
        &latest_summary.assessment,
    ) else {
        return;
    };
    let previous_overrides = state.config_overrides.clone();
    if previous_overrides.as_ref() == Some(&candidate_overrides) {
        return;
    }

    state.config_overrides = Some(candidate_overrides.clone());
    state.backtest_auto_tune.active_trial = Some(state::BacktestAutoTuneTrial {
        started_at: now,
        history_len_at_start: state.backtest_history.len(),
        baseline_mean_profit_pct: baseline_stats.mean,
        baseline_median_profit_pct: baseline_stats.median,
        baseline_samples: baseline_slice.len(),
        previous_overrides,
        candidate_overrides,
        trigger_assessment: latest_summary.assessment.to_string(),
    });
    let action = format!(
        "Started adaptive tuning trial after {} backtest (profit {:+.2}%): {}",
        latest_summary.assessment, latest_profit_pct, rationale
    );
    state.backtest_auto_tune.last_action = Some(action.clone());
    state.backtest_auto_tune.last_action_at = Some(now);
    state.log(
        "warn",
        "backtest_tuning",
        action,
        json!({
            "baseline_runs": baseline_slice.len(),
            "baseline_mean_profit_pct": baseline_stats.mean,
            "baseline_median_profit_pct": baseline_stats.median,
            "latest_profit_pct": latest_profit_pct,
            "latest_assessment": latest_summary.assessment.to_string(),
            "validation_runs_required": BACKTEST_AUTOTUNE_VALIDATION_RUNS,
            "min_mean_improvement_pct": BACKTEST_AUTOTUNE_MIN_MEAN_IMPROVEMENT_PCT,
        }),
    );
}

fn adaptive_schema_has_observations(schema: &adaptive_schema::AdaptiveApiSchema) -> bool {
    !schema.endpoints.is_empty()
        || !schema.semantic_hints.is_empty()
        || !schema.numeric_hints.is_empty()
        || !schema.recent_adjustments.is_empty()
}

pub(crate) fn degraded_live_execution_reason(
    consensus: &advisor::AiConsensus,
    config: &TradingConfig,
) -> Option<String> {
    if consensus.responders == 0 {
        return Some("No advisor responses available".to_string());
    }

    let requested = config.max_parallel_advisors.max(1);
    let attempted = (consensus.responders + consensus.failures).max(1);
    let expected = requested.min(attempted);
    let coverage = consensus_coverage(consensus);
    let average_risk = consensus_average_risk(consensus);
    let agreement = consensus_agreement(consensus);
    let single_responder_is_strong = consensus.responders == 1
        && consensus.confidence >= 0.62
        && average_risk < 0.68
        && agreement >= 0.55;

    if expected >= 2 && consensus.responders * 2 < expected && !single_responder_is_strong {
        return Some(format!(
            "Advisor quorum too low ({}/{} responders)",
            consensus.responders, expected
        ));
    }

    if consensus.failures > 0 && coverage < 0.35 && !single_responder_is_strong {
        return Some(format!(
            "Advisor coverage too low after failures ({:.0}% coverage)",
            coverage * 100.0
        ));
    }

    if average_risk >= 0.72 {
        return Some(format!(
            "Consensus risk too high ({average_risk:.2} >= 0.72)"
        ));
    }

    if consensus.responders >= 3 && agreement < 0.30 {
        return Some(format!(
            "Advisor disagreement too high ({agreement:.2} agreement)"
        ));
    }

    None
}

fn consensus_coverage(consensus: &advisor::AiConsensus) -> f64 {
    let total = consensus.responders + consensus.failures;
    if total == 0 {
        0.0
    } else {
        consensus.responders as f64 / total as f64
    }
}

fn consensus_agreement(consensus: &advisor::AiConsensus) -> f64 {
    consensus
        .vote_distribution
        .get("agreement")
        .and_then(serde_json::Value::as_f64)
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(1.0)
}

fn consensus_average_risk(consensus: &advisor::AiConsensus) -> f64 {
    if let Some(value) = consensus
        .vote_distribution
        .get("average_risk")
        .and_then(serde_json::Value::as_f64)
        .map(|value| value.clamp(0.0, 1.0))
    {
        return value;
    }

    let mut weighted_risk = 0.0;
    let mut weighted_total = 0.0;
    for advice in consensus.advices.iter().filter(|advice| advice.parsed_ok) {
        let weight = advice.weight.max(0.05);
        weighted_risk += advice.risk_score.clamp(0.0, 1.0) * weight;
        weighted_total += weight;
    }
    if weighted_total > 0.0 {
        (weighted_risk / weighted_total).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

async fn resolve_trade_markouts(
    state: &SharedTradingState,
    snapshots: &[MarketSnapshot],
    config: &TradingConfig,
) {
    if snapshots.is_empty() {
        return;
    }
    let round_trip_cost_bps = 2.0 * (config.estimated_fee_bps + config.estimated_slippage_bps);
    let mut state = state.0.lock().await;
    let resolved = state
        .outcome_ledger
        .resolve_due(snapshots, now_ts(), round_trip_cost_bps);
    if resolved > 0 {
        state.log(
            "info",
            "outcomes",
            format!("Resolved {resolved} fixed-horizon trade markout(s)"),
            json!({
                "resolved": resolved,
                "horizon_seconds": config.markout_horizon_seconds,
                "round_trip_cost_bps": round_trip_cost_bps,
            }),
        );
    }
}

// ---------------------------------------------------------------------------
// Trade execution
// ---------------------------------------------------------------------------

async fn execute_if_warranted(
    octobot: &OctobotClient,
    decision: &TradeDecision,
    state: &SharedTradingState,
    config: &TradingConfig,
) {
    let mut execution_amount_usd = decision.amount_usd;
    let mut execution_exchange = decision.exchange.clone();
    let mut execution_reference_price = decision.reference_price;
    let mut execution_round_trip_cost_bps = decision.economics.estimated_round_trip_cost_bps;
    let min_execution_usd = effective_micro_trade_floor_usd(state, config).await;

    match &decision.action {
        TradeAction::Hold => {
            debug!("trading: hold — no trade placed");
            return;
        }
        TradeAction::Cancel => {
            // Cancel pending override order if any.
            debug!("trading: cancel action — no new order to place");
            let mut s = state.0.lock().await;
            s.pending_override = None;
            return;
        }
        _ => {}
    }

    let side = match &decision.action {
        TradeAction::Buy | TradeAction::StrongBuy => "buy",
        TradeAction::Sell | TradeAction::StrongSell => "sell",
        _ => return,
    };

    if decision.exchange.is_empty() || decision.symbol.is_empty() {
        warn!("trading: decision has no target exchange/symbol — skipping");
        state
            .log_warn("execute", "No target exchange/symbol — trade skipped")
            .await;
        return;
    }

    if !decision.override_applied {
        let now = now_ts();
        let advisory_age = (now - decision.created_at).max(0.0);
        if advisory_age > config.advisory_ttl_seconds {
            state
                .log_warn(
                    "execute",
                    format!(
                        "Trade skipped: advisory expired ({advisory_age:.1}s > {:.1}s)",
                        config.advisory_ttl_seconds
                    ),
                )
                .await;
            return;
        }
        let Some(market_fetched_at) = decision.market_fetched_at else {
            state
                .log_warn("execute", "Trade skipped: decision has no market timestamp")
                .await;
            return;
        };
        let market_age = (now - market_fetched_at).max(0.0);
        if market_age > config.market_snapshot_ttl_seconds {
            state
                .log_warn(
                    "execute",
                    format!(
                        "Trade skipped: source market snapshot expired ({market_age:.1}s > {:.1}s)",
                        config.market_snapshot_ttl_seconds
                    ),
                )
                .await;
            return;
        }
        if !decision.economics.is_worthwhile() {
            state
                .log_warn(
                    "execute",
                    format!(
                        "Trade skipped: expected net edge {:.1}bps below {:.1}bps",
                        decision.economics.expected_net_edge_bps,
                        decision.economics.required_net_edge_bps
                    ),
                )
                .await;
            return;
        }
    }

    if side == "buy"
        && let Some((rerouted_exchange, reason)) = maybe_reroute_execution_exchange(
            state,
            &execution_exchange,
            &decision.symbol,
            side,
            execution_amount_usd,
        )
        .await
    {
        warn!(
            "trading: rerouting buy execution for {} from {} to {} ({})",
            decision.symbol, execution_exchange, rerouted_exchange, reason
        );
        state
            .log_warn(
                "execute",
                format!(
                    "Rerouted BUY execution for {} from {} to {} ({})",
                    decision.symbol, execution_exchange, rerouted_exchange, reason
                ),
            )
            .await;
        execution_exchange = rerouted_exchange;
    }

    if side == "buy" {
        let quote_asset = symbol_quote_asset(&decision.symbol).unwrap_or("quote");
        let Some(max_buy_amount_usd) =
            max_buy_amount_usd_from_balance(octobot, state, &execution_exchange, &decision.symbol)
                .await
        else {
            warn!(
                "trading: buy skipped — {} balance unavailable for {}/{} after portfolio refresh",
                quote_asset, execution_exchange, decision.symbol
            );
            state
                .log_warn(
                    "execute",
                    format!(
                        "Buy skipped for {}/{}: {} balance unavailable after portfolio refresh",
                        execution_exchange, decision.symbol, quote_asset
                    ),
                )
                .await;
            return;
        };

        if max_buy_amount_usd <= 0.0 {
            warn!(
                "trading: buy skipped — non-positive available {} balance for {}/{}",
                quote_asset, execution_exchange, decision.symbol
            );
            state
                .log_warn(
                    "execute",
                    format!(
                        "Buy skipped for {}/{}: non-positive available {} balance",
                        execution_exchange, decision.symbol, quote_asset
                    ),
                )
                .await;
            return;
        }

        if execution_amount_usd > max_buy_amount_usd + f64::EPSILON {
            warn!(
                "trading: capping buy amount for {} on {} from ${:.2} to ${:.2} based on available {} balance",
                decision.symbol,
                execution_exchange,
                execution_amount_usd,
                max_buy_amount_usd,
                quote_asset
            );
            state
                .log_warn(
                    "execute",
                    format!(
                        "Capped buy amount for {} on {} from ${:.2} to ${:.2} based on available {} balance",
                        decision.symbol,
                        execution_exchange,
                        execution_amount_usd,
                        max_buy_amount_usd,
                        quote_asset
                    ),
                )
                .await;
            execution_amount_usd = max_buy_amount_usd;
        }
    }

    if side == "sell" {
        let base_asset = decision
            .symbol
            .split('/')
            .next()
            .map(str::trim)
            .unwrap_or_default();

        if !base_asset.is_empty() {
            let mut sell_availability = ensure_sell_balance_available(
                octobot,
                state,
                &execution_exchange,
                base_asset,
                &decision.symbol,
                config.strict_exchange_selection,
            )
            .await;

            // The market candidate may come from an exchange with no base
            // asset. Route to the exchange that actually holds the asset,
            // even when strict exchange selection is enabled; strict means
            // balances must be exchange-scoped, not that an unfunded venue
            // must be used.
            if let Some((rerouted_exchange, reason)) = maybe_reroute_execution_exchange(
                state,
                &execution_exchange,
                &decision.symbol,
                side,
                execution_amount_usd,
            )
            .await
            {
                warn!(
                    "trading: rerouting sell execution for {} from {} to {} ({})",
                    decision.symbol, execution_exchange, rerouted_exchange, reason
                );
                state
                    .log_warn(
                        "execute",
                        format!(
                            "Rerouted SELL execution for {} from {} to {} ({})",
                            decision.symbol, execution_exchange, rerouted_exchange, reason
                        ),
                    )
                    .await;
                execution_exchange = rerouted_exchange;
                sell_availability = ensure_sell_balance_available(
                    octobot,
                    state,
                    &execution_exchange,
                    base_asset,
                    &decision.symbol,
                    config.strict_exchange_selection,
                )
                .await;
            }

            match sell_availability {
                SellBalanceAvailability::Available {
                    free,
                    total,
                    value_usd,
                } => {
                    if let Some(max_sell_amount_usd) = max_sell_amount_usd_from_balance(
                        octobot,
                        &execution_exchange,
                        &decision.symbol,
                        free,
                        total,
                        value_usd,
                    )
                    .await
                    {
                        if max_sell_amount_usd < 0.01 {
                            warn!(
                                "trading: sell skipped — estimated sellable {} value too small for {} (${:.4})",
                                base_asset, decision.symbol, max_sell_amount_usd
                            );
                            state
                                .log_warn(
                                    "execute",
                                    format!(
                                        "Sell skipped for {}: estimated sellable {base_asset} value too small (${max_sell_amount_usd:.4})",
                                        decision.symbol
                                    ),
                                )
                                .await;
                            return;
                        }

                        if execution_amount_usd > max_sell_amount_usd + f64::EPSILON {
                            warn!(
                                "trading: capping sell amount for {} from ${:.2} to ${:.2} based on available {} balance (free={}, total={})",
                                decision.symbol,
                                execution_amount_usd,
                                max_sell_amount_usd,
                                base_asset,
                                free,
                                total
                            );
                            state
                                .log_warn(
                                    "execute",
                                    format!(
                                        "Capped sell amount for {} from ${:.2} to ${:.2} based on available {base_asset} balance",
                                        decision.symbol, execution_amount_usd, max_sell_amount_usd
                                    ),
                                )
                                .await;
                            execution_amount_usd = max_sell_amount_usd;
                        }
                    }
                }
                SellBalanceAvailability::NonPositive { free, total } => {
                    warn!(
                        "trading: sell skipped — non-positive {base_asset} balance for {} (free={}, total={})",
                        decision.symbol, free, total
                    );
                    state
                        .log_warn(
                            "execute",
                            format!(
                                "Sell skipped for {}: non-positive {} balance (free={}, total={})",
                                decision.symbol, base_asset, free, total
                            ),
                        )
                        .await;
                    return;
                }
                SellBalanceAvailability::Missing => {
                    warn!(
                        "trading: sell skipped — {base_asset} balance unavailable in OctoBot portfolio for {} after refresh",
                        decision.symbol
                    );
                    state
                        .log_warn(
                            "execute",
                            format!(
                                "Sell skipped for {}: {base_asset} balance unavailable in OctoBot portfolio after refresh",
                                decision.symbol
                            ),
                        )
                        .await;
                    return;
                }
            }
        }
    }

    if execution_amount_usd + f64::EPSILON < min_execution_usd {
        warn!(
            "trading: {} skipped — execution amount for {} below effective micro-trade floor (${:.2} < ${:.2})",
            side, decision.symbol, execution_amount_usd, min_execution_usd
        );
        state
            .log_warn(
                "execute",
                format!(
                    "{} skipped for {}: amount ${:.2} below effective micro-trade floor ${:.2}",
                    side.to_ascii_uppercase(),
                    decision.symbol,
                    execution_amount_usd,
                    min_execution_usd
                ),
            )
            .await;
        return;
    }

    // Reprice only after venue routing and balance-based amount capping. This
    // ensures both the economics gate and subsequent fill telemetry refer to
    // the exact venue, symbol and notional that will be submitted.
    if !decision.override_applied {
        let fresh_snapshot = match octobot
            .get_market_snapshot(&execution_exchange, &decision.symbol)
            .await
        {
            Ok(snapshot) => snapshot,
            Err(error) => {
                state
                    .log_warn(
                        "execute",
                        format!("Trade skipped: immediate repricing failed: {error}"),
                    )
                    .await;
                return;
            }
        };
        let fresh_age = (now_ts() - fresh_snapshot.fetched_at).max(0.0);
        if fresh_age > config.market_snapshot_ttl_seconds {
            state
                .log_warn(
                    "execute",
                    format!("Trade skipped: repricing snapshot is {fresh_age:.1}s old"),
                )
                .await;
            return;
        }
        let Some(reference_price) = decision.reference_price else {
            state
                .log_warn("execute", "Trade skipped: decision has no reference price")
                .await;
            return;
        };
        let adverse_drift_bps =
            adverse_reprice_drift_bps(side, reference_price, fresh_snapshot.price);
        let fresh_cost = {
            let current = state.0.lock().await;
            current.execution_telemetry.estimate_round_trip(
                &fresh_snapshot,
                side,
                execution_amount_usd,
                config.estimated_fee_bps,
                config.estimated_slippage_bps,
                &config.quantitative.execution_telemetry,
            )
        };
        execution_round_trip_cost_bps = fresh_cost.round_trip_cost_bps;
        let repriced = reprice_net_edge(
            decision.economics.expected_net_edge_bps,
            decision.economics.estimated_round_trip_cost_bps,
            fresh_cost.round_trip_cost_bps,
            adverse_drift_bps,
        );
        let repriced_net_edge_bps = repriced.net_edge_bps;
        if adverse_drift_bps > config.max_reprice_drift_bps
            || repriced_net_edge_bps + f64::EPSILON < decision.economics.required_net_edge_bps
        {
            state
                .log(
                    "warn",
                    "execute",
                    "Trade skipped after immediate price-drift/economics check",
                    json!({
                        "exchange": execution_exchange,
                        "symbol": decision.symbol,
                        "side": side,
                        "reference_price": reference_price,
                        "current_price": fresh_snapshot.price,
                        "adverse_drift_bps": adverse_drift_bps,
                        "fresh_round_trip_cost_bps": fresh_cost.round_trip_cost_bps,
                        "decision_round_trip_cost_bps": decision.economics.estimated_round_trip_cost_bps,
                        "cost_regression_bps": repriced.cost_regression_bps,
                        "projected_market_cost_bps": fresh_cost.projected_market_cost_bps,
                        "empirical_market_cost_bps": fresh_cost.empirical_market_cost_bps,
                        "empirical_samples": fresh_cost.empirical_samples,
                        "used_cost_fallback": fresh_cost.used_fallback,
                        "max_reprice_drift_bps": config.max_reprice_drift_bps,
                        "repriced_net_edge_bps": repriced_net_edge_bps,
                        "required_net_edge_bps": decision.economics.required_net_edge_bps,
                    }),
                )
                .await;
            return;
        }
        execution_reference_price = Some(fresh_snapshot.price);
    }

    let Some(intent_key) = execution_intent_key(side, &execution_exchange, &decision.symbol) else {
        state
            .log_warn("execute", "Unable to derive order intent — trade skipped")
            .await;
        return;
    };
    let paper_policy = PaperQualificationPolicy {
        min_evaluations: config.paper_qualification_min_evaluations,
        min_validated_intents: config.paper_qualification_min_validated_intents,
        validity_seconds: config.paper_qualification_validity_seconds as f64,
        intent_lease_seconds: config.execution_lease_seconds,
    };
    let build_revision = crate::build_info::revision();
    let paper_qualified = {
        let current = state.0.lock().await;
        current
            .paper_qualification
            .is_qualified(&build_revision, now_ts(), paper_policy)
    };
    let profitability_qualified = {
        let current = state.0.lock().await;
        if !config.backtesting_enabled {
            true
        } else {
            current.last_backtest.as_ref().is_some_and(|summary| {
                matches!(summary.assessment, backtest::ApproachAssessment::Viable)
                    && summary.profitability_pct.is_some_and(|value| {
                        value.is_finite() && value >= config.backtest_profitability_threshold
                    })
            })
        }
    };
    let auto_live_ready =
        config.live_execution_auto_gate_enabled && paper_qualified && profitability_qualified;
    // `live_execution_enabled` is an operator permission, not a bypass of
    // qualification.  In particular, a deployment with that flag set must
    // still remain in paper-observation mode until the current build has
    // qualified.  This allows a clean paper window to accumulate instead of
    // entering the live path and silently starving qualification evidence.
    let effective_live_execution = (config.live_execution_enabled || auto_live_ready)
        && paper_qualified
        && profitability_qualified;

    if !effective_live_execution {
        if decision.override_applied {
            state
                .log_warn(
                    "paper",
                    "Operator overrides are not counted as paper qualification evidence",
                )
                .await;
            return;
        }
        let (is_new, qualified, validated_intents, observed_evaluations) = {
            let mut current = state.0.lock().await;
            let evaluation_count = current.evaluation_count.saturating_add(1);
            let now = now_ts();
            let is_new = current.paper_qualification.observe_intent(
                &build_revision,
                &intent_key,
                evaluation_count,
                now,
                paper_policy,
            );
            let qualified =
                current
                    .paper_qualification
                    .is_qualified(&build_revision, now, paper_policy);
            (
                is_new,
                qualified,
                current.paper_qualification.validated_intents,
                current.paper_qualification.observed_evaluations,
            )
        };
        state.persist(&PathBuf::from(&config.data_path)).await;
        state
            .log(
                if is_new { "info" } else { "warn" },
                "paper",
                if is_new {
                    "Paper intent passed every read-only execution gate"
                } else {
                    "Duplicate paper intent rejected"
                },
                json!({
                    "build_revision": build_revision,
                    "intent_key": intent_key,
                    "qualified": qualified,
                    "validated_intents": validated_intents,
                    "observed_evaluations": observed_evaluations,
                }),
            )
            .await;
        return;
    }
    if auto_live_ready && !config.live_execution_enabled {
        state
            .log(
                "info",
                "execute",
                "LIVE_EXECUTION_AUTO_ENABLED",
                json!({
                    "paper_qualified": paper_qualified,
                    "profitability_qualified": profitability_qualified,
                    "backtest_profitability_threshold": config.backtest_profitability_threshold,
                }),
            )
            .await;
    }
    if config.paper_qualification_required {
        if !paper_qualified {
            state
                .log(
                    "error",
                    "execute",
                    "Live order blocked: current build has not passed paper qualification",
                    json!({
                        "build_revision": build_revision,
                        "required_evaluations": config.paper_qualification_min_evaluations,
                        "required_validated_intents": config.paper_qualification_min_validated_intents,
                    }),
                )
                .await;
            return;
        }
    }
    if config.backtesting_enabled && !profitability_qualified {
        state
            .log(
                "warn",
                "execute",
                "Live order blocked: latest backtest has not met profitability gate",
                json!({
                    "required_profitability_pct": config.backtest_profitability_threshold,
                }),
            )
            .await;
        return;
    }
    if let Err(reason) =
        claim_execution_intent(state, &intent_key, config, decision.override_applied).await
    {
        warn!(
            intent_key = %intent_key,
            reason = %reason,
            "trading: duplicate/in-flight order intent skipped"
        );
        state
            .log(
                "warn",
                "execute",
                format!("Duplicate order intent skipped: {reason}"),
                json!({ "intent_key": intent_key }),
            )
            .await;
        return;
    }
    // The lease must be durable before any mutating OctoBot endpoint is called.
    if let Err(err) = state
        .persist_checked(&PathBuf::from(&config.data_path))
        .await
    {
        release_execution_intent(state, &intent_key).await;
        state
            .log_error(
                "execute",
                format!("Trade skipped: unable to persist execution intent lease: {err}"),
            )
            .await;
        return;
    }

    let pair_activation = match octobot
        .ensure_trading_pair_active_for_order(&execution_exchange, &decision.symbol)
        .await
    {
        Ok(status) => status,
        Err(err) => {
            release_and_persist_execution_intent(state, &intent_key, config).await;
            warn!(
                "trading: {} skipped — failed to validate OctoBot market-status activation for {}/{}: {}",
                side, execution_exchange, decision.symbol, err
            );
            state
                .log_error(
                    "execute",
                    format!(
                        "{} skipped for {}/{}: failed to validate OctoBot market-status activation ({err})",
                        side.to_ascii_uppercase(),
                        execution_exchange,
                        decision.symbol
                    ),
                )
                .await;
            return;
        }
    };

    if !pair_activation.ready {
        release_and_persist_execution_intent(state, &intent_key, config).await;
        warn!(
            "trading: {} skipped — OctoBot pair activation pending for {}/{}: {}",
            side, execution_exchange, decision.symbol, pair_activation.message
        );
        state
            .log_warn(
                "execute",
                format!(
                    "{} skipped for {}/{}: {}",
                    side.to_ascii_uppercase(),
                    execution_exchange,
                    decision.symbol,
                    pair_activation.message
                ),
            )
            .await;
        return;
    }

    let result = if side == "buy" {
        octobot
            .place_buy_order(&execution_exchange, &decision.symbol, execution_amount_usd)
            .await
    } else {
        octobot
            .place_sell_order(&execution_exchange, &decision.symbol, execution_amount_usd)
            .await
    };

    match result {
        Ok(order) => {
            let executed_at = now_ts();
            info!(
                "trading: {} order placed — id={} {}/{} ${:.2}",
                side, order.order_id, execution_exchange, decision.symbol, execution_amount_usd
            );
            let trade = ExecutedTrade {
                ts: executed_at,
                exchange: execution_exchange.clone(),
                symbol: decision.symbol.clone(),
                action: decision.action.clone(),
                amount_usd: execution_amount_usd,
                price: order.price.or(execution_reference_price),
                order_id: Some(order.order_id.clone()),
                confidence: decision.confidence,
                rationale: decision.rationale.clone(),
                ai_votes: serde_json::Value::Null,
                fuzzy_confidence: decision.fuzzy_confidence,
                ai_confidence: decision.ai_confidence,
            };
            {
                let mut s = state.0.lock().await;
                s.in_flight_order_intents.remove(&intent_key);
                if let (Some(reference_price), Some(fill_price)) =
                    (execution_reference_price, order.price)
                {
                    s.execution_telemetry.record_fill(
                        executed_at,
                        &execution_exchange,
                        &decision.symbol,
                        side,
                        &order.order_id,
                        execution_amount_usd,
                        reference_price,
                        fill_price,
                        config.estimated_fee_bps,
                        &config.quantitative.execution_telemetry,
                    );
                }
                s.record_trade(trade);
                if let Some(entry_price) = order.price.or(execution_reference_price) {
                    let holding_horizon_seconds = decision
                        .holding_horizon_seconds
                        .unwrap_or(config.markout_horizon_seconds)
                        .clamp(60, 30 * 86_400);
                    s.outcome_ledger.record(
                        TradeMarkout {
                            order_id: order.order_id.clone(),
                            executed_at,
                            due_at: executed_at + holding_horizon_seconds as f64,
                            exchange: execution_exchange.clone(),
                            symbol: decision.symbol.clone(),
                            action: decision.action.clone(),
                            amount_usd: execution_amount_usd,
                            entry_price,
                            providers: decision.provider_keys.clone(),
                            regime: decision.market_regime.clone(),
                            ..TradeMarkout::default()
                        },
                        config.markout_ledger_size,
                    );
                }
                s.pending_override = None; // Clear override once executed.
            }
            state
                .log(
                    "info",
                    "execute",
                    format!(
                        "{side} order placed: {}/{} ${:.2} id={}",
                        execution_exchange, decision.symbol, execution_amount_usd, order.order_id
                    ),
                    json!({
                        "order_id": order.order_id,
                        "status": order.status,
                        "execution_round_trip_cost_bps": execution_round_trip_cost_bps,
                    }),
                )
                .await;
            // Filled trades are safety-critical history. Persist immediately so
            // a pod restart cannot erase deduplication or ROI feedback records.
            state.persist(&PathBuf::from(&config.data_path)).await;
        }
        Err(err) => {
            // A transport failure after request submission is ambiguous: the
            // exchange may have accepted the order even though Gail never saw
            // the acknowledgement. Retain the durable lease until expiry so
            // the next evaluation cannot duplicate that economic intent.
            state.persist(&PathBuf::from(&config.data_path)).await;
            warn!(
                "trading: {} order failed or acknowledgement was uncertain; retaining intent lease: {}",
                side, err
            );
            state
                .log_error(
                    "execute",
                    format!(
                        "{side} order failed or acknowledgement was uncertain; intent lease retained until expiry: {err}"
                    ),
                )
                .await;
        }
    }
}

async fn claim_execution_intent(
    state: &SharedTradingState,
    intent_key: &str,
    config: &TradingConfig,
    operator_override: bool,
) -> Result<(), String> {
    let now = now_ts();
    {
        let mut locked = state.0.lock().await;
        locked
            .in_flight_order_intents
            .retain(|_, claim| claim.expires_at > now);
        if let Some(existing) = locked.in_flight_order_intents.get(intent_key) {
            return Err(format!(
                "an equivalent order is leased by authority {} for another {:.0}s",
                existing.authority,
                (existing.expires_at - now).max(0.0)
            ));
        }

        if !operator_override
            && let Some(previous) = locked.recent_trades.iter().rev().find(|trade| {
                trade_action_side(&trade.action)
                    .and_then(|side| execution_intent_key(side, &trade.exchange, &trade.symbol))
                    .as_deref()
                    == Some(intent_key)
            })
        {
            let age = (now - previous.ts).max(0.0);
            if age < config.min_trade_interval_seconds as f64 {
                return Err(format!(
                    "equivalent order filled {:.0}s ago; {:.0}s cooldown remains",
                    age,
                    config.min_trade_interval_seconds as f64 - age,
                ));
            }
        }

        locked.in_flight_order_intents.insert(
            intent_key.to_string(),
            ExecutionIntentClaim {
                authority: config.execution_authority.clone(),
                claimed_at: now,
                expires_at: now + config.execution_lease_seconds,
            },
        );
    }
    Ok(())
}

async fn release_execution_intent(state: &SharedTradingState, intent_key: &str) {
    state
        .0
        .lock()
        .await
        .in_flight_order_intents
        .remove(intent_key);
}

async fn release_and_persist_execution_intent(
    state: &SharedTradingState,
    intent_key: &str,
    config: &TradingConfig,
) {
    release_execution_intent(state, intent_key).await;
    state.persist(&PathBuf::from(&config.data_path)).await;
}

async fn effective_micro_trade_floor_usd(
    state: &SharedTradingState,
    config: &TradingConfig,
) -> f64 {
    let s = state.0.lock().await;
    s.config_overrides
        .as_ref()
        .and_then(|overrides| overrides.micro_trade_min_usd)
        .unwrap_or(config.micro_trade_min_usd)
        .max(0.01)
}

#[derive(Clone, Copy, Debug)]
enum SellBalanceAvailability {
    Available {
        free: f64,
        total: f64,
        value_usd: Option<f64>,
    },
    Missing,
    NonPositive {
        free: f64,
        total: f64,
    },
}

#[derive(Clone, Debug)]
struct ExchangeAssetBalanceCandidate {
    exchange: String,
    free: f64,
    total: f64,
    value_usd: Option<f64>,
}

async fn maybe_reroute_execution_exchange(
    state: &SharedTradingState,
    requested_exchange: &str,
    symbol: &str,
    side: &str,
    amount_usd: f64,
) -> Option<(String, String)> {
    let s = state.0.lock().await;
    let portfolio = s.current_portfolio.as_ref()?;
    select_execution_exchange_from_portfolio(
        requested_exchange,
        symbol,
        side,
        amount_usd,
        portfolio,
        &s.available_exchanges,
    )
}

fn select_execution_exchange_from_portfolio(
    requested_exchange: &str,
    symbol: &str,
    side: &str,
    amount_usd: f64,
    portfolio: &OctobotPortfolio,
    available_exchanges: &[OctobotExchange],
) -> Option<(String, String)> {
    if portfolio.exchange_currencies.is_empty() {
        return None;
    }

    let tracked_asset = if side.eq_ignore_ascii_case("sell") {
        symbol_base_asset(symbol)?
    } else if side.eq_ignore_ascii_case("buy") {
        symbol_quote_asset(symbol)?
    } else {
        return None;
    };

    let mut candidates: Vec<ExchangeAssetBalanceCandidate> = portfolio
        .exchange_currencies
        .iter()
        .filter(|(exchange, _)| exchange_supports_symbol(available_exchanges, exchange, symbol))
        .filter_map(|(exchange, balances)| {
            let balance = balances
                .iter()
                .find(|(asset, _)| asset.eq_ignore_ascii_case(tracked_asset))
                .map(|(_, balance)| balance)?;
            if !balance.free.is_finite() || balance.free <= 0.0 {
                return None;
            }
            Some(ExchangeAssetBalanceCandidate {
                exchange: exchange.clone(),
                free: balance.free,
                total: balance.total,
                value_usd: balance.value_usd,
            })
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|left, right| {
        right
            .free
            .partial_cmp(&left.free)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                right
                    .total
                    .partial_cmp(&left.total)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                right
                    .value_usd
                    .unwrap_or(0.0)
                    .partial_cmp(&left.value_usd.unwrap_or(0.0))
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.exchange.cmp(&right.exchange))
    });

    let requested = candidates
        .iter()
        .find(|candidate| candidate.exchange.eq_ignore_ascii_case(requested_exchange));

    if side.eq_ignore_ascii_case("sell") {
        if let Some(requested) = requested
            && sell_balance_is_sufficient(requested, amount_usd)
        {
            return None;
        }
    } else if let Some(requested) = requested
        && buy_balance_is_sufficient(requested, tracked_asset, amount_usd)
    {
        return None;
    }

    let best = candidates.first()?;
    if best.exchange.eq_ignore_ascii_case(requested_exchange) {
        return None;
    }

    let reason = if side.eq_ignore_ascii_case("sell") {
        format!(
            "{} balance for {} available on {} (free={:.8})",
            tracked_asset, symbol, best.exchange, best.free
        )
    } else {
        format!(
            "{} buy balance for {} is stronger on {} (free={:.8})",
            tracked_asset, symbol, best.exchange, best.free
        )
    };
    Some((best.exchange.clone(), reason))
}

fn exchange_supports_symbol(
    available_exchanges: &[OctobotExchange],
    exchange: &str,
    symbol: &str,
) -> bool {
    if available_exchanges.is_empty() {
        return true;
    }

    let Some(entry) = available_exchanges
        .iter()
        .find(|entry| entry.name.eq_ignore_ascii_case(exchange))
    else {
        // Exchange discovery can be temporarily stale; don't block routing
        // when portfolio data points to a viable exchange.
        return true;
    };
    if entry.symbols.is_empty() {
        return true;
    }
    entry
        .symbols
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(symbol))
}

fn buy_balance_is_sufficient(
    candidate: &ExchangeAssetBalanceCandidate,
    quote_asset: &str,
    amount_usd: f64,
) -> bool {
    if !amount_usd.is_finite() || amount_usd <= 0.0 {
        return candidate.free > 0.0;
    }

    let required = amount_usd * 0.98;
    if is_stablecoin(quote_asset) {
        return candidate.free + f64::EPSILON >= required;
    }

    if let Some(value_usd) = candidate
        .value_usd
        .filter(|value| value.is_finite() && *value >= 0.0)
    {
        return value_usd + f64::EPSILON >= required;
    }

    candidate.free > 0.0
}

fn sell_balance_is_sufficient(candidate: &ExchangeAssetBalanceCandidate, amount_usd: f64) -> bool {
    if !amount_usd.is_finite() || amount_usd <= 0.0 {
        return candidate.free > 0.0;
    }

    let required = amount_usd * 0.98;
    if let Some(value_usd) = candidate
        .value_usd
        .filter(|value| value.is_finite() && *value >= 0.0)
    {
        return value_usd + f64::EPSILON >= required;
    }

    candidate.free > 0.0
}

async fn ensure_sell_balance_available(
    octobot: &OctobotClient,
    state: &SharedTradingState,
    exchange: &str,
    base_asset: &str,
    symbol: &str,
    strict_exchange_selection: bool,
) -> SellBalanceAvailability {
    let initial = cached_sell_balance(state, exchange, base_asset, strict_exchange_selection).await;

    // Always request one explicit OctoBot refresh cycle before executing a sell.
    // This keeps free balances and lock state aligned with the exchange.
    debug!(
        "trading: refreshing OctoBot portfolio before sell precheck for {} ({})",
        symbol, base_asset
    );
    if let Err(err) = octobot.refresh_portfolio().await {
        warn!(
            "trading: portfolio refresh request failed before sell precheck for {}: {}",
            symbol, err
        );
    }

    match octobot.get_portfolio().await {
        Ok(portfolio) => {
            let availability = if strict_exchange_selection {
                portfolio_balance_state_for_exchange(&portfolio, exchange, base_asset)
            } else {
                portfolio_balance_state(&portfolio, base_asset)
            };
            let mut s = state.0.lock().await;
            s.current_portfolio = Some(portfolio);
            availability
        }
        Err(err) => {
            warn!(
                "trading: portfolio refetch failed after refresh for {}: {}",
                symbol, err
            );
            initial
        }
    }
}

async fn cached_sell_balance(
    state: &SharedTradingState,
    exchange: &str,
    base_asset: &str,
    strict_exchange_selection: bool,
) -> SellBalanceAvailability {
    let s = state.0.lock().await;
    if let Some(portfolio) = s.current_portfolio.as_ref() {
        if strict_exchange_selection {
            portfolio_balance_state_for_exchange(portfolio, exchange, base_asset)
        } else {
            portfolio_balance_state(portfolio, base_asset)
        }
    } else {
        SellBalanceAvailability::Missing
    }
}

async fn max_buy_amount_usd_from_balance(
    octobot: &OctobotClient,
    state: &SharedTradingState,
    exchange: &str,
    symbol: &str,
) -> Option<f64> {
    // Refresh portfolio before buy execution so quote balances are current.
    debug!(
        "trading: refreshing OctoBot portfolio before buy precheck for {} ({})",
        symbol, exchange
    );
    if let Err(err) = octobot.refresh_portfolio().await {
        warn!(
            "trading: portfolio refresh request failed before buy precheck for {}/{}: {}",
            exchange, symbol, err
        );
    }

    match octobot.get_portfolio().await {
        Ok(portfolio) => {
            let max_buy = max_buy_amount_usd_for_exchange(&portfolio, exchange, symbol);
            let mut s = state.0.lock().await;
            s.current_portfolio = Some(portfolio);
            max_buy
        }
        Err(err) => {
            warn!(
                "trading: portfolio refetch failed after refresh for buy precheck {}/{}: {}",
                exchange, symbol, err
            );
            None
        }
    }
}

fn max_buy_amount_usd_for_exchange(
    portfolio: &OctobotPortfolio,
    exchange: &str,
    symbol: &str,
) -> Option<f64> {
    let quote_asset = symbol_quote_asset(symbol)?;
    let exchange_balances = portfolio
        .exchange_currencies
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(exchange))
        .map(|(_, balances)| balances)?;
    let quote_balance = exchange_balances
        .iter()
        .find(|(asset, _)| asset.eq_ignore_ascii_case(quote_asset))
        .map(|(_, balance)| balance)?;

    let buyable_usd = if is_stablecoin(quote_asset) {
        quote_balance.free.max(0.0)
    } else {
        match sellable_value_usd(
            quote_balance.free,
            quote_balance.total,
            quote_balance.value_usd,
            None,
        ) {
            Some(value) => value.max(0.0),
            None => return None,
        }
    };
    if !buyable_usd.is_finite() {
        return None;
    }

    Some(((buyable_usd * BUY_BALANCE_USD_SAFETY_FACTOR) * 100.0).floor() / 100.0)
}

fn portfolio_balance_state(
    portfolio: &OctobotPortfolio,
    base_asset: &str,
) -> SellBalanceAvailability {
    match portfolio.currencies.get(base_asset) {
        Some(balance) if balance.free > 0.0 => SellBalanceAvailability::Available {
            free: balance.free,
            total: balance.total,
            value_usd: balance.value_usd,
        },
        Some(balance) => SellBalanceAvailability::NonPositive {
            free: balance.free,
            total: balance.total,
        },
        None => SellBalanceAvailability::Missing,
    }
}

fn portfolio_balance_state_for_exchange(
    portfolio: &OctobotPortfolio,
    exchange: &str,
    base_asset: &str,
) -> SellBalanceAvailability {
    let Some(balances) = portfolio
        .exchange_currencies
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(exchange))
        .map(|(_, balances)| balances)
    else {
        return SellBalanceAvailability::Missing;
    };
    match balances
        .iter()
        .find(|(asset, _)| asset.eq_ignore_ascii_case(base_asset))
        .map(|(_, balance)| balance)
    {
        Some(balance) if balance.free.is_finite() && balance.free > 0.0 => {
            SellBalanceAvailability::Available {
                free: balance.free,
                total: balance.total,
                value_usd: balance.value_usd,
            }
        }
        Some(balance) => SellBalanceAvailability::NonPositive {
            free: balance.free,
            total: balance.total,
        },
        None => SellBalanceAvailability::Missing,
    }
}

async fn max_sell_amount_usd_from_balance(
    octobot: &OctobotClient,
    exchange: &str,
    symbol: &str,
    free: f64,
    total: f64,
    value_usd: Option<f64>,
) -> Option<f64> {
    let mut sellable_usd = sellable_value_usd(free, total, value_usd, None);
    if sellable_usd.is_none()
        && let Ok(snapshot) = octobot.get_market_snapshot(exchange, symbol).await
    {
        sellable_usd = sellable_value_usd(free, total, value_usd, Some(snapshot.price));
    }
    let sellable_usd = sellable_usd?;
    if !sellable_usd.is_finite() || sellable_usd <= 0.0 {
        return None;
    }
    let capped = ((sellable_usd * SELL_BALANCE_USD_SAFETY_FACTOR) * 100.0).floor() / 100.0;
    if capped > 0.0 { Some(capped) } else { None }
}

fn sellable_value_usd(
    free: f64,
    total: f64,
    value_usd: Option<f64>,
    fallback_price: Option<f64>,
) -> Option<f64> {
    if !free.is_finite() || free <= 0.0 {
        return Some(0.0);
    }

    if let Some(value) = value_usd.filter(|value| value.is_finite() && *value > 0.0) {
        let ratio = if total.is_finite() && total > 0.0 {
            (free / total).clamp(0.0, 1.0)
        } else {
            1.0
        };
        return Some(value * ratio);
    }

    fallback_price
        .filter(|price| price.is_finite() && *price > 0.0)
        .map(|price| free * price)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TARGET_LOCK_CONSENSUS_CONFIDENCE_MIN: f64 = 0.70;
const TARGET_LOCK_CONSENSUS_SIGNAL_MIN: f64 = 0.30;
const TARGET_LOCK_SUPPORT_MIN: f64 = 0.35;
const BUY_BALANCE_USD_SAFETY_FACTOR: f64 = 0.95;
const SELL_BALANCE_USD_SAFETY_FACTOR: f64 = 0.99;
const MARKET_REGIME_CONTAGION_PRICE_WEIGHT: f64 = 0.35;
const MARKET_REGIME_CONTAGION_LEADER_COUNT_MIN: usize = 2;
const MARKET_REGIME_CONTAGION_LEADER_COUNT_MAX: usize = 6;

#[derive(Clone, Debug)]
struct DecisionMarketSelection {
    snapshot: Option<MarketSnapshot>,
    note: String,
    override_reason: Option<String>,
    used_target_signal: bool,
    target_support: f64,
    high_confidence_target: bool,
}

#[derive(Clone, Debug)]
struct TargetSnapshotSupport {
    snapshot: MarketSnapshot,
    signed_support: f64,
    total_support: f64,
    advisors: usize,
}

/// Quant primary owns both direction and market selection. Keeping its exact
/// venue/symbol avoids silently applying a measured strategy to the legacy
/// momentum fallback, while retaining the existing spot-inventory floor.
fn select_quant_primary_market(
    snapshots: &[MarketSnapshot],
    signal: &quant::QuantSignal,
    portfolio: &OctobotPortfolio,
    min_sellable_usd: f64,
) -> DecisionMarketSelection {
    let selected = snapshots
        .iter()
        .find(|snapshot| {
            snapshot.exchange.eq_ignore_ascii_case(&signal.exchange)
                && snapshot.symbol.eq_ignore_ascii_case(&signal.symbol)
                && snapshot_is_usable(snapshot)
        })
        .cloned();
    let Some(snapshot) = selected else {
        let reason = format!(
            "Quant primary target {}/{} is unavailable in the current market snapshot",
            signal.exchange, signal.symbol
        );
        return DecisionMarketSelection {
            snapshot: None,
            note: reason.clone(),
            override_reason: Some(reason),
            used_target_signal: true,
            target_support: 1.0,
            high_confidence_target: true,
        };
    };

    let override_reason = (signal.signal < 0.0
        && !snapshot_sellable_above_floor(&snapshot, portfolio, min_sellable_usd))
    .then(|| {
        format!(
            "Quant sell target {} is below the effective sellable inventory floor ${:.2}",
            snapshot_label(&snapshot),
            min_sellable_usd.max(0.01)
        )
    });
    DecisionMarketSelection {
        note: if override_reason.is_some() {
            "Quant primary target retained for evaluation but blocked by inventory validation"
                .to_string()
        } else {
            format!(
                "Decision market locked to quant primary target {}",
                snapshot_label(&snapshot)
            )
        },
        snapshot: Some(snapshot),
        override_reason,
        used_target_signal: true,
        target_support: 1.0,
        high_confidence_target: true,
    }
}

#[derive(Clone, Debug, Serialize)]
struct ContagionLeader {
    exchange: String,
    symbol: String,
    pressure: f64,
    influence: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MarketRegimeContagion {
    signal: f64,
    confidence: f64,
    leaders: Vec<ContagionLeader>,
}

impl MarketRegimeContagion {
    fn neutral() -> Self {
        Self {
            signal: 0.0,
            confidence: 0.0,
            leaders: Vec::new(),
        }
    }
}

fn quantitative_regime_label(regime: &MarketRegimeContagion) -> &'static str {
    if regime.confidence < 0.30 || regime.signal.abs() < 0.20 {
        "neutral"
    } else if regime.signal > 0.0 {
        "bullish_trend"
    } else {
        "bearish_trend"
    }
}

#[allow(dead_code)]
fn choose_decision_market_candidate(
    snapshots: &[MarketSnapshot],
    consensus: &advisor::AiConsensus,
    fallback_snapshot: Option<&MarketSnapshot>,
    portfolio: &OctobotPortfolio,
    min_sellable_usd: f64,
) -> DecisionMarketSelection {
    let market_regime = compute_market_regime_contagion(snapshots);
    choose_decision_market_candidate_with_regime(
        snapshots,
        consensus,
        fallback_snapshot,
        portfolio,
        min_sellable_usd,
        &market_regime,
    )
}

fn choose_decision_market_candidate_with_regime(
    snapshots: &[MarketSnapshot],
    consensus: &advisor::AiConsensus,
    fallback_snapshot: Option<&MarketSnapshot>,
    portfolio: &OctobotPortfolio,
    min_sellable_usd: f64,
    market_regime: &MarketRegimeContagion,
) -> DecisionMarketSelection {
    if snapshots.is_empty() {
        return DecisionMarketSelection {
            snapshot: None,
            note: "No market snapshots available for decision targeting".to_string(),
            override_reason: None,
            used_target_signal: false,
            target_support: 0.0,
            high_confidence_target: false,
        };
    }

    let consensus_direction = consensus_direction_sign(consensus.signal);
    let enforce_sell_floor = consensus_direction < 0.0;
    let is_sell_eligible = |snapshot: &MarketSnapshot| {
        snapshot_sellable_above_floor(snapshot, portfolio, min_sellable_usd)
    };
    let snapshot_allowed =
        |snapshot: &MarketSnapshot| !enforce_sell_floor || is_sell_eligible(snapshot);

    let fallback = if enforce_sell_floor {
        select_dynamic_sell_market_candidate(snapshots, portfolio, min_sellable_usd, market_regime)
            .or_else(|| fallback_snapshot.cloned().filter(snapshot_allowed))
            .or_else(|| select_best_market_candidate_with_filter(snapshots, snapshot_allowed))
    } else {
        fallback_snapshot
            .cloned()
            .filter(snapshot_allowed)
            .or_else(|| select_best_market_candidate_with_filter(snapshots, snapshot_allowed))
    };
    let fallback_label = fallback
        .as_ref()
        .map(snapshot_label)
        .unwrap_or_else(|| "none".to_string());

    let mut supports: HashMap<String, TargetSnapshotSupport> = HashMap::new();
    let mut strongest_unresolved_target: Option<(String, f64)> = None;
    for advice in consensus.advices.iter().filter(|advice| advice.parsed_ok) {
        let direction = advisory_action_signal(advice.action.as_str());
        if direction.abs() < f64::EPSILON {
            continue;
        }
        let Some(target_hint) = advice
            .target_symbol
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };

        let weight = advisory_target_support_weight(advice);
        if weight <= 0.0 {
            continue;
        }
        let signed_support = direction * weight;
        let support_abs = signed_support.abs();

        if let Some(snapshot) = resolve_target_snapshot_hint(snapshots, target_hint) {
            if !snapshot_allowed(snapshot) {
                continue;
            }
            let key = market_feature_key(&snapshot.exchange, &snapshot.symbol);
            let entry = supports
                .entry(key)
                .or_insert_with(|| TargetSnapshotSupport {
                    snapshot: snapshot.clone(),
                    signed_support: 0.0,
                    total_support: 0.0,
                    advisors: 0,
                });
            entry.signed_support += signed_support;
            entry.total_support += support_abs;
            entry.advisors += 1;
        } else {
            let should_replace = strongest_unresolved_target
                .as_ref()
                .map(|(_, current_support)| support_abs > *current_support + f64::EPSILON)
                .unwrap_or(true);
            if should_replace {
                strongest_unresolved_target = Some((target_hint.to_string(), support_abs));
            }
        }
    }

    let mut strongest_target: Option<TargetSnapshotSupport> = None;
    for support in supports.values() {
        if consensus_direction > 0.0 && support.signed_support <= 0.0 {
            continue;
        }
        if consensus_direction < 0.0 && support.signed_support >= 0.0 {
            continue;
        }
        let should_replace = strongest_target
            .as_ref()
            .map(|current| {
                support.signed_support.abs() > current.signed_support.abs() + f64::EPSILON
                    || ((support.signed_support.abs() - current.signed_support.abs()).abs()
                        <= f64::EPSILON
                        && snapshot_label(&support.snapshot) < snapshot_label(&current.snapshot))
            })
            .unwrap_or(true);
        if should_replace {
            strongest_target = Some(support.clone());
        }
    }

    if let Some(target) = strongest_target {
        let support_abs = target.signed_support.abs();
        let high_confidence_target =
            should_lock_target_by_consensus(consensus, support_abs, target.advisors);
        if high_confidence_target || fallback.is_none() {
            let selected_label = snapshot_label(&target.snapshot);
            let note = if fallback
                .as_ref()
                .is_some_and(|candidate| same_market(candidate, &target.snapshot))
            {
                format!(
                    "Decision market {} aligns with high-confidence AI target support {:.2}",
                    selected_label, support_abs
                )
            } else {
                format!(
                    "Decision market locked to AI target {} (support {:.2}, responders {})",
                    selected_label, support_abs, target.advisors
                )
            };
            return DecisionMarketSelection {
                snapshot: Some(target.snapshot),
                note,
                override_reason: None,
                used_target_signal: true,
                target_support: support_abs,
                high_confidence_target,
            };
        }

        let selected = fallback.clone();
        let note = format!(
            "Decision market kept at {}: strongest AI target {} support {:.2} below lock threshold",
            fallback_label,
            snapshot_label(&target.snapshot),
            support_abs
        );
        return DecisionMarketSelection {
            snapshot: selected,
            note,
            override_reason: None,
            used_target_signal: false,
            target_support: support_abs,
            high_confidence_target: false,
        };
    }

    if let Some((target_hint, support)) = strongest_unresolved_target
        && should_lock_target_by_consensus(consensus, support, 1)
    {
        let reason = format!(
            "High-confidence AI target `{}` could not be mapped to a live tradable market snapshot",
            target_hint
        );
        let note = format!("{reason}; using fallback market {}", fallback_label);
        return DecisionMarketSelection {
            snapshot: fallback,
            note,
            override_reason: Some(reason),
            used_target_signal: false,
            target_support: support,
            high_confidence_target: true,
        };
    }

    let no_sell_eligible_fallback = enforce_sell_floor && fallback.is_none();
    DecisionMarketSelection {
        snapshot: fallback,
        note: if no_sell_eligible_fallback {
            format!(
                "Decision market unavailable: no sell-eligible symbols above effective floor ${:.2}",
                min_sellable_usd.max(0.01)
            )
        } else {
            format!(
                "Decision market defaulted to {} (no aligned AI target support)",
                fallback_label
            )
        },
        override_reason: None,
        used_target_signal: false,
        target_support: 0.0,
        high_confidence_target: false,
    }
}

fn resolve_target_snapshot_hint<'a>(
    snapshots: &'a [MarketSnapshot],
    target_hint: &str,
) -> Option<&'a MarketSnapshot> {
    let normalized = normalize_target_hint(target_hint)?;
    let normalized_key = normalize_symbol_key(&normalized);
    if !normalized_key.is_empty()
        && let Some(snapshot) = snapshots
            .iter()
            .filter(|snapshot| snapshot_is_usable(snapshot))
            .find(|snapshot| normalize_symbol_key(&snapshot.symbol) == normalized_key)
    {
        return Some(snapshot);
    }

    let base_asset =
        if normalized.contains('/') || normalized.contains('-') || normalized.contains('_') {
            normalized
                .split(['/', '-', '_'])
                .next()
                .map(str::trim)
                .unwrap_or_default()
                .to_string()
        } else {
            normalized.clone()
        };

    if base_asset.is_empty() {
        return None;
    }

    preferred_snapshot_for_asset_ref(snapshots, &base_asset).or_else(|| {
        snapshots
            .iter()
            .filter(|snapshot| snapshot_is_usable(snapshot))
            .find(|snapshot| {
                symbol_base_asset(&snapshot.symbol)
                    .is_some_and(|asset| asset.eq_ignore_ascii_case(base_asset.as_str()))
            })
    })
}

fn normalize_target_hint(value: &str) -> Option<String> {
    let trimmed = value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim_matches('`');
    if trimmed.is_empty() || matches!(trimmed.to_ascii_lowercase().as_str(), "null" | "none") {
        return None;
    }
    Some(trimmed.to_ascii_uppercase())
}

fn normalize_symbol_key(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

fn advisory_target_support_weight(advice: &advisor::AiAdvice) -> f64 {
    advice.weight.max(0.05) * advice.confidence.clamp(0.0, 1.0)
}

fn advisory_action_signal(action: &str) -> f64 {
    match action.to_ascii_lowercase().as_str() {
        "strong_buy" => 1.0,
        "buy" => 0.5,
        "sell" => -0.5,
        "strong_sell" => -1.0,
        _ => 0.0,
    }
}

fn consensus_direction_sign(signal: f64) -> f64 {
    if signal > 0.05 {
        1.0
    } else if signal < -0.05 {
        -1.0
    } else {
        0.0
    }
}

fn should_lock_target_by_consensus(
    consensus: &advisor::AiConsensus,
    target_support: f64,
    advisors: usize,
) -> bool {
    consensus.confidence >= TARGET_LOCK_CONSENSUS_CONFIDENCE_MIN
        && consensus.signal.abs() >= TARGET_LOCK_CONSENSUS_SIGNAL_MIN
        && target_support >= TARGET_LOCK_SUPPORT_MIN
        && advisors >= 1
}

fn snapshot_is_usable(snapshot: &MarketSnapshot) -> bool {
    snapshot.price.is_finite() && snapshot.price > 0.0
}

fn same_market(left: &MarketSnapshot, right: &MarketSnapshot) -> bool {
    left.exchange.eq_ignore_ascii_case(&right.exchange)
        && left.symbol.eq_ignore_ascii_case(&right.symbol)
}

fn snapshot_label(snapshot: &MarketSnapshot) -> String {
    format!("{}/{}", snapshot.exchange, snapshot.symbol)
}

/// Select the best candidate market for trading based on signal quality.
/// Prefers high-volume, high-momentum markets.
fn select_best_market_candidate(snapshots: &[MarketSnapshot]) -> Option<MarketSnapshot> {
    select_best_market_candidate_with_filter(snapshots, |_| true)
}

fn select_best_market_candidate_with_filter<F>(
    snapshots: &[MarketSnapshot],
    predicate: F,
) -> Option<MarketSnapshot>
where
    F: Fn(&MarketSnapshot) -> bool,
{
    if snapshots.is_empty() {
        return None;
    }
    // Score: abs(24h change) * log(volume + 1) — highest momentum + volume.
    let best = snapshots
        .iter()
        .filter(|snapshot| predicate(snapshot))
        .max_by(|a, b| {
            let score_a = market_score(a);
            let score_b = market_score(b);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    best.cloned()
}

fn compute_market_regime_contagion(snapshots: &[MarketSnapshot]) -> MarketRegimeContagion {
    if snapshots.is_empty() {
        return MarketRegimeContagion::neutral();
    }

    let adaptive_count = ((snapshots.len() as f64).sqrt().round() as usize)
        .clamp(
            MARKET_REGIME_CONTAGION_LEADER_COUNT_MIN,
            MARKET_REGIME_CONTAGION_LEADER_COUNT_MAX,
        )
        .min(snapshots.len().max(1));
    let leaders = select_contagion_leaders(snapshots, adaptive_count);
    if leaders.is_empty() {
        return MarketRegimeContagion::neutral();
    }

    let mut weighted_signal = 0.0;
    let mut total_weight = 0.0;
    for leader in &leaders {
        let weight = leader.influence.max(0.05);
        weighted_signal += leader.pressure * weight;
        total_weight += weight;
    }
    if total_weight <= f64::EPSILON {
        return MarketRegimeContagion::neutral();
    }

    let signal = (weighted_signal / total_weight).clamp(-1.0, 1.0);
    let average_influence = (leaders.iter().map(|leader| leader.influence).sum::<f64>()
        / leaders.len().max(1) as f64)
        .clamp(0.0, 1.0);
    let breadth = contagion_breadth_alignment(snapshots, signal);
    let confidence = (average_influence * 0.6 + breadth * 0.4).clamp(0.0, 1.0);

    MarketRegimeContagion {
        signal,
        confidence,
        leaders,
    }
}

fn select_contagion_leaders(
    snapshots: &[MarketSnapshot],
    leader_count: usize,
) -> Vec<ContagionLeader> {
    if snapshots.is_empty() {
        return Vec::new();
    }

    let mut ranked = snapshots
        .iter()
        .filter(|snapshot| snapshot_is_usable(snapshot))
        .filter(|snapshot| snapshot_has_stable_quote(snapshot))
        .filter_map(|snapshot| {
            let pressure = normalized_price_trend_signal(snapshot);
            let move_strength = pressure.abs().clamp(0.0, 1.0);
            let liquidity = normalized_snapshot_liquidity(snapshot);
            let influence = (move_strength * 0.55 + liquidity * 0.45).clamp(0.0, 1.0);
            if influence <= f64::EPSILON {
                return None;
            }
            Some(ContagionLeader {
                exchange: snapshot.exchange.clone(),
                symbol: snapshot.symbol.clone(),
                pressure,
                influence,
            })
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|left, right| {
        right
            .influence
            .partial_cmp(&left.influence)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                right
                    .pressure
                    .abs()
                    .partial_cmp(&left.pressure.abs())
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| right.symbol.cmp(&left.symbol))
    });

    let mut seen_assets = HashSet::new();
    ranked
        .into_iter()
        .filter(|leader| {
            symbol_base_asset(&leader.symbol)
                .is_some_and(|asset| seen_assets.insert(asset.trim().to_ascii_uppercase()))
        })
        .take(leader_count.max(1))
        .collect()
}

fn contagion_breadth_alignment(snapshots: &[MarketSnapshot], regime_signal: f64) -> f64 {
    let direction = consensus_direction_sign(regime_signal);
    if direction.abs() < f64::EPSILON {
        return 0.0;
    }

    let mut total = 0usize;
    let mut aligned = 0usize;
    for snapshot in snapshots
        .iter()
        .filter(|snapshot| snapshot_is_usable(snapshot))
        .filter(|snapshot| snapshot_has_stable_quote(snapshot))
    {
        let pressure = normalized_price_trend_signal(snapshot);
        let candidate_direction = consensus_direction_sign(pressure);
        if candidate_direction.abs() < f64::EPSILON {
            continue;
        }
        total += 1;
        if (candidate_direction - direction).abs() <= f64::EPSILON {
            aligned += 1;
        }
    }

    if total == 0 {
        0.0
    } else {
        (aligned as f64 / total as f64).clamp(0.0, 1.0)
    }
}

fn select_dynamic_sell_market_candidate(
    snapshots: &[MarketSnapshot],
    portfolio: &OctobotPortfolio,
    min_sellable_usd: f64,
    market_regime: &MarketRegimeContagion,
) -> Option<MarketSnapshot> {
    if snapshots.is_empty() {
        return None;
    }

    let floor = min_sellable_usd.max(0.01);
    let candidates = snapshots
        .iter()
        .filter(|snapshot| snapshot_is_usable(snapshot))
        .filter(|snapshot| snapshot_has_stable_quote(snapshot))
        .filter_map(|snapshot| {
            let sellable = sellable_value_usd_for_snapshot(portfolio, snapshot)?;
            if sellable + f64::EPSILON < floor {
                return None;
            }
            Some((snapshot, sellable))
        })
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return None;
    }

    let total_sellable_usd = candidates
        .iter()
        .map(|(_, sellable)| *sellable)
        .sum::<f64>()
        .max(0.01);
    let regime_sell_pressure = (-market_regime.signal).max(0.0) * market_regime.confidence;
    let leader_influence_by_market = market_regime
        .leaders
        .iter()
        .map(|leader| {
            (
                market_feature_key(&leader.exchange, &leader.symbol),
                leader.influence,
            )
        })
        .collect::<HashMap<_, _>>();

    candidates
        .into_iter()
        .max_by(
            |(left_snapshot, left_sellable), (right_snapshot, right_sellable)| {
                let left_score = dynamic_sell_candidate_score(
                    left_snapshot,
                    *left_sellable,
                    total_sellable_usd,
                    regime_sell_pressure,
                    &leader_influence_by_market,
                );
                let right_score = dynamic_sell_candidate_score(
                    right_snapshot,
                    *right_sellable,
                    total_sellable_usd,
                    regime_sell_pressure,
                    &leader_influence_by_market,
                );
                left_score
                    .partial_cmp(&right_score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| {
                        left_sellable
                            .partial_cmp(right_sellable)
                            .unwrap_or(Ordering::Equal)
                    })
                    .then_with(|| right_snapshot.symbol.cmp(&left_snapshot.symbol))
            },
        )
        .map(|(snapshot, _)| snapshot.clone())
}

fn dynamic_sell_candidate_score(
    snapshot: &MarketSnapshot,
    sellable_usd: f64,
    total_sellable_usd: f64,
    regime_sell_pressure: f64,
    leader_influence_by_market: &HashMap<String, f64>,
) -> f64 {
    let local_sell_pressure = (-normalized_price_trend_signal(snapshot)).max(0.0);
    let holding_share = (sellable_usd / total_sellable_usd).clamp(0.0, 1.0);
    let contagion_alignment = (regime_sell_pressure * local_sell_pressure).clamp(0.0, 1.0);
    let leader_influence = leader_influence_by_market
        .get(&market_feature_key(&snapshot.exchange, &snapshot.symbol))
        .copied()
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let liquidity = normalized_snapshot_liquidity(snapshot);

    let base_score = local_sell_pressure * 0.42
        + holding_share * 0.33
        + contagion_alignment * 0.20
        + leader_influence * 0.05;
    (base_score * (0.75 + 0.25 * liquidity)).clamp(0.0, 1.0)
}

fn normalized_price_trend_signal(snapshot: &MarketSnapshot) -> f64 {
    snapshot
        .price_change_pct_24h
        .map(|value| (value / 12.0).clamp(-1.0, 1.0))
        .unwrap_or(0.0)
}

fn normalized_snapshot_liquidity(snapshot: &MarketSnapshot) -> f64 {
    ((snapshot.volume_24h.unwrap_or(0.0) + 1.0).ln() / 18.0).clamp(0.0, 1.0)
}

fn market_score(snap: &MarketSnapshot) -> f64 {
    let change_abs = snap.price_change_pct_24h.unwrap_or(0.0).abs();
    let vol = snap.volume_24h.unwrap_or(0.0);
    change_abs * (vol + 1.0).ln()
}

fn build_research_query(config: &TradingConfig, snap: Option<&MarketSnapshot>) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let date = {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        utc_date_from_unix_days((secs / 86_400) as i64)
    };
    let (currency, exchange) = snap
        .map(|s| (s.symbol.clone(), s.exchange.clone()))
        .unwrap_or_else(|| ("BTC/USDT".to_string(), "all".to_string()));

    config
        .research_query_template
        .replace("{currency}", &currency)
        .replace("{exchange}", &exchange)
        .replace("{date}", &date)
}

fn utc_date_from_unix_days(days_since_epoch: i64) -> String {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + if month <= 2 { 1 } else { 0 };
    format!("{year:04}-{month:02}-{day:02}")
}

fn compute_fuzzy_inputs(
    best_market: Option<&MarketSnapshot>,
    historical_market: Option<&MarketHistoricalFeatures>,
    consensus: &advisor::AiConsensus,
    research: &refiner::ResearchContext,
    portfolio: &OctobotPortfolio,
    market_regime: Option<&MarketRegimeContagion>,
    config: &TradingConfig,
) -> FuzzyInputs {
    let live_price_trend = best_market
        .and_then(|m| m.price_change_pct_24h)
        .map(|p| (p / 5.0).clamp(-1.0, 1.0))
        .unwrap_or(0.0);

    let live_volume_ratio = best_market
        .and_then(|m| m.volume_24h)
        .map(|v| (v / 1_000_000.0).clamp(0.0, 2.0)) // rough normalisation
        .unwrap_or(1.0);

    let ai_consensus = consensus.signal.clamp(-1.0, 1.0);

    // Rough research sentiment from match scores.
    let mut research_sentiment = if research.is_empty() {
        0.0
    } else {
        let avg_score: f64 = research.matches.iter().map(|m| m.score).sum::<f64>()
            / research.matches.len().max(1) as f64;
        // High score (close to 1.0) from RAG means relevant content found; treat as neutral.
        // We lean on the AI to interpret; keep sentiment neutral unless explicitly negative.
        (avg_score - 0.5) * 0.4 // gentle signal
    };

    // Portfolio exposure: ratio of non-stablecoin holdings to total.
    let portfolio_exposure = {
        let total = portfolio.total_value_usd.unwrap_or(0.0);
        if total < 0.01 {
            0.0
        } else {
            let stable: f64 = portfolio
                .currencies
                .iter()
                .filter(|(sym, _)| is_stablecoin(sym))
                .map(|(_, b)| b.value_usd.unwrap_or(0.0))
                .sum();
            ((total - stable) / total).clamp(0.0, 1.0)
        }
    };

    let mut price_trend = live_price_trend;
    let mut volume_ratio = live_volume_ratio;
    if let Some(history) = historical_market {
        let weight = config.market_datalake_feature_weight.clamp(0.0, 1.0);
        let historical_momentum = history.momentum_signal();
        let historical_volume = history.volume_regime_ratio().clamp(0.0, 2.0);
        price_trend =
            (live_price_trend * (1.0 - weight) + historical_momentum * weight).clamp(-1.0, 1.0);
        volume_ratio =
            (live_volume_ratio * (1.0 - weight) + historical_volume * weight).clamp(0.0, 2.0);
        let risk_pressure = history.risk_pressure();
        research_sentiment = (research_sentiment - risk_pressure * 0.25).clamp(-1.0, 1.0);
    }
    if let Some(regime) = market_regime {
        let contagion_confidence = regime.confidence.clamp(0.0, 1.0);
        if contagion_confidence > f64::EPSILON {
            let contagion_weight =
                (MARKET_REGIME_CONTAGION_PRICE_WEIGHT * contagion_confidence).clamp(0.0, 1.0);
            let contagion_signal = regime.signal.clamp(-1.0, 1.0);
            price_trend = (price_trend * (1.0 - contagion_weight)
                + contagion_signal * contagion_weight)
                .clamp(-1.0, 1.0);
        }
    }

    FuzzyInputs {
        price_trend,
        volume_ratio,
        ai_consensus,
        research_sentiment,
        portfolio_exposure,
    }
}

fn is_stablecoin(sym: &str) -> bool {
    let lower = sym.to_ascii_lowercase();
    lower.contains("usdt")
        || lower.contains("usdc")
        || lower.contains("busd")
        || lower.contains("dai")
        || lower.contains("usd")
        || lower.contains("eur")
}
