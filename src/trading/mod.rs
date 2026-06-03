/// Gail Crypto Trading Bridge — main module.
///
/// Provides `TradingBridge`, a non-blocking background service that:
///  1. Fetches live market data from OctoBot
///  2. Gathers research context from Refiner
///  3. Consults all configured AI providers in parallel (TradingAdvisor)
///  4. Applies Type-2 fuzzy logic (FuzzyEngine)
///  5. Blends fuzzy + AI signals and applies historical ROI feedback (DecisionEngine)
///  6. Executes only through supported OctoBot trading/command bridges
///  7. Logs all activity in a ring-buffer (SharedTradingState)
///  8. Persists state to disk periodically
///
/// The bridge is entirely non-blocking and runs in its own tokio task.
/// All HTTP handlers access state through `SharedTradingState` (Arc<Mutex<>>).
pub mod advisor;
pub mod backtest;
pub mod config;
pub mod datalake;
pub mod decision;
pub mod fuzzy;
pub mod octobot;
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
use fuzzy::{FuzzyEngine, FuzzyInputs};
use octobot::{MarketSnapshot, OctobotClient, OctobotLogEntry, OctobotPortfolio};
use refiner::RefinerClient;
use state::{ExecutedTrade, SharedTradingState, TradeAction, TradingState};

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

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
    let refiner = RefinerClient::new(
        &config.refiner_base_url,
        config.refiner_api_token.as_deref(),
        config.refiner_timeout_seconds,
    );
    let fuzzy_engine = FuzzyEngine::new();
    let postgres_dsn = service.config().storage.postgres_dsn.clone();
    let advisor = TradingAdvisor::new(service, config.advisor_timeout_seconds);
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

    // Initial OctoBot login.
    if let Err(err) = octobot.login().await {
        warn!("trading: OctoBot login failed at startup: {}", err);
        state
            .log_warn("startup", format!("OctoBot login failed: {err}"))
            .await;
    } else {
        state.log_info("startup", "Trading bridge started").await;
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
    // Persist state every 5 evaluations.
    let mut persist_counter: u32 = 0;
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
                    &octobot,
                    &refiner,
                    &fuzzy_engine,
                    &advisor,
                    &decision_engine,
                    market_data_lake.as_ref(),
                ).await;
                persist_counter += 1;
                if persist_counter >= 5 {
                    state.persist(&data_path).await;
                    persist_counter = 0;
                }

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
                if let Some(ref engine) = backtest_engine {
                    let due = now_ts() - last_backtest_ts >= config.backtest_interval_seconds as f64;
                    if due {
                        info!("trading: running periodic backtest");
                        state.log_info("backtest", "Starting periodic backtesting run").await;
                        let summary = engine.run_with_config(&config).await;
                        let assessment = summary.assessment.to_string();
                        let should_pause = config.backtest_pause_on_failure
                            && summary.assessment == backtest::ApproachAssessment::Unprofitable;
                        {
                            let mut s = state.0.lock().await;
                            s.record_backtest(summary);
                            if should_pause {
                                s.paused = true;
                                s.log_warn("backtest", "Trading paused: approach assessed as unprofitable");
                            }
                        }
                        if should_pause {
                            warn!("trading: bridge paused due to unprofitable backtest result");
                        } else {
                            info!("trading: backtest complete — assessment={}", assessment);
                        }
                        last_backtest_ts = now_ts();
                    }
                }
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

// ---------------------------------------------------------------------------
// Single evaluation cycle
// ---------------------------------------------------------------------------

async fn run_single_evaluation(
    config: &TradingConfig,
    state: &SharedTradingState,
    octobot: &OctobotClient,
    refiner: &RefinerClient,
    fuzzy_engine: &FuzzyEngine,
    advisor: &TradingAdvisor,
    decision_engine: &DecisionEngine,
    market_data_lake: Option<&MarketDataLake>,
) {
    let eval_start = now_ts();
    debug!("trading: starting evaluation cycle");
    state.log_info("eval", "Starting evaluation cycle").await;

    // --- 1. Fetch market data from OctoBot ---
    let (target_exchanges, target_currencies) = resolve_target_market_filters(config, state).await;

    let market_snapshots = octobot
        .get_all_market_snapshots(&target_exchanges, &target_currencies, 20)
        .await;
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

    // --- 2. Build research query ---
    // Keep pre-consensus market ranking for research context only.
    let research_snapshot = select_best_market_candidate(&market_snapshots);
    let research_query = build_research_query(config, research_snapshot.as_ref());

    // Run remaining service calls in parallel so one slow dependency
    // does not serialize the whole evaluation cycle.
    let (portfolio_result, open_orders_result, exchange_info_result, log_feedback_result, research) = tokio::join!(
        octobot.get_portfolio(),
        octobot.get_open_orders(),
        octobot.get_exchange_info(),
        octobot.get_recent_logs(25),
        refiner.research_with_site_hints_best_effort(
            &config.research_index_name,
            &research_query,
            &config.research_site_hints,
            config.research_top_k,
            config.research_max_parallel_queries,
        ),
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

    // --- 3. Consult AI advisors in parallel ---
    let consensus = advisor
        .consult_all(
            &market_snapshots,
            &historical_features,
            &research,
            &portfolio,
            config.max_parallel_advisors,
        )
        .await;

    debug!(
        "trading: AI consensus = action={} signal={:.3} confidence={:.2} responders={}",
        consensus.action, consensus.signal, consensus.confidence, consensus.responders
    );

    // Select final decision market after consensus so high-confidence
    // target symbols are not silently replaced by generic market ranking.
    let decision_market_selection =
        choose_decision_market_candidate(&market_snapshots, &consensus, research_snapshot.as_ref());
    if let Some(reason) = decision_market_selection.override_reason.as_deref() {
        warn!(
            "trading: consensus target override applied with explicit risk justification: {}",
            reason
        );
    }
    debug!(
        "trading: decision market selection => {}",
        decision_market_selection.note
    );
    let decision_snapshot = decision_market_selection.snapshot.clone();
    let decision_snapshot_history = decision_snapshot.as_ref().and_then(|snapshot| {
        historical_features
            .get(&market_feature_key(&snapshot.exchange, &snapshot.symbol))
            .cloned()
    });

    // --- 4. Compute fuzzy inputs ---
    let fuzzy_inputs = compute_fuzzy_inputs(
        decision_snapshot.as_ref(),
        decision_snapshot_history.as_ref(),
        &consensus,
        &research,
        &portfolio,
        config,
    );
    let fuzzy_out = fuzzy_engine.evaluate(&fuzzy_inputs);

    debug!(
        "trading: fuzzy = signal={:.3} confidence={:.2} label={}",
        fuzzy_out.signal, fuzzy_out.confidence, fuzzy_out.label
    );

    // --- 5. Make decision ---
    let mut decision = {
        let s = state.0.lock().await;
        decision_engine.decide(
            &fuzzy_out,
            &consensus,
            decision_snapshot.as_ref(),
            &s,
            config,
        )
    };

    if !decision.override_applied
        && !matches!(decision.action, TradeAction::Hold | TradeAction::Cancel)
        && let Some(reason) = degraded_live_execution_reason(&consensus, config)
    {
        let previous_action = decision.action.clone();
        let previous_confidence = decision.confidence;
        let previous_amount = decision.amount_usd;
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
                    "previous_action": previous_action.to_string(),
                    "previous_confidence": previous_confidence,
                    "previous_amount_usd": previous_amount,
                    "responders": consensus.responders,
                    "failures": consensus.failures,
                    "coverage": consensus_coverage(&consensus),
                    "average_risk": consensus_average_risk(&consensus),
                    "agreement": consensus_agreement(&consensus),
                }),
            )
            .await;
        decision = TradeDecision {
            action: TradeAction::Hold,
            exchange: String::new(),
            symbol: String::new(),
            amount_usd: 0.0,
            rationale: format!("Execution gated: {reason}"),
            ..decision
        };
    }

    decision.rationale =
        merge_rationale_note(&decision.rationale, decision_market_selection.note.as_str());

    info!(
        "trading: decision = {:?} exchange={} symbol={} amount=${:.2} confidence={:.2}",
        decision.action,
        decision.exchange,
        decision.symbol,
        decision.amount_usd,
        decision.confidence
    );

    state
        .log(
            "info",
            "decision",
            format!(
                "{:?} {}/{} ${:.2} conf={:.2}",
                decision.action,
                decision.exchange,
                decision.symbol,
                decision.amount_usd,
                decision.confidence
            ),
            json!({
                "fuzzy_signal": fuzzy_out.signal,
                "fuzzy_confidence": fuzzy_out.confidence,
                "ai_signal": consensus.signal,
                "ai_confidence": consensus.confidence,
                "ai_action": consensus.action,
                "ai_responders": consensus.responders,
                "ai_failures": consensus.failures,
                "ai_vote_distribution": consensus.vote_distribution,
                "market_history": decision_snapshot_history,
                "research_market": research_snapshot.as_ref().map(|snapshot| {
                    json!({
                        "exchange": snapshot.exchange,
                        "symbol": snapshot.symbol,
                    })
                }),
                "decision_market": decision_snapshot.as_ref().map(|snapshot| {
                    json!({
                        "exchange": snapshot.exchange,
                        "symbol": snapshot.symbol,
                    })
                }),
                "market_selection_note": decision_market_selection.note,
                "market_selection_override_reason": decision_market_selection.override_reason,
                "market_selection_used_target": decision_market_selection.used_target_signal,
                "market_selection_target_support": decision_market_selection.target_support,
                "market_selection_target_support_advisors": decision_market_selection.target_support_advisors,
                "market_selection_high_confidence_target": decision_market_selection.high_confidence_target,
                "blended_signal": decision.blended_signal,
                "roi_feedback_applied": decision.roi_feedback_applied,
                "roi_feedback_signal_adjustment": decision.roi_feedback_signal_adjustment,
                "roi_feedback_confidence_multiplier": decision.roi_feedback_confidence_multiplier,
                "roi_feedback_samples": decision.roi_feedback_samples,
                "roi_feedback_avg_directional_roi": decision.roi_feedback_avg_directional_roi,
                "roi_feedback_win_rate": decision.roi_feedback_win_rate,
                "rationale": decision.rationale
            }),
        )
        .await;

    // --- 6. Execute trade if warranted ---
    execute_if_warranted(octobot, &decision, state, config).await;

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

    let mut evaluated = Vec::new();
    for snapshot in &candidates {
        let review = evaluate_symbol_candidate(
            config,
            state,
            refiner,
            fuzzy_engine,
            advisor,
            decision_engine,
            &portfolio,
            snapshot,
            false,
            historical_features.get(&market_feature_key(&snapshot.exchange, &snapshot.symbol)),
        )
        .await;
        evaluated.push(review);
    }
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

    let mut evaluated = Vec::new();
    for snapshot in &candidates {
        let review = evaluate_symbol_candidate(
            config,
            state,
            refiner,
            fuzzy_engine,
            advisor,
            decision_engine,
            &portfolio,
            snapshot,
            true,
            historical_features.get(&market_feature_key(&snapshot.exchange, &snapshot.symbol)),
        )
        .await;
        evaluated.push(review);
    }
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
) -> EvaluatedSymbol {
    let research_query = build_research_query(config, Some(snapshot));
    let research = refiner
        .research_with_site_hints_best_effort(
            &config.research_index_name,
            &research_query,
            &config.research_site_hints,
            config.research_top_k,
            config.research_max_parallel_queries,
        )
        .await;
    let consensus = advisor
        .consult_all(
            std::slice::from_ref(snapshot),
            &historical_features_map(snapshot, historical_features),
            &research,
            portfolio,
            config.max_parallel_advisors,
        )
        .await;
    let fuzzy_inputs = compute_fuzzy_inputs(
        Some(snapshot),
        historical_features,
        &consensus,
        &research,
        portfolio,
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
                && balance.value_usd.unwrap_or(min_holding_usd) >= min_holding_usd
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
    portfolio
        .currencies
        .get(base_asset)
        .is_some_and(|balance| balance.free > 0.0 || balance.total > 0.0)
}

fn holding_value_usd_for_symbol(portfolio: &OctobotPortfolio, symbol: &str) -> Option<f64> {
    let base_asset = symbol_base_asset(symbol)?;
    portfolio
        .currencies
        .get(base_asset)
        .and_then(|balance| balance.value_usd)
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
    if expected >= 2 && consensus.responders * 2 < expected {
        return Some(format!(
            "Advisor quorum too low ({}/{} responders)",
            consensus.responders, expected
        ));
    }

    let coverage = consensus_coverage(consensus);
    if consensus.failures > 0 && coverage < 0.5 {
        return Some(format!(
            "Advisor coverage too low after failures ({:.0}% coverage)",
            coverage * 100.0
        ));
    }

    let average_risk = consensus_average_risk(consensus);
    if average_risk >= 0.72 {
        return Some(format!(
            "Consensus risk too high ({average_risk:.2} >= 0.72)"
        ));
    }

    let agreement = consensus_agreement(consensus);
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

// ---------------------------------------------------------------------------
// Trade execution
// ---------------------------------------------------------------------------

async fn execute_if_warranted(
    octobot: &OctobotClient,
    decision: &TradeDecision,
    state: &SharedTradingState,
    _config: &TradingConfig,
) {
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

    if !_config.live_execution_enabled {
        info!(
            "trading: live execution disabled — decision not sent to OctoBot exchange={} symbol={} action={:?} amount=${:.2}",
            decision.exchange, decision.symbol, decision.action, decision.amount_usd
        );
        state
            .log_warn(
                "execute",
                "Live execution disabled; decision was not sent to OctoBot",
            )
            .await;
        return;
    }

    if side == "sell" {
        let base_asset = decision
            .symbol
            .split('/')
            .next()
            .map(str::trim)
            .unwrap_or_default();

        if !base_asset.is_empty() {
            match ensure_sell_balance_available(octobot, state, base_asset, &decision.symbol).await
            {
                SellBalanceAvailability::Available => {}
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

    let result = if side == "buy" {
        octobot
            .place_buy_order(&decision.exchange, &decision.symbol, decision.amount_usd)
            .await
    } else {
        octobot
            .place_sell_order(&decision.exchange, &decision.symbol, decision.amount_usd)
            .await
    };

    match result {
        Ok(order) => {
            info!(
                "trading: {} order placed — id={} {}/{} ${:.2}",
                side, order.order_id, decision.exchange, decision.symbol, decision.amount_usd
            );
            let trade = ExecutedTrade {
                ts: now_ts(),
                exchange: decision.exchange.clone(),
                symbol: decision.symbol.clone(),
                action: decision.action.clone(),
                amount_usd: decision.amount_usd,
                price: order.price,
                order_id: Some(order.order_id.clone()),
                confidence: decision.confidence,
                rationale: decision.rationale.clone(),
                ai_votes: serde_json::Value::Null,
                fuzzy_confidence: decision.fuzzy_confidence,
                ai_confidence: decision.ai_confidence,
            };
            {
                let mut s = state.0.lock().await;
                s.record_trade(trade);
                s.pending_override = None; // Clear override once executed.
            }
            state
                .log(
                    "info",
                    "execute",
                    format!(
                        "{side} order placed: {}/{} ${:.2} id={}",
                        decision.exchange, decision.symbol, decision.amount_usd, order.order_id
                    ),
                    json!({ "order_id": order.order_id, "status": order.status }),
                )
                .await;
        }
        Err(err) => {
            warn!("trading: {} order failed: {}", side, err);
            state
                .log_error("execute", format!("{side} order failed: {err}"))
                .await;
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum SellBalanceAvailability {
    Available,
    Missing,
    NonPositive { free: f64, total: f64 },
}

async fn ensure_sell_balance_available(
    octobot: &OctobotClient,
    state: &SharedTradingState,
    base_asset: &str,
    symbol: &str,
) -> SellBalanceAvailability {
    let initial = cached_sell_balance(state, base_asset).await;
    if matches!(initial, SellBalanceAvailability::Available) {
        return initial;
    }

    // Try one explicit OctoBot refresh cycle before skipping a sell. This
    // closes the gap between exchange state and cached portfolio snapshots.
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
            let availability = portfolio_balance_state(&portfolio, base_asset);
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
    base_asset: &str,
) -> SellBalanceAvailability {
    let s = state.0.lock().await;
    if let Some(portfolio) = s.current_portfolio.as_ref() {
        portfolio_balance_state(portfolio, base_asset)
    } else {
        SellBalanceAvailability::Missing
    }
}

fn portfolio_balance_state(
    portfolio: &OctobotPortfolio,
    base_asset: &str,
) -> SellBalanceAvailability {
    match portfolio.currencies.get(base_asset) {
        Some(balance) if balance.free > 0.0 || balance.total > 0.0 => {
            SellBalanceAvailability::Available
        }
        Some(balance) => SellBalanceAvailability::NonPositive {
            free: balance.free,
            total: balance.total,
        },
        None => SellBalanceAvailability::Missing,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TARGET_LOCK_CONSENSUS_CONFIDENCE_MIN: f64 = 0.70;
const TARGET_LOCK_CONSENSUS_SIGNAL_MIN: f64 = 0.30;
const TARGET_LOCK_SUPPORT_MIN: f64 = 0.35;

#[derive(Clone, Debug)]
struct DecisionMarketSelection {
    snapshot: Option<MarketSnapshot>,
    note: String,
    override_reason: Option<String>,
    used_target_signal: bool,
    target_support: f64,
    target_support_advisors: usize,
    high_confidence_target: bool,
}

#[derive(Clone, Debug)]
struct TargetSnapshotSupport {
    snapshot: MarketSnapshot,
    signed_support: f64,
    total_support: f64,
    advisors: usize,
}

fn choose_decision_market_candidate(
    snapshots: &[MarketSnapshot],
    consensus: &advisor::AiConsensus,
    fallback_snapshot: Option<&MarketSnapshot>,
) -> DecisionMarketSelection {
    if snapshots.is_empty() {
        return DecisionMarketSelection {
            snapshot: None,
            note: "No market snapshots available for decision targeting".to_string(),
            override_reason: None,
            used_target_signal: false,
            target_support: 0.0,
            target_support_advisors: 0,
            high_confidence_target: false,
        };
    }

    let fallback = fallback_snapshot
        .cloned()
        .or_else(|| select_best_market_candidate(snapshots));
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
    let consensus_direction = consensus_direction_sign(consensus.signal);
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
                target_support_advisors: target.advisors,
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
            target_support_advisors: target.advisors,
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
            target_support_advisors: 0,
            high_confidence_target: true,
        };
    }

    DecisionMarketSelection {
        snapshot: fallback,
        note: format!(
            "Decision market defaulted to {} (no aligned AI target support)",
            fallback_label
        ),
        override_reason: None,
        used_target_signal: false,
        target_support: 0.0,
        target_support_advisors: 0,
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

fn merge_rationale_note(rationale: &str, note: &str) -> String {
    let rationale = rationale.trim();
    let note = note.trim();
    if note.is_empty() {
        rationale.to_string()
    } else if rationale.is_empty() {
        note.to_string()
    } else if rationale.contains(note) {
        rationale.to_string()
    } else {
        format!("{rationale} | {note}")
    }
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
    if snapshots.is_empty() {
        return None;
    }
    // Score: abs(24h change) * log(volume + 1) — highest momentum + volume.
    let best = snapshots.iter().max_by(|a, b| {
        let score_a = market_score(a);
        let score_b = market_score(b);
        score_a
            .partial_cmp(&score_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    best.cloned()
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
