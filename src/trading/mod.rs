/// Gail Crypto Trading Bridge — main module.
///
/// Provides `TradingBridge`, a non-blocking background service that:
///  1. Fetches live market data from OctoBot
///  2. Gathers research context from Refiner
///  3. Consults all configured AI providers in parallel (TradingAdvisor)
///  4. Applies Type-2 fuzzy logic (FuzzyEngine)
///  5. Blends fuzzy + AI signals into a decision (DecisionEngine)
///  6. Executes only through supported OctoBot trading/command bridges
///  7. Logs all activity in a ring-buffer (SharedTradingState)
///  8. Persists state to disk periodically
///
/// The bridge is entirely non-blocking and runs in its own tokio task.
/// All HTTP handlers access state through `SharedTradingState` (Arc<Mutex<>>).
pub mod advisor;
pub mod backtest;
pub mod config;
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
    let advisor = TradingAdvisor::new(service, config.advisor_timeout_seconds);
    let decision_engine = DecisionEngine::new(config.fuzzy_weight);
    let data_path = PathBuf::from(&config.data_path);

    // Initial OctoBot login.
    if let Err(err) = octobot.login().await {
        warn!("trading: OctoBot login failed at startup: {}", err);
        state
            .log_warn("startup", format!("OctoBot login failed: {err}"))
            .await;
    } else {
        state.log_info("startup", "Trading bridge started").await;
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
                run_single_evaluation(
                    &config,
                    &state,
                    &octobot,
                    &refiner,
                    &fuzzy_engine,
                    &advisor,
                    &decision_engine,
                ).await;
                persist_counter += 1;
                if persist_counter >= 5 {
                    state.persist(&data_path).await;
                    persist_counter = 0;
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
) {
    let eval_start = now_ts();
    debug!("trading: starting evaluation cycle");
    state.log_info("eval", "Starting evaluation cycle").await;

    // --- 1. Fetch market data from OctoBot ---
    let (target_exchanges, target_currencies) = {
        let s = state.0.lock().await;
        let ov = s.config_overrides.as_ref();
        let exch = ov
            .and_then(|o| o.target_exchanges.clone())
            .unwrap_or_else(|| config.target_exchanges.clone());
        let curr = ov
            .and_then(|o| o.target_currencies.clone())
            .unwrap_or_else(|| config.target_currencies.clone());
        (exch, curr)
    };

    let market_snapshots = octobot
        .get_all_market_snapshots(&target_exchanges, &target_currencies, 20)
        .await;

    // --- 2. Build research query ---
    let best_snapshot = select_best_market_candidate(&market_snapshots);
    let research_query = build_research_query(config, best_snapshot.as_ref());

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
            &research,
            &portfolio,
            config.max_parallel_advisors,
        )
        .await;

    debug!(
        "trading: AI consensus = action={} signal={:.3} confidence={:.2} responders={}",
        consensus.action, consensus.signal, consensus.confidence, consensus.responders
    );

    // --- 4. Compute fuzzy inputs ---
    let fuzzy_inputs =
        compute_fuzzy_inputs(best_snapshot.as_ref(), &consensus, &research, &portfolio);
    let fuzzy_out = fuzzy_engine.evaluate(&fuzzy_inputs);

    debug!(
        "trading: fuzzy = signal={:.3} confidence={:.2} label={}",
        fuzzy_out.signal, fuzzy_out.confidence, fuzzy_out.label
    );

    // --- 5. Make decision ---
    let mut decision = {
        let s = state.0.lock().await;
        decision_engine.decide(&fuzzy_out, &consensus, best_snapshot.as_ref(), &s, config)
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
                "blended_signal": decision.blended_signal,
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
            let balance_opt = {
                let s = state.0.lock().await;
                s.current_portfolio
                    .as_ref()
                    .and_then(|portfolio| portfolio.currencies.get(base_asset).cloned())
            };

            match balance_opt {
                Some(balance) if balance.free <= 0.0 && balance.total <= 0.0 => {
                    warn!(
                        "trading: sell skipped — non-positive {base_asset} balance for {} (free={}, total={})",
                        decision.symbol, balance.free, balance.total
                    );
                    state
                        .log_warn(
                            "execute",
                            format!(
                                "Sell skipped for {}: non-positive {} balance (free={}, total={})",
                                decision.symbol, base_asset, balance.free, balance.total
                            ),
                        )
                        .await;
                    return;
                }
                None => {
                    warn!(
                        "trading: sell skipped — {base_asset} balance unavailable in OctoBot portfolio for {}",
                        decision.symbol
                    );
                    state
                        .log_warn(
                            "execute",
                            format!(
                                "Sell skipped for {}: {base_asset} balance unavailable in OctoBot portfolio",
                                decision.symbol
                            ),
                        )
                        .await;
                    return;
                }
                _ => {}
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
    consensus: &advisor::AiConsensus,
    research: &refiner::ResearchContext,
    portfolio: &OctobotPortfolio,
) -> FuzzyInputs {
    let price_trend = best_market
        .and_then(|m| m.price_change_pct_24h)
        .map(|p| (p / 5.0).clamp(-1.0, 1.0))
        .unwrap_or(0.0);

    let volume_ratio = best_market
        .and_then(|m| m.volume_24h)
        .map(|v| (v / 1_000_000.0).clamp(0.0, 2.0)) // rough normalisation
        .unwrap_or(1.0);

    let ai_consensus = consensus.signal.clamp(-1.0, 1.0);

    // Rough research sentiment from match scores.
    let research_sentiment = if research.is_empty() {
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
