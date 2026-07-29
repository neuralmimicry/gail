//! Gail-native, event-driven replay for deterministic quant policies.
//!
//! OctoBot backtests exercise OctoBot's configured strategy, not Gail's Rust
//! decision policy.  This engine instead replays [`crate::trading::quant`]
//! directly.  Candidate arms are evaluated concurrently with Rayon and the
//! asynchronous entry point uses `spawn_blocking`, ensuring CPU-heavy research
//! cannot obstruct Tokio's I/O workers.

use std::{cmp::Ordering, collections::HashMap};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::super::{
    datalake::{MarketHistoricalFeatures, market_feature_key},
    octobot::{MarketSnapshot, OctobotPortfolio},
    quant::{QuantParameters, evaluate_universe_for_parameters},
    quantitative::portfolio::CrossSectionalConfig,
};

/// One point-in-time market universe supplied to the replay engine.
///
/// Frames should contain only information available at `timestamp`; callers
/// are responsible for constructing historical features without look-ahead.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantBacktestFrame {
    pub timestamp: f64,
    pub snapshots: Vec<MarketSnapshot>,
    pub historical_features: HashMap<String, MarketHistoricalFeatures>,
}

/// Configuration for deterministic native replay and promotion validation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NativeBacktestConfig {
    /// Nominal equity used for readily interpretable monetary P&L.
    pub initial_equity_usd: f64,
    /// Fixed notional allocated to each independent markout.
    pub trade_notional_usd: f64,
    /// Holding period used by the policy label and simulated exit.
    pub holding_horizon_seconds: u64,
    /// One-way fee charged at both entry and exit.
    pub fee_bps: f64,
    /// Conservative one-way fallback slippage charged at both legs.
    pub slippage_bps: f64,
    /// Minimum training frames in each walk-forward fold.
    pub minimum_training_frames: usize,
    /// Number of subsequent frames reserved for promotion validation.
    pub validation_frames: usize,
    /// Frames omitted between train and validation to reduce leakage.
    pub embargo_frames: usize,
    /// Minimum actionable validation trades required for promotion.
    pub minimum_validation_trades: usize,
    /// One-sided confidence multiplier applied to mean net return.
    pub confidence_z_score: f64,
    /// Required conservative net edge after all simulated costs.
    pub minimum_net_edge_bps: f64,
    /// Maximum permitted probability of backtest overfitting.
    pub maximum_pbo: f64,
    /// Maximum frames materialised from the live datalake for one run.
    pub maximum_frames: usize,
    /// Maximum liquid symbols replayed in one scheduled run.
    pub symbol_limit: usize,
    /// Spacing between frames built from higher-frequency datalake samples.
    pub frame_interval_seconds: u64,
}

impl Default for NativeBacktestConfig {
    fn default() -> Self {
        Self {
            initial_equity_usd: 10_000.0,
            trade_notional_usd: 25.0,
            holding_horizon_seconds: 3_600,
            fee_bps: 10.0,
            slippage_bps: 15.0,
            minimum_training_frames: 96,
            validation_frames: 48,
            embargo_frames: 4,
            minimum_validation_trades: 16,
            confidence_z_score: 1.645,
            minimum_net_edge_bps: 10.0,
            maximum_pbo: 0.20,
            maximum_frames: 20_000,
            symbol_limit: 40,
            frame_interval_seconds: 900,
        }
    }
}

impl NativeBacktestConfig {
    pub fn normalise(&mut self) {
        self.initial_equity_usd = self.initial_equity_usd.clamp(100.0, 1_000_000_000.0);
        self.trade_notional_usd = self.trade_notional_usd.clamp(0.01, self.initial_equity_usd);
        self.holding_horizon_seconds = self.holding_horizon_seconds.clamp(60, 30 * 86_400);
        self.fee_bps = self.fee_bps.clamp(0.0, 500.0);
        self.slippage_bps = self.slippage_bps.clamp(0.0, 2_500.0);
        self.minimum_training_frames = self.minimum_training_frames.clamp(10, 1_000_000);
        self.validation_frames = self.validation_frames.clamp(5, 1_000_000);
        self.embargo_frames = self.embargo_frames.min(self.validation_frames);
        self.minimum_validation_trades = self.minimum_validation_trades.clamp(1, 100_000);
        self.confidence_z_score = self.confidence_z_score.clamp(0.0, 4.0);
        self.minimum_net_edge_bps = self.minimum_net_edge_bps.clamp(0.0, 2_500.0);
        self.maximum_pbo = self.maximum_pbo.clamp(0.0, 1.0);
        self.maximum_frames = self.maximum_frames.clamp(100, 1_000_000);
        self.symbol_limit = self.symbol_limit.clamp(1, 500);
        self.frame_interval_seconds = self.frame_interval_seconds.clamp(60, 86_400);
    }

    pub fn round_trip_cost_bps(&self) -> f64 {
        2.0 * (self.fee_bps + self.slippage_bps)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NativeBacktestTrade {
    pub parameter_id: String,
    pub exchange: String,
    pub symbol: String,
    pub entered_at: f64,
    pub exited_at: f64,
    pub direction: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub gross_return_bps: f64,
    /// Projected entry market impact, excluding exchange fees.
    pub entry_market_cost_bps: f64,
    /// Projected exit market impact, excluding exchange fees.
    pub exit_market_cost_bps: f64,
    /// Complete simulated round-trip fee and market-impact cost.
    pub round_trip_cost_bps: f64,
    pub net_return_bps: f64,
    pub pnl_usd: f64,
}

/// Statistics calculated only from actionable observations.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ActionableStatistics {
    pub samples: usize,
    pub mean_net_return_bps: f64,
    pub standard_deviation_bps: f64,
    pub lower_confidence_bound_bps: f64,
    pub win_rate: f64,
    pub downside_mean_bps: f64,
    pub cumulative_pnl_usd: f64,
}

impl ActionableStatistics {
    fn from_trades(trades: &[&NativeBacktestTrade], confidence_z_score: f64) -> Self {
        if trades.is_empty() {
            return Self::default();
        }
        let values = trades
            .iter()
            .map(|trade| trade.net_return_bps)
            .collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = if values.len() > 1 {
            values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (values.len() - 1) as f64
        } else {
            0.0
        };
        let standard_deviation = variance.sqrt();
        let standard_error = standard_deviation / (values.len() as f64).sqrt();
        let losses = values
            .iter()
            .copied()
            .filter(|value| *value < 0.0)
            .collect::<Vec<_>>();
        Self {
            samples: values.len(),
            mean_net_return_bps: mean,
            standard_deviation_bps: standard_deviation,
            lower_confidence_bound_bps: mean - confidence_z_score * standard_error,
            win_rate: values.iter().filter(|value| **value > 0.0).count() as f64
                / values.len() as f64,
            downside_mean_bps: if losses.is_empty() {
                0.0
            } else {
                losses.iter().map(|value| value.abs()).sum::<f64>() / losses.len() as f64
            },
            cumulative_pnl_usd: trades.iter().map(|trade| trade.pnl_usd).sum(),
        }
    }

    fn score(&self) -> f64 {
        self.lower_confidence_bound_bps - 0.25 * self.downside_mean_bps
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CandidateBacktestReport {
    pub parameter_id: String,
    pub trades: usize,
    pub ending_equity_usd: f64,
    pub maximum_drawdown_pct: f64,
    pub statistics: ActionableStatistics,
    pub trade_ledger: Vec<NativeBacktestTrade>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WalkForwardFoldReport {
    pub training_start: f64,
    pub training_end: f64,
    pub validation_start: f64,
    pub validation_end: f64,
    pub selected_parameter_id: String,
    pub training_statistics: ActionableStatistics,
    pub validation_statistics: ActionableStatistics,
}

/// Auditable result used by the promotion controller.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct NativeBacktestReport {
    pub generated_at: f64,
    pub frames: usize,
    pub candidates: Vec<CandidateBacktestReport>,
    pub walk_forward_folds: Vec<WalkForwardFoldReport>,
    pub out_of_sample_statistics: ActionableStatistics,
    pub probability_backtest_overfit: f64,
    pub selected_parameter_id: Option<String>,
    pub promotion_qualified: bool,
    pub rejection_reasons: Vec<String>,
}

/// Pure replay engine.  Network and file I/O are intentionally absent.
#[derive(Clone, Debug)]
pub struct NativeQuantBacktester {
    config: NativeBacktestConfig,
    portfolio_config: CrossSectionalConfig,
}

impl NativeQuantBacktester {
    pub fn new(mut config: NativeBacktestConfig) -> Self {
        config.normalise();
        Self {
            config,
            portfolio_config: CrossSectionalConfig::default(),
        }
    }

    pub fn with_portfolio_config(mut self, mut portfolio_config: CrossSectionalConfig) -> Self {
        portfolio_config.normalise();
        self.portfolio_config = portfolio_config;
        self
    }

    /// Run replay away from Tokio's I/O workers.
    pub async fn run_async(
        &self,
        frames: Vec<QuantBacktestFrame>,
        parameter_sets: Vec<QuantParameters>,
    ) -> Result<NativeBacktestReport, String> {
        let engine = self.clone();
        tokio::task::spawn_blocking(move || engine.run(&frames, &parameter_sets))
            .await
            .map_err(|error| format!("native quant backtest worker failed: {error}"))
    }

    /// Replay all parameter arms concurrently, then perform chronological
    /// walk-forward selection and promotion checks.
    pub fn run(
        &self,
        frames: &[QuantBacktestFrame],
        parameter_sets: &[QuantParameters],
    ) -> NativeBacktestReport {
        let mut ordered_frames = frames.to_vec();
        ordered_frames.sort_by(|left, right| {
            left.timestamp
                .partial_cmp(&right.timestamp)
                .unwrap_or(Ordering::Equal)
        });
        let candidates = parameter_sets
            .par_iter()
            .map(|parameters| self.replay_candidate(&ordered_frames, parameters))
            .collect::<Vec<_>>();
        let walk_forward_folds = self.walk_forward(&ordered_frames, &candidates);
        let selected_parameter_id = walk_forward_folds
            .last()
            .map(|fold| fold.selected_parameter_id.clone());
        let validation_trades = walk_forward_folds
            .iter()
            .flat_map(|fold| {
                let candidate = candidates
                    .iter()
                    .find(|candidate| candidate.parameter_id == fold.selected_parameter_id);
                candidate.into_iter().flat_map(|candidate| {
                    candidate.trade_ledger.iter().filter(|trade| {
                        trade.entered_at + f64::EPSILON >= fold.validation_start
                            && trade.entered_at <= fold.validation_end + f64::EPSILON
                    })
                })
            })
            .collect::<Vec<_>>();
        let out_of_sample_statistics =
            ActionableStatistics::from_trades(&validation_trades, self.config.confidence_z_score);
        let probability_backtest_overfit = estimate_pbo(&walk_forward_folds, &candidates);
        let mut rejection_reasons = Vec::new();
        if walk_forward_folds.is_empty() {
            rejection_reasons.push("no complete walk-forward fold".to_string());
        }
        if out_of_sample_statistics.samples < self.config.minimum_validation_trades {
            rejection_reasons.push(format!(
                "only {} actionable validation trades; {} required",
                out_of_sample_statistics.samples, self.config.minimum_validation_trades
            ));
        }
        if out_of_sample_statistics.lower_confidence_bound_bps <= self.config.minimum_net_edge_bps {
            rejection_reasons.push(format!(
                "validation lower-bound edge {:.2}bps does not exceed {:.2}bps",
                out_of_sample_statistics.lower_confidence_bound_bps,
                self.config.minimum_net_edge_bps
            ));
        }
        if probability_backtest_overfit > self.config.maximum_pbo {
            rejection_reasons.push(format!(
                "backtest-overfitting probability {:.3} exceeds {:.3}",
                probability_backtest_overfit, self.config.maximum_pbo
            ));
        }
        NativeBacktestReport {
            generated_at: ordered_frames
                .last()
                .map(|frame| frame.timestamp)
                .unwrap_or(0.0),
            frames: ordered_frames.len(),
            candidates,
            walk_forward_folds,
            out_of_sample_statistics,
            probability_backtest_overfit,
            selected_parameter_id,
            promotion_qualified: rejection_reasons.is_empty(),
            rejection_reasons,
        }
    }

    fn replay_candidate(
        &self,
        frames: &[QuantBacktestFrame],
        parameters: &QuantParameters,
    ) -> CandidateBacktestReport {
        #[derive(Clone)]
        struct PendingPosition {
            exchange: String,
            symbol: String,
            entered_at: f64,
            due_at: f64,
            direction: f64,
            entry_price: f64,
            entry_market_cost_bps: f64,
        }

        let portfolio = OctobotPortfolio::default();
        let mut positions: Vec<PendingPosition> = Vec::new();
        let mut trades = Vec::new();
        let mut equity = self.config.initial_equity_usd;
        let mut peak_equity = equity;
        let mut maximum_drawdown_pct: f64 = 0.0;
        for frame in frames {
            let mut remaining = Vec::with_capacity(positions.len());
            for position in positions.drain(..) {
                if position.due_at > frame.timestamp {
                    remaining.push(position);
                    continue;
                }
                let exit = latest_market(&frame.snapshots, &position.exchange, &position.symbol);
                let Some(exit) = exit else {
                    remaining.push(position);
                    continue;
                };
                let gross_return_bps =
                    position.direction * ((exit.price / position.entry_price) - 1.0) * 10_000.0;
                let exit_side = if position.direction < 0.0 {
                    "buy"
                } else {
                    "sell"
                };
                let exit_market_cost_bps = exit.projected_one_way_slippage_bps(
                    exit_side,
                    self.config.trade_notional_usd,
                    self.config.slippage_bps,
                );
                let round_trip_cost_bps = 2.0 * self.config.fee_bps
                    + position.entry_market_cost_bps
                    + exit_market_cost_bps;
                let net_return_bps = gross_return_bps - round_trip_cost_bps;
                let pnl_usd = self.config.trade_notional_usd * net_return_bps / 10_000.0;
                equity += pnl_usd;
                peak_equity = peak_equity.max(equity);
                if peak_equity > 0.0 {
                    maximum_drawdown_pct = maximum_drawdown_pct
                        .max((peak_equity - equity).max(0.0) / peak_equity * 100.0);
                }
                trades.push(NativeBacktestTrade {
                    parameter_id: parameters.id.clone(),
                    exchange: position.exchange,
                    symbol: position.symbol,
                    entered_at: position.entered_at,
                    exited_at: frame.timestamp,
                    direction: position.direction,
                    entry_price: position.entry_price,
                    exit_price: exit.price,
                    gross_return_bps,
                    entry_market_cost_bps: position.entry_market_cost_bps,
                    exit_market_cost_bps,
                    round_trip_cost_bps,
                    net_return_bps,
                    pnl_usd,
                });
            }
            positions = remaining;

            let signal = evaluate_universe_for_parameters(
                parameters,
                &frame.snapshots,
                &frame.historical_features,
                &portfolio,
                &self.portfolio_config,
            );
            if !signal.actionable || signal.signal.abs() < 0.2 || signal.symbol.is_empty() {
                continue;
            }
            if positions.iter().any(|position| {
                position.exchange.eq_ignore_ascii_case(&signal.exchange)
                    && position.symbol.eq_ignore_ascii_case(&signal.symbol)
            }) {
                continue;
            }
            let Some(entry) = latest_market(&frame.snapshots, &signal.exchange, &signal.symbol)
            else {
                continue;
            };
            positions.push(PendingPosition {
                exchange: signal.exchange,
                symbol: signal.symbol,
                entered_at: frame.timestamp,
                due_at: frame.timestamp + self.config.holding_horizon_seconds as f64,
                direction: signal.signal.signum(),
                entry_price: entry.price,
                entry_market_cost_bps: entry.projected_one_way_slippage_bps(
                    if signal.signal < 0.0 { "sell" } else { "buy" },
                    self.config.trade_notional_usd,
                    self.config.slippage_bps,
                ),
            });
        }
        let trade_refs = trades.iter().collect::<Vec<_>>();
        CandidateBacktestReport {
            parameter_id: parameters.id.clone(),
            trades: trades.len(),
            ending_equity_usd: equity,
            maximum_drawdown_pct,
            statistics: ActionableStatistics::from_trades(
                &trade_refs,
                self.config.confidence_z_score,
            ),
            trade_ledger: trades,
        }
    }

    fn walk_forward(
        &self,
        frames: &[QuantBacktestFrame],
        candidates: &[CandidateBacktestReport],
    ) -> Vec<WalkForwardFoldReport> {
        let training = self.config.minimum_training_frames;
        let validation = self.config.validation_frames;
        let embargo = self.config.embargo_frames;
        if frames.len() < training + embargo + validation || candidates.is_empty() {
            return Vec::new();
        }
        let mut folds = Vec::new();
        let mut validation_end_index = training + embargo + validation - 1;
        while validation_end_index < frames.len() {
            let validation_start_index = validation_end_index + 1 - validation;
            let training_end_index = validation_start_index.saturating_sub(embargo + 1);
            let training_start_index =
                training_end_index + 1 - training.min(training_end_index + 1);
            let training_start = frames[training_start_index].timestamp;
            let training_end = frames[training_end_index].timestamp;
            let validation_start = frames[validation_start_index].timestamp;
            let validation_end = frames[validation_end_index].timestamp;
            let selected = candidates.iter().max_by(|left, right| {
                let left_stats = statistics_between(
                    &left.trade_ledger,
                    training_start,
                    training_end,
                    self.config.confidence_z_score,
                );
                let right_stats = statistics_between(
                    &right.trade_ledger,
                    training_start,
                    training_end,
                    self.config.confidence_z_score,
                );
                left_stats
                    .score()
                    .partial_cmp(&right_stats.score())
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| right.parameter_id.cmp(&left.parameter_id))
            });
            if let Some(selected) = selected {
                folds.push(WalkForwardFoldReport {
                    training_start,
                    training_end,
                    validation_start,
                    validation_end,
                    selected_parameter_id: selected.parameter_id.clone(),
                    training_statistics: statistics_between(
                        &selected.trade_ledger,
                        training_start,
                        training_end,
                        self.config.confidence_z_score,
                    ),
                    validation_statistics: statistics_between(
                        &selected.trade_ledger,
                        validation_start,
                        validation_end,
                        self.config.confidence_z_score,
                    ),
                });
            }
            validation_end_index = validation_end_index.saturating_add(validation);
        }
        folds
    }
}

fn statistics_between(
    trades: &[NativeBacktestTrade],
    start: f64,
    end: f64,
    confidence_z_score: f64,
) -> ActionableStatistics {
    let selected = trades
        .iter()
        .filter(|trade| trade.entered_at + f64::EPSILON >= start)
        .filter(|trade| trade.entered_at <= end + f64::EPSILON)
        .collect::<Vec<_>>();
    ActionableStatistics::from_trades(&selected, confidence_z_score)
}

fn latest_market<'a>(
    snapshots: &'a [MarketSnapshot],
    exchange: &str,
    symbol: &str,
) -> Option<&'a MarketSnapshot> {
    snapshots
        .iter()
        .filter(|snapshot| snapshot.price.is_finite() && snapshot.price > 0.0)
        .filter(|snapshot| snapshot.exchange.eq_ignore_ascii_case(exchange))
        .filter(|snapshot| snapshot.symbol.eq_ignore_ascii_case(symbol))
        .max_by(|left, right| {
            left.fetched_at
                .partial_cmp(&right.fetched_at)
                .unwrap_or(Ordering::Equal)
        })
}

/// Approximate combinatorially symmetric validation using the independent
/// walk-forward folds available to a continuously running service.  For each
/// fold, the in-sample winner is ranked against every arm out of sample.  PBO
/// is the fraction of selections whose out-of-sample rank is below the median.
fn estimate_pbo(folds: &[WalkForwardFoldReport], candidates: &[CandidateBacktestReport]) -> f64 {
    if folds.is_empty() || candidates.len() < 2 {
        return 1.0;
    }
    let mut below_median = 0usize;
    let mut eligible = 0usize;
    for fold in folds {
        let mut validation_scores = candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.parameter_id.as_str(),
                    statistics_between(
                        &candidate.trade_ledger,
                        fold.validation_start,
                        fold.validation_end,
                        0.0,
                    )
                    .score(),
                )
            })
            .collect::<Vec<_>>();
        validation_scores
            .sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal));
        let Some(selected_rank) = validation_scores
            .iter()
            .position(|(parameter_id, _)| *parameter_id == fold.selected_parameter_id)
        else {
            continue;
        };
        eligible += 1;
        if selected_rank * 2 < validation_scores.len() {
            below_median += 1;
        }
    }
    if eligible == 0 {
        1.0
    } else {
        below_median as f64 / eligible as f64
    }
}

/// Helper used by datalake adapters and tests to create a feature map without
/// duplicating Gail's normalised exchange/symbol key convention.
pub fn feature_map(
    rows: impl IntoIterator<Item = (MarketSnapshot, MarketHistoricalFeatures)>,
) -> QuantBacktestFrame {
    let rows = rows.into_iter().collect::<Vec<_>>();
    let timestamp = rows
        .iter()
        .map(|(snapshot, _)| snapshot.fetched_at)
        .fold(0.0, f64::max);
    let historical_features = rows
        .iter()
        .map(|(snapshot, features)| {
            (
                market_feature_key(&snapshot.exchange, &snapshot.symbol),
                features.clone(),
            )
        })
        .collect();
    QuantBacktestFrame {
        timestamp,
        snapshots: rows.into_iter().map(|(snapshot, _)| snapshot).collect(),
        historical_features,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(id: &str, threshold: f64) -> QuantParameters {
        QuantParameters {
            id: id.to_string(),
            entry_threshold: threshold,
            ..QuantParameters::default()
        }
    }

    fn frames(count: usize, drift: f64) -> Vec<QuantBacktestFrame> {
        (0..count)
            .map(|index| {
                let price = 100.0 * (1.0 + drift * index as f64);
                let snapshot = MarketSnapshot {
                    exchange: "test".to_string(),
                    symbol: "BTC/USDT".to_string(),
                    price,
                    price_change_pct_1h: Some(4.0),
                    price_change_pct_24h: Some(8.0),
                    volume_24h: Some(10_000_000.0),
                    volume_change_pct: Some(20.0),
                    high_24h: Some(price * 1.02),
                    low_24h: Some(price * 0.98),
                    fetched_at: index as f64 * 60.0,
                    microstructure: Default::default(),
                };
                feature_map([(
                    snapshot,
                    MarketHistoricalFeatures {
                        samples: 100,
                        momentum_short_pct: Some(4.0),
                        momentum_mid_pct: Some(6.0),
                        momentum_long_pct: Some(10.0),
                        volatility_pct: Some(1.0),
                        drawdown_pct: Some(-1.0),
                        volume_ratio_short_long: Some(1.4),
                        ..MarketHistoricalFeatures::default()
                    },
                )])
            })
            .collect()
    }

    #[test]
    fn native_replay_uses_costs_and_walk_forward_validation() {
        let config = NativeBacktestConfig {
            holding_horizon_seconds: 60,
            fee_bps: 1.0,
            slippage_bps: 1.0,
            minimum_training_frames: 20,
            validation_frames: 10,
            embargo_frames: 1,
            minimum_validation_trades: 2,
            minimum_net_edge_bps: 0.0,
            maximum_pbo: 1.0,
            ..NativeBacktestConfig::default()
        };
        let report = NativeQuantBacktester::new(config).run(
            &frames(80, 0.001),
            &[parameters("active", 0.20), parameters("inactive", 0.80)],
        );
        assert_eq!(report.candidates.len(), 2);
        assert!(!report.walk_forward_folds.is_empty());
        let active = report
            .candidates
            .iter()
            .find(|candidate| candidate.parameter_id == "active")
            .unwrap();
        assert!(active.trades > 0);
        assert!(
            active.trade_ledger.iter().all(|trade| {
                (trade.net_return_bps - (trade.gross_return_bps - 4.0)).abs() < 1e-9
            })
        );
    }

    #[tokio::test]
    async fn async_replay_returns_without_running_on_the_io_worker() {
        let config = NativeBacktestConfig {
            holding_horizon_seconds: 60,
            minimum_training_frames: 10,
            validation_frames: 5,
            embargo_frames: 1,
            ..NativeBacktestConfig::default()
        };
        let result = NativeQuantBacktester::new(config)
            .run_async(frames(30, 0.001), vec![parameters("active", 0.20)])
            .await;
        assert!(result.is_ok());
    }
}
