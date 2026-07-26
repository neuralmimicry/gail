//! Fixed-horizon, fee-adjusted trade outcome ledger.
//!
//! Each accepted order creates one pending markout. Once a market observation
//! reaches the configured horizon, the ledger resolves the directional return
//! using the same round-trip cost assumptions as the pre-trade gate. Resolved
//! samples drive bounded provider/symbol/regime calibration; they never compare
//! a trade with an unrelated next trade.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::{octobot::MarketSnapshot, state::TradeAction};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TradeMarkout {
    pub order_id: String,
    pub executed_at: f64,
    pub due_at: f64,
    pub resolved_at: Option<f64>,
    pub exchange: String,
    pub symbol: String,
    pub action: TradeAction,
    pub amount_usd: f64,
    pub entry_price: f64,
    pub observed_price: Option<f64>,
    pub providers: Vec<String>,
    pub regime: String,
    pub gross_directional_return_bps: Option<f64>,
    pub net_directional_return_bps: Option<f64>,
}

impl Default for TradeMarkout {
    fn default() -> Self {
        Self {
            order_id: String::new(),
            executed_at: 0.0,
            due_at: 0.0,
            resolved_at: None,
            exchange: String::new(),
            symbol: String::new(),
            action: TradeAction::Hold,
            amount_usd: 0.0,
            entry_price: 0.0,
            observed_price: None,
            providers: Vec::new(),
            regime: "unknown".to_string(),
            gross_directional_return_bps: None,
            net_directional_return_bps: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OutcomePerformance {
    pub samples: usize,
    pub average_net_return_bps: f64,
    pub win_rate: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct OutcomeCalibration {
    pub multiplier: f64,
    pub performance: OutcomePerformance,
}

impl Default for OutcomeCalibration {
    fn default() -> Self {
        Self {
            multiplier: 1.0,
            performance: OutcomePerformance::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OutcomeLedger {
    pub observations: VecDeque<TradeMarkout>,
}

impl OutcomeLedger {
    pub fn record(&mut self, markout: TradeMarkout, capacity: usize) {
        if markout.entry_price <= 0.0 || !markout.entry_price.is_finite() {
            return;
        }
        if self
            .observations
            .iter()
            .any(|item| !markout.order_id.is_empty() && item.order_id == markout.order_id)
        {
            return;
        }
        while self.observations.len() >= capacity.max(1) {
            self.observations.pop_front();
        }
        self.observations.push_back(markout);
    }

    /// Resolve every due observation for which an exact exchange/symbol price
    /// is present. Returns the number newly resolved in this call.
    pub fn resolve_due(
        &mut self,
        snapshots: &[MarketSnapshot],
        now: f64,
        round_trip_cost_bps: f64,
    ) -> usize {
        let mut resolved = 0;
        for observation in self.observations.iter_mut().filter(|item| {
            item.resolved_at.is_none() && item.due_at <= now && item.entry_price > 0.0
        }) {
            let Some(snapshot) = snapshots.iter().find(|snapshot| {
                snapshot
                    .exchange
                    .eq_ignore_ascii_case(&observation.exchange)
                    && snapshot.symbol.eq_ignore_ascii_case(&observation.symbol)
                    && snapshot.price.is_finite()
                    && snapshot.price > 0.0
            }) else {
                continue;
            };
            let market_return_bps =
                (snapshot.price - observation.entry_price) / observation.entry_price * 10_000.0;
            let direction = if matches!(
                observation.action,
                TradeAction::Buy | TradeAction::StrongBuy
            ) {
                1.0
            } else {
                -1.0
            };
            let gross = market_return_bps * direction;
            observation.observed_price = Some(snapshot.price);
            observation.gross_directional_return_bps = Some(gross);
            observation.net_directional_return_bps = Some(gross - round_trip_cost_bps.max(0.0));
            observation.resolved_at = Some(now);
            resolved += 1;
        }
        resolved
    }

    pub fn directional_performance(
        &self,
        symbol: Option<&str>,
        buying: bool,
        lookback: usize,
    ) -> OutcomePerformance {
        let values = self
            .observations
            .iter()
            .rev()
            .filter(|item| {
                let item_buying = matches!(item.action, TradeAction::Buy | TradeAction::StrongBuy);
                item_buying == buying
                    && symbol.is_none_or(|symbol| item.symbol.eq_ignore_ascii_case(symbol))
            })
            .filter_map(|item| item.net_directional_return_bps)
            .take(lookback.max(1))
            .collect::<Vec<_>>();
        summarize(&values)
    }

    /// Prefer the most specific sample set with enough observations, falling
    /// back from symbol+provider+regime to symbol, then global outcomes.
    pub fn calibration_for(
        &self,
        symbol: &str,
        providers: &[String],
        regime: &str,
        min_samples: usize,
    ) -> OutcomeCalibration {
        let resolved = self
            .observations
            .iter()
            .rev()
            .filter(|item| item.net_directional_return_bps.is_some())
            .collect::<Vec<_>>();
        let provider_matches = |item: &&TradeMarkout| {
            providers.is_empty()
                || providers.iter().any(|provider| {
                    item.providers
                        .iter()
                        .any(|seen| seen.eq_ignore_ascii_case(provider))
                })
        };
        let exact = resolved
            .iter()
            .copied()
            .filter(|item| item.symbol.eq_ignore_ascii_case(symbol))
            .filter(provider_matches)
            .filter(|item| item.regime.eq_ignore_ascii_case(regime))
            .filter_map(|item| item.net_directional_return_bps)
            .take(250)
            .collect::<Vec<_>>();
        let symbol_only = resolved
            .iter()
            .copied()
            .filter(|item| item.symbol.eq_ignore_ascii_case(symbol))
            .filter_map(|item| item.net_directional_return_bps)
            .take(250)
            .collect::<Vec<_>>();
        let global = resolved
            .iter()
            .filter_map(|item| item.net_directional_return_bps)
            .take(250)
            .collect::<Vec<_>>();
        let values = if exact.len() >= min_samples {
            &exact
        } else if symbol_only.len() >= min_samples {
            &symbol_only
        } else if global.len() >= min_samples {
            &global
        } else {
            return OutcomeCalibration::default();
        };
        let performance = summarize(values);
        let return_score = (performance.average_net_return_bps / 100.0).clamp(-1.0, 1.0);
        let win_score = ((performance.win_rate - 0.5) * 2.0).clamp(-1.0, 1.0);
        let score = return_score * 0.7 + win_score * 0.3;
        OutcomeCalibration {
            multiplier: (1.0 + score * 0.35).clamp(0.55, 1.25),
            performance,
        }
    }
}

fn summarize(values: &[f64]) -> OutcomePerformance {
    if values.is_empty() {
        return OutcomePerformance::default();
    }
    OutcomePerformance {
        samples: values.len(),
        average_net_return_bps: values.iter().sum::<f64>() / values.len() as f64,
        win_rate: values.iter().filter(|value| **value > 0.0).count() as f64 / values.len() as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markout_is_directional_and_fee_adjusted() {
        let mut ledger = OutcomeLedger::default();
        ledger.record(
            TradeMarkout {
                order_id: "1".to_string(),
                due_at: 10.0,
                exchange: "bitget".to_string(),
                symbol: "BTC/USDT".to_string(),
                action: TradeAction::Sell,
                entry_price: 100.0,
                ..TradeMarkout::default()
            },
            100,
        );
        let resolved = ledger.resolve_due(
            &[MarketSnapshot {
                exchange: "bitget".to_string(),
                symbol: "BTC/USDT".to_string(),
                price: 99.0,
                ..MarketSnapshot::default()
            }],
            11.0,
            20.0,
        );
        assert_eq!(resolved, 1);
        assert_eq!(
            ledger.observations[0].net_directional_return_bps,
            Some(80.0)
        );
    }
}
