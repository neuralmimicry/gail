//! Liquidity-aware cross-sectional factor allocation.
//!
//! Absolute percentage thresholds are unstable across tokens with radically
//! different volatility and liquidity.  The allocator instead winsorises and
//! volatility-normalises directional factors, removes the cross-sectional
//! market component, filters markets that cannot be executed economically and
//! returns diversified target weights plus an explicit cash residual.

use std::{cmp::Ordering, collections::HashMap};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CrossSectionalConfig {
    pub enabled: bool,
    pub top_k: usize,
    pub minimum_quote_volume_usd: f64,
    pub maximum_spread_bps: f64,
    pub minimum_depth_usd: f64,
    pub minimum_listing_age_days: f64,
    pub winsor_quantile: f64,
    pub volatility_floor_pct: f64,
    pub volatility_target_pct: f64,
    pub maximum_asset_weight: f64,
    pub maximum_cluster_weight: f64,
    pub minimum_factor_score: f64,
    pub allow_illiquid_reversal: bool,
    /// Bounded entry-timing contribution from order/trade-flow imbalance.
    pub microstructure_signal_weight: f64,
}

impl Default for CrossSectionalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            top_k: 3,
            minimum_quote_volume_usd: 1_000_000.0,
            maximum_spread_bps: 40.0,
            minimum_depth_usd: 5_000.0,
            minimum_listing_age_days: 14.0,
            winsor_quantile: 0.05,
            volatility_floor_pct: 0.25,
            volatility_target_pct: 2.0,
            maximum_asset_weight: 0.45,
            maximum_cluster_weight: 0.65,
            minimum_factor_score: 0.05,
            allow_illiquid_reversal: false,
            microstructure_signal_weight: 0.15,
        }
    }
}

impl CrossSectionalConfig {
    pub fn normalise(&mut self) {
        self.top_k = self.top_k.clamp(1, 50);
        self.minimum_quote_volume_usd = self.minimum_quote_volume_usd.clamp(0.0, 1e15);
        self.maximum_spread_bps = self.maximum_spread_bps.clamp(0.1, 5_000.0);
        self.minimum_depth_usd = self.minimum_depth_usd.clamp(0.0, 1e15);
        self.minimum_listing_age_days = self.minimum_listing_age_days.clamp(0.0, 10_000.0);
        self.winsor_quantile = self.winsor_quantile.clamp(0.0, 0.25);
        self.volatility_floor_pct = self.volatility_floor_pct.clamp(0.01, 100.0);
        self.volatility_target_pct = self.volatility_target_pct.clamp(0.01, 100.0);
        self.maximum_asset_weight = self.maximum_asset_weight.clamp(0.01, 1.0);
        self.maximum_cluster_weight = self
            .maximum_cluster_weight
            .clamp(self.maximum_asset_weight, 1.0);
        self.minimum_factor_score = self.minimum_factor_score.clamp(0.0, 10.0);
        self.microstructure_signal_weight = self.microstructure_signal_weight.clamp(0.0, 0.5);
    }
}

/// Strategy-neutral input so the allocator can be reused by momentum, pairs
/// and future factor sleeves.
#[derive(Clone, Debug, Default)]
pub struct CrossSectionalInput {
    pub exchange: String,
    pub symbol: String,
    pub directional_signal: f64,
    pub confidence: f64,
    pub risk_score: f64,
    pub volatility_pct: f64,
    pub quote_volume_usd: Option<f64>,
    pub spread_bps: Option<f64>,
    pub depth_usd: Option<f64>,
    pub listing_age_days: Option<f64>,
    pub inventory_eligible: bool,
    pub actionable: bool,
    pub correlation_cluster: String,
    pub current_weight: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PortfolioAllocation {
    pub exchange: String,
    pub symbol: String,
    pub rank: usize,
    pub directional_signal: f64,
    pub factor_score: f64,
    pub target_weight: f64,
    pub current_weight: f64,
    /// Signed target-minus-current portfolio weight.
    pub rebalance_weight: f64,
    pub volatility_pct: f64,
    pub liquidity_score: f64,
    pub correlation_cluster: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PortfolioRecommendation {
    pub allocations: Vec<PortfolioAllocation>,
    pub cash_weight: f64,
    pub eligible_markets: usize,
    pub excluded_markets: usize,
}

#[derive(Clone, Debug)]
struct ScoredInput {
    input: CrossSectionalInput,
    factor_score: f64,
    liquidity_score: f64,
}

#[derive(Clone, Debug)]
pub struct CrossSectionalAllocator {
    config: CrossSectionalConfig,
}

impl CrossSectionalAllocator {
    pub fn new(mut config: CrossSectionalConfig) -> Self {
        config.normalise();
        Self { config }
    }

    pub fn allocate(&self, inputs: Vec<CrossSectionalInput>) -> PortfolioRecommendation {
        if !self.config.enabled {
            return self.allocate_without_cross_section(inputs);
        }
        let received = inputs.len();
        let eligible = inputs
            .into_iter()
            .filter(|input| input.actionable && input.inventory_eligible)
            .filter(|input| {
                input
                    .quote_volume_usd
                    .is_some_and(|volume| volume >= self.config.minimum_quote_volume_usd)
            })
            .filter(|input| {
                input
                    .spread_bps
                    .is_none_or(|spread| spread <= self.config.maximum_spread_bps)
            })
            .filter(|input| {
                input
                    .depth_usd
                    .is_none_or(|depth| depth >= self.config.minimum_depth_usd)
            })
            .filter(|input| {
                input
                    .listing_age_days
                    .is_none_or(|age| age >= self.config.minimum_listing_age_days)
            })
            .collect::<Vec<_>>();
        if eligible.is_empty() {
            return PortfolioRecommendation {
                cash_weight: 1.0,
                eligible_markets: 0,
                excluded_markets: received,
                ..PortfolioRecommendation::default()
            };
        }

        let market_component = if eligible.len() >= 3 {
            median(
                &eligible
                    .iter()
                    .map(|input| input.directional_signal)
                    .collect::<Vec<_>>(),
            )
        } else {
            0.0
        };
        let raw_scores = eligible
            .iter()
            .map(|input| {
                let volatility = input
                    .volatility_pct
                    .abs()
                    .max(self.config.volatility_floor_pct);
                let market_adjusted = input.directional_signal - market_component;
                market_adjusted / volatility
                    * input.confidence.clamp(0.0, 1.0)
                    * (1.0 - input.risk_score.clamp(0.0, 1.0))
            })
            .collect::<Vec<_>>();
        let (lower, upper) = winsor_bounds(&raw_scores, self.config.winsor_quantile);
        let mut scored = eligible
            .into_iter()
            .zip(raw_scores)
            .map(|(input, raw_score)| {
                let quote_volume = input.quote_volume_usd.unwrap_or(0.0).max(0.0);
                let liquidity_score = ((quote_volume + 1.0).log10() / 10.0).clamp(0.0, 1.0);
                let mut factor_score = raw_score.clamp(lower, upper);
                if self.config.allow_illiquid_reversal
                    && liquidity_score < 0.45
                    && input.directional_signal.abs() > 0.0
                {
                    factor_score = -factor_score;
                }
                ScoredInput {
                    input,
                    factor_score,
                    liquidity_score,
                }
            })
            .filter(|row| row.factor_score.abs() >= self.config.minimum_factor_score)
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .factor_score
                .abs()
                .partial_cmp(&left.factor_score.abs())
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    right
                        .liquidity_score
                        .partial_cmp(&left.liquidity_score)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| left.input.symbol.cmp(&right.input.symbol))
        });

        let mut cluster_weights: HashMap<String, f64> = HashMap::new();
        let mut allocations = Vec::new();
        let mut target_invested = 0.0;
        let mut target_count = 0usize;
        for row in scored {
            if target_count >= self.config.top_k {
                break;
            }
            let volatility = row
                .input
                .volatility_pct
                .abs()
                .max(self.config.volatility_floor_pct);
            let volatility_weight = (self.config.volatility_target_pct / volatility)
                .clamp(0.0, self.config.maximum_asset_weight);
            let cluster = if row.input.correlation_cluster.trim().is_empty() {
                row.input.symbol.clone()
            } else {
                row.input.correlation_cluster.clone()
            };
            let used_cluster_weight = cluster_weights.get(&cluster).copied().unwrap_or(0.0);
            let remaining_cluster_weight =
                (self.config.maximum_cluster_weight - used_cluster_weight).max(0.0);
            let target_weight = volatility_weight.min(remaining_cluster_weight);
            if target_weight <= f64::EPSILON {
                continue;
            }
            let target_weight = if row.input.directional_signal >= 0.0 {
                target_weight
            } else {
                0.0
            };
            target_count += 1;
            target_invested += target_weight;
            cluster_weights.insert(cluster.clone(), used_cluster_weight + target_weight);
            let rebalance_weight = target_weight - row.input.current_weight.clamp(0.0, 1.0);
            if (row.input.directional_signal >= 0.0 && rebalance_weight <= 0.005)
                || (row.input.directional_signal < 0.0 && rebalance_weight >= -0.005)
            {
                continue;
            }
            allocations.push(PortfolioAllocation {
                exchange: row.input.exchange,
                symbol: row.input.symbol,
                rank: allocations.len() + 1,
                directional_signal: row.input.directional_signal,
                factor_score: row.factor_score,
                target_weight,
                current_weight: row.input.current_weight.clamp(0.0, 1.0),
                rebalance_weight,
                volatility_pct: volatility,
                liquidity_score: row.liquidity_score,
                correlation_cluster: cluster,
            });
        }
        allocations.sort_by(|left, right| {
            right
                .rebalance_weight
                .abs()
                .partial_cmp(&left.rebalance_weight.abs())
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.symbol.cmp(&right.symbol))
        });
        for (index, allocation) in allocations.iter_mut().enumerate() {
            allocation.rank = index + 1;
        }
        PortfolioRecommendation {
            eligible_markets: target_count,
            excluded_markets: received.saturating_sub(target_count),
            allocations,
            cash_weight: 1.0 - target_invested.min(1.0),
        }
    }

    fn allocate_without_cross_section(
        &self,
        mut inputs: Vec<CrossSectionalInput>,
    ) -> PortfolioRecommendation {
        let received = inputs.len();
        inputs.retain(|input| input.actionable && input.inventory_eligible);
        inputs.sort_by(|left, right| {
            right
                .directional_signal
                .abs()
                .partial_cmp(&left.directional_signal.abs())
                .unwrap_or(Ordering::Equal)
        });
        let allocations = inputs
            .into_iter()
            .take(self.config.top_k)
            .enumerate()
            .map(|(index, input)| PortfolioAllocation {
                exchange: input.exchange,
                symbol: input.symbol.clone(),
                rank: index + 1,
                directional_signal: input.directional_signal,
                factor_score: input.directional_signal,
                target_weight: self.config.maximum_asset_weight,
                current_weight: input.current_weight,
                rebalance_weight: self.config.maximum_asset_weight - input.current_weight,
                volatility_pct: input.volatility_pct,
                liquidity_score: 0.0,
                correlation_cluster: input.symbol,
            })
            .collect::<Vec<_>>();
        let invested = allocations
            .iter()
            .map(|allocation| allocation.target_weight)
            .sum::<f64>()
            .min(1.0);
        PortfolioRecommendation {
            eligible_markets: allocations.len(),
            excluded_markets: received.saturating_sub(allocations.len()),
            allocations,
            cash_weight: 1.0 - invested,
        }
    }
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let middle = ordered.len() / 2;
    if ordered.len().is_multiple_of(2) {
        (ordered[middle - 1] + ordered[middle]) / 2.0
    } else {
        ordered[middle]
    }
}

fn winsor_bounds(values: &[f64], quantile: f64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let tail =
        ((ordered.len().saturating_sub(1)) as f64 * quantile.clamp(0.0, 0.25)).round() as usize;
    (ordered[tail], ordered[ordered.len() - 1 - tail])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(symbol: &str, signal: f64, volatility: f64, volume: f64) -> CrossSectionalInput {
        CrossSectionalInput {
            exchange: "test".to_string(),
            symbol: symbol.to_string(),
            directional_signal: signal,
            confidence: 0.9,
            risk_score: 0.1,
            volatility_pct: volatility,
            quote_volume_usd: Some(volume),
            inventory_eligible: true,
            actionable: true,
            correlation_cluster: symbol.to_string(),
            ..CrossSectionalInput::default()
        }
    }

    #[test]
    fn allocator_filters_illiquid_markets_and_retains_cash() {
        let config = CrossSectionalConfig {
            minimum_quote_volume_usd: 1_000_000.0,
            minimum_factor_score: 0.0,
            ..CrossSectionalConfig::default()
        };
        let result = CrossSectionalAllocator::new(config).allocate(vec![
            input("BTC/USDT", 0.8, 2.0, 100_000_000.0),
            input("DUST/USDT", 1.0, 20.0, 100.0),
        ]);
        assert_eq!(result.allocations.len(), 1);
        assert_eq!(result.allocations[0].symbol, "BTC/USDT");
        assert!(result.cash_weight > 0.0);
    }

    #[test]
    fn volatility_target_reduces_weight_for_riskier_asset() {
        let config = CrossSectionalConfig {
            minimum_quote_volume_usd: 0.0,
            minimum_factor_score: 0.0,
            top_k: 2,
            ..CrossSectionalConfig::default()
        };
        let result = CrossSectionalAllocator::new(config).allocate(vec![
            input("LOW/USDT", 0.8, 1.0, 10_000_000.0),
            input("HIGH/USDT", 0.8, 10.0, 10_000_000.0),
        ]);
        let low = result
            .allocations
            .iter()
            .find(|allocation| allocation.symbol == "LOW/USDT")
            .unwrap();
        let high = result
            .allocations
            .iter()
            .find(|allocation| allocation.symbol == "HIGH/USDT")
            .unwrap();
        assert!(low.target_weight > high.target_weight);
    }

    #[test]
    fn cross_section_removes_common_market_component() {
        let config = CrossSectionalConfig {
            minimum_quote_volume_usd: 0.0,
            minimum_factor_score: 0.0,
            top_k: 3,
            ..CrossSectionalConfig::default()
        };
        let result = CrossSectionalAllocator::new(config).allocate(vec![
            input("A/USDT", 0.9, 1.0, 10_000_000.0),
            input("B/USDT", 0.5, 1.0, 10_000_000.0),
            input("C/USDT", 0.4, 1.0, 10_000_000.0),
        ]);
        assert_eq!(result.allocations[0].symbol, "A/USDT");
        assert!(result.allocations[0].factor_score > 0.0);
    }
}
