/// Type-2 Interval Fuzzy Logic System for crypto trading decisions.
///
/// Uses interval-valued (Type-2) membership functions to handle the inherent
/// uncertainty in financial data.  Each membership value is represented as
/// [lower, upper] bounds.  Inference follows the Mamdani approach and type
/// reduction uses a simplified Karnik-Mendel centroid algorithm.
///
/// Input linguistic variables:
///   - price_trend       : down / flat / up          (from % price change)
///   - volume_strength   : low / medium / high        (relative volume)
///   - ai_consensus      : bearish / neutral / bullish (aggregated AI votes)
///   - research_sentiment: negative / neutral / positive (Refiner context score)
///   - portfolio_exposure: underweight / balanced / overweight
///
/// Output linguistic variable:
///   - trade_action (strong_sell=-1.0, sell=-0.5, hold=0.0, buy=0.5, strong_buy=1.0)
///     with combined confidence score (0.0–1.0)
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Interval type alias
// ---------------------------------------------------------------------------

/// [lower_bound, upper_bound] membership pair (both in [0,1])
type Interval = [f64; 2];

fn interval(lower: f64, upper: f64) -> Interval {
    [lower.clamp(0.0, 1.0), upper.clamp(0.0, 1.0)]
}

fn interval_max(a: Interval, b: Interval) -> Interval {
    interval(a[0].max(b[0]), a[1].max(b[1]))
}

fn interval_min(a: Interval, b: Interval) -> Interval {
    interval(a[0].min(b[0]), a[1].min(b[1]))
}

fn interval_centre(iv: Interval) -> f64 {
    (iv[0] + iv[1]) / 2.0
}

// ---------------------------------------------------------------------------
// Membership functions (Gaussian with uncertainty band)
// ---------------------------------------------------------------------------

/// Interval Gaussian MF: mu ± sigma_range produces upper MF; outer band adds uncertainty.
fn gaussian_it2(x: f64, centre: f64, sigma: f64, uncertainty: f64) -> Interval {
    let upper = (-((x - centre).powi(2)) / (2.0 * sigma.powi(2))).exp();
    let lower = (upper - uncertainty).max(0.0);
    interval(lower, upper)
}

/// Trapezoidal MF for plateau regions (e.g. strong signals at extremes).
fn trapezoid_it2(x: f64, a: f64, b: f64, c: f64, d: f64, uncertainty: f64) -> Interval {
    let upper = if x < a || x > d {
        0.0
    } else if x >= b && x <= c {
        1.0
    } else if x < b {
        (x - a) / (b - a)
    } else {
        (d - x) / (d - c)
    };
    let lower = (upper - uncertainty).max(0.0);
    interval(lower, upper)
}

// ---------------------------------------------------------------------------
// Input encoding
// ---------------------------------------------------------------------------

/// All inputs normalised to [−1, +1] or [0, 1].
#[derive(Clone, Debug)]
pub struct FuzzyInputs {
    /// Normalised price trend: −1 (strong down) to +1 (strong up).
    /// Derived from % price change: clamp(change_pct / 5.0, -1, 1).
    pub price_trend: f64,

    /// Volume relative to 24 h average: 0 (no volume) to 2+ (double average).
    /// Clamped to [0, 2] before use.
    pub volume_ratio: f64,

    /// AI consensus: −1 (unanimous bearish) to +1 (unanimous bullish).
    pub ai_consensus: f64,

    /// Research sentiment: −1 (very negative) to +1 (very positive).
    pub research_sentiment: f64,

    /// Portfolio exposure: 0 (no exposure) to 1 (fully invested).
    pub portfolio_exposure: f64,
}

impl Default for FuzzyInputs {
    fn default() -> Self {
        Self {
            price_trend: 0.0,
            volume_ratio: 1.0,
            ai_consensus: 0.0,
            research_sentiment: 0.0,
            portfolio_exposure: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FuzzyDecision {
    /// Crisp output in [−1, +1]: −1 = strong sell, +1 = strong buy.
    pub signal: f64,
    /// Confidence in [0, 1].
    pub confidence: f64,
    /// Human-readable label.
    pub label: String,
    /// Per-output-term activation strengths (for diagnostics).
    pub term_activations: FuzzyTermActivations,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FuzzyTermActivations {
    pub strong_sell: f64,
    pub sell: f64,
    pub hold: f64,
    pub buy: f64,
    pub strong_buy: f64,
}

impl FuzzyDecision {
    pub fn hold() -> Self {
        Self {
            signal: 0.0,
            confidence: 0.0,
            label: "hold".to_string(),
            term_activations: FuzzyTermActivations::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Rule base
// ---------------------------------------------------------------------------

/// A single fuzzy rule evaluated to produce a weighted output term.
struct Rule {
    /// Name of the output term.
    term: &'static str,
    /// Closure: receives all five inputs and returns interval activation strength.
    antecedent: Box<dyn Fn(&FuzzyInputs) -> Interval + Send + Sync>,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct FuzzyEngine {
    rules: Vec<Rule>,
}

impl FuzzyEngine {
    pub fn new() -> Self {
        let rules = build_rules();
        Self { rules }
    }

    pub fn evaluate(&self, inputs: &FuzzyInputs) -> FuzzyDecision {
        // Accumulate activated output regions per term using Mamdani max-min.
        let mut strong_sell_acc: Interval = interval(0.0, 0.0);
        let mut sell_acc: Interval = interval(0.0, 0.0);
        let mut hold_acc: Interval = interval(0.0, 0.0);
        let mut buy_acc: Interval = interval(0.0, 0.0);
        let mut strong_buy_acc: Interval = interval(0.0, 0.0);

        for rule in &self.rules {
            let strength = (rule.antecedent)(inputs);
            match rule.term {
                "strong_sell" => strong_sell_acc = interval_max(strong_sell_acc, strength),
                "sell" => sell_acc = interval_max(sell_acc, strength),
                "hold" => hold_acc = interval_max(hold_acc, strength),
                "buy" => buy_acc = interval_max(buy_acc, strength),
                "strong_buy" => strong_buy_acc = interval_max(strong_buy_acc, strength),
                _ => {}
            }
        }

        let activations = FuzzyTermActivations {
            strong_sell: interval_centre(strong_sell_acc),
            sell: interval_centre(sell_acc),
            hold: interval_centre(hold_acc),
            buy: interval_centre(buy_acc),
            strong_buy: interval_centre(strong_buy_acc),
        };

        // Karnik-Mendel type reduction — weighted centroid using output crisp values.
        // We iterate over lower and upper bounds separately then average.
        let terms = [
            (-1.0_f64, strong_sell_acc),
            (-0.5, sell_acc),
            (0.0, hold_acc),
            (0.5, buy_acc),
            (1.0, strong_buy_acc),
        ];

        let signal_l = weighted_centroid(&terms, |iv| iv[0]);
        let signal_u = weighted_centroid(&terms, |iv| iv[1]);
        let signal = (signal_l + signal_u) / 2.0;

        // Confidence = max activated strength across all terms, averaged L+U / 2.
        let max_l = terms.iter().map(|(_, iv)| iv[0]).fold(0.0_f64, f64::max);
        let max_u = terms.iter().map(|(_, iv)| iv[1]).fold(0.0_f64, f64::max);
        let confidence = ((max_l + max_u) / 2.0).clamp(0.0, 1.0);

        let label = signal_to_label(signal);

        FuzzyDecision { signal, confidence, label, term_activations: activations }
    }
}

impl Default for FuzzyEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn weighted_centroid(terms: &[(f64, Interval)], pick: impl Fn(Interval) -> f64) -> f64 {
    let num: f64 = terms.iter().map(|(v, iv)| v * pick(*iv)).sum();
    let den: f64 = terms.iter().map(|(_, iv)| pick(*iv)).sum();
    if den < 1e-9 { 0.0 } else { num / den }
}

fn signal_to_label(signal: f64) -> String {
    match signal {
        s if s < -0.65 => "strong_sell",
        s if s < -0.2 => "sell",
        s if s < 0.2 => "hold",
        s if s < 0.65 => "buy",
        _ => "strong_buy",
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Helper accessors for membership in input space
// ---------------------------------------------------------------------------

// price_trend: [-1, 1]
fn pt_down(x: f64) -> Interval { trapezoid_it2(x, -1.0, -1.0, -0.5, -0.1, 0.05) }
fn pt_flat(x: f64) -> Interval { gaussian_it2(x, 0.0, 0.25, 0.08) }
fn pt_up(x: f64) -> Interval   { trapezoid_it2(x, 0.1, 0.5, 1.0, 1.0, 0.05) }

// volume_ratio: [0, 2]
fn vr_low(x: f64) -> Interval  { trapezoid_it2(x, 0.0, 0.0, 0.5, 0.8, 0.06) }
fn vr_high(x: f64) -> Interval { trapezoid_it2(x, 1.2, 1.6, 2.0, 2.0, 0.06) }

// ai_consensus: [-1, 1]
fn ai_bearish(x: f64) -> Interval  { trapezoid_it2(x, -1.0, -1.0, -0.4, -0.1, 0.05) }
fn ai_neutral(x: f64) -> Interval  { gaussian_it2(x, 0.0, 0.25, 0.07) }
fn ai_bullish(x: f64) -> Interval  { trapezoid_it2(x, 0.1, 0.4, 1.0, 1.0, 0.05) }

// research_sentiment: [-1, 1]
fn rs_negative(x: f64) -> Interval { trapezoid_it2(x, -1.0, -1.0, -0.3, 0.0, 0.05) }
fn rs_neutral(x: f64) -> Interval  { gaussian_it2(x, 0.0, 0.3, 0.07) }
fn rs_positive(x: f64) -> Interval { trapezoid_it2(x, 0.0, 0.3, 1.0, 1.0, 0.05) }

// portfolio_exposure: [0, 1]
fn pe_under(x: f64) -> Interval    { trapezoid_it2(x, 0.0, 0.0, 0.2, 0.4, 0.05) }
fn pe_balanced(x: f64) -> Interval { gaussian_it2(x, 0.5, 0.2, 0.06) }
fn pe_over(x: f64) -> Interval     { trapezoid_it2(x, 0.6, 0.8, 1.0, 1.0, 0.05) }

// ---------------------------------------------------------------------------
// Rule builder
// ---------------------------------------------------------------------------

fn build_rules() -> Vec<Rule> {
    macro_rules! rule {
        ($_out_signal:expr, $term:literal, |$inp:ident| $body:expr) => {
            Rule {
                term: $term,
                antecedent: Box::new(move |$inp: &FuzzyInputs| $body),
            }
        };
    }

    vec![
        // --- Strong buy ---
        rule!( 1.0, "strong_buy", |i| {
            let a = interval_min(pt_up(i.price_trend),  ai_bullish(i.ai_consensus));
            let b = interval_min(a, rs_positive(i.research_sentiment));
            interval_min(b, pe_under(i.portfolio_exposure))
        }),
        rule!( 1.0, "strong_buy", |i| {
            let a = interval_min(pt_up(i.price_trend),  vr_high(i.volume_ratio));
            interval_min(a, ai_bullish(i.ai_consensus))
        }),

        // --- Buy ---
        rule!( 0.5, "buy", |i| {
            interval_min(pt_up(i.price_trend), ai_bullish(i.ai_consensus))
        }),
        rule!( 0.5, "buy", |i| {
            let a = interval_min(ai_bullish(i.ai_consensus), rs_positive(i.research_sentiment));
            interval_min(a, pe_under(i.portfolio_exposure))
        }),
        rule!( 0.5, "buy", |i| {
            let a = interval_min(pt_flat(i.price_trend), vr_high(i.volume_ratio));
            interval_min(a, ai_bullish(i.ai_consensus))
        }),
        rule!( 0.5, "buy", |i| {
            interval_min(pt_up(i.price_trend), rs_positive(i.research_sentiment))
        }),
        rule!( 0.5, "buy", |i| {
            let a = interval_min(ai_bullish(i.ai_consensus), pe_under(i.portfolio_exposure));
            interval_min(a, rs_neutral(i.research_sentiment))
        }),

        // --- Hold ---
        rule!( 0.0, "hold", |i| {
            interval_min(pt_flat(i.price_trend), ai_neutral(i.ai_consensus))
        }),
        rule!( 0.0, "hold", |i| {
            interval_min(ai_neutral(i.ai_consensus), pe_balanced(i.portfolio_exposure))
        }),
        rule!( 0.0, "hold", |i| {
            let a = interval_min(pt_flat(i.price_trend), rs_neutral(i.research_sentiment));
            interval_min(a, vr_low(i.volume_ratio))
        }),
        rule!( 0.0, "hold", |i| {
            let a = interval_min(ai_bullish(i.ai_consensus), pe_over(i.portfolio_exposure));
            interval_min(a, rs_neutral(i.research_sentiment))
        }),
        rule!( 0.0, "hold", |i| {
            let a = interval_min(ai_bearish(i.ai_consensus), pe_under(i.portfolio_exposure));
            interval_min(a, rs_neutral(i.research_sentiment))
        }),
        rule!( 0.0, "hold", |i| {
            interval_min(pt_down(i.price_trend), vr_low(i.volume_ratio))
        }),

        // --- Sell ---
        rule!(-0.5, "sell", |i| {
            interval_min(pt_down(i.price_trend), ai_bearish(i.ai_consensus))
        }),
        rule!(-0.5, "sell", |i| {
            let a = interval_min(ai_bearish(i.ai_consensus), rs_negative(i.research_sentiment));
            interval_min(a, pe_over(i.portfolio_exposure))
        }),
        rule!(-0.5, "sell", |i| {
            interval_min(pt_down(i.price_trend), rs_negative(i.research_sentiment))
        }),
        rule!(-0.5, "sell", |i| {
            let a = interval_min(pt_flat(i.price_trend), ai_bearish(i.ai_consensus));
            interval_min(a, pe_over(i.portfolio_exposure))
        }),
        rule!(-0.5, "sell", |i| {
            interval_min(ai_bearish(i.ai_consensus), pe_over(i.portfolio_exposure))
        }),

        // --- Strong sell ---
        rule!(-1.0, "strong_sell", |i| {
            let a = interval_min(pt_down(i.price_trend), ai_bearish(i.ai_consensus));
            let b = interval_min(a, rs_negative(i.research_sentiment));
            interval_min(b, pe_over(i.portfolio_exposure))
        }),
        rule!(-1.0, "strong_sell", |i| {
            let a = interval_min(pt_down(i.price_trend), vr_high(i.volume_ratio));
            interval_min(a, ai_bearish(i.ai_consensus))
        }),
        rule!(-1.0, "strong_sell", |i| {
            let a = interval_min(ai_bearish(i.ai_consensus), rs_negative(i.research_sentiment));
            interval_min(a, pe_over(i.portfolio_exposure))
        }),

        // --- Edge cases: high volume + neutral → buy/sell bias ---
        rule!( 0.4, "buy", |i| {
            let a = interval_min(vr_high(i.volume_ratio), pt_up(i.price_trend));
            interval_min(a, rs_positive(i.research_sentiment))
        }),
        rule!(-0.4, "sell", |i| {
            let a = interval_min(vr_high(i.volume_ratio), pt_down(i.price_trend));
            interval_min(a, rs_negative(i.research_sentiment))
        }),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> FuzzyEngine { FuzzyEngine::new() }

    #[test]
    fn test_strong_buy_conditions() {
        let e = engine();
        let out = e.evaluate(&FuzzyInputs {
            price_trend: 0.8,
            volume_ratio: 1.8,
            ai_consensus: 0.9,
            research_sentiment: 0.8,
            portfolio_exposure: 0.1,
        });
        assert!(out.signal > 0.3, "strong buy should produce positive signal, got {}", out.signal);
        assert!(out.confidence > 0.1, "confidence should be non-trivial");
    }

    #[test]
    fn test_strong_sell_conditions() {
        let e = engine();
        let out = e.evaluate(&FuzzyInputs {
            price_trend: -0.8,
            volume_ratio: 1.8,
            ai_consensus: -0.9,
            research_sentiment: -0.8,
            portfolio_exposure: 0.9,
        });
        assert!(out.signal < -0.3, "strong sell should produce negative signal, got {}", out.signal);
    }

    #[test]
    fn test_neutral_holds() {
        let e = engine();
        let out = e.evaluate(&FuzzyInputs::default());
        assert!(out.signal.abs() < 0.4, "neutral inputs should produce hold-ish signal, got {}", out.signal);
    }

    #[test]
    fn test_signal_in_range() {
        let e = engine();
        for pt in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            for ai in [-1.0, 0.0, 1.0] {
                let out = e.evaluate(&FuzzyInputs {
                    price_trend: pt,
                    ai_consensus: ai,
                    ..Default::default()
                });
                assert!(out.signal >= -1.0 && out.signal <= 1.0,
                    "signal out of range: {}", out.signal);
                assert!(out.confidence >= 0.0 && out.confidence <= 1.0,
                    "confidence out of range: {}", out.confidence);
            }
        }
    }
}
