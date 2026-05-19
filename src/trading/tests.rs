/// Comprehensive test suite for the Gail crypto trading bridge.
///
/// Tests cover:
/// - TradingConfig: defaults, normalisation, viability check, runtime overrides
/// - TradingState / SharedTradingState: ring buffers, logging, trade recording, persistence
/// - FuzzyEngine: all five input dimensions, boundary conditions, monotonicity, range invariants
/// - Advisor consensus aggregation: weighting, failure handling, edge cases
/// - DecisionEngine: blending, all three risk gates, override mechanism, trade sizing
/// - Pipeline integration: end-to-end signal flow from raw inputs to trade decision
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use crate::trading::advisor::{AiAdvice, AiConsensus};
    use crate::trading::config::{TradingConfig, TradingConfigOverride};
    use crate::trading::decision::DecisionEngine;
    use crate::trading::degraded_live_execution_reason;
    use crate::trading::fuzzy::{FuzzyEngine, FuzzyInputs};
    use crate::trading::octobot::{
        CurrencyBalance, MarketSnapshot, OctobotOrder, OctobotPortfolio,
    };
    use crate::trading::state::{ExecutedTrade, SharedTradingState, TradeAction, TradeOverride};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn bullish_inputs() -> FuzzyInputs {
        FuzzyInputs {
            price_trend: 0.8,
            volume_ratio: 1.8,
            ai_consensus: 0.9,
            research_sentiment: 0.7,
            portfolio_exposure: 0.1,
        }
    }

    fn bearish_inputs() -> FuzzyInputs {
        FuzzyInputs {
            price_trend: -0.8,
            volume_ratio: 1.8,
            ai_consensus: -0.9,
            research_sentiment: -0.7,
            portfolio_exposure: 0.9,
        }
    }

    fn make_advice(action: &str, confidence: f64, weight: f64) -> AiAdvice {
        AiAdvice {
            provider: "test".to_string(),
            model: Some("test-model".to_string()),
            action: action.to_string(),
            confidence,
            reasoning: format!("test reasoning for {action}"),
            suggested_amount_usd: None,
            risk_score: 0.25,
            risk_flags: Vec::new(),
            target_symbol: Some("BTC/USDT".to_string()),
            raw_response: format!(
                r#"{{"action":"{action}","confidence":{confidence},"reasoning":"test"}}"#
            ),
            parsed_ok: true,
            weight,
        }
    }

    fn make_failed_advice(weight: f64) -> AiAdvice {
        AiAdvice {
            provider: "failing-provider".to_string(),
            model: None,
            action: "hold".to_string(),
            confidence: 0.0,
            reasoning: "provider error: timeout".to_string(),
            suggested_amount_usd: None,
            risk_score: 1.0,
            risk_flags: vec!["provider_error".to_string()],
            target_symbol: None,
            raw_response: String::new(),
            parsed_ok: false,
            weight,
        }
    }

    fn consensus_from_advices(advices: Vec<AiAdvice>, failures: usize) -> AiConsensus {
        // Mirror the aggregate_consensus logic from advisor.rs via a helper
        // that constructs an AiConsensus directly from computed values.
        let responders = advices.iter().filter(|a| a.parsed_ok).count();
        if responders == 0 {
            return AiConsensus {
                action: "hold".to_string(),
                confidence: 0.0,
                signal: 0.0,
                vote_distribution: json!({}),
                advices,
                responders: 0,
                failures: failures + responders,
            };
        }
        let action_to_signal = |a: &str| match a {
            "strong_buy" => 1.0,
            "buy" => 0.5,
            "sell" => -0.5,
            "strong_sell" => -1.0,
            _ => 0.0_f64,
        };
        let mut weighted_signal = 0.0_f64;
        let mut total_weight = 0.0_f64;
        for a in &advices {
            if !a.parsed_ok {
                continue;
            }
            let ew = (a.weight * a.confidence).max(0.01);
            weighted_signal += action_to_signal(&a.action) * ew;
            total_weight += ew;
        }
        let signal = if total_weight > 0.0 {
            weighted_signal / total_weight
        } else {
            0.0
        };
        let action = match signal {
            s if s >= 0.65 => "strong_buy",
            s if s >= 0.2 => "buy",
            s if s <= -0.65 => "strong_sell",
            s if s <= -0.2 => "sell",
            _ => "hold",
        }
        .to_string();
        let conf_num: f64 = advices
            .iter()
            .filter(|a| a.parsed_ok)
            .map(|a| a.confidence * a.weight)
            .sum();
        let conf_den: f64 = advices
            .iter()
            .filter(|a| a.parsed_ok)
            .map(|a| a.weight)
            .sum();
        let confidence = if conf_den > 0.0 {
            (conf_num / conf_den).clamp(0.0, 1.0)
        } else {
            0.0
        };
        AiConsensus {
            action,
            confidence,
            signal,
            vote_distribution: json!({}),
            advices,
            responders,
            failures,
        }
    }

    fn make_strong_bullish_consensus() -> AiConsensus {
        consensus_from_advices(
            vec![
                make_advice("strong_buy", 0.9, 1.0),
                make_advice("strong_buy", 0.85, 1.0),
                make_advice("buy", 0.8, 0.9),
            ],
            0,
        )
    }

    fn make_strong_bearish_consensus() -> AiConsensus {
        consensus_from_advices(
            vec![
                make_advice("strong_sell", 0.9, 1.0),
                make_advice("strong_sell", 0.85, 1.0),
                make_advice("sell", 0.8, 0.9),
            ],
            0,
        )
    }

    fn make_neutral_consensus() -> AiConsensus {
        consensus_from_advices(
            vec![make_advice("hold", 0.5, 1.0), make_advice("hold", 0.5, 1.0)],
            0,
        )
    }

    fn make_snapshot(
        exchange: &str,
        symbol: &str,
        price: f64,
        change_pct: f64,
        volume: f64,
    ) -> MarketSnapshot {
        MarketSnapshot {
            exchange: exchange.to_string(),
            symbol: symbol.to_string(),
            price,
            price_change_pct_1h: None,
            price_change_pct_24h: Some(change_pct),
            volume_24h: Some(volume),
            volume_change_pct: None,
            high_24h: None,
            low_24h: None,
            fetched_at: 0.0,
        }
    }

    fn make_portfolio(total_usd: f64) -> OctobotPortfolio {
        let mut currencies = std::collections::HashMap::new();
        currencies.insert(
            "BTC".to_string(),
            CurrencyBalance {
                free: 0.001,
                locked: 0.0,
                total: 0.001,
                value_usd: Some(total_usd * 0.8),
            },
        );
        currencies.insert(
            "USDT".to_string(),
            CurrencyBalance {
                free: total_usd * 0.2,
                locked: 0.0,
                total: total_usd * 0.2,
                value_usd: Some(total_usd * 0.2),
            },
        );
        OctobotPortfolio {
            currencies,
            total_value_usd: Some(total_usd),
        }
    }

    fn make_open_order(id: &str) -> OctobotOrder {
        OctobotOrder {
            id: id.to_string(),
            exchange: "binance".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: "buy".to_string(),
            order_type: "limit".to_string(),
            amount: 0.0001,
            price: Some(50000.0),
            status: "open".to_string(),
            timestamp: Some(1_700_000_000.0),
        }
    }

    fn default_config() -> TradingConfig {
        TradingConfig {
            enabled: true,
            octobot_base_url: "http://localhost:5001".to_string(),
            ..TradingConfig::default()
        }
    }

    // -----------------------------------------------------------------------
    // TradingConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn config_defaults_are_disabled() {
        let cfg = TradingConfig::default();
        assert!(!cfg.enabled, "trading must be disabled by default");
        assert_eq!(cfg.micro_trade_max_usd, 25.0);
        assert_eq!(cfg.micro_trade_min_usd, 1.0);
        assert_eq!(cfg.fuzzy_confidence_threshold, 0.65);
        assert_eq!(cfg.fuzzy_weight, 0.4);
        assert_eq!(cfg.max_open_positions, 5);
        assert_eq!(cfg.evaluation_interval_seconds, 60);
        assert_eq!(cfg.research_index_name, "crypto");
        assert_eq!(cfg.research_site_hints, vec!["bloomberg.com".to_string()]);
        assert_eq!(cfg.research_max_parallel_queries, 3);
        assert!(cfg.live_execution_enabled);
        assert!(!cfg.backtesting_enabled);
        assert!(cfg.backtest_data_collection_enabled);
        assert_eq!(cfg.backtest_data_collection_exchange, "binance");
        assert_eq!(cfg.backtest_data_collection_time_frames, vec!["1h", "1d"]);
    }

    #[test]
    fn config_not_viable_when_disabled() {
        let cfg = TradingConfig::default();
        assert!(!cfg.is_viable(), "disabled config must not be viable");
    }

    #[test]
    fn config_not_viable_without_octobot_url() {
        let cfg = TradingConfig {
            enabled: true,
            ..TradingConfig::default()
        };
        assert!(
            !cfg.is_viable(),
            "config without octobot_base_url must not be viable"
        );
    }

    #[test]
    fn config_viable_with_enabled_and_url() {
        let cfg = default_config();
        assert!(cfg.is_viable());
    }

    #[test]
    fn config_normalize_clamps_values() {
        let mut cfg = TradingConfig {
            enabled: true,
            octobot_base_url: "http://x".to_string(),
            micro_trade_max_usd: -5.0,
            micro_trade_min_usd: 100.0, // min > max before normalise
            fuzzy_confidence_threshold: 1.5,
            fuzzy_weight: -0.1,
            evaluation_interval_seconds: 3, // below minimum of 10
            max_parallel_advisors: 0,       // below minimum of 1
            max_open_positions: 0,
            research_index_name: "   ".to_string(),
            research_site_hints: vec![
                " bloomberg.com ".to_string(),
                "".to_string(),
                "BLOOMBERG.com".to_string(),
            ],
            research_max_parallel_queries: 0,
            research_top_k: 0,
            log_ring_size: 0,
            trade_ring_size: 0,
            backtest_data_collection_exchange: "  ".to_string(),
            backtest_data_collection_time_frames: vec![
                " 1h ".to_string(),
                "".to_string(),
                "1H".to_string(),
            ],
            backtest_data_collection_cooldown_seconds: 0,
            ..TradingConfig::default()
        };
        cfg.normalize();
        assert!(cfg.micro_trade_max_usd >= 0.01, "max_usd must be clamped");
        assert!(
            cfg.micro_trade_min_usd <= cfg.micro_trade_max_usd,
            "min must not exceed max after normalize"
        );
        assert!(cfg.fuzzy_confidence_threshold <= 1.0);
        assert!(cfg.fuzzy_weight >= 0.0);
        assert!(cfg.evaluation_interval_seconds >= 10);
        assert!(cfg.max_parallel_advisors >= 1);
        assert!(cfg.max_open_positions >= 1);
        assert_eq!(cfg.research_index_name, "crypto");
        assert_eq!(cfg.research_site_hints, vec!["bloomberg.com".to_string()]);
        assert_eq!(cfg.research_max_parallel_queries, 1);
        assert_eq!(cfg.research_top_k, 1);
        assert!(cfg.log_ring_size >= 10);
        assert!(cfg.trade_ring_size >= 10);
        assert_eq!(cfg.backtest_data_collection_exchange, "binance");
        assert_eq!(cfg.backtest_data_collection_time_frames, vec!["1h"]);
        assert!(cfg.backtest_data_collection_cooldown_seconds >= 60);
    }

    #[test]
    fn config_normalize_fills_empty_template() {
        let mut cfg = TradingConfig {
            research_query_template: "  ".to_string(),
            ..TradingConfig::default()
        };
        cfg.normalize();
        assert!(
            !cfg.research_query_template.trim().is_empty(),
            "empty template should be replaced with default"
        );
    }

    #[test]
    fn config_normalize_fills_empty_data_path() {
        let mut cfg = TradingConfig {
            data_path: "".to_string(),
            ..TradingConfig::default()
        };
        cfg.normalize();
        assert!(
            !cfg.data_path.is_empty(),
            "empty data_path should be replaced"
        );
    }

    #[test]
    fn config_override_fields_are_optional() {
        let ov = TradingConfigOverride::default();
        assert!(ov.evaluation_interval_seconds.is_none());
        assert!(ov.micro_trade_max_usd.is_none());
        assert!(ov.fuzzy_confidence_threshold.is_none());
        assert!(ov.target_exchanges.is_none());
        assert!(ov.target_currencies.is_none());
    }

    #[test]
    fn trade_floor_feedback_does_not_create_empty_override() {
        use crate::trading::state::TradingState;

        let config = TradingConfig {
            micro_trade_min_usd: 5.0,
            micro_trade_max_usd: 25.0,
            ..TradingConfig::default()
        };
        let mut state = TradingState::new(100, 100);

        crate::trading::apply_trade_floor_feedback(&config, &mut state, 4.0);

        assert!(state.config_overrides.is_none());
    }

    #[test]
    fn trade_floor_feedback_raises_minimum_and_maximum() {
        use crate::trading::state::TradingState;

        let config = TradingConfig {
            micro_trade_min_usd: 1.0,
            micro_trade_max_usd: 3.0,
            ..TradingConfig::default()
        };
        let mut state = TradingState::new(100, 100);

        crate::trading::apply_trade_floor_feedback(&config, &mut state, 5.0);

        let overrides = state.config_overrides.expect("adaptive overrides");
        assert_eq!(overrides.micro_trade_min_usd, Some(5.0));
        assert_eq!(overrides.micro_trade_max_usd, Some(5.0));
        assert_eq!(state.activity_log.len(), 1);
    }

    // -----------------------------------------------------------------------
    // TradingState tests
    // -----------------------------------------------------------------------

    #[test]
    fn state_log_ring_enforces_size() {
        use crate::trading::state::TradingState;
        let mut state = TradingState::new(5, 10);
        for i in 0..10 {
            state.log_info("test", format!("message {i}"));
        }
        assert_eq!(
            state.activity_log.len(),
            5,
            "ring buffer must cap at log_ring_size"
        );
        // Most recent message should be the last one written.
        assert!(
            state
                .activity_log
                .back()
                .unwrap()
                .message
                .contains("message 9")
        );
    }

    #[test]
    fn state_trade_ring_enforces_size() {
        use crate::trading::state::TradingState;
        let mut state = TradingState::new(100, 3);
        for i in 0..6u64 {
            state.record_trade(ExecutedTrade {
                ts: i as f64,
                exchange: "binance".to_string(),
                symbol: "BTC/USDT".to_string(),
                action: TradeAction::Buy,
                amount_usd: 5.0,
                price: Some(50000.0),
                order_id: Some(format!("ord-{i}")),
                confidence: 0.8,
                rationale: "test".to_string(),
                ai_votes: serde_json::Value::Null,
                fuzzy_confidence: 0.7,
                ai_confidence: 0.9,
            });
        }
        assert_eq!(
            state.recent_trades.len(),
            3,
            "trade ring must cap at trade_ring_size"
        );
        assert_eq!(
            state.trade_count, 6,
            "trade_count must reflect all recorded trades"
        );
    }

    #[test]
    fn state_record_trade_updates_last_trade_at() {
        use crate::trading::state::TradingState;
        let mut state = TradingState::new(100, 100);
        assert!(state.last_trade_at.is_none());
        state.record_trade(ExecutedTrade {
            ts: 12345.0,
            exchange: "binance".to_string(),
            symbol: "ETH/USDT".to_string(),
            action: TradeAction::Sell,
            amount_usd: 10.0,
            price: None,
            order_id: None,
            confidence: 0.7,
            rationale: String::new(),
            ai_votes: serde_json::Value::Null,
            fuzzy_confidence: 0.6,
            ai_confidence: 0.8,
        });
        assert_eq!(state.last_trade_at, Some(12345.0));
    }

    #[test]
    fn state_log_error_sets_last_error() {
        use crate::trading::state::TradingState;
        let mut state = TradingState::new(100, 100);
        assert!(state.last_error.is_none());
        state.log_error("test", "something went wrong");
        assert_eq!(state.last_error.as_deref(), Some("something went wrong"));
    }

    #[test]
    fn state_status_snapshot_reflects_state() {
        use crate::trading::state::TradingState;
        let mut state = TradingState::new(100, 100);
        state.evaluation_count = 7;
        state.trade_count = 3;
        state.paused = true;
        state.open_positions = vec![make_open_order("o1"), make_open_order("o2")];
        let snap = state.status_snapshot(true);
        assert!(snap.enabled);
        assert!(snap.paused);
        assert_eq!(snap.evaluation_count, 7);
        assert_eq!(snap.trade_count, 3);
        assert_eq!(snap.open_positions, 2);
    }

    #[test]
    fn state_take_override_clears_it() {
        use crate::trading::state::TradingState;
        let mut state = TradingState::new(100, 100);
        state.pending_override = Some(TradeOverride {
            action: TradeAction::Buy,
            exchange: Some("binance".to_string()),
            symbol: Some("BTC/USDT".to_string()),
            amount_usd: Some(10.0),
            reason: None,
            issued_at: 0.0,
            issued_by: "pbisaacs".to_string(),
        });
        let ov = state.take_override();
        assert!(ov.is_some());
        assert!(
            state.pending_override.is_none(),
            "override must be cleared after take"
        );
    }

    #[tokio::test]
    async fn shared_state_persist_and_restore() {
        let state = SharedTradingState::new(100, 50);

        // Write some data.
        {
            let mut s = state.0.lock().await;
            s.evaluation_count = 42;
            s.trade_count = 6; // record_trade will increment this to 7
            s.record_trade(ExecutedTrade {
                ts: 99999.0,
                exchange: "kraken".to_string(),
                symbol: "ETH/USDT".to_string(),
                action: TradeAction::Buy,
                amount_usd: 5.0,
                price: Some(3000.0),
                order_id: Some("test-order".to_string()),
                confidence: 0.75,
                rationale: "persist test".to_string(),
                ai_votes: serde_json::Value::Null,
                fuzzy_confidence: 0.6,
                ai_confidence: 0.9,
            });
        }

        let tmp = PathBuf::from("/tmp/gail_trading_test_state.json");
        state.persist(&tmp).await;

        // Restore into a fresh state.
        let restored = SharedTradingState::new(100, 50);
        restored.restore(&tmp).await;
        let s = restored.0.lock().await;
        assert_eq!(s.evaluation_count, 42);
        assert_eq!(s.trade_count, 7);
        assert_eq!(s.recent_trades.len(), 1);
        assert_eq!(s.recent_trades[0].exchange, "kraken");

        // Clean up.
        let _ = std::fs::remove_file(&tmp);
    }

    #[tokio::test]
    async fn shared_state_persist_includes_null_log_context_field() {
        let state = SharedTradingState::new(100, 50);
        state.log_info("startup", "no context payload").await;

        let tmp = tempfile::NamedTempFile::new().expect("temp file");
        state.persist(&tmp.path().to_path_buf()).await;

        let payload: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.path()).expect("state json"))
                .expect("valid state payload");
        assert!(
            payload["activity_log"][0].get("context").is_some(),
            "persisted activity_log entries must include context for backward compatibility"
        );
        assert!(payload["activity_log"][0]["context"].is_null());
    }

    #[tokio::test]
    async fn shared_state_restore_accepts_legacy_log_entries_without_context() {
        use crate::trading::state::TradingState;

        let mut legacy = TradingState::new(100, 50);
        legacy.evaluation_count = 17;
        legacy.log_info("startup", "legacy entry");
        let mut payload = serde_json::to_value(&legacy).expect("state json");
        payload["activity_log"][0]
            .as_object_mut()
            .expect("legacy log object")
            .remove("context");

        let tmp = tempfile::NamedTempFile::new().expect("temp file");
        std::fs::write(
            tmp.path(),
            serde_json::to_string_pretty(&payload).expect("legacy state json"),
        )
        .expect("write legacy state");

        let restored = SharedTradingState::new(100, 50);
        restored.restore(&tmp.path().to_path_buf()).await;
        let s = restored.0.lock().await;
        assert_eq!(s.evaluation_count, 17);
        assert_eq!(s.activity_log.front().unwrap().message, "legacy entry");
        assert!(s.activity_log.front().unwrap().context.is_null());
        drop(s);

        let repaired: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(tmp.path()).expect("read repaired state"),
        )
        .expect("repaired state json");
        assert!(repaired["activity_log"][0].get("context").is_some());
    }

    #[tokio::test]
    async fn shared_state_restore_missing_file_is_noop() {
        let state = SharedTradingState::new(100, 50);
        // Should not panic.
        state
            .restore(&PathBuf::from("/tmp/gail_nonexistent_state_abc.json"))
            .await;
        let s = state.0.lock().await;
        assert_eq!(s.evaluation_count, 0);
    }

    // -----------------------------------------------------------------------
    // FuzzyEngine tests
    // -----------------------------------------------------------------------

    #[test]
    fn fuzzy_signal_always_in_range() {
        let engine = FuzzyEngine::new();
        let test_values = [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0];
        for &pt in &test_values {
            for &ai in &test_values {
                for &rs in &[-0.5_f64, 0.0, 0.5] {
                    for &pe in &[0.1_f64, 0.5, 0.9] {
                        let out = engine.evaluate(&FuzzyInputs {
                            price_trend: pt,
                            volume_ratio: 1.0,
                            ai_consensus: ai,
                            research_sentiment: rs,
                            portfolio_exposure: pe,
                        });
                        assert!(
                            out.signal >= -1.0 && out.signal <= 1.0,
                            "signal {:.4} out of [-1,1] for pt={pt} ai={ai}",
                            out.signal
                        );
                        assert!(
                            out.confidence >= 0.0 && out.confidence <= 1.0,
                            "confidence {:.4} out of [0,1]",
                            out.confidence
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fuzzy_bullish_inputs_produce_positive_signal() {
        let engine = FuzzyEngine::new();
        let out = engine.evaluate(&bullish_inputs());
        assert!(
            out.signal > 0.2,
            "strongly bullish inputs should produce positive signal, got {:.4}",
            out.signal
        );
    }

    #[test]
    fn fuzzy_bearish_inputs_produce_negative_signal() {
        let engine = FuzzyEngine::new();
        let out = engine.evaluate(&bearish_inputs());
        assert!(
            out.signal < -0.2,
            "strongly bearish inputs should produce negative signal, got {:.4}",
            out.signal
        );
    }

    #[test]
    fn fuzzy_neutral_inputs_produce_hold_ish_signal() {
        let engine = FuzzyEngine::new();
        let out = engine.evaluate(&FuzzyInputs::default());
        assert!(
            out.signal.abs() < 0.4,
            "neutral inputs should produce hold-ish signal, got {:.4}",
            out.signal
        );
    }

    #[test]
    fn fuzzy_signal_directional_monotonicity() {
        // Increasing ai_consensus while keeping other inputs constant should
        // not decrease the signal.
        let engine = FuzzyEngine::new();
        let mut prev_signal = f64::NEG_INFINITY;
        for &ai in &[-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
            let out = engine.evaluate(&FuzzyInputs {
                price_trend: 0.5,
                volume_ratio: 1.2,
                ai_consensus: ai,
                research_sentiment: 0.0,
                portfolio_exposure: 0.3,
            });
            assert!(
                out.signal >= prev_signal - 0.2,
                "signal should not sharply decrease as ai_consensus increases: prev={:.4} cur={:.4} ai={ai}",
                prev_signal,
                out.signal
            );
            prev_signal = out.signal;
        }
    }

    #[test]
    fn fuzzy_high_overexposure_inhibits_strong_buy() {
        let engine = FuzzyEngine::new();
        // Portfolio fully invested (overweight) with bullish signal.
        let out_overweight = engine.evaluate(&FuzzyInputs {
            price_trend: 0.8,
            volume_ratio: 1.5,
            ai_consensus: 0.8,
            research_sentiment: 0.5,
            portfolio_exposure: 0.95, // overweight
        });
        // Portfolio underweight with same signal.
        let out_underweight = engine.evaluate(&FuzzyInputs {
            price_trend: 0.8,
            volume_ratio: 1.5,
            ai_consensus: 0.8,
            research_sentiment: 0.5,
            portfolio_exposure: 0.05, // underweight
        });
        // Underweight should be at least as bullish or more bullish.
        assert!(
            out_underweight.signal >= out_overweight.signal - 0.1,
            "underweight exposure should not be less bullish than overweight; \
             underweight={:.4} overweight={:.4}",
            out_underweight.signal,
            out_overweight.signal
        );
    }

    #[test]
    fn fuzzy_term_activations_sum_reasonable() {
        let engine = FuzzyEngine::new();
        let out = engine.evaluate(&bullish_inputs());
        let sum = out.term_activations.strong_sell
            + out.term_activations.sell
            + out.term_activations.hold
            + out.term_activations.buy
            + out.term_activations.strong_buy;
        assert!(sum > 0.0, "at least one term should be activated");
        assert!(sum <= 5.0, "sum of activations should be bounded");
    }

    #[test]
    fn fuzzy_label_matches_signal() {
        let engine = FuzzyEngine::new();
        let out = engine.evaluate(&bullish_inputs());
        if out.signal >= 0.65 {
            assert_eq!(out.label, "strong_buy");
        } else if out.signal >= 0.2 {
            assert_eq!(out.label, "buy");
        } else if out.signal <= -0.65 {
            assert_eq!(out.label, "strong_sell");
        } else if out.signal <= -0.2 {
            assert_eq!(out.label, "sell");
        } else {
            assert_eq!(out.label, "hold");
        }
    }

    #[test]
    fn fuzzy_volume_high_amplifies_directional_signal() {
        let engine = FuzzyEngine::new();
        let high_vol = engine.evaluate(&FuzzyInputs {
            price_trend: 0.7,
            volume_ratio: 1.8,
            ai_consensus: 0.7,
            research_sentiment: 0.3,
            portfolio_exposure: 0.2,
        });
        let low_vol = engine.evaluate(&FuzzyInputs {
            price_trend: 0.7,
            volume_ratio: 0.2,
            ai_consensus: 0.7,
            research_sentiment: 0.3,
            portfolio_exposure: 0.2,
        });
        // High volume with bullish price/AI should be at least as bullish as low volume.
        assert!(
            high_vol.signal >= low_vol.signal - 0.15,
            "high volume should not inhibit bullish signal; high={:.4} low={:.4}",
            high_vol.signal,
            low_vol.signal
        );
    }

    // -----------------------------------------------------------------------
    // Advisor consensus aggregation tests
    // -----------------------------------------------------------------------

    #[test]
    fn consensus_all_strong_buy_produces_high_positive_signal() {
        let c = make_strong_bullish_consensus();
        assert!(
            c.signal > 0.4,
            "all buy advices should produce positive signal: {:.4}",
            c.signal
        );
        assert_eq!(c.responders, 3);
        assert_eq!(c.failures, 0);
    }

    #[test]
    fn consensus_all_strong_sell_produces_high_negative_signal() {
        let c = make_strong_bearish_consensus();
        assert!(
            c.signal < -0.4,
            "all sell advices should produce negative signal: {:.4}",
            c.signal
        );
    }

    #[test]
    fn consensus_all_hold_produces_near_zero_signal() {
        let c = make_neutral_consensus();
        assert!(
            c.signal.abs() < 0.1,
            "all hold advices should produce ~0 signal: {:.4}",
            c.signal
        );
    }

    #[test]
    fn consensus_all_failures_returns_uncertain() {
        let advices = vec![make_failed_advice(1.0), make_failed_advice(1.0)];
        let c = consensus_from_advices(advices, 0);
        assert_eq!(c.signal, 0.0);
        assert_eq!(c.confidence, 0.0);
        assert_eq!(c.action, "hold");
    }

    #[test]
    fn consensus_mixed_advices_signal_between_extremes() {
        let advices = vec![make_advice("buy", 0.8, 1.0), make_advice("sell", 0.8, 1.0)];
        let c = consensus_from_advices(advices, 0);
        // Equal buy/sell with equal weights → signal near 0.
        assert!(
            c.signal.abs() < 0.2,
            "balanced buy/sell should produce near-zero signal: {:.4}",
            c.signal
        );
    }

    #[test]
    fn consensus_higher_weight_provider_dominates() {
        let advices = vec![
            make_advice("strong_buy", 0.9, 10.0), // dominant weight
            make_advice("strong_sell", 0.9, 1.0),
            make_advice("strong_sell", 0.9, 1.0),
        ];
        let c = consensus_from_advices(advices, 0);
        assert!(
            c.signal > 0.0,
            "high-weight buy provider should dominate: signal={:.4}",
            c.signal
        );
    }

    #[test]
    fn consensus_failures_are_counted() {
        let advices = vec![
            make_advice("buy", 0.8, 1.0),
            make_failed_advice(1.0),
            make_failed_advice(1.0),
        ];
        let c = consensus_from_advices(advices, 1); // 1 extra failure from timeout
        assert_eq!(c.failures, 1, "failures parameter should be passed through");
        assert_eq!(c.responders, 1);
    }

    #[test]
    fn consensus_confidence_is_clamped() {
        let advices = vec![make_advice("buy", 1.0, 100.0)];
        let c = consensus_from_advices(advices, 0);
        assert!(c.confidence <= 1.0, "confidence must not exceed 1.0");
        assert!(c.confidence >= 0.0, "confidence must not be negative");
    }

    #[test]
    fn consensus_single_provider_preserves_signal() {
        let advices = vec![make_advice("strong_buy", 0.9, 1.0)];
        let c = consensus_from_advices(advices, 0);
        assert!(
            c.signal > 0.5,
            "single strong_buy should produce strong positive signal"
        );
    }

    #[test]
    fn degraded_execution_guard_blocks_when_no_advisors_respond() {
        let config = default_config();
        let consensus = AiConsensus {
            action: "hold".to_string(),
            confidence: 0.0,
            signal: 0.0,
            vote_distribution: json!({}),
            advices: Vec::new(),
            responders: 0,
            failures: 2,
        };
        let reason = degraded_live_execution_reason(&consensus, &config);
        assert!(
            reason.is_some(),
            "guard should block execution when no advisor responds"
        );
    }

    #[test]
    fn degraded_execution_guard_blocks_high_risk_consensus() {
        let config = default_config();
        let consensus = AiConsensus {
            action: "buy".to_string(),
            confidence: 0.8,
            signal: 0.5,
            vote_distribution: json!({
                "average_risk": 0.85,
                "coverage": 1.0,
                "agreement": 1.0
            }),
            advices: vec![make_advice("buy", 0.8, 1.0)],
            responders: 1,
            failures: 0,
        };
        let reason = degraded_live_execution_reason(&consensus, &config);
        assert!(
            reason
                .as_deref()
                .is_some_and(|value| value.contains("risk")),
            "guard should block high-risk consensus, reason={reason:?}"
        );
    }

    #[test]
    fn degraded_execution_guard_allows_stable_low_risk_consensus() {
        let config = default_config();
        let consensus = AiConsensus {
            action: "buy".to_string(),
            confidence: 0.8,
            signal: 0.5,
            vote_distribution: json!({
                "average_risk": 0.2,
                "coverage": 1.0,
                "agreement": 1.0
            }),
            advices: vec![make_advice("buy", 0.8, 1.0)],
            responders: 1,
            failures: 0,
        };
        let reason = degraded_live_execution_reason(&consensus, &config);
        assert!(reason.is_none(), "stable consensus should not be blocked");
    }

    // -----------------------------------------------------------------------
    // DecisionEngine tests
    // -----------------------------------------------------------------------

    fn make_default_state() -> crate::trading::state::TradingState {
        crate::trading::state::TradingState::new(100, 100)
    }

    #[test]
    fn decision_strong_bullish_signals_produce_buy() {
        let engine = DecisionEngine::new(0.4);
        let fuzzy = FuzzyEngine::new().evaluate(&bullish_inputs());
        let consensus = make_strong_bullish_consensus();
        let state = make_default_state();
        let config = default_config();
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, 5.0, 2_000_000.0);
        let decision = engine.decide(&fuzzy, &consensus, Some(&snap), &state, &config);
        assert!(
            matches!(decision.action, TradeAction::Buy | TradeAction::StrongBuy),
            "strong bullish signals should produce a buy decision, got {:?}",
            decision.action
        );
    }

    #[test]
    fn decision_strong_bearish_signals_produce_sell() {
        let engine = DecisionEngine::new(0.4);
        let fuzzy = FuzzyEngine::new().evaluate(&bearish_inputs());
        let consensus = make_strong_bearish_consensus();
        let state = make_default_state();
        let config = default_config();
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, -5.0, 2_000_000.0);
        let decision = engine.decide(&fuzzy, &consensus, Some(&snap), &state, &config);
        assert!(
            matches!(decision.action, TradeAction::Sell | TradeAction::StrongSell),
            "strong bearish signals should produce a sell decision, got {:?}",
            decision.action
        );
    }

    #[test]
    fn decision_neutral_signals_produce_hold() {
        let engine = DecisionEngine::new(0.4);
        let fuzzy = FuzzyEngine::new().evaluate(&FuzzyInputs::default());
        let consensus = make_neutral_consensus();
        let state = make_default_state();
        let config = default_config();
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, 0.0, 1_000_000.0);
        let decision = engine.decide(&fuzzy, &consensus, Some(&snap), &state, &config);
        assert_eq!(
            decision.action,
            TradeAction::Hold,
            "neutral signals should hold, blended_signal={:.4}",
            decision.blended_signal
        );
    }

    #[test]
    fn decision_confidence_gate_blocks_low_confidence_trades() {
        let engine = DecisionEngine::new(0.5);
        // Construct very low confidence inputs.
        use crate::trading::fuzzy::FuzzyDecision;
        let fuzzy = FuzzyDecision {
            signal: 0.8,     // strong directional signal
            confidence: 0.1, // but very low confidence
            label: "buy".to_string(),
            term_activations: Default::default(),
        };
        let consensus = AiConsensus {
            action: "buy".to_string(),
            confidence: 0.1,
            signal: 0.8,
            vote_distribution: json!({}),
            advices: vec![],
            responders: 1,
            failures: 0,
        };
        let state = make_default_state();
        let config = TradingConfig {
            enabled: true,
            octobot_base_url: "http://x".to_string(),
            fuzzy_confidence_threshold: 0.65, // threshold >> 0.1
            ..TradingConfig::default()
        };
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, 5.0, 1_000_000.0);
        let decision = engine.decide(&fuzzy, &consensus, Some(&snap), &state, &config);
        assert_eq!(
            decision.action,
            TradeAction::Hold,
            "low confidence should be blocked by confidence gate"
        );
        assert!(
            decision.rationale.contains("Confidence"),
            "rationale should mention confidence gate"
        );
    }

    #[test]
    fn decision_position_gate_blocks_new_buys_when_full() {
        let engine = DecisionEngine::new(0.4);
        let fuzzy = FuzzyEngine::new().evaluate(&bullish_inputs());
        let consensus = make_strong_bullish_consensus();
        let mut state = make_default_state();
        // Fill to max open positions.
        let config = TradingConfig {
            enabled: true,
            octobot_base_url: "http://x".to_string(),
            max_open_positions: 2,
            ..TradingConfig::default()
        };
        state.open_positions = vec![make_open_order("o1"), make_open_order("o2")];
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, 5.0, 2_000_000.0);
        let decision = engine.decide(&fuzzy, &consensus, Some(&snap), &state, &config);
        assert_eq!(
            decision.action,
            TradeAction::Hold,
            "position gate should block buy when at max open positions"
        );
        assert!(
            decision.rationale.to_lowercase().contains("position")
                || decision.rationale.contains("Confidence"),
            "rationale should mention position gate or confidence gate"
        );
    }

    #[test]
    fn decision_cooldown_gate_blocks_trade_within_interval() {
        let engine = DecisionEngine::new(0.4);
        let fuzzy = FuzzyEngine::new().evaluate(&bullish_inputs());
        let consensus = make_strong_bullish_consensus();
        let mut state = make_default_state();
        let config = TradingConfig {
            enabled: true,
            octobot_base_url: "http://x".to_string(),
            min_trade_interval_seconds: 300,
            ..TradingConfig::default()
        };
        // Set last trade to "just now".
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        state.last_trade_at = Some(now - 10.0); // 10 seconds ago, cooldown is 300s
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, 5.0, 2_000_000.0);
        let decision = engine.decide(&fuzzy, &consensus, Some(&snap), &state, &config);
        assert_eq!(
            decision.action,
            TradeAction::Hold,
            "cooldown gate should block trade within cooldown interval"
        );
        assert!(
            decision.rationale.to_lowercase().contains("cooldown")
                || decision.rationale.contains("Confidence"),
            "rationale should mention cooldown"
        );
    }

    #[test]
    fn decision_operator_override_bypasses_pipeline() {
        let engine = DecisionEngine::new(0.4);
        let fuzzy = FuzzyEngine::new().evaluate(&FuzzyInputs::default()); // neutral
        let consensus = make_neutral_consensus();
        let mut state = make_default_state();
        let config = default_config();
        // Inject a buy override.
        state.pending_override = Some(TradeOverride {
            action: TradeAction::Buy,
            exchange: Some("kraken".to_string()),
            symbol: Some("ETH/USDT".to_string()),
            amount_usd: Some(15.0),
            reason: Some("manual signal".to_string()),
            issued_at: 0.0,
            issued_by: "pbisaacs".to_string(),
        });
        let decision = engine.decide(&fuzzy, &consensus, None, &state, &config);
        assert_eq!(
            decision.action,
            TradeAction::Buy,
            "override action should take effect"
        );
        assert_eq!(decision.exchange, "kraken");
        assert_eq!(decision.symbol, "ETH/USDT");
        assert_eq!(decision.amount_usd, 15.0);
        assert!(
            decision.override_applied,
            "override_applied flag must be set"
        );
        assert!(
            decision.rationale.contains("pbisaacs"),
            "rationale should identify who issued the override"
        );
    }

    #[test]
    fn decision_override_amount_clamped_to_config_bounds() {
        let engine = DecisionEngine::new(0.4);
        let fuzzy = FuzzyEngine::new().evaluate(&FuzzyInputs::default());
        let consensus = make_neutral_consensus();
        let mut state = make_default_state();
        let config = TradingConfig {
            enabled: true,
            octobot_base_url: "http://x".to_string(),
            micro_trade_min_usd: 2.0,
            micro_trade_max_usd: 20.0,
            ..TradingConfig::default()
        };
        state.pending_override = Some(TradeOverride {
            action: TradeAction::Sell,
            exchange: Some("binance".to_string()),
            symbol: Some("BTC/USDT".to_string()),
            amount_usd: Some(999.0), // way above max
            reason: None,
            issued_at: 0.0,
            issued_by: "pbisaacs".to_string(),
        });
        let decision = engine.decide(&fuzzy, &consensus, None, &state, &config);
        assert!(
            decision.amount_usd <= 20.0,
            "override amount should be clamped to micro_trade_max_usd"
        );
        assert!(
            decision.amount_usd >= 2.0,
            "override amount should be at least micro_trade_min_usd"
        );
    }

    #[test]
    fn decision_trade_sizing_scales_with_signal_and_confidence() {
        let engine = DecisionEngine::new(0.4);
        let config = default_config(); // min=1.0, max=25.0
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, 5.0, 2_000_000.0);

        // High signal + high confidence → large trade.
        let high = engine.decide(
            &FuzzyEngine::new().evaluate(&bullish_inputs()),
            &make_strong_bullish_consensus(),
            Some(&snap),
            &make_default_state(),
            &config,
        );

        // Lower signal (but still buys) → smaller trade.
        let low_signal_consensus = consensus_from_advices(vec![make_advice("buy", 0.7, 1.0)], 0);
        let moderate_inputs = FuzzyInputs {
            price_trend: 0.3,
            volume_ratio: 1.0,
            ai_consensus: 0.3,
            research_sentiment: 0.1,
            portfolio_exposure: 0.3,
        };
        let low = engine.decide(
            &FuzzyEngine::new().evaluate(&moderate_inputs),
            &low_signal_consensus,
            Some(&snap),
            &make_default_state(),
            &config,
        );

        if matches!(high.action, TradeAction::Buy | TradeAction::StrongBuy)
            && matches!(low.action, TradeAction::Buy | TradeAction::StrongBuy)
        {
            assert!(
                high.amount_usd >= low.amount_usd,
                "stronger signal should produce larger or equal trade size; \
                 high={:.2} low={:.2}",
                high.amount_usd,
                low.amount_usd
            );
        }
        // In any case, trade size must stay within configured bounds.
        if high.amount_usd > 0.0 {
            assert!(high.amount_usd >= config.micro_trade_min_usd);
            assert!(high.amount_usd <= config.micro_trade_max_usd);
        }
    }

    #[test]
    fn decision_no_market_snapshot_produces_hold() {
        let engine = DecisionEngine::new(0.4);
        let fuzzy = FuzzyEngine::new().evaluate(&bullish_inputs());
        let consensus = make_strong_bullish_consensus();
        let state = make_default_state();
        let config = default_config();
        let decision = engine.decide(&fuzzy, &consensus, None, &state, &config);
        assert_eq!(
            decision.action,
            TradeAction::Hold,
            "missing market snapshot should produce hold"
        );
    }

    #[test]
    fn decision_blended_signal_correct_weights() {
        // With fuzzy_weight=0.4: blended = fuzzy*0.4 + ai*0.6
        let engine = DecisionEngine::new(0.4);
        let config = TradingConfig {
            enabled: true,
            octobot_base_url: "http://x".to_string(),
            fuzzy_confidence_threshold: 0.0, // disable gate
            min_trade_interval_seconds: 0,
            ..TradingConfig::default()
        };
        use crate::trading::fuzzy::{FuzzyDecision, FuzzyTermActivations};
        let fuzzy = FuzzyDecision {
            signal: 1.0,
            confidence: 1.0,
            label: "strong_buy".to_string(),
            term_activations: FuzzyTermActivations::default(),
        };
        let consensus = AiConsensus {
            action: "strong_sell".to_string(),
            confidence: 1.0,
            signal: -1.0,
            vote_distribution: json!({}),
            advices: vec![],
            responders: 1,
            failures: 0,
        };
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, 0.0, 1_000_000.0);
        let decision = engine.decide(
            &fuzzy,
            &consensus,
            Some(&snap),
            &make_default_state(),
            &config,
        );
        // Expected: 1.0 * 0.4 + (-1.0) * 0.6 = -0.2 → sell threshold
        let expected = 1.0 * 0.4 + -0.6;
        assert!(
            (decision.blended_signal - expected).abs() < 0.01,
            "blended_signal should be {expected:.3} got {:.3}",
            decision.blended_signal
        );
    }

    #[test]
    fn decision_config_override_affects_threshold() {
        // If runtime config override raises the threshold above the signal,
        // the decision should be hold even though the base config would allow a trade.
        let engine = DecisionEngine::new(0.4);
        let fuzzy_out = FuzzyEngine::new().evaluate(&FuzzyInputs {
            price_trend: 0.5,
            volume_ratio: 1.2,
            ai_consensus: 0.5,
            research_sentiment: 0.2,
            portfolio_exposure: 0.2,
        });
        let consensus = consensus_from_advices(vec![make_advice("buy", 0.7, 1.0)], 0);
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, 3.0, 1_500_000.0);
        let mut state = make_default_state();
        // Set runtime override to require very high confidence.
        state.config_overrides = Some(TradingConfigOverride {
            fuzzy_confidence_threshold: Some(0.99),
            ..TradingConfigOverride::default()
        });
        let config = default_config();
        let decision = engine.decide(&fuzzy_out, &consensus, Some(&snap), &state, &config);
        assert_eq!(
            decision.action,
            TradeAction::Hold,
            "runtime config override should raise threshold and block trade"
        );
    }

    // -----------------------------------------------------------------------
    // End-to-end pipeline integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn pipeline_full_bullish_scenario_produces_buy_decision() {
        let fuzzy_engine = FuzzyEngine::new();
        let decision_engine = DecisionEngine::new(0.4);

        let inputs = FuzzyInputs {
            price_trend: 0.75,
            volume_ratio: 1.7,
            ai_consensus: 0.85,
            research_sentiment: 0.6,
            portfolio_exposure: 0.15,
        };
        let fuzzy = fuzzy_engine.evaluate(&inputs);
        let consensus = make_strong_bullish_consensus();
        let snap = make_snapshot("binance", "BTC/USDT", 65000.0, 4.5, 3_000_000.0);
        let state = make_default_state();
        let config = default_config();
        let decision = decision_engine.decide(&fuzzy, &consensus, Some(&snap), &state, &config);

        // The pipeline should recommend buying.
        assert!(
            matches!(decision.action, TradeAction::Buy | TradeAction::StrongBuy),
            "full bullish scenario should produce buy/strong_buy, got {:?} \
             (fuzzy.signal={:.4}, ai.signal={:.4}, blended={:.4}, confidence={:.4})",
            decision.action,
            fuzzy.signal,
            consensus.signal,
            decision.blended_signal,
            decision.confidence
        );
        assert!(decision.amount_usd >= config.micro_trade_min_usd);
        assert!(decision.amount_usd <= config.micro_trade_max_usd);
        assert!(!decision.exchange.is_empty());
        assert!(!decision.symbol.is_empty());
    }

    #[test]
    fn pipeline_full_bearish_scenario_produces_sell_decision() {
        let fuzzy_engine = FuzzyEngine::new();
        let decision_engine = DecisionEngine::new(0.4);

        let inputs = FuzzyInputs {
            price_trend: -0.75,
            volume_ratio: 1.7,
            ai_consensus: -0.85,
            research_sentiment: -0.6,
            portfolio_exposure: 0.85,
        };
        let fuzzy = fuzzy_engine.evaluate(&inputs);
        let consensus = make_strong_bearish_consensus();
        let snap = make_snapshot("binance", "BTC/USDT", 65000.0, -4.5, 3_000_000.0);
        let state = make_default_state();
        let config = default_config();
        let decision = decision_engine.decide(&fuzzy, &consensus, Some(&snap), &state, &config);

        assert!(
            matches!(decision.action, TradeAction::Sell | TradeAction::StrongSell),
            "full bearish scenario should produce sell/strong_sell, got {:?} \
             (fuzzy.signal={:.4}, ai.signal={:.4}, blended={:.4})",
            decision.action,
            fuzzy.signal,
            consensus.signal,
            decision.blended_signal
        );
    }

    #[test]
    fn pipeline_conflicting_fuzzy_ai_signals_produce_moderate_decision() {
        let fuzzy_engine = FuzzyEngine::new();
        let decision_engine = DecisionEngine::new(0.5); // equal weights
        // Fuzzy says sell (bearish price/sentiment), AI says buy.
        let bearish_for_fuzzy = FuzzyInputs {
            price_trend: -0.6,
            volume_ratio: 0.3,
            ai_consensus: 0.8, // AI is bullish
            research_sentiment: -0.4,
            portfolio_exposure: 0.8,
        };
        let fuzzy = fuzzy_engine.evaluate(&bearish_for_fuzzy);
        let bullish_ai = make_strong_bullish_consensus();
        let snap = make_snapshot("binance", "ETH/USDT", 3000.0, -2.0, 1_000_000.0);
        let config = default_config();
        let decision = decision_engine.decide(
            &fuzzy,
            &bullish_ai,
            Some(&snap),
            &make_default_state(),
            &config,
        );
        // With conflicting signals the blended signal should be moderate; it may hold.
        assert!(
            decision.blended_signal.abs() <= 0.8,
            "conflicting signals should produce moderate blended signal: {:.4}",
            decision.blended_signal
        );
    }

    #[test]
    fn pipeline_rationale_is_non_empty() {
        let fuzzy_engine = FuzzyEngine::new();
        let decision_engine = DecisionEngine::new(0.4);
        let fuzzy = fuzzy_engine.evaluate(&bullish_inputs());
        let consensus = make_strong_bullish_consensus();
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, 5.0, 2_000_000.0);
        let decision = decision_engine.decide(
            &fuzzy,
            &consensus,
            Some(&snap),
            &make_default_state(),
            &default_config(),
        );
        assert!(
            !decision.rationale.is_empty(),
            "rationale must always be non-empty"
        );
    }

    #[test]
    fn pipeline_all_signals_recorded_in_decision() {
        let fuzzy_engine = FuzzyEngine::new();
        let decision_engine = DecisionEngine::new(0.4);
        let fuzzy = fuzzy_engine.evaluate(&bullish_inputs());
        let consensus = make_strong_bullish_consensus();
        let snap = make_snapshot("binance", "BTC/USDT", 50000.0, 5.0, 2_000_000.0);
        let decision = decision_engine.decide(
            &fuzzy,
            &consensus,
            Some(&snap),
            &make_default_state(),
            &default_config(),
        );
        // Verify signal components are propagated into decision for auditability.
        assert!(
            (decision.fuzzy_signal - fuzzy.signal).abs() < 0.001,
            "fuzzy_signal must be recorded: expected {:.4} got {:.4}",
            fuzzy.signal,
            decision.fuzzy_signal
        );
        assert!(
            (decision.ai_signal - consensus.signal).abs() < 0.001,
            "ai_signal must be recorded: expected {:.4} got {:.4}",
            consensus.signal,
            decision.ai_signal
        );
        assert!((decision.fuzzy_confidence - fuzzy.confidence).abs() < 0.001);
        assert!((decision.ai_confidence - consensus.confidence).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // Market candidate selection tests
    // -----------------------------------------------------------------------

    #[test]
    fn market_score_prefers_high_momentum_and_volume() {
        // Replicate the market_score logic used in mod.rs
        fn market_score(snap: &MarketSnapshot) -> f64 {
            let change_abs = snap.price_change_pct_24h.unwrap_or(0.0).abs();
            let vol = snap.volume_24h.unwrap_or(0.0);
            change_abs * (vol + 1.0).ln()
        }

        let high_momentum = make_snapshot("binance", "BTC/USDT", 50000.0, 8.0, 5_000_000.0);
        let low_momentum = make_snapshot("binance", "ETH/USDT", 3000.0, 0.5, 500_000.0);
        assert!(
            market_score(&high_momentum) > market_score(&low_momentum),
            "high momentum/volume market should score higher"
        );
    }

    #[test]
    fn market_score_handles_missing_fields() {
        fn market_score(snap: &MarketSnapshot) -> f64 {
            let change_abs = snap.price_change_pct_24h.unwrap_or(0.0).abs();
            let vol = snap.volume_24h.unwrap_or(0.0);
            change_abs * (vol + 1.0).ln()
        }
        let no_data = MarketSnapshot {
            exchange: "x".to_string(),
            symbol: "A/B".to_string(),
            price: 1.0,
            price_change_pct_1h: None,
            price_change_pct_24h: None,
            volume_24h: None,
            volume_change_pct: None,
            high_24h: None,
            low_24h: None,
            fetched_at: 0.0,
        };
        let score = market_score(&no_data);
        assert!(
            score >= 0.0,
            "score with missing data should be non-negative: {score}"
        );
    }

    #[test]
    fn research_query_date_uses_real_utc_calendar_date() {
        assert_eq!(crate::trading::utc_date_from_unix_days(0), "1970-01-01");
        assert_eq!(
            crate::trading::utc_date_from_unix_days(20_578),
            "2026-05-05"
        );
    }

    // -----------------------------------------------------------------------
    // TradeAction display tests
    // -----------------------------------------------------------------------

    #[test]
    fn trade_action_display_strings() {
        assert_eq!(TradeAction::Buy.to_string(), "buy");
        assert_eq!(TradeAction::Sell.to_string(), "sell");
        assert_eq!(TradeAction::Hold.to_string(), "hold");
        assert_eq!(TradeAction::StrongBuy.to_string(), "strong_buy");
        assert_eq!(TradeAction::StrongSell.to_string(), "strong_sell");
        assert_eq!(TradeAction::Cancel.to_string(), "cancel");
    }

    #[test]
    fn trade_action_serialisation_roundtrip() {
        let actions = vec![
            TradeAction::Buy,
            TradeAction::Sell,
            TradeAction::Hold,
            TradeAction::StrongBuy,
            TradeAction::StrongSell,
            TradeAction::Cancel,
        ];
        for action in &actions {
            let json = serde_json::to_string(action).unwrap();
            let restored: TradeAction = serde_json::from_str(&json).unwrap();
            assert_eq!(action, &restored, "roundtrip failed for {action}");
        }
    }

    // -----------------------------------------------------------------------
    // OctobotPortfolio tests
    // -----------------------------------------------------------------------

    #[test]
    fn portfolio_stablecoin_filtering() {
        // Replicate is_stablecoin logic from mod.rs
        fn is_stablecoin(sym: &str) -> bool {
            let lower = sym.to_ascii_lowercase();
            lower.contains("usdt")
                || lower.contains("usdc")
                || lower.contains("busd")
                || lower.contains("dai")
                || lower.contains("usd")
                || lower.contains("eur")
        }
        assert!(is_stablecoin("USDT"));
        assert!(is_stablecoin("USDC"));
        assert!(is_stablecoin("DAI"));
        assert!(is_stablecoin("BUSD"));
        assert!(!is_stablecoin("BTC"));
        assert!(!is_stablecoin("ETH"));
        assert!(!is_stablecoin("SOL"));
    }

    #[test]
    fn portfolio_exposure_calculation() {
        // Replicate the exposure formula from compute_fuzzy_inputs in mod.rs.
        let portfolio = make_portfolio(100.0);
        let total = portfolio.total_value_usd.unwrap_or(0.0);
        let fn_is_stable = |sym: &str| {
            let lower = sym.to_ascii_lowercase();
            lower.contains("usdt") || lower.contains("usd")
        };
        let stable: f64 = portfolio
            .currencies
            .iter()
            .filter(|(sym, _)| fn_is_stable(sym))
            .map(|(_, b)| b.value_usd.unwrap_or(0.0))
            .sum();
        let exposure = ((total - stable) / total).clamp(0.0, 1.0);
        // Portfolio has 80% BTC (non-stable) + 20% USDT (stable).
        assert!(
            (exposure - 0.8).abs() < 0.01,
            "portfolio exposure should be ~0.8, got {exposure:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Fuzzy input normalisation boundary tests
    // -----------------------------------------------------------------------

    #[test]
    fn fuzzy_inputs_at_extreme_boundaries_do_not_panic() {
        let engine = FuzzyEngine::new();
        let extremes = [
            FuzzyInputs {
                price_trend: -1.0,
                volume_ratio: 0.0,
                ai_consensus: -1.0,
                research_sentiment: -1.0,
                portfolio_exposure: 0.0,
            },
            FuzzyInputs {
                price_trend: 1.0,
                volume_ratio: 2.0,
                ai_consensus: 1.0,
                research_sentiment: 1.0,
                portfolio_exposure: 1.0,
            },
            FuzzyInputs {
                price_trend: 0.0,
                volume_ratio: 1.0,
                ai_consensus: 0.0,
                research_sentiment: 0.0,
                portfolio_exposure: 0.5,
            },
        ];
        for inp in &extremes {
            let out = engine.evaluate(inp);
            assert!(out.signal >= -1.0 && out.signal <= 1.0);
            assert!(out.confidence >= 0.0 && out.confidence <= 1.0);
        }
    }

    // =======================================================================
    // Backtesting integration tests (using wiremock to simulate OctoBot)
    // =======================================================================

    use crate::trading::backtest::{ApproachAssessment, BacktestEngine, BacktestSummary};
    use crate::trading::octobot::{BacktestStartRequest, OctobotClient};
    use std::time::Duration;
    use tempfile::tempdir;
    use wiremock::matchers::{body_string_contains, method, path, path_regex, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn octobot_client_login_probes_ping_without_legacy_account_login() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/ping"))
            .respond_with(ResponseTemplate::new(200).set_body_json("Running since 2026-05-05."))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/accounts/login"))
            .respond_with(ResponseTemplate::new(500).set_body_string("legacy endpoint used"))
            .expect(0)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), Some("shared-auth-password"), 10.0);
        let result = client.login().await;
        assert!(result.is_ok(), "login should probe /api/ping: {result:?}");
        server.verify().await;
    }

    #[tokio::test]
    async fn octobot_client_get_open_orders_parses_current_octobot_shape() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/orders"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "orders": [{
                    "id": "order-1",
                    "exchange": "binance",
                    "symbol": "BTC/USDT",
                    "type": "BUY LIMIT",
                    "amount": 0.01,
                    "price": 65000.0,
                    "status": "Real",
                    "time": 1_777_777_777.0
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let orders = client.get_open_orders().await.unwrap();
        assert_eq!(orders.len(), 1);
        assert_eq!(orders[0].id, "order-1");
        assert_eq!(orders[0].side, "buy");
        assert_eq!(orders[0].order_type, "BUY LIMIT");
        assert_eq!(orders[0].timestamp, Some(1_777_777_777.0));
        server.verify().await;
    }

    #[tokio::test]
    async fn octobot_client_place_buy_order_uses_create_order_endpoint() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/orders"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/trades"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/orders"))
            .and(query_param("action", "create_order"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "created-order-123",
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": 5.0,
                "price": 65000.0,
                "status": "submitted"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let result = client
            .place_buy_order("binance", "BTC/USDT", 5.0)
            .await
            .expect("place buy order");
        assert_eq!(result.order_id, "created-order-123");
        assert_eq!(result.symbol, "BTC/USDT");
        assert_eq!(result.side, "buy");
        assert_eq!(result.amount, 5.0);
        assert_eq!(result.price, Some(65000.0));
        server.verify().await;
    }

    #[tokio::test]
    async fn octobot_client_place_order_reports_attempts_when_all_paths_fail() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/orders"))
            .respond_with(ResponseTemplate::new(500).set_body_string("not available"))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/trades"))
            .respond_with(ResponseTemplate::new(500).set_body_string("not available"))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/orders"))
            .respond_with(ResponseTemplate::new(404).set_body_string("unsupported"))
            .expect(4)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/user_command"))
            .respond_with(ResponseTemplate::new(404).set_body_string("unsupported"))
            .expect(3)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let err = client
            .place_sell_order("binance", "ETH/USDT", 4.0)
            .await
            .expect_err("order placement should fail");
        assert!(err.contains("OctoBot order placement failed"));
        assert!(err.contains("/api/orders"));
        assert!(err.contains("/api/user_command"));
        server.verify().await;
    }

    #[tokio::test]
    async fn octobot_client_get_exchange_info_uses_first_exchange_and_configured_symbols() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/exchanges"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/first_exchange_details"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "exchange_name": "binance",
                "exchange_id": "binance-id"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/get_config_currency"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "Bitcoin": {
                    "enabled": true,
                    "pairs": ["BTC/USDT", "BTC/USDC"]
                },
                "Ethereum": {
                    "enabled": false,
                    "pairs": ["ETH/USDT"]
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let exchanges = client.get_exchange_info().await.unwrap();
        assert_eq!(exchanges.len(), 1);
        assert_eq!(exchanges[0].name, "binance");
        assert_eq!(
            exchanges[0].symbols,
            vec!["BTC/USDC".to_string(), "BTC/USDT".to_string()]
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn octobot_client_get_market_snapshot_uses_dashboard_graph_fallback() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/market/ticker"))
            .and(query_param("exchange", "binance"))
            .and(query_param("symbol", "BTC/USDT"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/dashboard/watched_symbol/BTC(%7C|\|)USDT$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "exchange_id": "binance-id",
                "time_frame": "1h"
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(
                r"^/dashboard/currency_price_graph_update/binance-id/BTC(%7C|\|)USDT/1h/live$",
            ))
            .and(query_param("display_orders", "false"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "candles": {
                    "close": [100.0, 110.0],
                    "high": [101.0, 112.0],
                    "low": [99.0, 105.0],
                    "volume": [10.0, 15.0]
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let snapshot = client
            .get_market_snapshot("binance", "BTC/USDT")
            .await
            .unwrap();
        assert_eq!(snapshot.exchange, "binance");
        assert_eq!(snapshot.symbol, "BTC/USDT");
        assert_eq!(snapshot.price, 110.0);
        assert_eq!(snapshot.price_change_pct_24h, Some(10.0));
        assert_eq!(snapshot.volume_24h, Some(25.0));
        assert_eq!(snapshot.high_24h, Some(112.0));
        assert_eq!(snapshot.low_24h, Some(99.0));
        server.verify().await;
    }

    #[tokio::test]
    async fn octobot_client_schema_skips_known_missing_optional_ticker_route() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/market/ticker"))
            .and(query_param("exchange", "binance"))
            .and(query_param("symbol", "BTC/USDT"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/dashboard/watched_symbol/BTC(%7C|\|)USDT$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "exchange_id": "binance-id",
                "time_frame": "1h"
            })))
            .expect(2)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(
                r"^/dashboard/currency_price_graph_update/binance-id/BTC(%7C|\|)USDT/1h/live$",
            ))
            .and(query_param("display_orders", "false"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "candles": {
                    "close": [100.0, 110.0],
                    "high": [101.0, 112.0],
                    "low": [99.0, 105.0],
                    "volume": [10.0, 15.0]
                }
            })))
            .expect(2)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        client
            .get_market_snapshot("binance", "BTC/USDT")
            .await
            .unwrap();
        client
            .get_market_snapshot("binance", "BTC/USDT")
            .await
            .unwrap();
        let schema = client.api_schema_snapshot().await;
        assert!(
            schema
                .endpoints
                .get("GET /api/market/ticker")
                .is_some_and(|endpoint| endpoint.degraded)
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn octobot_client_logs_update_schema_hints() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/logs"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/logs"))
            .and(query_param("format", "json"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "Time\tLevel\tSource\tMessage\n\
                 2026-05-06 06:06:00\tWARNING\tAIToolsTeamManagerAgentProducer\t3 validation errors for ManagerToolCall tool_name Field required arguments Field required agent_name Extra inputs\n\
                 2026-05-06 06:04:10\tWARNING\tAIIndexTradingModeConsumer\tMissingMinimalExchangeTradeVolume {'cost': {'min': 5.0, 'max': 9000000.0}}\n",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let logs = client.get_recent_logs(10).await.unwrap();
        assert_eq!(logs.len(), 2);
        let schema = client.api_schema_snapshot().await;
        assert!(
            schema
                .semantic_hints
                .contains_key("manager_tool_call_shape")
        );
        assert_eq!(schema.numeric_hints.get("micro_trade_min_usd"), Some(&5.0));
        server.verify().await;
    }

    // -----------------------------------------------------------------------
    // Helper: build an OctoBot-format backtest report response body.
    // -----------------------------------------------------------------------
    fn octobot_report_body(profit_pct: f64, market_pct: f64, trades: usize) -> serde_json::Value {
        serde_json::json!({
            "report": {
                "bot_report": {
                    "profitability": { "binance": profit_pct },
                    "market_average_profitability": { "binance": market_pct },
                    "reference_market": "USDT",
                    "trading_mode": "GailStrategyMode",
                    "starting_portfolio": { "binance": { "USDT": 1000.0 } },
                    "end_portfolio": {
                        "binance": {
                            "USDT": 1000.0 * (1.0 + profit_pct / 100.0)
                        }
                    }
                },
                "symbol_report": [],
                "chart_identifiers": [
                    {
                        "symbol": "BTC/USDT",
                        "exchange_id": "binance",
                        "exchange_name": "binance",
                        "time_frames": ["1h"]
                    }
                ],
                "errors_count": 0
            },
            "trades": serde_json::Value::Array(
                (0..trades).map(|i| serde_json::json!({
                    "id": i.to_string(),
                    "symbol": "BTC/USDT",
                    "side": if i % 2 == 0 { "buy" } else { "sell" },
                    "amount": 0.001
                })).collect()
            )
        })
    }

    // -----------------------------------------------------------------------
    // OctoBot client: start_backtest sends correct POST
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_client_start_request_format() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/backtesting"))
            .and(query_param("action_type", "start_backtesting"))
            .respond_with(ResponseTemplate::new(200).set_body_string("\"Backtesting started\""))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let req = BacktestStartRequest {
            files: vec!["user/backtesting/collector/binance_BTC_USDT_1h.data".to_string()],
            start_timestamp: Some(1_700_000_000_000),
            end_timestamp: Some(1_702_678_400_000),
            enable_logs: false,
        };
        let result = client.start_backtest(&req).await;
        assert!(
            result.is_ok(),
            "start_backtest should succeed: {:?}",
            result
        );
        server.verify().await;
    }

    // -----------------------------------------------------------------------
    // OctoBot client: get_backtest_report parses a full report
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_client_get_report_parses_profitability() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_report"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(octobot_report_body(12.34, 5.0, 8)),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let report = client.get_backtest_report().await;
        assert!(
            report.is_ok(),
            "get_backtest_report should succeed: {:?}",
            report
        );
        let report = report.unwrap();
        assert!(report.is_some(), "report should be present (non-empty)");
        let report = report.unwrap();
        assert!(
            (report.best_profitability().unwrap() - 12.34).abs() < 0.01,
            "profitability should be 12.34, got {:?}",
            report.best_profitability()
        );
        assert_eq!(report.total_trades, 8);
        assert!(report.symbols.contains(&"BTC/USDT".to_string()));
        server.verify().await;
    }

    // -----------------------------------------------------------------------
    // OctoBot client: get_backtest_report returns None on empty body
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_client_empty_report_returns_none() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_report"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let result = client.get_backtest_report().await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none(), "empty object should return None");
        server.verify().await;
    }

    // -----------------------------------------------------------------------
    // OctoBot client: get_backtest_run_id
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_client_get_run_id() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting_run_id"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "backtesting_id": 42 })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let run_id = client.get_backtest_run_id().await;
        assert_eq!(run_id.unwrap(), Some(42));
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_client_get_run_id_accepts_raw_numeric_body() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting_run_id"))
            .respond_with(ResponseTemplate::new(200).set_body_json(42))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let run_id = client.get_backtest_run_id().await;
        assert_eq!(run_id.unwrap(), Some(42));
        server.verify().await;
    }

    // -----------------------------------------------------------------------
    // OctoBot client: list_backtest_data_files
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_client_list_data_files() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_data_files"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                "user/backtesting/collector/binance_BTC_USDT_1h.data",
                "user/backtesting/collector/binance_ETH_USDT_1h.data"
            ])))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let files = client.list_backtest_data_files().await.unwrap();
        assert_eq!(files.len(), 2);
        assert!(files[0].contains("BTC_USDT"));
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_client_list_data_files_accepts_wrapped_json() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_data_files"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data_files": [
                    "user/backtesting/collector/binance_BTC_USDT_1h.data",
                    "user/backtesting/collector/binance_ETH_USDT_1h.data"
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let files = client.list_backtest_data_files().await.unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|file| file.contains("BTC_USDT")));
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_client_list_data_files_parses_html_backtesting_page() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_data_files"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string(
                        r#"<input type="checkbox" class="dataFileCheckbox" data-file="user/backtesting/collector/binance_BTC_USDT_1h.data">"#,
                    ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let files = client.list_backtest_data_files().await.unwrap();
        assert_eq!(
            files,
            vec!["user/backtesting/collector/binance_BTC_USDT_1h.data"]
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_client_list_data_files_accepts_plain_collector_filenames() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_data_files"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string(
                        r#"<input type="checkbox" class="dataFileCheckbox" data-file="ExchangeHistoryDataCollector_1779190560.6877387.data">"#,
                    ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let files = client.list_backtest_data_files().await.unwrap();
        assert_eq!(
            files,
            vec!["ExchangeHistoryDataCollector_1779190560.6877387.data"]
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_client_list_data_files_ignores_non_backtesting_data_assets() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_data_files"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string(
                        r#"
                        <script src="https://cdn.datatables.net/2.0.8/css/dataTables.data"></script>
                        <td>user/backtesting/collector/binance_BTC_USDT_1h.data</td>
                        "#,
                    ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let files = client.list_backtest_data_files().await.unwrap();
        assert_eq!(
            files,
            vec!["user/backtesting/collector/binance_BTC_USDT_1h.data"]
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_client_list_data_files_ignores_partial_dot_data_tokens_from_cdn_hosts() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_data_files"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string(
                        r#"
                        <link rel="stylesheet" href="https://cdn.datatables.net/2.0.8/css/dataTables.dataTables.min.css">
                        <td>ExchangeHistoryDataCollector_1779194074.4955919.data</td>
                        "#,
                    ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let files = client.list_backtest_data_files().await.unwrap();
        assert_eq!(
            files,
            vec!["ExchangeHistoryDataCollector_1779194074.4955919.data"]
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_client_list_data_files_prefers_data_collector_page_when_available() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_data_files"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string("<html><body>No explicit list endpoint</body></html>"),
            )
            .expect(0)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/data_collector"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string(
                        r#"<td>user/backtesting/collector/binance_ETH_USDT_1h.data</td>"#,
                    ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let files = client.list_backtest_data_files().await.unwrap();
        assert_eq!(
            files,
            vec!["user/backtesting/collector/binance_ETH_USDT_1h.data"]
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_client_start_data_collector_request_format() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/data_collector"))
            .and(query_param("action_type", "start_collector"))
            .and(body_string_contains("\"exchange\":\"binance\""))
            .and(body_string_contains("\"symbols\":[\"BTC/USDT\"]"))
            .and(body_string_contains("\"time_frames\":[\"1h\"]"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json("Historical data collection started."),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let started = client
            .start_data_collector(
                "binance",
                &["BTC/USDT".to_string()],
                &["1h".to_string()],
                Some(1_700_000_000_000),
                Some(1_700_086_400_000),
            )
            .await;
        assert!(
            started.is_ok(),
            "start_data_collector should succeed: {started:?}"
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_engine_run_with_config_uses_cached_catalog_when_discovery_fails() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_data_files"))
            .respond_with(ResponseTemplate::new(500).set_body_string("not available"))
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/backtesting"))
            .and(query_param("action_type", "start_backtesting"))
            .and(body_string_contains("binance_BTC_USDT_1h.data"))
            .respond_with(ResponseTemplate::new(200).set_body_string("\"Backtesting started\""))
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting_run_id"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "backtesting_id": 77 })),
            )
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_report"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(octobot_report_body(6.0, 2.0, 9)),
            )
            .mount(&server)
            .await;

        let temp = tempdir().expect("tempdir");
        let catalog_path = temp.path().join("backtest_data_catalog.json");
        std::fs::write(
            &catalog_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "files": ["user/backtesting/collector/binance_BTC_USDT_1h.data"],
                "updated_at": 1_700_000_000.0
            }))
            .expect("catalog json"),
        )
        .expect("write catalog");

        let mut config = TradingConfig {
            data_path: temp
                .path()
                .join("trading_state.json")
                .to_string_lossy()
                .to_string(),
            backtest_data_catalog_path: catalog_path.to_string_lossy().to_string(),
            backtest_symbols: vec!["BTC/USDT".to_string()],
            backtest_data_collection_enabled: false,
            ..TradingConfig::default()
        };
        config.normalize();

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let engine = BacktestEngine::with_poll_params(client, 0.0, Duration::from_millis(1), 5);
        let summary = engine.run_with_config(&config).await;

        assert_eq!(summary.assessment, ApproachAssessment::Viable);
        assert_eq!(summary.run_id, Some(77));
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_engine_run_with_config_starts_collector_when_no_files_available() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_data_files"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string("<html><body>No data files yet</body></html>"),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/data_collector"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string("<html><body>collector page</body></html>"),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/data_collector"))
            .and(query_param("action_type", "start_collector"))
            .and(body_string_contains("\"symbols\":[\"BTC/USDT\"]"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json("Historical data collection started."),
            )
            .expect(1)
            .mount(&server)
            .await;

        let temp = tempdir().expect("tempdir");
        let catalog_path = temp.path().join("backtest_data_catalog.json");
        let mut config = TradingConfig {
            data_path: temp
                .path()
                .join("trading_state.json")
                .to_string_lossy()
                .to_string(),
            backtest_data_catalog_path: catalog_path.to_string_lossy().to_string(),
            backtest_symbols: vec!["BTC/USDT".to_string()],
            backtest_data_collection_enabled: true,
            backtest_data_collection_exchange: "binance".to_string(),
            backtest_data_collection_time_frames: vec!["1h".to_string()],
            backtest_data_collection_cooldown_seconds: 300,
            ..TradingConfig::default()
        };
        config.normalize();

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let engine = BacktestEngine::with_poll_params(client, 0.0, Duration::from_millis(1), 5);
        let summary = engine.run_with_config(&config).await;

        assert_eq!(summary.assessment, ApproachAssessment::Incomplete);
        assert!(
            summary
                .notes
                .contains("started OctoBot historical data collection"),
            "summary notes should mention collector start: {}",
            summary.notes
        );
        let persisted_catalog = std::fs::read_to_string(catalog_path).expect("catalog persisted");
        assert!(
            persisted_catalog.contains("last_collection_requested_at"),
            "catalog should include collection request timestamp"
        );
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_engine_run_with_config_uses_generic_collector_files_without_retriggering_collection()
     {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_data_files"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string(
                        r#"<td>ExchangeHistoryDataCollector_1779194074.4955919.data</td>"#,
                    ),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/data_collector"))
            .and(query_param("action_type", "start_collector"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json("Historical data collection started."),
            )
            .expect(0)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/backtesting"))
            .and(query_param("action_type", "start_backtesting"))
            .and(body_string_contains(
                "ExchangeHistoryDataCollector_1779194074.4955919.data",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_string("\"Backtesting started\""))
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting_run_id"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "backtesting_id": 88 })),
            )
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_report"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(octobot_report_body(4.0, 2.0, 6)),
            )
            .mount(&server)
            .await;

        let temp = tempdir().expect("tempdir");
        let catalog_path = temp.path().join("backtest_data_catalog.json");
        let mut config = TradingConfig {
            data_path: temp
                .path()
                .join("trading_state.json")
                .to_string_lossy()
                .to_string(),
            backtest_data_catalog_path: catalog_path.to_string_lossy().to_string(),
            backtest_symbols: vec!["BTC/USDT".to_string()],
            backtest_data_collection_enabled: true,
            backtest_data_collection_exchange: "binance".to_string(),
            backtest_data_collection_time_frames: vec!["1h".to_string()],
            backtest_data_collection_cooldown_seconds: 300,
            ..TradingConfig::default()
        };
        config.normalize();

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let engine = BacktestEngine::with_poll_params(client, 0.0, Duration::from_millis(1), 5);
        let summary = engine.run_with_config(&config).await;

        assert_eq!(summary.assessment, ApproachAssessment::Viable);
        assert_eq!(summary.run_id, Some(88));
        assert!(
            !summary
                .notes
                .contains("started OctoBot historical data collection"),
            "collector should not be retriggered when generic collector files are available: {}",
            summary.notes
        );
        let persisted_catalog = std::fs::read_to_string(catalog_path).expect("catalog persisted");
        assert!(
            persisted_catalog.contains("ExchangeHistoryDataCollector_1779194074.4955919.data"),
            "catalog should persist discovered generic collector file names"
        );
        server.verify().await;
    }

    // -----------------------------------------------------------------------
    // BacktestEngine: full run with mock OctoBot — profitable approach
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_engine_full_run_profitable_approach() {
        let server = MockServer::start().await;

        // 1. Start backtest → 200 OK
        Mock::given(method("POST"))
            .and(path("/backtesting"))
            .and(query_param("action_type", "start_backtesting"))
            .respond_with(ResponseTemplate::new(200).set_body_string("\"Backtesting started\""))
            .expect(1)
            .mount(&server)
            .await;

        // 2. Run ID
        Mock::given(method("GET"))
            .and(path("/backtesting_run_id"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "backtesting_id": 7 })),
            )
            .mount(&server)
            .await;

        // 3. Report (returns immediately — no polling delay needed in test)
        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_report"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(octobot_report_body(8.5, 3.2, 15)),
            )
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        // Use 1 ms poll interval and 5 max polls so the test is fast.
        let engine = BacktestEngine::with_poll_params(client, 0.0, Duration::from_millis(1), 5);

        let req = BacktestStartRequest::default();
        let summary = engine.run(&req).await;

        assert_eq!(
            summary.assessment,
            ApproachAssessment::Viable,
            "8.5% return should be viable (threshold=0.0): notes={}",
            summary.notes
        );
        assert!(summary.profitability_pct.unwrap() > 0.0);
        assert_eq!(summary.beats_market, Some(true));
        assert_eq!(summary.total_trades, 15);
        assert_eq!(summary.run_id, Some(7));
        assert!(!summary.notes.is_empty());
    }

    // -----------------------------------------------------------------------
    // BacktestEngine: unprofitable approach is correctly assessed
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_engine_full_run_unprofitable_approach() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/backtesting"))
            .and(query_param("action_type", "start_backtesting"))
            .respond_with(ResponseTemplate::new(200).set_body_string("\"Backtesting started\""))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting_run_id"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "backtesting_id": null })),
            )
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_report"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(octobot_report_body(-15.0, 4.0, 5)),
            )
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let engine = BacktestEngine::with_poll_params(client, 0.0, Duration::from_millis(1), 5);
        let req = BacktestStartRequest::default();
        let summary = engine.run(&req).await;

        assert_eq!(
            summary.assessment,
            ApproachAssessment::Unprofitable,
            "−15% return should be unprofitable"
        );
        assert!(summary.profitability_pct.unwrap() < 0.0);
        assert_eq!(summary.run_id, None, "null run_id should be None");
    }

    // -----------------------------------------------------------------------
    // BacktestEngine: OctoBot returns an error on start
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_engine_start_failure_returns_incomplete() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/backtesting"))
            .and(query_param("action_type", "start_backtesting"))
            .respond_with(ResponseTemplate::new(500).set_body_string("\"Internal server error\""))
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let engine = BacktestEngine::with_poll_params(client, 0.0, Duration::from_millis(1), 5);
        let req = BacktestStartRequest::default();
        let summary = engine.run(&req).await;

        assert_eq!(summary.assessment, ApproachAssessment::Incomplete);
        assert!(summary.notes.contains("start failed"));
        server.verify().await;
    }

    #[tokio::test]
    async fn backtest_engine_already_running_reuses_existing_run() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/backtesting"))
            .and(query_param("action_type", "start_backtesting"))
            .respond_with(
                ResponseTemplate::new(500).set_body_string("\"A backtesting is already running\""),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting_run_id"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "backtesting_id": 123 })),
            )
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_report"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(octobot_report_body(7.0, 2.0, 9)),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let engine = BacktestEngine::with_poll_params(client, 0.0, Duration::from_millis(1), 5);
        let req = BacktestStartRequest::default();
        let summary = engine.run(&req).await;

        assert_eq!(summary.assessment, ApproachAssessment::Viable);
        assert_eq!(summary.run_id, Some(123));
        assert!(summary.notes.contains("Profitable"));
        server.verify().await;
    }

    // -----------------------------------------------------------------------
    // BacktestEngine: times out when OctoBot keeps returning empty report
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_engine_times_out_when_no_report_ready() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/backtesting"))
            .and(query_param("action_type", "start_backtesting"))
            .respond_with(ResponseTemplate::new(200).set_body_string("\"Backtesting started\""))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting_run_id"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "backtesting_id": 1 })),
            )
            .mount(&server)
            .await;

        // Always return empty report (still running).
        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_report"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        // 3 polls with 1 ms interval → quick timeout.
        let engine = BacktestEngine::with_poll_params(client, 0.0, Duration::from_millis(1), 3);
        let req = BacktestStartRequest::default();
        let summary = engine.run(&req).await;

        assert_eq!(summary.assessment, ApproachAssessment::Incomplete);
        assert!(summary.notes.contains("timed out"));
    }

    // -----------------------------------------------------------------------
    // BacktestSummary: stored in SharedTradingState via record_backtest
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_summary_stored_in_state() {
        let state = SharedTradingState::new(100, 50);

        let summary = BacktestSummary::incomplete("unit test");
        {
            let mut s = state.0.lock().await;
            s.record_backtest(summary.clone());
        }

        let s = state.0.lock().await;
        assert!(s.last_backtest.is_some());
        assert_eq!(s.backtest_history.len(), 1);
        assert_eq!(
            s.last_backtest.as_ref().unwrap().assessment,
            ApproachAssessment::Incomplete
        );
    }

    // -----------------------------------------------------------------------
    // BacktestSummary: ring buffer enforces max 20 entries
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn backtest_history_ring_enforces_size() {
        let state = SharedTradingState::new(100, 50);
        {
            let mut s = state.0.lock().await;
            for i in 0..25u64 {
                s.record_backtest(BacktestSummary::incomplete(format!("run {i}")));
            }
        }
        let s = state.0.lock().await;
        assert!(
            s.backtest_history.len() <= 20,
            "backtest history should be capped at 20, got {}",
            s.backtest_history.len()
        );
    }

    // -----------------------------------------------------------------------
    // status_snapshot reflects last backtest assessment
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn status_snapshot_includes_backtest_assessment() {
        let state = SharedTradingState::new(100, 50);
        {
            let mut s = state.0.lock().await;
            s.record_backtest(BacktestSummary::incomplete("test snapshot"));
        }
        let s = state.0.lock().await;
        let snapshot = s.status_snapshot(true);
        assert_eq!(
            snapshot.last_backtest_assessment.as_deref(),
            Some("incomplete")
        );
        assert!(snapshot.last_backtest_at.is_some());
    }

    // -----------------------------------------------------------------------
    // A sample BTC/USDT trending-up approach proves viable via backtest
    //
    // This test proves the end-to-end capability: given a mock OctoBot that
    // returns a profitable backtest result for a trend-following strategy, the
    // engine correctly classifies the approach as Viable and the result is
    // surfaced in the trading state.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn sample_btc_trending_approach_proves_viable() {
        let server = MockServer::start().await;

        // Simulate OctoBot running a 30-day BTC/USDT trend-following backtest
        // that returned +9.2% vs +4.8% market average — a clear alpha of ~4.4%.
        let report = octobot_report_body(9.2, 4.8, 22);

        Mock::given(method("POST"))
            .and(path("/backtesting"))
            .and(query_param("action_type", "start_backtesting"))
            .respond_with(ResponseTemplate::new(200).set_body_string("\"Backtesting started\""))
            .expect(1)
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting_run_id"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "backtesting_id": 99 })),
            )
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/backtesting"))
            .and(query_param("update_type", "backtesting_report"))
            .respond_with(ResponseTemplate::new(200).set_body_json(report))
            .mount(&server)
            .await;

        let client = OctobotClient::new(&server.uri(), None, 10.0);
        let engine = BacktestEngine::with_poll_params(
            client,
            0.0, // threshold: any positive return is viable
            Duration::from_millis(1),
            5,
        );

        // Build a request representing a 30-day lookback for BTC/USDT.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let req = BacktestStartRequest {
            files: vec![], // let OctoBot choose data
            start_timestamp: Some(now_ms - 30 * 86_400_000),
            end_timestamp: Some(now_ms),
            enable_logs: false,
        };

        let summary = engine.run(&req).await;

        // --- Prove the approach is viable ---
        assert_eq!(
            summary.assessment,
            ApproachAssessment::Viable,
            "BTC trend-following over 30 days returned +9.2% — should be viable"
        );
        assert!(
            summary.profitability_pct.unwrap() > 0.0,
            "Profitability must be positive: {:?}",
            summary.profitability_pct
        );
        assert_eq!(
            summary.beats_market,
            Some(true),
            "Strategy should beat market (+9.2% vs +4.8%)"
        );
        assert_eq!(
            summary.total_trades, 22,
            "Should record all 22 simulated trades"
        );
        assert!(summary.symbols.contains(&"BTC/USDT".to_string()));
        assert_eq!(summary.run_id, Some(99));
        assert!(
            !summary.notes.is_empty(),
            "Notes should describe the result"
        );

        // --- Persist to state and verify surfaced correctly ---
        let state = SharedTradingState::new(100, 50);
        {
            let mut s = state.0.lock().await;
            s.record_backtest(summary);
        }
        let s = state.0.lock().await;
        let snap = s.status_snapshot(true);
        assert_eq!(snap.last_backtest_assessment.as_deref(), Some("viable"));
        assert!(snap.last_backtest_at.is_some());

        server.verify().await;
    }
}
