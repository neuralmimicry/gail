//! Independent, bounded research scheduling. Provider rounds, discovery and
//! venue maintenance cannot starve a manual replay. The worker owns no order
//! submission path and uses the same replay and promotion rules as before.

use std::{
    future::Future,
    panic::AssertUnwindSafe,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use futures::FutureExt;
use serde_json::json;
use tokio::sync::{mpsc, watch};
use tokio::time::{Instant, MissedTickBehavior, interval_at};
use tracing::{info, warn};

use super::{
    apply_backtest_auto_tuning, backtest::BacktestEngine, config::TradingConfig,
    datalake::MarketDataLake, native_backtest_summary, now_ts, octobot::OctobotClient,
    quantitative::backtest::NativeQuantBacktester, state::SharedTradingState,
};

/// At most one queued request in addition to the replay currently running.
/// The atomic flag also makes that queue visible to read-only API clients.
#[derive(Clone)]
pub(super) struct BacktestRequests {
    sender: mpsc::Sender<()>,
    queued: Arc<AtomicBool>,
}

impl BacktestRequests {
    pub(super) fn channel() -> (Self, mpsc::Receiver<()>) {
        let (sender, receiver) = mpsc::channel(1);
        (
            Self {
                sender,
                queued: Arc::new(AtomicBool::new(false)),
            },
            receiver,
        )
    }

    pub(super) fn request(&self) -> bool {
        if self
            .queued
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        if self.sender.try_send(()).is_err() {
            self.queued.store(false, Ordering::SeqCst);
            return false;
        }
        true
    }

    pub(super) fn is_queued(&self) -> bool {
        self.queued.load(Ordering::SeqCst)
    }
}

/// Run one replay at a time. Automatic work first becomes due after one
/// interval; an administrator can request it immediately, including while
/// trading is paused. Reset the interval after completion to avoid catch-up
/// bursts. Shutdown cancels the awaiting future and clears transient flags;
/// already-started CPU replay may finish on Tokio's blocking pool but cannot
/// publish a report or submit orders after this worker exits.
pub(super) async fn run<F, Fut>(
    state: SharedTradingState,
    data_path: PathBuf,
    requests: BacktestRequests,
    mut receiver: mpsc::Receiver<()>,
    period: Duration,
    mut shutdown: watch::Receiver<()>,
    mut replay: F,
) where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<(), String>>,
{
    let period = period.max(Duration::from_millis(1));
    let mut tick = interval_at(Instant::now() + period, period);
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        let manual = tokio::select! {
            biased;
            _ = shutdown.changed() => break,
            Some(()) = receiver.recv() => true,
            _ = tick.tick() => false,
        };
        let claimed = {
            let mut current = state.0.lock().await;
            if current.backtest_in_progress {
                false
            } else {
                current.backtest_in_progress = true;
                true
            }
        };
        if manual {
            requests.queued.store(false, Ordering::SeqCst);
        }
        if !claimed {
            continue;
        }
        state
            .log(
                "info",
                "backtest",
                "Starting independently scheduled backtest",
                json!({"manual": manual}),
            )
            .await;
        // Catch a worker panic so a failed replay cannot wedge the running bit
        // or permanently close the research queue. Financial gates stay intact.
        let result = tokio::select! {
            biased;
            _ = shutdown.changed() => None,
            result = AssertUnwindSafe(async { replay().await }).catch_unwind() => Some(result),
        };
        state.0.lock().await.backtest_in_progress = false;
        let Some(result) = result else {
            break;
        };
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => state.log_error("backtest", error).await,
            Err(_) => {
                state
                    .log_error(
                        "backtest",
                        "Research worker panicked; replay did not qualify",
                    )
                    .await
            }
        }
        state.persist(&data_path).await;
        tick.reset();
    }
    receiver.close();
    requests.queued.store(false, Ordering::SeqCst);
    state.0.lock().await.backtest_in_progress = false;
    state
        .log_info("shutdown", "Independent backtest worker stopped")
        .await;
    state.persist(&data_path).await;
}

pub(super) async fn run_configured_backtest(
    config: Arc<TradingConfig>,
    state: SharedTradingState,
    lake: Option<MarketDataLake>,
) -> Result<(), String> {
    if !config.uses_native_backtest() {
        // Explicit compatibility configuration retains its existing assessment
        // and tuning behavior. These reports never promote native quant.
        let engine = BacktestEngine::new(
            OctobotClient::new(
                &config.octobot_base_url,
                config.octobot_password.as_deref(),
                config.octobot_timeout_seconds,
            ),
            config.backtest_profitability_threshold,
        );
        let summary = engine.run_with_config(&config).await;
        let mut current = state.0.lock().await;
        current.record_backtest(summary.clone());
        apply_backtest_auto_tuning(&config, &mut current, &summary);
        return Ok(());
    }
    // Catalog maintenance has its own task. Native replay must not wait on
    // OctoBot's catalog/collector HTTP routes before reading the local lake.
    let lake = lake.ok_or("Native replay requires the market datalake")?;
    let mut native_config = config.quantitative.native_backtest.clone();
    native_config.fee_bps = config.estimated_fee_bps;
    native_config.slippage_bps = config.estimated_slippage_bps;
    let frames = lake.native_backtest_frames(&native_config).await?;
    let parameter_sets = state.0.lock().await.quant_migration.parameter_sets.clone();
    let report = NativeQuantBacktester::new(native_config.clone())
        .with_portfolio_config(config.quantitative.portfolio.clone())
        .run_async(frames, parameter_sets)
        .await?;
    let summary = native_backtest_summary(&report, &native_config);
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
        current.record_backtest(summary);
        current.log(
            if report.promotion_qualified {
                "info"
            } else {
                "warn"
            },
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
        let pause =
            config.backtest_pause_on_failure && quant_primary && !report.promotion_qualified;
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::Semaphore;

    /// A blocked evaluator retains its own queue, while independent research
    /// starts, coalesces pending requests, serializes runs and stops on shutdown.
    #[tokio::test]
    async fn blocked_evaluation_cannot_starve_bounded_replay_or_shutdown() {
        let directory = tempfile::tempdir().unwrap();
        let state = SharedTradingState::new(100, 100);
        let (requests, receiver) = BacktestRequests::channel();
        let (shutdown, stop) = watch::channel(());
        let started = Arc::new(Semaphore::new(0));
        let release = Arc::new(Semaphore::new(0));
        let (evaluation, mut evaluation_rx) = mpsc::channel(1);
        evaluation.try_send(()).unwrap();
        let blocked = tokio::spawn(async move {
            std::future::pending::<()>().await;
            evaluation_rx.recv().await;
        });
        let worker = tokio::spawn(run(
            state.clone(),
            directory.path().join("state.json"),
            requests.clone(),
            receiver,
            Duration::from_secs(3600),
            stop,
            {
                let started = started.clone();
                let release = release.clone();
                move || {
                    let started = started.clone();
                    let release = release.clone();
                    async move {
                        started.add_permits(1);
                        release.acquire().await.unwrap().forget();
                        Ok(())
                    }
                }
            },
        ));
        assert!(requests.request());
        tokio::time::timeout(Duration::from_secs(2), started.acquire())
            .await
            .unwrap()
            .unwrap()
            .forget();
        assert!(state.0.lock().await.backtest_in_progress);
        assert!(!requests.is_queued());
        assert!(requests.request());
        assert!(!requests.request(), "pending work must coalesce");
        assert!(
            evaluation.try_send(()).is_err(),
            "live evaluation is still blocked"
        );
        assert_eq!(started.available_permits(), 0, "only one replay runs");
        release.add_permits(1);
        tokio::time::timeout(Duration::from_secs(2), started.acquire())
            .await
            .unwrap()
            .unwrap()
            .forget();
        assert!(requests.request());
        drop(shutdown);
        tokio::time::timeout(Duration::from_secs(2), worker)
            .await
            .unwrap()
            .unwrap();
        assert!(!state.0.lock().await.backtest_in_progress);
        assert!(!requests.is_queued());
        assert!(!requests.request(), "closed worker rejects new work");
        assert_eq!(state.0.lock().await.evaluation_count, 0);
        blocked.abort();
    }

    #[tokio::test]
    async fn error_and_panic_clear_running_and_allow_retry() {
        let directory = tempfile::tempdir().unwrap();
        let state = SharedTradingState::new(100, 100);
        let (requests, receiver) = BacktestRequests::channel();
        let (shutdown, stop) = watch::channel(());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker = tokio::spawn(run(
            state.clone(),
            directory.path().join("state.json"),
            requests.clone(),
            receiver,
            Duration::from_secs(3600),
            stop,
            {
                let calls = calls.clone();
                move || {
                    let calls = calls.clone();
                    async move {
                        match calls.fetch_add(1, Ordering::SeqCst) {
                            0 => Err("injected datalake failure".into()),
                            1 => panic!("injected replay panic"),
                            _ => Ok(()),
                        }
                    }
                }
            },
        ));
        for expected in 1..=3 {
            assert!(requests.request());
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    if calls.load(Ordering::SeqCst) == expected
                        && !state.0.lock().await.backtest_in_progress
                    {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
        }
        drop(shutdown);
        worker.await.unwrap();
        assert!(state.0.lock().await.last_native_quant_backtest.is_none());
    }

    #[tokio::test]
    async fn wall_clock_replay_runs_without_evaluation_or_manual_request() {
        let directory = tempfile::tempdir().unwrap();
        let state = SharedTradingState::new(100, 100);
        state.0.lock().await.paused = true;
        let (requests, receiver) = BacktestRequests::channel();
        let (shutdown, stop) = watch::channel(());
        let (completed, mut completion) = mpsc::channel(1);
        let worker = tokio::spawn(run(
            state.clone(),
            directory.path().join("state.json"),
            requests,
            receiver,
            Duration::from_millis(10),
            stop,
            move || {
                let completed = completed.clone();
                async move {
                    let _ = completed.try_send(());
                    Ok(())
                }
            },
        ));
        tokio::time::timeout(Duration::from_secs(2), completion.recv())
            .await
            .unwrap()
            .unwrap();
        drop(shutdown);
        worker.await.unwrap();
        let current = state.0.lock().await;
        assert!(current.paused);
        assert_eq!(current.evaluation_count, 0);
    }
}
