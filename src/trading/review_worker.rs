//! A single bounded read-only maintenance review. Slow candidate research must
//! not occupy the primary evaluation loop. Completed plans return to that loop
//! for fresh checks and serial execution; this worker owns no order executor.

use std::future::Future;

use tokio::task::{JoinError, JoinHandle};

pub(super) struct ReadOnlyReview<T> {
    task: Option<JoinHandle<T>>,
}

impl<T: Send + 'static> ReadOnlyReview<T> {
    pub(super) fn new() -> Self {
        Self { task: None }
    }

    pub(super) fn is_pending(&self) -> bool {
        self.task.is_some()
    }

    /// Repeated due ticks coalesce while the current bounded review is pending.
    pub(super) fn start<F>(&mut self, review: F) -> bool
    where
        F: Future<Output = T> + Send + 'static,
    {
        if self.is_pending() {
            return false;
        }
        self.task = Some(tokio::spawn(review));
        true
    }

    /// Cancellation-safe select branch: ticks may keep evaluating while a
    /// review waits, and completion wakes the serial execution owner promptly.
    /// A completed result is consumed exactly once; no review means no wakeup.
    pub(super) async fn wait_ready(&mut self) -> Result<T, JoinError> {
        let Some(task) = self.task.as_mut() else {
            return std::future::pending().await;
        };
        let result = task.await;
        self.task.take();
        result
    }
}

impl<T> Drop for ReadOnlyReview<T> {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            // Planning performs no orders; cancellation cannot leave an order
            // submission outcome ambiguous or detach a future write executor.
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::sync::Semaphore;

    use super::*;

    #[tokio::test]
    async fn blocked_review_does_not_block_evaluations_or_accumulate_work() {
        let release = Arc::new(Semaphore::new(0));
        let started = Arc::new(Semaphore::new(0));
        let mut worker = ReadOnlyReview::new();
        let release_task = release.clone();
        let started_task = started.clone();
        assert!(worker.start(async move {
            started_task.add_permits(1);
            release_task.acquire().await.unwrap().forget();
            42
        }));
        started.acquire().await.unwrap().forget();
        for _ in 0..100 {
            tokio::select! {
                result = worker.wait_ready() => panic!("blocked review completed: {result:?}"),
                _ = tokio::task::yield_now() => {},
            }
            assert!(!worker.start(async { 99 }));
        }
        release.add_permits(1);
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), worker.wait_ready())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result, 42);
        assert!(!worker.is_pending());
    }

    #[tokio::test]
    async fn completion_wakes_before_the_next_evaluation_tick() {
        let mut worker = ReadOnlyReview::new();
        worker.start(async { 42 });
        let next_tick = tokio::time::sleep(std::time::Duration::from_secs(3600));
        let completed = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            tokio::select! {
                result = worker.wait_ready() => result.unwrap(),
                _ = next_tick => panic!("review waited for the unrelated evaluation tick"),
            }
        })
        .await
        .unwrap();
        assert_eq!(completed, 42);
        assert!(!worker.is_pending());
    }

    #[tokio::test]
    async fn drop_cancels_planning_before_it_can_publish_a_result() {
        let release = Arc::new(Semaphore::new(0));
        let started = Arc::new(Semaphore::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        let mut worker = ReadOnlyReview::new();
        let release_task = release.clone();
        let started_task = started.clone();
        let completed_task = completed.clone();
        worker.start(async move {
            started_task.add_permits(1);
            release_task.acquire().await.unwrap().forget();
            completed_task.fetch_add(1, Ordering::SeqCst);
        });
        started.acquire().await.unwrap().forget();
        drop(worker);
        release.add_permits(1);
        tokio::task::yield_now().await;
        assert_eq!(completed.load(Ordering::SeqCst), 0);
    }
}
