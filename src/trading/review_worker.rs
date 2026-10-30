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

    /// Never wait for provider I/O. Awaiting a finished task only retrieves its
    /// result or panic, leaving the next primary evaluation free to proceed.
    pub(super) async fn take_ready(&mut self) -> Result<Option<T>, JoinError> {
        if !self.task.as_ref().is_some_and(JoinHandle::is_finished) {
            return Ok(None);
        }
        self.task
            .take()
            .expect("finished review exists")
            .await
            .map(Some)
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
            assert!(worker.take_ready().await.unwrap().is_none());
            assert!(!worker.start(async { 99 }));
        }
        release.add_permits(1);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if let Some(result) = worker.take_ready().await.unwrap() {
                    assert_eq!(result, 42);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
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
