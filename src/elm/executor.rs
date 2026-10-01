//! Fixed-size inference execution with bounded admission and deadline abstention.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::sync::{Semaphore, oneshot};

use super::{ElmArtifact, ElmError, Result};

#[derive(Clone)]
pub struct InferenceExecutor {
    pool: Arc<rayon::ThreadPool>,
    slots: Arc<Semaphore>,
}

impl InferenceExecutor {
    pub fn new(threads: usize, queue_capacity: usize) -> Result<Self> {
        let threads = threads.clamp(1, 64);
        let queue_capacity = queue_capacity.clamp(1, 65_536);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("gail-elm-infer-{index}"))
            .build()
            .map_err(|error| ElmError::Artifact(format!("create inference pool: {error}")))?;
        Ok(Self {
            pool: Arc::new(pool),
            slots: Arc::new(Semaphore::new(threads.saturating_add(queue_capacity))),
        })
    }

    pub async fn predict(
        &self,
        model: Arc<ElmArtifact>,
        features: Vec<f64>,
        deadline: Duration,
    ) -> Result<InferenceResult> {
        let permit = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| ElmError::QueueFull)?;
        let (sender, receiver) = oneshot::channel();
        let started = Instant::now();
        self.pool.spawn(move || {
            // Holding the permit in this closure bounds work even after a caller
            // has timed out and stopped awaiting the result.
            let _permit = permit;
            let result = if model.task_kind == super::TaskKind::Classification {
                model
                    .predict_probabilities(std::slice::from_ref(&features))
                    .map(|values| values.into_iter().next().unwrap_or_default())
            } else {
                model
                    .raw_predict(std::slice::from_ref(&features))
                    .map(|values| values.into_iter().next().unwrap_or_default())
            };
            let _ = sender.send(result.map(|values| InferenceResult {
                values,
                elapsed: started.elapsed(),
            }));
        });
        match tokio::time::timeout(deadline, receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(ElmError::Numerical(
                "inference worker exited before returning a result".into(),
            )),
            Err(_) => Err(ElmError::Numerical(
                "inference deadline exceeded; queued computation remains bounded".into(),
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct InferenceResult {
    pub values: Vec<f64>,
    pub elapsed: Duration,
}
