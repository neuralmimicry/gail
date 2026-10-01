//! Bounded Rust-native Extreme Learning Machines and lifecycle support.
//!
//! Domain integrations consume this façade; model code does not own routing,
//! training admission, mirror delivery, or trading execution authority.

pub mod data;
pub mod executor;
pub mod gates;
pub mod integrations;
pub mod lifecycle;
pub mod math;
pub mod model;
pub mod worker;

use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub use model::{ElmArtifact, FeatureSchema, TaskKind};

pub type Result<T> = std::result::Result<T, ElmError>;

#[derive(Debug, Error)]
pub enum ElmError {
    #[error("invalid ELM input: {0}")]
    InvalidInput(String),
    #[error("ELM numerical error: {0}")]
    Numerical(String),
    #[error("ELM artifact error: {0}")]
    Artifact(String),
    #[error("feature schema mismatch")]
    SchemaMismatch,
    #[error("classification artifact is not calibrated")]
    Uncalibrated,
    #[error("bounded ELM inference queue is full")]
    QueueFull,
    #[error("model registry revision changed")]
    RevisionConflict,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleStage {
    Candidate,
    Trained,
    Evaluated,
    Qualified,
    Shadow,
    Canary,
    Active,
    Rejected,
    Quarantined,
    Retired,
    RolledBack,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegistryRecord {
    pub revision: u64,
    pub model_id: String,
    pub task_id: String,
    pub digest: String,
    pub stage: LifecycleStage,
    pub artifact_file: String,
    pub reason: Option<String>,
    pub updated_at: u64,
    pub fixture_only: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct RegistryFile {
    revision: u64,
    records: Vec<RegistryRecord>,
    champion_by_task: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    rollback_history: Vec<RollbackEvent>,
    /// Durable per-model budget for bounded shadow-to-canary exposure.
    #[serde(default)]
    canary_decisions: std::collections::BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollbackEvent {
    pub idempotency_key: String,
    pub task_id: String,
    pub rolled_back_model_id: String,
    pub fallback_model_id: Option<String>,
    pub revision: u64,
    pub reason: String,
}

#[derive(Default)]
struct RegistryState {
    file: RegistryFile,
    champions: std::collections::BTreeMap<String, Arc<ElmArtifact>>,
}

/// Registry backed by atomic local files and an OS advisory writer lock.
/// Processes sharing one host-mounted directory coordinate revisions safely;
/// multi-host deployments still require a distributed fenced registry.
pub struct ModelRegistry {
    root: PathBuf,
    max_artifact_bytes: usize,
    state: RwLock<RegistryState>,
}

impl ModelRegistry {
    pub fn open(root: impl Into<PathBuf>, max_artifact_bytes: usize) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(root.join("artifacts"))
            .map_err(|error| ElmError::Artifact(format!("create registry: {error}")))?;
        let file = root.join("registry-v1.json");
        let registry_file = if file.exists() {
            let metadata =
                std::fs::metadata(&file).map_err(|error| ElmError::Artifact(error.to_string()))?;
            if metadata.len() as usize > max_artifact_bytes {
                return Err(ElmError::Artifact(
                    "registry index exceeds configured size limit".into(),
                ));
            }
            let raw = std::fs::read(file).map_err(|error| ElmError::Artifact(error.to_string()))?;
            serde_json::from_slice::<RegistryFile>(&raw)
                .map_err(|error| ElmError::Artifact(error.to_string()))?
        } else {
            RegistryFile::default()
        };
        let registry = Self {
            root,
            max_artifact_bytes,
            state: RwLock::new(RegistryState {
                file: registry_file,
                champions: Default::default(),
            }),
        };
        registry.restore_champions()?;
        Ok(registry)
    }

    /// Refresh a follower's immutable serving snapshots from the durable
    /// registry index. Call on a bounded background interval, never per request.
    pub fn refresh(&self) -> Result<bool> {
        let mut state = self
            .state
            .write()
            .map_err(|_| ElmError::Artifact("registry lock poisoned".into()))?;
        self.refresh_locked(&mut state)
    }

    fn acquire_writer_lock(&self) -> Result<File> {
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.root.join("registry-v1.lock"))
            .map_err(|error| ElmError::Artifact(format!("open registry writer lock: {error}")))?;
        FileExt::lock_exclusive(&lock)
            .map_err(|error| ElmError::Artifact(format!("lock registry writer: {error}")))?;
        Ok(lock)
    }

    fn read_registry_file(&self) -> Result<RegistryFile> {
        let path = self.root.join("registry-v1.json");
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(RegistryFile::default());
            }
            Err(error) => {
                return Err(ElmError::Artifact(format!(
                    "read registry metadata: {error}"
                )));
            }
        };
        if metadata.len() as usize > self.max_artifact_bytes {
            return Err(ElmError::Artifact(
                "registry index exceeds configured size limit".into(),
            ));
        }
        let raw = std::fs::read(path)
            .map_err(|error| ElmError::Artifact(format!("read registry index: {error}")))?;
        serde_json::from_slice(&raw)
            .map_err(|error| ElmError::Artifact(format!("invalid registry index: {error}")))
    }

    fn load_champions(
        &self,
        file: &RegistryFile,
    ) -> Result<std::collections::BTreeMap<String, Arc<ElmArtifact>>> {
        let mut loaded = std::collections::BTreeMap::new();
        for (task, model_id) in &file.champion_by_task {
            let record = file
                .records
                .iter()
                .find(|record| record.model_id == *model_id)
                .ok_or_else(|| ElmError::Artifact("registry champion record is missing".into()))?;
            if record.task_id != *task
                || record.fixture_only
                || !matches!(
                    record.stage,
                    LifecycleStage::Qualified
                        | LifecycleStage::Shadow
                        | LifecycleStage::Canary
                        | LifecycleStage::Active
                )
            {
                continue;
            }
            let artifact = load_artifact(
                &self.root.join(&record.artifact_file),
                self.max_artifact_bytes,
            )?;
            if artifact.metadata.model_id != *model_id
                || artifact.metadata.task_id != *task
                || artifact.content_sha256 != record.digest
            {
                return Err(ElmError::Artifact(
                    "registry record and immutable model artifact do not match".into(),
                ));
            }
            loaded.insert(task.clone(), Arc::new(artifact));
        }
        Ok(loaded)
    }

    fn refresh_locked(&self, state: &mut RegistryState) -> Result<bool> {
        let fresh = self.read_registry_file()?;
        if fresh.revision <= state.file.revision {
            return Ok(false);
        }
        let champions = self.load_champions(&fresh)?;
        state.file = fresh;
        state.champions = champions;
        Ok(true)
    }

    fn restore_champions(&self) -> Result<()> {
        let champions = {
            let state = self
                .state
                .read()
                .map_err(|_| ElmError::Artifact("registry lock poisoned".into()))?;
            state
                .file
                .champion_by_task
                .iter()
                .map(|(task, model)| (task.clone(), model.clone()))
                .collect::<Vec<_>>()
        };
        let mut loaded = Vec::new();
        for (task, model_id) in champions {
            let record = self
                .record(&model_id)
                .ok_or_else(|| ElmError::Artifact("registry champion record is missing".into()))?;
            if record.task_id != task
                || record.fixture_only
                || !matches!(
                    record.stage,
                    LifecycleStage::Qualified
                        | LifecycleStage::Shadow
                        | LifecycleStage::Canary
                        | LifecycleStage::Active
                )
            {
                continue;
            }
            let artifact = load_artifact(
                &self.root.join(&record.artifact_file),
                self.max_artifact_bytes,
            )?;
            if artifact.metadata.model_id != model_id
                || artifact.metadata.task_id != task
                || artifact.content_sha256 != record.digest
            {
                continue;
            }
            loaded.push((task, Arc::new(artifact)));
        }
        let mut state = self
            .state
            .write()
            .map_err(|_| ElmError::Artifact("registry lock poisoned".into()))?;
        state.champions.extend(loaded);
        Ok(())
    }

    pub fn champion(&self, task_id: &str) -> Option<Arc<ElmArtifact>> {
        self.state.read().ok()?.champions.get(task_id).cloned()
    }

    pub fn revision(&self) -> u64 {
        self.state
            .read()
            .map(|state| state.file.revision)
            .unwrap_or(0)
    }

    pub fn records(&self) -> Vec<RegistryRecord> {
        self.state
            .read()
            .map(|state| state.file.records.clone())
            .unwrap_or_default()
    }

    pub fn record(&self, model_id: &str) -> Option<RegistryRecord> {
        self.state
            .read()
            .ok()?
            .file
            .records
            .iter()
            .find(|item| item.model_id == model_id)
            .cloned()
    }

    /// Return the durable canary-decision count for one immutable model.
    pub fn canary_decisions(&self, model_id: &str) -> u64 {
        self.state
            .read()
            .map(|state| {
                state
                    .file
                    .canary_decisions
                    .get(model_id)
                    .copied()
                    .unwrap_or(0)
            })
            .unwrap_or(u64::MAX)
    }

    /// Claim one bounded canary decision and persist the count before inference.
    /// When the final permitted decision is claimed, the model is atomically
    /// returned to shadow so subsequent requests retain the baseline path.
    pub fn claim_canary_decision(&self, model_id: &str, maximum: u64) -> Result<bool> {
        if maximum == 0 {
            return Ok(false);
        }
        let _writer_lock = self.acquire_writer_lock()?;
        let mut state = self
            .state
            .write()
            .map_err(|_| ElmError::Artifact("registry lock poisoned".into()))?;
        self.refresh_locked(&mut state)?;
        let Some(index) = state
            .file
            .records
            .iter()
            .position(|record| record.model_id == model_id)
        else {
            return Err(ElmError::Artifact("canary model not found".into()));
        };
        if state.file.records[index].stage != LifecycleStage::Canary
            || state.file.records[index].fixture_only
        {
            return Ok(false);
        }
        let used = state
            .file
            .canary_decisions
            .get(model_id)
            .copied()
            .unwrap_or(0);
        if used >= maximum {
            return Ok(false);
        }
        let mut next = state.file.clone();
        let count = used.saturating_add(1);
        next.canary_decisions.insert(model_id.to_string(), count);
        let revision = next.revision.saturating_add(1);
        if count >= maximum {
            next.records[index].stage = LifecycleStage::Shadow;
            next.records[index].reason =
                Some("canary decision budget exhausted; automatically returned to shadow".into());
            next.records[index].updated_at = unix_seconds();
            next.records[index].revision = revision;
        }
        next.revision = revision;
        persist_registry(&self.root, &next, self.max_artifact_bytes)?;
        state.file = next;
        Ok(true)
    }

    pub fn find_dataset(&self, task_id: &str, dataset_digest: &str) -> Option<RegistryRecord> {
        self.state
            .read()
            .ok()?
            .file
            .records
            .iter()
            .rev()
            .find(|record| {
                if record.task_id != task_id {
                    return false;
                }
                let artifact = load_artifact(
                    &self.root.join(&record.artifact_file),
                    self.max_artifact_bytes,
                )
                .ok();
                artifact.is_some_and(|item| item.metadata.dataset_digest == dataset_digest)
            })
            .cloned()
    }

    pub fn publish_candidate(
        &self,
        artifact: ElmArtifact,
        expected_revision: u64,
    ) -> Result<RegistryRecord> {
        artifact.verify()?;
        let bytes =
            serde_json::to_vec(&artifact).map_err(|error| ElmError::Artifact(error.to_string()))?;
        if bytes.len() > self.max_artifact_bytes {
            return Err(ElmError::Artifact(
                "artifact exceeds configured size limit".into(),
            ));
        }
        let _writer_lock = self.acquire_writer_lock()?;
        let mut state = self
            .state
            .write()
            .map_err(|_| ElmError::Artifact("registry lock poisoned".into()))?;
        self.refresh_locked(&mut state)?;
        if state.file.revision != expected_revision {
            return Err(ElmError::RevisionConflict);
        }
        let filename = format!("artifacts/{}.json", artifact.metadata.model_id);
        write_atomic(&self.root.join(&filename), &bytes)?;
        let revision = state.file.revision.saturating_add(1);
        let record = RegistryRecord {
            revision,
            model_id: artifact.metadata.model_id.clone(),
            task_id: artifact.metadata.task_id.clone(),
            digest: artifact.content_sha256.clone(),
            stage: LifecycleStage::Trained,
            artifact_file: filename,
            reason: Some(
                if artifact.metadata.fixture_only {
                    "fixture_only: cannot be qualified"
                } else {
                    "evaluation report attached; promotion requires policy evidence"
                }
                .into(),
            ),
            updated_at: unix_seconds(),
            fixture_only: artifact.metadata.fixture_only,
        };
        let mut next = state.file.clone();
        next.revision = revision;
        next.records.push(record.clone());
        persist_registry(&self.root, &next, self.max_artifact_bytes)?;
        state.file = next;
        Ok(record)
    }

    pub fn store_evaluation(&self, model_id: &str, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.max_artifact_bytes {
            return Err(ElmError::Artifact(
                "evaluation report exceeds configured size limit".into(),
            ));
        }
        let record = self
            .record(model_id)
            .ok_or_else(|| ElmError::Artifact("model not found".into()))?;
        write_atomic(
            &self
                .root
                .join("evaluations")
                .join(format!("{}.json", record.model_id)),
            bytes,
        )
    }

    pub fn evaluation_json(&self, model_id: &str) -> Result<serde_json::Value> {
        let record = self
            .record(model_id)
            .ok_or_else(|| ElmError::Artifact("model not found".into()))?;
        let path = self
            .root
            .join("evaluations")
            .join(format!("{}.json", record.model_id));
        let metadata =
            std::fs::metadata(&path).map_err(|error| ElmError::Artifact(error.to_string()))?;
        if metadata.len() as usize > self.max_artifact_bytes {
            return Err(ElmError::Artifact(
                "evaluation report exceeds configured size limit".into(),
            ));
        }
        let raw = std::fs::read(path).map_err(|error| ElmError::Artifact(error.to_string()))?;
        serde_json::from_slice(&raw).map_err(|error| ElmError::Artifact(error.to_string()))
    }

    pub fn mark_evaluated(
        &self,
        model_id: &str,
        expected_revision: u64,
        reason: &str,
    ) -> Result<RegistryRecord> {
        self.promote(
            model_id,
            LifecycleStage::Evaluated,
            expected_revision,
            reason,
        )
    }

    pub fn promote(
        &self,
        model_id: &str,
        target: LifecycleStage,
        expected_revision: u64,
        reason: &str,
    ) -> Result<RegistryRecord> {
        let _writer_lock = self.acquire_writer_lock()?;
        let mut state = self
            .state
            .write()
            .map_err(|_| ElmError::Artifact("registry lock poisoned".into()))?;
        self.refresh_locked(&mut state)?;
        if state.file.revision != expected_revision {
            return Err(ElmError::RevisionConflict);
        }
        let mut next = state.file.clone();
        let index = next
            .records
            .iter()
            .position(|record| record.model_id == model_id)
            .ok_or_else(|| ElmError::Artifact("model not found".into()))?;
        let previous = next.records[index].stage.clone();
        if !lifecycle::allowed_transition(&previous, &target) {
            return Err(ElmError::Artifact(format!(
                "invalid lifecycle transition: {previous:?} -> {target:?}"
            )));
        }
        if next.records[index].fixture_only
            && matches!(
                target,
                LifecycleStage::Qualified
                    | LifecycleStage::Shadow
                    | LifecycleStage::Canary
                    | LifecycleStage::Active
            )
        {
            return Err(ElmError::Artifact(
                "fixture-only models cannot enter a serving stage".into(),
            ));
        }
        let path = self.root.join(&next.records[index].artifact_file);
        let artifact = load_artifact(&path, self.max_artifact_bytes)?;
        if artifact.metadata.model_id != model_id
            || artifact.metadata.task_id != next.records[index].task_id
            || artifact.content_sha256 != next.records[index].digest
        {
            return Err(ElmError::Artifact(
                "registry record and immutable model artifact do not match".into(),
            ));
        }
        let task_id = artifact.metadata.task_id.clone();
        let target_is_serving = matches!(
            target,
            LifecycleStage::Qualified
                | LifecycleStage::Shadow
                | LifecycleStage::Canary
                | LifecycleStage::Active
        ) && !artifact.metadata.fixture_only;
        let current_champion = next.champion_by_task.get(&task_id).cloned();
        let revision = next.revision.saturating_add(1);
        if target_is_serving {
            if let Some(previous_id) = current_champion.as_deref()
                && previous_id != model_id
                && let Some(previous_record) = next
                    .records
                    .iter_mut()
                    .find(|record| record.model_id == previous_id)
                && matches!(
                    previous_record.stage,
                    LifecycleStage::Qualified | LifecycleStage::Canary | LifecycleStage::Active
                )
            {
                previous_record.stage = LifecycleStage::Shadow;
                previous_record.reason =
                    Some("replaced by a newer compatible lifecycle snapshot".into());
                previous_record.updated_at = unix_seconds();
                previous_record.revision = revision;
            }
            next.champion_by_task
                .insert(task_id.clone(), model_id.to_string());
        }
        next.records[index].stage = target.clone();
        next.records[index].reason = Some(reason.to_string());
        next.records[index].updated_at = unix_seconds();
        next.records[index].revision = revision;
        next.revision = revision;

        let mut champions = state.champions.clone();
        if target_is_serving {
            champions.insert(task_id.clone(), Arc::new(artifact));
        } else if current_champion.as_deref() == Some(model_id) {
            let fallback = next
                .records
                .iter()
                .filter(|record| {
                    record.task_id == task_id
                        && !record.fixture_only
                        && matches!(
                            record.stage,
                            LifecycleStage::Qualified
                                | LifecycleStage::Shadow
                                | LifecycleStage::Canary
                                | LifecycleStage::Active
                        )
                })
                .max_by_key(|record| record.revision)
                .and_then(|record| {
                    load_artifact(
                        &self.root.join(&record.artifact_file),
                        self.max_artifact_bytes,
                    )
                    .ok()
                    .filter(|candidate| {
                        candidate.content_sha256 == record.digest
                            && candidate.feature_schema == artifact.feature_schema
                            && candidate.metadata.tenant_scope == artifact.metadata.tenant_scope
                            && candidate.metadata.domain_scope == artifact.metadata.domain_scope
                    })
                    .map(Arc::new)
                    .map(|candidate| (record.model_id.clone(), candidate))
                });
            if let Some((fallback_id, fallback)) = fallback {
                next.champion_by_task.insert(task_id.clone(), fallback_id);
                champions.insert(task_id.clone(), fallback);
            } else {
                next.champion_by_task.remove(&task_id);
                champions.remove(&task_id);
            }
        }
        let record = next.records[index].clone();
        persist_registry(&self.root, &next, self.max_artifact_bytes)?;
        state.file = next;
        state.champions = champions;
        Ok(record)
    }

    /// Atomically roll a task back to its newest compatible prior serving
    /// snapshot. Repeating the same idempotency key returns the recorded event.
    pub fn rollback_champion(
        &self,
        task_id: &str,
        idempotency_key: &str,
        expected_revision: u64,
        reason: &str,
    ) -> Result<RollbackEvent> {
        if idempotency_key.trim().is_empty() || reason.trim().is_empty() {
            return Err(ElmError::InvalidInput(
                "rollback requires an idempotency key and reason".into(),
            ));
        }
        let _writer_lock = self.acquire_writer_lock()?;
        let mut state = self
            .state
            .write()
            .map_err(|_| ElmError::Artifact("registry lock poisoned".into()))?;
        self.refresh_locked(&mut state)?;
        if let Some(event) = state
            .file
            .rollback_history
            .iter()
            .find(|event| event.idempotency_key == idempotency_key)
        {
            if event.task_id != task_id {
                return Err(ElmError::Artifact(
                    "rollback idempotency key was already used for another task".into(),
                ));
            }
            return Ok(event.clone());
        }
        if state.file.revision != expected_revision {
            return Err(ElmError::RevisionConflict);
        }
        let current_id = state
            .file
            .champion_by_task
            .get(task_id)
            .cloned()
            .ok_or_else(|| {
                ElmError::Artifact("task has no serving champion to roll back".into())
            })?;
        let current_artifact =
            state.champions.get(task_id).cloned().ok_or_else(|| {
                ElmError::Artifact("serving champion snapshot is unavailable".into())
            })?;
        let mut next = state.file.clone();
        let revision = next.revision.saturating_add(1);
        let current = next
            .records
            .iter_mut()
            .find(|record| record.model_id == current_id)
            .ok_or_else(|| ElmError::Artifact("serving champion record is missing".into()))?;
        if !lifecycle::allowed_transition(&current.stage, &LifecycleStage::RolledBack) {
            return Err(ElmError::Artifact(format!(
                "serving stage {:?} cannot be rolled back",
                current.stage
            )));
        }
        current.stage = LifecycleStage::RolledBack;
        current.reason = Some(reason.to_string());
        current.updated_at = unix_seconds();
        current.revision = revision;
        let fallback = next
            .records
            .iter()
            .filter(|record| {
                record.task_id == task_id
                    && record.model_id != current_id
                    && !record.fixture_only
                    && matches!(
                        record.stage,
                        LifecycleStage::Qualified
                            | LifecycleStage::Shadow
                            | LifecycleStage::Canary
                            | LifecycleStage::Active
                    )
            })
            .max_by_key(|record| record.revision)
            .and_then(|record| {
                load_artifact(
                    &self.root.join(&record.artifact_file),
                    self.max_artifact_bytes,
                )
                .ok()
                .filter(|candidate| {
                    candidate.content_sha256 == record.digest
                        && candidate.feature_schema == current_artifact.feature_schema
                        && candidate.metadata.tenant_scope == current_artifact.metadata.tenant_scope
                        && candidate.metadata.domain_scope == current_artifact.metadata.domain_scope
                })
                .map(Arc::new)
                .map(|artifact| (record.model_id.clone(), artifact))
            });
        let mut champions = state.champions.clone();
        let fallback_model_id = if let Some((model_id, artifact)) = fallback {
            next.champion_by_task
                .insert(task_id.to_string(), model_id.clone());
            champions.insert(task_id.to_string(), artifact);
            Some(model_id)
        } else {
            next.champion_by_task.remove(task_id);
            champions.remove(task_id);
            None
        };
        next.revision = revision;
        let event = RollbackEvent {
            idempotency_key: idempotency_key.to_string(),
            task_id: task_id.to_string(),
            rolled_back_model_id: current_id,
            fallback_model_id,
            revision,
            reason: reason.to_string(),
        };
        next.rollback_history.push(event.clone());
        persist_registry(&self.root, &next, self.max_artifact_bytes)?;
        state.file = next;
        state.champions = champions;
        Ok(event)
    }

    pub fn status_json(&self) -> serde_json::Value {
        serde_json::json!({ "revision": self.revision(), "models": self.records().iter().map(|record| serde_json::json!({
            "model_id": record.model_id, "task_id": record.task_id, "digest": record.digest,
            "stage": record.stage, "fixture_only": record.fixture_only, "reason": record.reason,
        })).collect::<Vec<_>>() })
    }
}

pub fn save_artifact(path: &Path, artifact: &ElmArtifact, max_bytes: usize) -> Result<()> {
    artifact.verify()?;
    let bytes =
        serde_json::to_vec(artifact).map_err(|error| ElmError::Artifact(error.to_string()))?;
    if bytes.len() > max_bytes {
        return Err(ElmError::Artifact(
            "artifact exceeds configured size limit".into(),
        ));
    }
    write_atomic(path, &bytes)
}

pub fn load_artifact(path: &Path, max_bytes: usize) -> Result<ElmArtifact> {
    let metadata =
        std::fs::metadata(path).map_err(|error| ElmError::Artifact(error.to_string()))?;
    if metadata.len() as usize > max_bytes {
        return Err(ElmError::Artifact(
            "artifact exceeds configured size limit".into(),
        ));
    }
    let bytes = std::fs::read(path).map_err(|error| ElmError::Artifact(error.to_string()))?;
    let artifact: ElmArtifact =
        serde_json::from_slice(&bytes).map_err(|error| ElmError::Artifact(error.to_string()))?;
    artifact.verify()?;
    Ok(artifact)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| ElmError::Artifact("artifact path has no parent".into()))?;
    std::fs::create_dir_all(parent).map_err(|error| ElmError::Artifact(error.to_string()))?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        Uuid::new_v4()
    ));
    let mut file =
        File::create(&temporary).map_err(|error| ElmError::Artifact(error.to_string()))?;
    file.write_all(bytes)
        .map_err(|error| ElmError::Artifact(error.to_string()))?;
    file.sync_all()
        .map_err(|error| ElmError::Artifact(error.to_string()))?;
    std::fs::rename(&temporary, path).map_err(|error| ElmError::Artifact(error.to_string()))?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| ElmError::Artifact(error.to_string()))?;
    Ok(())
}

fn persist_registry(root: &Path, registry: &RegistryFile, max_bytes: usize) -> Result<()> {
    let bytes =
        serde_json::to_vec(registry).map_err(|error| ElmError::Artifact(error.to_string()))?;
    if bytes.len() > max_bytes {
        return Err(ElmError::Artifact(
            "registry index exceeds configured size limit".into(),
        ));
    }
    write_atomic(&root.join("registry-v1.json"), &bytes)
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |value| value.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::{AlgorithmFamily, FitOptions};

    fn artifact() -> ElmArtifact {
        let schema = FeatureSchema {
            id: "registry-test:v1".into(),
            version: 1,
            names: vec!["signal".into()],
            units: vec!["unitless".into()],
        };
        let features = (0..32).map(|index| vec![index as f64]).collect::<Vec<_>>();
        let targets = features
            .iter()
            .map(|row| vec![row[0] * 0.25 + 2.0])
            .collect::<Vec<_>>();
        ElmArtifact::fit_regression(
            schema,
            &features,
            &targets,
            None,
            FitOptions {
                task_id: "routing_candidate_utility".into(),
                family: AlgorithmFamily::Ridge,
                hidden_units: 0,
                seed: 7,
                lambda: 1e-3,
                target_names: vec!["utility".into()],
                target_units: vec!["score".into()],
                class_mapping: Vec::new(),
                training_cutoff: 100.0,
                dataset_digest: "registry-test-digest".into(),
                tenant_scope: "local".into(),
                domain_scope: "test".into(),
                horizon_seconds: None,
                fixture_only: false,
            },
        )
        .unwrap()
    }

    fn advance_to_evaluated(registry: &ModelRegistry) -> String {
        let item = artifact();
        let id = item.metadata.model_id.clone();
        registry
            .publish_candidate(item, registry.revision())
            .unwrap();
        registry
            .mark_evaluated(&id, registry.revision(), "test evaluation")
            .unwrap();
        id
    }

    #[test]
    fn rollback_restores_last_compatible_snapshot_and_is_idempotent() {
        let directory = tempfile::tempdir().unwrap();
        let registry = ModelRegistry::open(directory.path(), 4 * 1024 * 1024).unwrap();
        let first = advance_to_evaluated(&registry);
        registry
            .promote(
                &first,
                LifecycleStage::Qualified,
                registry.revision(),
                "test evidence",
            )
            .unwrap();
        registry
            .promote(
                &first,
                LifecycleStage::Shadow,
                registry.revision(),
                "shadow",
            )
            .unwrap();
        let second = advance_to_evaluated(&registry);
        registry
            .promote(
                &second,
                LifecycleStage::Qualified,
                registry.revision(),
                "test evidence",
            )
            .unwrap();
        registry
            .promote(
                &second,
                LifecycleStage::Shadow,
                registry.revision(),
                "shadow",
            )
            .unwrap();
        registry
            .promote(
                &second,
                LifecycleStage::Active,
                registry.revision(),
                "authorised test stage",
            )
            .unwrap();
        let event = registry
            .rollback_champion(
                "routing_candidate_utility",
                "test-rollback-1",
                registry.revision(),
                "fault-injection test",
            )
            .unwrap();
        assert_eq!(event.rolled_back_model_id, second);
        assert_eq!(event.fallback_model_id.as_deref(), Some(first.as_str()));
        assert_eq!(
            registry
                .champion("routing_candidate_utility")
                .unwrap()
                .metadata
                .model_id,
            first
        );
        assert_eq!(
            registry
                .rollback_champion("routing_candidate_utility", "test-rollback-1", 0, "retry",)
                .unwrap()
                .revision,
            event.revision
        );
    }

    #[test]
    fn canary_decisions_are_durable_bounded_and_return_the_model_to_shadow() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().to_path_buf();
        let registry = ModelRegistry::open(&root, 4 * 1024 * 1024).unwrap();
        let model_id = advance_to_evaluated(&registry);
        registry
            .promote(
                &model_id,
                LifecycleStage::Qualified,
                registry.revision(),
                "test evidence",
            )
            .unwrap();
        registry
            .promote(
                &model_id,
                LifecycleStage::Shadow,
                registry.revision(),
                "test shadow",
            )
            .unwrap();
        registry
            .promote(
                &model_id,
                LifecycleStage::Canary,
                registry.revision(),
                "test canary",
            )
            .unwrap();

        assert!(registry.claim_canary_decision(&model_id, 2).unwrap());
        assert_eq!(registry.canary_decisions(&model_id), 1);
        assert_eq!(
            registry.record(&model_id).unwrap().stage,
            LifecycleStage::Canary
        );
        assert!(registry.claim_canary_decision(&model_id, 2).unwrap());
        assert_eq!(registry.canary_decisions(&model_id), 2);
        assert_eq!(
            registry.record(&model_id).unwrap().stage,
            LifecycleStage::Shadow
        );
        assert!(!registry.claim_canary_decision(&model_id, 2).unwrap());
        drop(registry);

        let restored = ModelRegistry::open(&root, 4 * 1024 * 1024).unwrap();
        assert_eq!(restored.canary_decisions(&model_id), 2);
        assert_eq!(
            restored.record(&model_id).unwrap().stage,
            LifecycleStage::Shadow
        );
    }

    #[test]
    fn registry_followers_refresh_complete_snapshots_and_reject_stale_writers() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let writer = ModelRegistry::open(root, 4 * 1024 * 1024).unwrap();
        let follower = ModelRegistry::open(root, 4 * 1024 * 1024).unwrap();
        let model_id = advance_to_evaluated(&writer);
        writer
            .promote(
                &model_id,
                LifecycleStage::Qualified,
                writer.revision(),
                "test evidence",
            )
            .unwrap();
        writer
            .promote(
                &model_id,
                LifecycleStage::Shadow,
                writer.revision(),
                "test shadow",
            )
            .unwrap();

        assert!(follower.refresh().unwrap());
        assert_eq!(
            follower
                .champion("routing_candidate_utility")
                .unwrap()
                .metadata
                .model_id,
            model_id
        );
        assert_eq!(follower.revision(), writer.revision());
        let stale = artifact();
        assert!(matches!(
            follower.publish_candidate(stale, 0),
            Err(ElmError::RevisionConflict)
        ));
        assert_eq!(follower.revision(), writer.revision());
    }

    #[test]
    fn registry_publication_failure_does_not_change_the_visible_revision() {
        let directory = tempfile::tempdir().unwrap();
        let registry = ModelRegistry::open(directory.path(), 4 * 1024 * 1024).unwrap();
        std::fs::create_dir(directory.path().join("registry-v1.json")).unwrap();
        assert!(registry.publish_candidate(artifact(), 0).is_err());
        assert_eq!(registry.revision(), 0);
        assert!(registry.records().is_empty());
    }

    #[test]
    fn failed_promotion_persistence_keeps_the_previous_snapshot() {
        let directory = tempfile::tempdir().unwrap();
        let registry = ModelRegistry::open(directory.path(), 4 * 1024 * 1024).unwrap();
        let item = artifact();
        let id = item.metadata.model_id.clone();
        registry
            .publish_candidate(item, registry.revision())
            .unwrap();
        registry
            .mark_evaluated(&id, registry.revision(), "test")
            .unwrap();
        let revision = registry.revision();
        let index_path = directory.path().join("registry-v1.json");
        std::fs::remove_file(&index_path).unwrap();
        std::fs::create_dir(&index_path).unwrap();
        assert!(
            registry
                .promote(&id, LifecycleStage::Qualified, revision, "test promotion",)
                .is_err()
        );
        assert_eq!(registry.revision(), revision);
        assert_eq!(
            registry.record(&id).unwrap().stage,
            LifecycleStage::Evaluated
        );
        assert!(registry.champion("routing_candidate_utility").is_none());
    }
}
