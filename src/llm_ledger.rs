//! Durable LLM interaction ledger.
//!
//! Gail appends interaction records to a local JSONL file and, when configured,
//! mirrors the same records into Postgres for downstream mirror/training
//! workers.

use std::{
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    fs::{self, OpenOptions},
    io::AsyncWriteExt,
    sync::mpsc,
    time::timeout,
};
use tokio_postgres::NoTls;

use crate::config::GailConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmLedgerRecord {
    pub request_id: String,
    pub conversation_id: String,
    pub workflow: String,
    pub role: String,
    pub provider_requested: Option<String>,
    pub model_requested: Option<String>,
    pub provider_resolved: Option<String>,
    pub model_resolved: Option<String>,
    pub request_category: Option<String>,
    pub system_prompt: Option<String>,
    pub prompt_text: String,
    pub response_text: Option<String>,
    pub message_roles: Vec<String>,
    pub status: String,
    pub error_text: Option<String>,
    pub latency_ms: Option<u64>,
    pub usage: Option<Value>,
    pub raw: Option<Value>,
    pub metadata: Option<Value>,
    pub created_ts: f64,
}

impl Default for LlmLedgerRecord {
    fn default() -> Self {
        Self {
            request_id: String::new(),
            conversation_id: String::new(),
            workflow: "general".to_string(),
            role: "general".to_string(),
            provider_requested: None,
            model_requested: None,
            provider_resolved: None,
            model_resolved: None,
            request_category: None,
            system_prompt: None,
            prompt_text: String::new(),
            response_text: None,
            message_roles: Vec::new(),
            status: "ok".to_string(),
            error_text: None,
            latency_ms: None,
            usage: None,
            raw: None,
            metadata: None,
            created_ts: now_ts(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LedgerInteraction {
    // Postgres-backed interaction row consumed by mirror/trainer workers.
    // Field names intentionally match `gail_llm_interactions` columns so replay
    // workers can rebuild bridge/training payloads without lossy transforms.
    pub id: i64,
    pub request_id: String,
    pub conversation_id: String,
    pub workflow: String,
    pub role: String,
    pub provider_requested: Option<String>,
    pub model_requested: Option<String>,
    pub provider_resolved: Option<String>,
    pub model_resolved: Option<String>,
    pub request_category: Option<String>,
    pub system_prompt: Option<String>,
    pub prompt_text: String,
    pub response_text: Option<String>,
    pub message_roles: Vec<String>,
    pub status: String,
    pub error_text: Option<String>,
    pub latency_ms: Option<u64>,
    pub usage: Option<Value>,
    pub raw: Option<Value>,
}

#[derive(Clone)]
pub struct LlmLedger {
    queue_tx: mpsc::Sender<LlmLedgerRecord>,
    queue_timeout: Duration,
    max_prompt_chars: usize,
    max_response_chars: usize,
}

impl LlmLedger {
    pub async fn from_config(config: &GailConfig) -> Option<Self> {
        if !config.llm_ledger.enabled {
            return None;
        }
        let path = PathBuf::from(config.storage.llm_ledger_path.clone());
        let postgres_dsn = config.storage.postgres_dsn.clone();
        if let Some(dsn) = postgres_dsn.as_deref()
            && let Err(error) = initialize_schema(dsn).await
        {
            tracing::warn!(
                error = %error,
                "failed to initialise Postgres LLM ledger schema; Gail will continue with file-only ledger"
            );
        }
        let queue_capacity = config.llm_ledger.queue_capacity;
        let queue_timeout = Duration::from_millis(config.llm_ledger.enqueue_timeout_ms);
        let max_prompt_chars = config.llm_ledger.max_prompt_chars;
        let max_response_chars = config.llm_ledger.max_response_chars;
        let (queue_tx, mut queue_rx) = mpsc::channel(queue_capacity);
        tokio::spawn(async move {
            while let Some(record) = queue_rx.recv().await {
                if let Err(error) = persist_file_record(&path, &record).await {
                    tracing::warn!(
                        error = %error,
                        path = %path.display(),
                        "failed to persist Gail LLM ledger record to file"
                    );
                }
                if let Some(dsn) = postgres_dsn.as_deref()
                    && let Err(error) = persist_postgres_record(dsn, &record).await
                {
                    tracing::warn!(
                        error = %error,
                        "failed to persist Gail LLM ledger record to Postgres"
                    );
                }
            }
        });
        Some(Self {
            queue_tx,
            queue_timeout,
            max_prompt_chars,
            max_response_chars,
        })
    }

    pub async fn record(&self, mut record: LlmLedgerRecord) {
        record.prompt_text = truncate_chars(&record.prompt_text, self.max_prompt_chars);
        record.response_text = record
            .response_text
            .as_deref()
            .map(|value| truncate_chars(value, self.max_response_chars))
            .filter(|value| !value.trim().is_empty());
        if record.conversation_id.trim().is_empty() {
            record.conversation_id = record.request_id.clone();
        }
        if record.workflow.trim().is_empty() {
            record.workflow = "general".to_string();
        }
        if record.role.trim().is_empty() {
            record.role = "general".to_string();
        }
        if record.created_ts <= 0.0 {
            record.created_ts = now_ts();
        }
        let send = self.queue_tx.send(record);
        match timeout(self.queue_timeout, send).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(error = %error, "LLM ledger queue is closed; dropping record");
            }
            Err(_) => {
                tracing::warn!("LLM ledger queue is saturated; dropping record");
            }
        }
    }
}

pub async fn initialize_schema(dsn: &str) -> Result<(), tokio_postgres::Error> {
    let client = connect_client(dsn).await?;
    client
        .batch_execute(
            r#"
            SET client_min_messages TO WARNING;
            CREATE TABLE IF NOT EXISTS gail_llm_interactions (
                id BIGSERIAL PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL,
                workflow TEXT NOT NULL,
                role TEXT NOT NULL,
                provider_requested TEXT,
                model_requested TEXT,
                provider_resolved TEXT,
                model_resolved TEXT,
                request_category TEXT,
                system_prompt TEXT,
                prompt_text TEXT NOT NULL,
                response_text TEXT,
                message_roles JSONB NOT NULL DEFAULT '[]'::jsonb,
                status TEXT NOT NULL DEFAULT 'ok',
                error_text TEXT,
                latency_ms BIGINT,
                usage JSONB,
                raw JSONB,
                metadata JSONB,
                created_ts DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                mirror_attempts INTEGER NOT NULL DEFAULT 0,
                mirror_status TEXT,
                mirror_error TEXT,
                next_mirror_at TIMESTAMPTZ,
                mirrored_at TIMESTAMPTZ,
                train_attempts INTEGER NOT NULL DEFAULT 0,
                train_status TEXT,
                train_error TEXT,
                next_train_at TIMESTAMPTZ,
                trained_at TIMESTAMPTZ,
                training_snapshot TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_gail_llm_interactions_mirror
                ON gail_llm_interactions (mirrored_at, next_mirror_at, id);
            CREATE INDEX IF NOT EXISTS idx_gail_llm_interactions_train
                ON gail_llm_interactions (trained_at, next_train_at, id);
            CREATE INDEX IF NOT EXISTS idx_gail_llm_interactions_created_at
                ON gail_llm_interactions (created_at DESC);
            "#,
        )
        .await
}

pub async fn fetch_pending_mirror(
    dsn: &str,
    batch_size: usize,
) -> Result<Vec<LedgerInteraction>, tokio_postgres::Error> {
    let client = connect_client(dsn).await?;
    // Mirror worker replays any row not marked mirrored yet where either prompt
    // or response text is available and the retry schedule is due.
    let rows = client
        .query(
            r#"
            SELECT
                id,
                request_id,
                conversation_id,
                workflow,
                role,
                provider_requested,
                model_requested,
                provider_resolved,
                model_resolved,
                request_category,
                system_prompt,
                prompt_text,
                response_text,
                message_roles,
                status,
                error_text,
                latency_ms,
                usage,
                raw
            FROM gail_llm_interactions
            WHERE mirrored_at IS NULL
              AND (next_mirror_at IS NULL OR next_mirror_at <= now())
              AND (
                    COALESCE(prompt_text, '') <> ''
                 OR COALESCE(response_text, '') <> ''
              )
            ORDER BY id ASC
            LIMIT $1
            "#,
            &[&(batch_size as i64)],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| LedgerInteraction {
            id: row.get("id"),
            request_id: row.get("request_id"),
            conversation_id: row.get("conversation_id"),
            workflow: row.get("workflow"),
            role: row.get("role"),
            provider_requested: row.get("provider_requested"),
            model_requested: row.get("model_requested"),
            provider_resolved: row.get("provider_resolved"),
            model_resolved: row.get("model_resolved"),
            request_category: row.get("request_category"),
            system_prompt: row.get("system_prompt"),
            prompt_text: row.get("prompt_text"),
            response_text: row.get("response_text"),
            message_roles: json_array_to_strings(row.get("message_roles")),
            status: row.get("status"),
            error_text: row.get("error_text"),
            latency_ms: row
                .get::<_, Option<i64>>("latency_ms")
                .and_then(|value| value.try_into().ok()),
            usage: row.get("usage"),
            raw: row.get("raw"),
        })
        .collect())
}

pub async fn mark_mirror_success(
    dsn: &str,
    id: i64,
    status: &str,
) -> Result<(), tokio_postgres::Error> {
    let client = connect_client(dsn).await?;
    // A successful replay closes mirror scheduling for this interaction.
    client
        .execute(
            r#"
            UPDATE gail_llm_interactions
            SET
                mirror_attempts = mirror_attempts + 1,
                mirror_status = $2,
                mirror_error = NULL,
                next_mirror_at = NULL,
                mirrored_at = now()
            WHERE id = $1
            "#,
            &[&id, &status],
        )
        .await?;
    Ok(())
}

pub async fn mark_mirror_retry(
    dsn: &str,
    id: i64,
    error: &str,
    max_attempts: u32,
    retry_backoff_seconds: u64,
) -> Result<(), tokio_postgres::Error> {
    let client = connect_client(dsn).await?;
    // Retry state is bounded; once attempts are exhausted the row is marked
    // failed and no further mirror scheduling is attempted.
    client
        .execute(
            r#"
            UPDATE gail_llm_interactions
            SET
                mirror_attempts = mirror_attempts + 1,
                mirror_status = CASE
                    WHEN mirror_attempts + 1 >= $2 THEN 'failed'
                    ELSE 'retry'
                END,
                mirror_error = $3,
                next_mirror_at = CASE
                    WHEN mirror_attempts + 1 >= $2 THEN NULL
                    ELSE now() + make_interval(secs => $4::int)
                END,
                mirrored_at = CASE
                    WHEN mirror_attempts + 1 >= $2 THEN now()
                    ELSE NULL
                END
            WHERE id = $1
            "#,
            &[
                &id,
                &(max_attempts as i32),
                &truncate_chars(error, 4000),
                &(retry_backoff_seconds as i32),
            ],
        )
        .await?;
    Ok(())
}

pub async fn fetch_pending_training(
    dsn: &str,
    batch_size: usize,
    include_degraded: bool,
) -> Result<Vec<LedgerInteraction>, tokio_postgres::Error> {
    let client = connect_client(dsn).await?;
    let status_filter = if include_degraded {
        "status IN ('ok', 'degraded')"
    } else {
        "status = 'ok'"
    };
    let query = format!(
        r#"
        SELECT
            id,
            request_id,
            conversation_id,
            workflow,
            role,
            provider_requested,
            model_requested,
            provider_resolved,
            model_resolved,
            request_category,
            system_prompt,
            prompt_text,
            response_text,
            message_roles,
            status,
            error_text,
            latency_ms,
            usage,
            raw
        FROM gail_llm_interactions
        WHERE trained_at IS NULL
          AND (next_train_at IS NULL OR next_train_at <= now())
          AND COALESCE(response_text, '') <> ''
          AND {status_filter}
        ORDER BY id ASC
        LIMIT $1
        "#
    );
    let rows = client
        .query(query.as_str(), &[&(batch_size as i64)])
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| LedgerInteraction {
            id: row.get("id"),
            request_id: row.get("request_id"),
            conversation_id: row.get("conversation_id"),
            workflow: row.get("workflow"),
            role: row.get("role"),
            provider_requested: row.get("provider_requested"),
            model_requested: row.get("model_requested"),
            provider_resolved: row.get("provider_resolved"),
            model_resolved: row.get("model_resolved"),
            request_category: row.get("request_category"),
            system_prompt: row.get("system_prompt"),
            prompt_text: row.get("prompt_text"),
            response_text: row.get("response_text"),
            message_roles: json_array_to_strings(row.get("message_roles")),
            status: row.get("status"),
            error_text: row.get("error_text"),
            latency_ms: row
                .get::<_, Option<i64>>("latency_ms")
                .and_then(|value| value.try_into().ok()),
            usage: row.get("usage"),
            raw: row.get("raw"),
        })
        .collect())
}

pub async fn mark_training_success(
    dsn: &str,
    ids: &[i64],
    snapshot: &str,
    status: &str,
) -> Result<(), tokio_postgres::Error> {
    if ids.is_empty() {
        return Ok(());
    }
    let client = connect_client(dsn).await?;
    client
        .execute(
            r#"
            UPDATE gail_llm_interactions
            SET
                train_attempts = train_attempts + 1,
                train_status = $2,
                train_error = NULL,
                next_train_at = NULL,
                trained_at = now(),
                training_snapshot = $3
            WHERE id = ANY($1)
            "#,
            &[&ids, &status, &snapshot],
        )
        .await?;
    Ok(())
}

pub async fn mark_training_retry(
    dsn: &str,
    id: i64,
    error: &str,
    max_attempts: u32,
    retry_backoff_seconds: u64,
) -> Result<(), tokio_postgres::Error> {
    let client = connect_client(dsn).await?;
    client
        .execute(
            r#"
            UPDATE gail_llm_interactions
            SET
                train_attempts = train_attempts + 1,
                train_status = CASE
                    WHEN train_attempts + 1 >= $2 THEN 'failed'
                    ELSE 'retry'
                END,
                train_error = $3,
                next_train_at = CASE
                    WHEN train_attempts + 1 >= $2 THEN NULL
                    ELSE now() + make_interval(secs => $4::int)
                END,
                trained_at = CASE
                    WHEN train_attempts + 1 >= $2 THEN now()
                    ELSE NULL
                END
            WHERE id = $1
            "#,
            &[
                &id,
                &(max_attempts as i32),
                &truncate_chars(error, 4000),
                &(retry_backoff_seconds as i32),
            ],
        )
        .await?;
    Ok(())
}

async fn connect_client(dsn: &str) -> Result<tokio_postgres::Client, tokio_postgres::Error> {
    let (client, connection) = tokio_postgres::connect(dsn, NoTls).await?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            tracing::debug!(error = %error, "LLM ledger Postgres connection closed");
        }
    });
    Ok(client)
}

async fn persist_file_record(path: &Path, record: &LlmLedgerRecord) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    let mut line = serde_json::to_string(record).unwrap_or_else(|_| "{}".to_string());
    line.push('\n');
    file.write_all(line.as_bytes()).await?;
    file.flush().await?;
    Ok(())
}

async fn persist_postgres_record(
    dsn: &str,
    record: &LlmLedgerRecord,
) -> Result<(), tokio_postgres::Error> {
    let client = connect_client(dsn).await?;
    let message_roles = serde_json::to_value(&record.message_roles).unwrap_or_else(|_| json!([]));
    let latency_ms = record
        .latency_ms
        .map(|value| value.min(i64::MAX as u64) as i64);
    client
        .execute(
            r#"
            INSERT INTO gail_llm_interactions (
                request_id,
                conversation_id,
                workflow,
                role,
                provider_requested,
                model_requested,
                provider_resolved,
                model_resolved,
                request_category,
                system_prompt,
                prompt_text,
                response_text,
                message_roles,
                status,
                error_text,
                latency_ms,
                usage,
                raw,
                metadata,
                created_ts
            ) VALUES (
                $1,  $2,  $3,  $4,  $5,
                $6,  $7,  $8,  $9,  $10,
                $11, $12, $13, $14, $15,
                $16, $17, $18, $19, $20
            )
            ON CONFLICT (request_id) DO UPDATE SET
                conversation_id = EXCLUDED.conversation_id,
                workflow = EXCLUDED.workflow,
                role = EXCLUDED.role,
                provider_requested = EXCLUDED.provider_requested,
                model_requested = EXCLUDED.model_requested,
                provider_resolved = EXCLUDED.provider_resolved,
                model_resolved = EXCLUDED.model_resolved,
                request_category = EXCLUDED.request_category,
                system_prompt = EXCLUDED.system_prompt,
                prompt_text = EXCLUDED.prompt_text,
                response_text = EXCLUDED.response_text,
                message_roles = EXCLUDED.message_roles,
                status = EXCLUDED.status,
                error_text = EXCLUDED.error_text,
                latency_ms = EXCLUDED.latency_ms,
                usage = EXCLUDED.usage,
                raw = EXCLUDED.raw,
                metadata = EXCLUDED.metadata,
                created_ts = EXCLUDED.created_ts
            "#,
            &[
                &record.request_id,
                &record.conversation_id,
                &record.workflow,
                &record.role,
                &record.provider_requested,
                &record.model_requested,
                &record.provider_resolved,
                &record.model_resolved,
                &record.request_category,
                &record.system_prompt,
                &record.prompt_text,
                &record.response_text,
                &message_roles,
                &record.status,
                &record.error_text,
                &latency_ms,
                &record.usage,
                &record.raw,
                &record.metadata,
                &record.created_ts,
            ],
        )
        .await?;
    Ok(())
}

fn truncate_chars(value: &str, limit: usize) -> String {
    value.chars().take(limit.max(1)).collect()
}

fn json_array_to_strings(value: Option<Value>) -> Vec<String> {
    value
        .and_then(|item| item.as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| item.as_str().map(ToOwned::to_owned))
        .collect()
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}
