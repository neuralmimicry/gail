//! Encrypted, runtime-managed provider credentials for model intake.
//!
//! Credentials are encrypted before they reach durable storage. The store is
//! shared by Gail API and trainer replicas through the configured data volume;
//! an advisory lock and atomic replacement keep concurrent updates coherent.

use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use fs2::FileExt;
use ring::{
    aead::{AES_256_GCM, Aad, LessSafeKey, Nonce, UnboundKey},
    rand::{SecureRandom, SystemRandom},
};
use serde::{Deserialize, Serialize};
use tokio::task;
use zeroize::Zeroize;

const VAULT_VERSION: u32 = 1;
const NONCE_BYTES: usize = 12;

/// Credentials supplied by an administrator. Values are cleared when dropped.
#[derive(Default, Deserialize, Serialize)]
pub struct ProviderCredentials {
    pub username: Option<String>,
    pub password: Option<String>,
    pub token: Option<String>,
}

impl ProviderCredentials {
    pub fn has_any_value(&self) -> bool {
        [
            self.username.as_deref(),
            self.password.as_deref(),
            self.token.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|value| !value.trim().is_empty())
    }

    pub fn has_token(&self) -> bool {
        self.token
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
    }
}

impl Drop for ProviderCredentials {
    fn drop(&mut self) {
        if let Some(value) = &mut self.username {
            value.zeroize();
        }
        if let Some(value) = &mut self.password {
            value.zeroize();
        }
        if let Some(value) = &mut self.token {
            value.zeroize();
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ProviderCredentialSummary {
    pub provider: String,
    pub username_configured: bool,
    pub password_configured: bool,
    pub token_configured: bool,
    pub updated_at: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct EncryptedCredential {
    nonce: String,
    ciphertext: String,
    username_configured: bool,
    password_configured: bool,
    token_configured: bool,
    updated_at: u64,
}

#[derive(Default, Deserialize, Serialize)]
struct VaultFile {
    version: u32,
    entries: BTreeMap<String, EncryptedCredential>,
}

/// Shared encrypted store for third-party model-provider credentials.
pub struct ModelCredentialVault {
    path: PathBuf,
    key: [u8; 32],
}

impl ModelCredentialVault {
    /// Opens the durable store with a 32-byte hexadecimal encryption key.
    pub async fn open(path: impl Into<PathBuf>, key_hex: &str) -> anyhow::Result<Self> {
        let mut key_bytes = hex::decode(key_hex.trim())?;
        if key_bytes.len() != 32 {
            key_bytes.zeroize();
            anyhow::bail!("credential encryption key must be 64 hexadecimal characters");
        }
        let mut key = [0_u8; 32];
        key.copy_from_slice(&key_bytes);
        key_bytes.zeroize();
        let vault = Self {
            path: path.into(),
            key,
        };
        let path = vault.path.clone();
        let key = vault.key;
        task::spawn_blocking(move || {
            let mut key = key;
            let result = with_locked_file(&path, false, |_lock| {
                let stored = read_vault(&path)?;
                for (provider, record) in &stored.entries {
                    let _ = decrypt_record(&key, provider, record)?;
                }
                Ok(())
            });
            key.zeroize();
            result
        })
        .await??;
        Ok(vault)
    }

    /// Returns metadata only; secret material is never included in summaries.
    pub async fn list(&self) -> anyhow::Result<Vec<ProviderCredentialSummary>> {
        let path = self.path.clone();
        let key = self.key;
        task::spawn_blocking(move || {
            let mut key = key;
            let result = with_locked_file(&path, false, |_lock| {
                let stored = read_vault(&path)?;
                Ok(stored
                    .entries
                    .into_iter()
                    .map(|(provider, record)| ProviderCredentialSummary {
                        provider,
                        username_configured: record.username_configured,
                        password_configured: record.password_configured,
                        token_configured: record.token_configured,
                        updated_at: record.updated_at,
                    })
                    .collect())
            });
            key.zeroize();
            result
        })
        .await?
    }

    /// Saves or replaces one provider's credentials without returning them.
    pub async fn put(
        &self,
        provider: String,
        mut credentials: ProviderCredentials,
    ) -> anyhow::Result<()> {
        let provider = normalize_provider(&provider)?;
        if !credentials.has_any_value() {
            return Err(anyhow::anyhow!("at least one credential field is required"));
        }
        let username_configured = credentials
            .username
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty());
        let password_configured = credentials
            .password
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty());
        let token_configured = credentials.has_token();
        let path = self.path.clone();
        let key = self.key;
        task::spawn_blocking(move || {
            let mut key = key;
            let result = with_locked_file(&path, true, |_lock| {
                let mut stored = read_vault(&path)?;
                let mut plaintext = serde_json::to_vec(&credentials)?;
                let encrypted_record = encrypt_record(
                    &key,
                    &provider,
                    &mut plaintext,
                    username_configured,
                    password_configured,
                    token_configured,
                );
                plaintext.zeroize();
                let record = encrypted_record?;
                stored.version = VAULT_VERSION;
                stored.entries.insert(provider, record);
                write_vault_atomic(&path, &stored)
            });
            key.zeroize();
            credentials.zeroize_fields();
            result
        })
        .await?
    }

    /// Removes a provider credential after the encrypted-file update succeeds.
    pub async fn delete(&self, provider: String) -> anyhow::Result<bool> {
        let provider = normalize_provider(&provider)?;
        let path = self.path.clone();
        let key = self.key;
        task::spawn_blocking(move || {
            let mut key = key;
            let result = with_locked_file(&path, true, |_lock| {
                let mut stored = read_vault(&path)?;
                let removed = stored.entries.remove(&provider).is_some();
                if removed {
                    write_vault_atomic(&path, &stored)?;
                }
                Ok(removed)
            });
            key.zeroize();
            result
        })
        .await?
    }

    /// Loads a provider credential for an authorised model-download process.
    pub async fn get(&self, provider: String) -> anyhow::Result<Option<ProviderCredentials>> {
        let provider = normalize_provider(&provider)?;
        let path = self.path.clone();
        let key = self.key;
        task::spawn_blocking(move || {
            let mut key = key;
            let result = with_locked_file(&path, false, |_lock| {
                let stored = read_vault(&path)?;
                stored
                    .entries
                    .get(&provider)
                    .map(|record| decrypt_record(&key, &provider, record))
                    .transpose()
            });
            key.zeroize();
            result
        })
        .await?
    }
}

impl Drop for ModelCredentialVault {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

impl ProviderCredentials {
    fn zeroize_fields(&mut self) {
        if let Some(value) = &mut self.username {
            value.zeroize();
        }
        if let Some(value) = &mut self.password {
            value.zeroize();
        }
        if let Some(value) = &mut self.token {
            value.zeroize();
        }
    }
}

fn normalize_provider(provider: &str) -> anyhow::Result<String> {
    let normalized = provider.trim().to_ascii_lowercase();
    if normalized.is_empty()
        || normalized.len() > 64
        || !normalized
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"-_".contains(&byte))
    {
        return Err(anyhow::anyhow!(
            "provider must contain only letters, digits, '-' or '_'"
        ));
    }
    Ok(normalized)
}

fn encryption_key(key: &[u8; 32]) -> anyhow::Result<LessSafeKey> {
    let unbound = UnboundKey::new(&AES_256_GCM, key)
        .map_err(|_| anyhow::anyhow!("could not initialise credential encryption"))?;
    Ok(LessSafeKey::new(unbound))
}

fn encrypt_record(
    key: &[u8; 32],
    provider: &str,
    plaintext: &mut Vec<u8>,
    username_configured: bool,
    password_configured: bool,
    token_configured: bool,
) -> anyhow::Result<EncryptedCredential> {
    let mut nonce_bytes = [0_u8; NONCE_BYTES];
    SystemRandom::new()
        .fill(&mut nonce_bytes)
        .map_err(|_| anyhow::anyhow!("could not generate credential encryption nonce"))?;
    encryption_key(key)?
        .seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce_bytes),
            Aad::from(provider.as_bytes()),
            plaintext,
        )
        .map_err(|_| anyhow::anyhow!("could not encrypt provider credentials"))?;
    Ok(EncryptedCredential {
        nonce: STANDARD.encode(nonce_bytes),
        ciphertext: STANDARD.encode(plaintext.as_slice()),
        username_configured,
        password_configured,
        token_configured,
        updated_at: now_epoch_seconds(),
    })
}

fn decrypt_record(
    key: &[u8; 32],
    provider: &str,
    record: &EncryptedCredential,
) -> anyhow::Result<ProviderCredentials> {
    let nonce_bytes = STANDARD.decode(&record.nonce)?;
    let nonce: [u8; NONCE_BYTES] = nonce_bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid credential encryption nonce"))?;
    let mut ciphertext = STANDARD.decode(&record.ciphertext)?;
    let plaintext = encryption_key(key)?
        .open_in_place(
            Nonce::assume_unique_for_key(nonce),
            Aad::from(provider.as_bytes()),
            &mut ciphertext,
        )
        .map_err(|_| anyhow::anyhow!("provider credential decryption failed"))?;
    let parsed_credentials = serde_json::from_slice(plaintext);
    ciphertext.zeroize();
    Ok(parsed_credentials?)
}

fn now_epoch_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn lock_path(path: &Path) -> anyhow::Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
    let name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("credential path has no filename"))?
        .to_string_lossy();
    Ok(parent.join(format!(".{name}.lock")))
}

fn with_locked_file<T>(
    path: &Path,
    exclusive: bool,
    operation: impl FnOnce(&File) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let lock_path = lock_path(path)?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .open(lock_path)?;
    lock.set_permissions(fs::Permissions::from_mode(0o600))?;
    if exclusive {
        lock.lock_exclusive()?;
    } else {
        lock.lock_shared()?;
    }
    operation(&lock)
}

fn read_vault(path: &Path) -> anyhow::Result<VaultFile> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(VaultFile {
                version: VAULT_VERSION,
                entries: BTreeMap::new(),
            });
        }
        Err(error) => return Err(error.into()),
    };
    let mode = file.metadata()?.permissions().mode() & 0o777;
    if mode != 0o600 {
        anyhow::bail!("credential store permissions must be 0600");
    }
    let mut body = String::new();
    file.read_to_string(&mut body)?;
    let stored: VaultFile = serde_json::from_str(&body)?;
    if stored.version != VAULT_VERSION {
        anyhow::bail!("unsupported credential store version");
    }
    Ok(stored)
}

fn write_vault_atomic(path: &Path, vault: &VaultFile) -> anyhow::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
    let body = serde_json::to_vec(vault)?;
    let temp_path = parent.join(format!(".model-credentials-{}.tmp", uuid::Uuid::new_v4()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temp_path)?;
    file.write_all(&body)?;
    file.sync_all()?;
    file.set_permissions(fs::Permissions::from_mode(0o600))?;
    fs::rename(&temp_path, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use tempfile::tempdir;

    use super::{ModelCredentialVault, ProviderCredentials};

    const KEY: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

    #[tokio::test]
    async fn provider_credentials_are_encrypted_and_round_trip_without_secret_readback_api() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("credentials.enc");
        let vault = ModelCredentialVault::open(&path, KEY).await.expect("vault");
        vault
            .put(
                "HuggingFace".to_string(),
                ProviderCredentials {
                    username: Some("model-user".to_string()),
                    password: Some("private-password".to_string()),
                    token: Some("hf_secret_token".to_string()),
                },
            )
            .await
            .expect("save credentials");

        let stored = std::fs::read_to_string(&path).expect("stored file");
        assert!(!stored.contains("model-user"));
        assert!(!stored.contains("private-password"));
        assert!(!stored.contains("hf_secret_token"));
        assert_eq!(
            std::fs::metadata(&path)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );

        let summaries = vault.list().await.expect("summaries");
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].provider, "huggingface");
        assert!(summaries[0].token_configured);

        let loaded = vault
            .get("huggingface".to_string())
            .await
            .expect("read credential")
            .expect("credential exists");
        assert_eq!(loaded.username.as_deref(), Some("model-user"));
        assert_eq!(loaded.password.as_deref(), Some("private-password"));
        assert_eq!(loaded.token.as_deref(), Some("hf_secret_token"));
    }

    #[tokio::test]
    async fn reopening_with_a_different_key_fails_closed() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("credentials.enc");
        let vault = ModelCredentialVault::open(&path, KEY).await.expect("vault");
        vault
            .put(
                "huggingface".to_string(),
                ProviderCredentials {
                    username: None,
                    password: None,
                    token: Some("hf_secret_token".to_string()),
                },
            )
            .await
            .expect("save credentials");
        let result = ModelCredentialVault::open(
            &path,
            "ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100",
        )
        .await;
        let error = result.err().expect("wrong key must fail");
        assert!(error.to_string().contains("decryption failed"));
    }

    #[tokio::test]
    async fn deleting_a_provider_removes_it_from_the_shared_store() {
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("credentials.enc");
        let vault = ModelCredentialVault::open(&path, KEY).await.expect("vault");
        vault
            .put(
                "huggingface".to_string(),
                ProviderCredentials {
                    username: None,
                    password: None,
                    token: Some("hf_secret_token".to_string()),
                },
            )
            .await
            .expect("save credentials");
        assert!(
            vault
                .delete("huggingface".to_string())
                .await
                .expect("delete credential")
        );
        assert!(
            vault
                .get("huggingface".to_string())
                .await
                .expect("read credential")
                .is_none()
        );
    }
}
