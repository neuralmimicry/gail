//! Helpers for producing operator-visible JSON without disclosing credentials.
//!
//! Public/status endpoints should call [`redact_sensitive_json`] on snapshots
//! derived from internal configuration. The matcher intentionally protects
//! common future credential names as well as Gail's current token/password
//! fields, so adding a secret-bearing config field does not silently expose it.

use serde_json::Value;

const SENSITIVE_KEY_PARTS: &[&str] = &[
    "access_token",
    "api_key",
    "authorization",
    "bearer",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
];

/// Recursively removes credential-shaped properties from a JSON value.
///
/// Arrays are traversed because configuration may contain provider lists. A
/// removed property is not replaced with a marker: even secret length and
/// representation should not cross the API boundary.
pub fn redact_sensitive_json(value: &mut Value) {
    match value {
        Value::Object(object) => {
            object.retain(|key, _| !is_sensitive_key(key));
            for child in object.values_mut() {
                redact_sensitive_json(child);
            }
        }
        Value::Array(values) => {
            for child in values {
                redact_sensitive_json(child);
            }
        }
        _ => {}
    }
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key.trim().to_ascii_lowercase().replace('-', "_");
    SENSITIVE_KEY_PARTS
        .iter()
        .any(|part| normalized == *part || normalized.ends_with(&format!("_{part}")))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::redact_sensitive_json;

    #[test]
    fn recursively_removes_credentials_without_hiding_safe_configuration() {
        let mut value = json!({
            "refiner_api_token": "do-not-return",
            "octobot-password": "do-not-return",
            "nested": [{
                "api_key": "do-not-return",
                "timeout_seconds": 30
            }],
            "token_discovery_enabled": true,
            "minimum_net_edge_bps": 15
        });

        redact_sensitive_json(&mut value);

        let encoded = serde_json::to_string(&value).expect("json");
        assert!(!encoded.contains("do-not-return"));
        assert!(value.get("refiner_api_token").is_none());
        assert!(value.get("octobot-password").is_none());
        assert_eq!(value["nested"][0]["timeout_seconds"], 30);
        assert_eq!(value["token_discovery_enabled"], true);
    }
}
