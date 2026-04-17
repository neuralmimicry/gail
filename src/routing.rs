use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};

use crate::errors::{GailError, Result};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RoutingProfiles {
    pub version: u64,
    pub workflow_profiles: BTreeMap<String, BTreeMap<String, Vec<String>>>,
    pub keyword_tags: BTreeMap<String, Vec<String>>,
    pub provider_specialties: BTreeMap<String, Vec<String>>,
}

impl RoutingProfiles {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let path = resolve_routing_profiles_path(path)?;
        let raw = fs::read_to_string(&path).map_err(|error| {
            GailError::invalid_config(format!(
                "failed to read routing contract {}: {}",
                path.display(),
                error
            ))
        })?;
        let mut profiles: Self = serde_json::from_str(&raw).map_err(|error| {
            GailError::invalid_config(format!(
                "failed to parse routing contract {}: {}",
                path.display(),
                error
            ))
        })?;
        profiles.normalize();
        Ok(profiles)
    }

    pub fn workflow_tags(&self, workflow: &str, role: &str, text: &str) -> HashSet<String> {
        let mut tags = HashSet::new();
        let workflow = normalize_key(workflow, "general");
        let role = normalize_key(role, "general");
        if let Some(entries) = self.workflow_profiles.get(&workflow) {
            if let Some(values) = entries.get("general") {
                tags.extend(values.iter().cloned());
            }
            if let Some(values) = entries.get(&role) {
                tags.extend(values.iter().cloned());
            }
        }
        let lowered = text.to_ascii_lowercase();
        for (tag, keywords) in &self.keyword_tags {
            if keywords.iter().any(|keyword| lowered.contains(keyword)) {
                tags.insert(tag.clone());
            }
        }
        tags
    }

    pub fn base_provider_specialties(&self, provider_type: &str) -> HashSet<String> {
        self.provider_specialties
            .get(&normalize_key(provider_type, ""))
            .map(|items| items.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn normalize(&mut self) {
        self.version = self.version.max(1);
        self.workflow_profiles =
            normalize_nested_profiles(std::mem::take(&mut self.workflow_profiles));
        self.keyword_tags = normalize_string_map(std::mem::take(&mut self.keyword_tags));
        self.provider_specialties =
            normalize_string_map(std::mem::take(&mut self.provider_specialties));
    }
}

pub fn default_routing_profiles() -> &'static RoutingProfiles {
    static DEFAULT_ROUTING_PROFILES: OnceCell<RoutingProfiles> = OnceCell::new();
    DEFAULT_ROUTING_PROFILES.get_or_init(|| {
        RoutingProfiles::load(None)
            .unwrap_or_else(|error| panic!("failed to load Gail routing contract: {error}"))
    })
}

pub fn resolve_routing_profiles_path(explicit: Option<&Path>) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = explicit {
        candidates.push(path.to_path_buf());
    }
    if let Ok(env_path) = std::env::var("GAIL_ROUTING_PROFILES_PATH") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            candidates.push(PathBuf::from(trimmed));
        }
    }
    candidates.push(PathBuf::from("config/ai-routing-profiles.json"));
    candidates.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("config/ai-routing-profiles.json"));

    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    Err(GailError::invalid_config(
        "no Gail routing-profiles contract file could be resolved",
    ))
}

fn normalize_nested_profiles(
    input: BTreeMap<String, BTreeMap<String, Vec<String>>>,
) -> BTreeMap<String, BTreeMap<String, Vec<String>>> {
    let mut output = BTreeMap::new();
    for (workflow, role_profiles) in input {
        let workflow_key = normalize_key(&workflow, "");
        if workflow_key.is_empty() {
            continue;
        }
        let mut normalized_roles = BTreeMap::new();
        for (role, tags) in role_profiles {
            let role_key = normalize_key(&role, "general");
            normalized_roles.insert(role_key, normalize_values(tags));
        }
        output.insert(workflow_key, normalized_roles);
    }
    output
}

fn normalize_string_map(input: BTreeMap<String, Vec<String>>) -> BTreeMap<String, Vec<String>> {
    let mut output = BTreeMap::new();
    for (key, values) in input {
        let key = normalize_key(&key, "");
        if key.is_empty() {
            continue;
        }
        output.insert(key, normalize_values(values));
    }
    output
}

fn normalize_values(values: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    values
        .into_iter()
        .map(|value| normalize_key(&value, ""))
        .filter(|value| !value.is_empty())
        .filter(|value| seen.insert(value.clone()))
        .collect()
}

fn normalize_key(value: &str, default: &str) -> String {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        default.to_string()
    } else {
        normalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_contract_contains_expected_workflow_tags() {
        let tags = default_routing_profiles().workflow_tags(
            "assistant_requirements",
            "assistant",
            "Need JSON schema for a reading quiz",
        );
        assert!(tags.contains("json"));
        assert!(tags.contains("requirements"));
    }

    #[test]
    fn default_contract_matches_refiner_copy_when_available() {
        let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/ai-routing-profiles.json");
        let sibling = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../rag_demo/config/ai-routing-profiles.json");
        if !sibling.is_file() {
            return;
        }
        let left = fs::read_to_string(local).expect("local contract");
        let right = fs::read_to_string(sibling).expect("sibling contract");
        assert_eq!(left, right);
    }
}
