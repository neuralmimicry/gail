//! Compile-time source provenance exposed to health and safety gates.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildInfo {
    pub version: String,
    pub build_version: String,
    pub git_commit: String,
    pub source_tree: String,
    pub source_dirty: String,
}

pub fn current() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_version: env!("GAIL_BUILD_VERSION").to_string(),
        git_commit: env!("GAIL_GIT_COMMIT").to_string(),
        source_tree: env!("GAIL_SOURCE_TREE").to_string(),
        source_dirty: env!("GAIL_SOURCE_DIRTY").to_string(),
    }
}

/// Stable identity used to bind paper qualification to executable content.
pub fn revision() -> String {
    let info = current();
    format!(
        "{}:{}:{}",
        short(&info.git_commit),
        short(&info.source_tree),
        info.source_dirty
    )
}

fn short(value: &str) -> &str {
    value.get(..12).unwrap_or(value)
}

#[cfg(test)]
mod tests {
    #[test]
    fn revision_contains_commit_tree_and_cleanliness() {
        let revision = super::revision();
        assert_eq!(revision.split(':').count(), 3);
    }
}
