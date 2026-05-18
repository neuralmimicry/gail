use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-env-changed=GAIL_VERSION");
    println!("cargo:rerun-if-env-changed=GAIL_VERSION_MAJOR");
    println!("cargo:rerun-if-env-changed=GAIL_VERSION_MINOR");
    println!("cargo:rerun-if-env-changed=GAIL_BUILD_NUMBER");
    println!("cargo:rerun-if-env-changed=BUILD_NUMBER");
    println!("cargo:rerun-if-env-changed=GAIL_GIT_COMMIT");
    println!("cargo:rerun-if-env-changed=GIT_COMMIT");

    if let Some(git_dir) = git_dir() {
        register_git_rerun_paths(&git_dir);
    }

    let release_version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.2.0".to_string());
    let (mut major, mut minor) = parse_major_minor(&release_version);
    if let Some(value) = env_int(&["GAIL_VERSION_MAJOR"]) {
        major = value;
    }
    if let Some(value) = env_int(&["GAIL_VERSION_MINOR"]) {
        minor = value;
    }

    let explicit_version = env_first(&["GAIL_VERSION"]);
    let mut build_number = env_int(&["GAIL_BUILD_NUMBER", "BUILD_NUMBER"]);
    let mut source = if build_number.is_some() || explicit_version.is_some() {
        "env"
    } else {
        "default"
    };

    if build_number.is_none()
        && let Some(value) =
            git_output(&["rev-list", "--count", "HEAD"]).and_then(|raw| raw.parse::<u64>().ok())
    {
        build_number = Some(value);
        source = "git";
    }

    let build_number = build_number.unwrap_or(0);
    let mut build_version = format!("{major}.{minor}.{build_number:04}");
    if let Some(value) = explicit_version
        .as_deref()
        .filter(|value| is_build_version(value))
    {
        build_version = value.to_string();
        source = "env";
    }

    let commit = env_first(&["GIT_COMMIT", "GAIL_GIT_COMMIT"])
        .or_else(|| git_output(&["rev-parse", "HEAD"]))
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GAIL_BUILD_VERSION={build_version}");
    println!("cargo:rustc-env=GAIL_RELEASE_VERSION={release_version}");
    println!("cargo:rustc-env=GAIL_BUILD_NUMBER={build_number}");
    println!("cargo:rustc-env=GAIL_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=GAIL_VERSION_SOURCE={source}");
}

fn env_first(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        let value = env::var(name).ok()?;
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn env_int(names: &[&str]) -> Option<u64> {
    env_first(names).and_then(|value| value.parse::<u64>().ok())
}

fn parse_major_minor(value: &str) -> (u64, u64) {
    let parts: Vec<u64> = value
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse::<u64>().ok())
        .collect();
    let major = parts.first().copied().unwrap_or(0);
    let minor = parts.get(1).copied().unwrap_or(1);
    (major, minor)
}

fn is_build_version(value: &str) -> bool {
    let mut parts = value.split('.');
    let Some(major) = parts.next() else {
        return false;
    };
    let Some(minor) = parts.next() else {
        return false;
    };
    let Some(build) = parts.next() else {
        return false;
    };
    parts.next().is_none()
        && major.chars().all(|ch| ch.is_ascii_digit())
        && minor.chars().all(|ch| ch.is_ascii_digit())
        && build.len() >= 4
        && build.chars().all(|ch| ch.is_ascii_digit())
}

fn git_output(args: &[&str]) -> Option<String> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    let output = Command::new("git")
        .args(args)
        .current_dir(&manifest_dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn git_dir() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").ok()?);
    let raw = git_output(&["rev-parse", "--git-dir"])?;
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        Some(path)
    } else {
        Some(manifest_dir.join(path))
    }
}

fn register_git_rerun_paths(git_dir: &Path) {
    let head = git_dir.join("HEAD");
    println!("cargo:rerun-if-changed={}", head.display());
    let Ok(head_text) = fs::read_to_string(&head) else {
        return;
    };
    let Some(reference) = head_text.strip_prefix("ref: ").map(str::trim) else {
        return;
    };
    let ref_path = git_dir.join(reference);
    println!("cargo:rerun-if-changed={}", ref_path.display());
}
