#!/usr/bin/env bash
# Common release helpers for Gail packaging, versioning and CI scripts.
# This file is intentionally Bash-only because the surrounding release scripts
# already rely on Bash arrays, regex support and strict-mode behaviour.

set -euo pipefail

nm_die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

nm_log() {
  printf '%s\n' "$*"
}

nm_script_dir() {
  CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[1]}")" && pwd
}

nm_repo_root() {
  local script_dir
  script_dir="$(nm_script_dir)"
  CDPATH='' cd -- "${script_dir}/.." && pwd
}

nm_require_command() {
  local cmd
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || nm_die "${cmd} is required"
  done
}

nm_run() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

nm_trim() {
  local value="${1:-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "$value"
}

nm_env_first() {
  local name value
  for name in "$@"; do
    value="$(nm_trim "${!name:-}")"
    if [[ -n "$value" ]]; then
      printf '%s\n' "$value"
      return 0
    fi
  done
  return 1
}

nm_env_int() {
  local value
  value="$(nm_env_first "$@" || true)"
  [[ -n "$value" ]] || return 1
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

nm_validate_deb_arch() {
  case "${1:-}" in
    amd64|arm64) ;;
    *) nm_die "unsupported Debian architecture: ${1:-<empty>}" ;;
  esac
}

nm_validate_deb_version() {
  [[ "${1:-}" =~ ^[0-9][0-9A-Za-z.+:~,-]*$ ]] || nm_die "invalid Debian package version: ${1:-<empty>}"
}

nm_validate_cargo_semver() {
  [[ "${1:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z][0-9A-Za-z.-]*)?(\+[0-9A-Za-z][0-9A-Za-z.-]*)?$ ]] \
    || nm_die "invalid Cargo SemVer release version: ${1:-<empty>}"
}

nm_debian_package_version() {
  local version="${1:-}" sanitized
  [[ -n "$version" ]] || nm_die "Debian package version is empty"
  sanitized="$(printf '%s' "$version" | tr '-' '~' | sed -E 's/[^A-Za-z0-9.+:~]+/./g; s/^[^A-Za-z0-9]+//; s/[^A-Za-z0-9]+$//')"
  [[ -n "$sanitized" ]] || nm_die "unable to derive Debian package version from '${version}'"
  printf '%s\n' "$sanitized"
}

nm_checksum_tool() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf 'sha256sum\n'
  elif command -v shasum >/dev/null 2>&1; then
    printf 'shasum -a 256\n'
  else
    nm_die "sha256sum or shasum is required"
  fi
}

nm_default_platform() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$os" in
    darwin) os="macos" ;;
    mingw*|msys*|cygwin*) os="windows" ;;
  esac
  case "$arch" in
    x86_64|amd64) arch="amd64" ;;
    aarch64|arm64) arch="arm64" ;;
  esac
  printf '%s-%s\n' "$os" "$arch"
}

nm_default_archive_format() {
  case "${1:-}" in
    windows*) printf 'zip\n' ;;
    *) printf 'tar.gz\n' ;;
  esac
}

nm_default_binary_suffix() {
  case "${1:-}" in
    windows*) printf '.exe\n' ;;
    *) printf '\n' ;;
  esac
}

nm_binary_dir() {
  local repo_root="$1" target_triple="${2:-}"
  if [[ -n "$target_triple" ]]; then
    printf '%s/target/%s/release\n' "$repo_root" "$target_triple"
  else
    printf '%s/target/release\n' "$repo_root"
  fi
}

nm_resolve_build_user() {
  if [[ $(id -u) -eq 0 && -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    printf '%s\n' "$SUDO_USER"
  fi
}

nm_git_output() {
  local repo_root="$1"
  shift
  git -C "$repo_root" "$@" 2>/dev/null | sed -e 's/[[:space:]]*$//' | head -n 1
}

nm_read_package_version() {
  local repo_root="$1"
  sed -n '/^\[package\]/,/^\[/ s/^version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
    "${repo_root}/Cargo.toml" | head -n 1
}

nm_compute_deb_depends() {
  if ! command -v dpkg-shlibdeps >/dev/null 2>&1; then
    printf '\n'
    return
  fi

  local work_dir output depends
  work_dir="$(mktemp -d)"
  output="$(cd "$work_dir" && dpkg-shlibdeps -O "$@" 2>/dev/null || true)"
  rm -rf "$work_dir"
  depends="$(printf '%s\n' "$output" | sed -n 's/^shlibs:Depends=//p' | tail -n 1)"
  printf '%s\n' "$depends"
}
