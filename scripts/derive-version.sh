#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/release-common.sh
source "${SCRIPT_DIR}/release-common.sh"

usage() {
  cat <<'USAGE'
Usage: derive-version.sh [options]

Derive Gail release, build, and tag metadata from Cargo.toml, git, and the
same environment overrides consumed by build.rs.

Options:
  --build-version     Print only the runtime build version.
  --release-version   Print only the Cargo package release version.
  --tag               Print only the version tag.
  --github-output     Append workflow outputs to $GITHUB_OUTPUT.
  --tag-prefix PREFIX Prefix for generated tags. Default: v.
  -h, --help          Show this help text.

Environment overrides:
  GAIL_VERSION, GAIL_VERSION_MAJOR, GAIL_VERSION_MINOR,
  GAIL_BUILD_NUMBER, BUILD_NUMBER, GAIL_GIT_COMMIT, GIT_COMMIT.
USAGE
}

version_numbers() {
  printf '%s' "$1" | tr -cs '0-9' ' ' | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

is_build_version() {
  [[ "${1:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]{4,}$ ]]
}

OUTPUT=env
TAG_PREFIX=v

while (($#)); do
  case "$1" in
    --build-version) OUTPUT=build-version ;;
    --release-version) OUTPUT=release-version ;;
    --tag) OUTPUT=tag ;;
    --github-output) OUTPUT=github-output ;;
    --tag-prefix)
      shift
      (($#)) || nm_die "--tag-prefix requires a value"
      TAG_PREFIX="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    *) nm_die "unknown option: $1" ;;
  esac
  shift
done

REPO_ROOT=$(nm_repo_root)
release_version="$(nm_read_package_version "$REPO_ROOT")"
[[ -n "$release_version" ]] || nm_die "could not read package version from Cargo.toml"

read -r -a release_parts <<<"$(version_numbers "$release_version")"
major="${release_parts[0]:-0}"
minor="${release_parts[1]:-1}"

major_override="$(nm_env_int GAIL_VERSION_MAJOR || true)"
minor_override="$(nm_env_int GAIL_VERSION_MINOR || true)"
[[ -n "$major_override" ]] && major="$major_override"
[[ -n "$minor_override" ]] && minor="$minor_override"

explicit_version="$(nm_env_first GAIL_VERSION || true)"
build_number="$(nm_env_int GAIL_BUILD_NUMBER BUILD_NUMBER || true)"
version_source=default
[[ -n "$build_number" || -n "$explicit_version" ]] && version_source=env

if [[ -z "$build_number" ]]; then
  build_number="$(nm_git_output "$REPO_ROOT" rev-list --count HEAD || true)"
  if [[ -n "$build_number" && "$build_number" =~ ^[0-9]+$ ]]; then
    version_source=git
  else
    build_number=0
  fi
fi

printf -v padded_build '%04d' "$build_number"
build_version="${major}.${minor}.${padded_build}"
if [[ -n "$explicit_version" ]] && is_build_version "$explicit_version"; then
  build_version="$explicit_version"
  version_source=env
fi

git_commit="$(nm_env_first GIT_COMMIT GAIL_GIT_COMMIT || true)"
[[ -z "$git_commit" ]] && git_commit="$(nm_git_output "$REPO_ROOT" rev-parse HEAD || true)"
git_commit="${git_commit:-unknown}"
commit_short=unknown
[[ "$git_commit" != unknown ]] && commit_short="${git_commit:0:8}"
tag="${TAG_PREFIX}${build_version}"

emit_github_output() {
  [[ -n "${GITHUB_OUTPUT:-}" ]] || nm_die "GITHUB_OUTPUT is not set"
  {
    printf 'release_version=%s\n' "$release_version"
    printf 'build_version=%s\n' "$build_version"
    printf 'build_number=%s\n' "$build_number"
    printf 'git_commit=%s\n' "$git_commit"
    printf 'commit_short=%s\n' "$commit_short"
    printf 'version_source=%s\n' "$version_source"
    printf 'tag=%s\n' "$tag"
  } >>"$GITHUB_OUTPUT"
  printf 'Gail build version: %s\n' "$build_version"
  printf 'Gail release tag: %s\n' "$tag"
}

case "$OUTPUT" in
  build-version) printf '%s\n' "$build_version" ;;
  release-version) printf '%s\n' "$release_version" ;;
  tag) printf '%s\n' "$tag" ;;
  github-output) emit_github_output ;;
  env)
    printf 'GAIL_RELEASE_VERSION=%s\n' "$release_version"
    printf 'GAIL_BUILD_VERSION=%s\n' "$build_version"
    printf 'GAIL_BUILD_NUMBER=%s\n' "$build_number"
    printf 'GAIL_GIT_COMMIT=%s\n' "$git_commit"
    printf 'GAIL_GIT_COMMIT_SHORT=%s\n' "$commit_short"
    printf 'GAIL_VERSION_SOURCE=%s\n' "$version_source"
    printf 'GAIL_TAG=%s\n' "$tag"
    ;;
  *) nm_die "unsupported output mode: $OUTPUT" ;;
esac
