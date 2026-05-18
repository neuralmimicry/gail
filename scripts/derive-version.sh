#!/usr/bin/env bash
set -euo pipefail

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

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

env_first() {
  local name value
  for name in "$@"; do
    value="${!name:-}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ -n "$value" ]]; then
      printf '%s\n' "$value"
      return 0
    fi
  done
  return 1
}

env_int() {
  local value
  value=$(env_first "$@" || true)
  [[ -n "$value" ]] || return 1
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

read_release_version() {
  sed -n '/^\[package\]/,/^\[/ s/^version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$REPO_ROOT/Cargo.toml" \
    | head -n 1
}

version_numbers() {
  printf '%s' "$1" \
    | tr -cs '0-9' ' ' \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

is_build_version() {
  [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]{4,}$ ]]
}

git_output() {
  git -C "$REPO_ROOT" "$@" 2>/dev/null | sed -e 's/[[:space:]]*$//' | head -n 1
}

OUTPUT=env
TAG_PREFIX=v

while (($#)); do
  case "$1" in
    --build-version)
      OUTPUT=build-version
      ;;
    --release-version)
      OUTPUT=release-version
      ;;
    --tag)
      OUTPUT=tag
      ;;
    --github-output)
      OUTPUT=github-output
      ;;
    --tag-prefix)
      shift
      (($#)) || die "--tag-prefix requires a value"
      TAG_PREFIX="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
  shift
done

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)

release_version=$(read_release_version)
[[ -n "$release_version" ]] || die "could not read package version from Cargo.toml"

read -r -a release_parts <<<"$(version_numbers "$release_version")"
major="${release_parts[0]:-0}"
minor="${release_parts[1]:-1}"

major_override=$(env_int GAIL_VERSION_MAJOR || true)
minor_override=$(env_int GAIL_VERSION_MINOR || true)
if [[ -n "$major_override" ]]; then
  major="$major_override"
fi
if [[ -n "$minor_override" ]]; then
  minor="$minor_override"
fi

explicit_version=$(env_first GAIL_VERSION || true)
build_number=$(env_int GAIL_BUILD_NUMBER BUILD_NUMBER || true)
version_source=default
if [[ -n "$build_number" || -n "$explicit_version" ]]; then
  version_source=env
fi

if [[ -z "$build_number" ]]; then
  build_number=$(git_output rev-list --count HEAD || true)
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

git_commit=$(env_first GIT_COMMIT GAIL_GIT_COMMIT || true)
if [[ -z "$git_commit" ]]; then
  git_commit=$(git_output rev-parse HEAD || true)
fi
git_commit="${git_commit:-unknown}"
if [[ "$git_commit" == "unknown" ]]; then
  commit_short=unknown
else
  commit_short="${git_commit:0:8}"
fi

tag="${TAG_PREFIX}${build_version}"

case "$OUTPUT" in
  build-version)
    printf '%s\n' "$build_version"
    ;;
  release-version)
    printf '%s\n' "$release_version"
    ;;
  tag)
    printf '%s\n' "$tag"
    ;;
  github-output)
    [[ -n "${GITHUB_OUTPUT:-}" ]] || die "GITHUB_OUTPUT is not set"
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
    ;;
  env)
    printf 'GAIL_RELEASE_VERSION=%s\n' "$release_version"
    printf 'GAIL_BUILD_VERSION=%s\n' "$build_version"
    printf 'GAIL_BUILD_NUMBER=%s\n' "$build_number"
    printf 'GAIL_GIT_COMMIT=%s\n' "$git_commit"
    printf 'GAIL_GIT_COMMIT_SHORT=%s\n' "$commit_short"
    printf 'GAIL_VERSION_SOURCE=%s\n' "$version_source"
    printf 'GAIL_TAG=%s\n' "$tag"
    ;;
  *)
    die "unsupported output mode: $OUTPUT"
    ;;
esac
