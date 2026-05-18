#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/release-common.sh
source "${SCRIPT_DIR}/release-common.sh"

usage() {
  cat <<'USAGE'
Usage: preflight.sh [options]

Run Gail source preflight checks that should fail fast before packaging,
deployment, or CI spends time on longer build steps.

Options:
  --quick             Run fast source checks only. This is the default.
  --ci                Run the full CI verification set.
  --skip-clippy       Skip clippy when --ci is used.
  --skip-tests        Skip tests when --ci is used.
  --deny-warnings     Treat clippy warnings as errors when --ci is used.
  -h, --help          Show this help text.
USAGE
}

MODE=quick
SKIP_CLIPPY=0
SKIP_TESTS=0
DENY_WARNINGS=0

while (($#)); do
  case "$1" in
    --quick) MODE=quick ;;
    --ci) MODE=ci ;;
    --skip-clippy) SKIP_CLIPPY=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --deny-warnings) DENY_WARNINGS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) nm_die "unknown option: $1" ;;
  esac
  shift
done

nm_require_command cargo
REPO_ROOT=$(nm_repo_root)
cd "$REPO_ROOT"

nm_log 'checking rustfmt formatting'
nm_run cargo fmt --all --check

if [[ "$MODE" == ci ]]; then
  nm_log 'checking Rust build graph'
  nm_run cargo check --locked --all-targets

  if (( ! SKIP_CLIPPY )); then
    nm_require_command cargo-clippy
    clippy_args=(cargo clippy --locked --all-targets)
    (( DENY_WARNINGS )) && clippy_args+=(-- -D warnings)
    nm_log 'checking clippy lints'
    nm_run "${clippy_args[@]}"
  fi

  if (( ! SKIP_TESTS )); then
    nm_log 'running Rust tests'
    nm_run cargo test --locked --all-targets
  fi
fi
