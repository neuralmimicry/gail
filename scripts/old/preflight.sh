#!/usr/bin/env bash
set -euo pipefail

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

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

run() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

MODE=quick
SKIP_CLIPPY=0
SKIP_TESTS=0
DENY_WARNINGS=0

while (($#)); do
  case "$1" in
    --quick)
      MODE=quick
      ;;
    --ci)
      MODE=ci
      ;;
    --skip-clippy)
      SKIP_CLIPPY=1
      ;;
    --skip-tests)
      SKIP_TESTS=1
      ;;
    --deny-warnings)
      DENY_WARNINGS=1
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

command -v cargo >/dev/null 2>&1 || die "cargo is required to run GAIL preflight checks"

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)

cd "$REPO_ROOT"
printf 'checking rustfmt formatting\n'
run cargo fmt --all --check

if [[ "$MODE" == "ci" ]]; then
  printf 'checking Rust build graph\n'
  run cargo check --locked --all-targets

  if (( ! SKIP_CLIPPY )); then
    clippy_args=(cargo clippy --locked --all-targets)
    command -v cargo-clippy >/dev/null 2>&1 || die "cargo clippy is required for --ci checks"
    printf 'checking clippy lints\n'
    if (( DENY_WARNINGS )); then
      clippy_args+=(-- -D warnings)
    fi
    run "${clippy_args[@]}"
  fi

  if (( ! SKIP_TESTS )); then
    printf 'running Rust tests\n'
    run cargo test --locked --all-targets
  fi
fi
