#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/release-common.sh
source "${SCRIPT_DIR}/release-common.sh"

nm_require_command cargo
REPO_ROOT=$(nm_repo_root)
cd "$REPO_ROOT"

nm_log 'running libtorch-free trading test profile'
nm_run cargo test --locked --lib trading::tests:: --no-default-features --features ci-trading-tests
