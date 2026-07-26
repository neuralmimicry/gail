#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/release-common.sh
source "${SCRIPT_DIR}/release-common.sh"

nm_require_command cargo
REPO_ROOT=$(nm_repo_root)
cd "$REPO_ROOT"

nm_log 'running libtorch-free trading test profile'
# Run every test nested under `trading`, including unit tests colocated with
# economics, outcomes, advisor, datalake, and execution modules. The previous
# `trading::tests::` filter silently excluded those modular safety checks.
nm_run cargo test --locked --lib trading:: --no-default-features --features ci-trading-tests
