#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/release-common.sh
source "${SCRIPT_DIR}/release-common.sh"

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/set-release-version.sh VERSION

Updates Cargo.toml and Cargo.lock in the current workspace to the supplied
release version.

VERSION may optionally start with "v". Numeric SemVer components are
normalised to remove leading zeroes, so values such as:

  v0.1.0072
  0.1.0072

become:

  0.1.72

The final version must be a Cargo-compatible SemVer value such as:
  1.2.3
  1.2.3-alpha.1
  1.2.3+build.4
USAGE
}

normalise_cargo_semver() {
  local raw="${1:-}"
  local version core build prerelease major minor patch rest

  [[ -n "$raw" ]] || {
    echo "missing version" >&2
    return 2
  }

  # Accept release tags such as v0.1.72.
  version="${raw#v}"

  build=""
  prerelease=""

  # Split build metadata first: 1.2.3-alpha.1+build.4
  core="${version%%+*}"
  if [[ "$version" == *"+"* ]]; then
    build="+${version#*+}"
  fi

  # Split prerelease from the numeric core.
  if [[ "$core" == *"-"* ]]; then
    prerelease="-${core#*-}"
    core="${core%%-*}"
  fi

  IFS='.' read -r major minor patch rest <<< "$core"

  if [[ -z "${major:-}" || -z "${minor:-}" || -z "${patch:-}" || -n "${rest:-}" ]]; then
    echo "invalid SemVer core '${core}' from input '${raw}'" >&2
    return 2
  fi

  if ! [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ && "$patch" =~ ^[0-9]+$ ]]; then
    echo "invalid SemVer numeric components in '${raw}'" >&2
    return 2
  fi

  # 10# avoids octal interpretation and strips leading zeroes.
  major="$((10#$major))"
  minor="$((10#$minor))"
  patch="$((10#$patch))"

  # Preserve prerelease/build metadata exactly. Cargo validation below will
  # reject invalid prerelease/build identifiers if present.
  printf '%s.%s.%s%s%s\n' "$major" "$minor" "$patch" "$prerelease" "$build"
}

raw_version="${1:-}"
[[ -n "$raw_version" ]] || {
  usage
  exit 2
}

version="$(normalise_cargo_semver "$raw_version")"

# Validate after normalisation, not before.
nm_validate_cargo_semver "$version"

REPO_ROOT=$(nm_repo_root)
cd "$REPO_ROOT"
[[ -f Cargo.toml && -f Cargo.lock ]] || nm_die "run this script from the Gail repository root, or keep it under scripts/"

export GAIL_RELEASE_VERSION="$version"

perl -0pi -e '
  our $updated;
  BEGIN { $version = $ENV{"GAIL_RELEASE_VERSION"} // die "GAIL_RELEASE_VERSION is not set\n"; }
  $updated += s/(\[package\]\s*(?:(?!\n\[).)*?\nversion\s*=\s*")[^"]+(")/${1}${version}${2}/s;
  END { die "failed to update package.version in Cargo.toml\n" unless $updated; }
' Cargo.toml

perl -0pi -e '
  our $updated;
  BEGIN { $version = $ENV{"GAIL_RELEASE_VERSION"} // die "GAIL_RELEASE_VERSION is not set\n"; }
  $updated += s/(\[\[package\]\]\s*name = "gail"\s*version = ")[^"]+(")/${1}${version}${2}/s;
  END { die "failed to update gail package version in Cargo.lock\n" unless $updated; }
' Cargo.lock

nm_run cargo metadata --locked --no-deps --format-version=1 >/dev/null

if [[ "$raw_version" != "$version" && "${raw_version#v}" != "$version" ]]; then
  printf 'Gail release version normalised from %s to %s\n' "$raw_version" "$version"
else
  printf 'Gail release version set to %s\n' "$version"
fi