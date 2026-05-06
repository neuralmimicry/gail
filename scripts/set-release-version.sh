#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/set-release-version.sh VERSION

Updates Cargo.toml and Cargo.lock in the current workspace to the supplied
release version. VERSION must be a Cargo-compatible SemVer value such as
1.2.3, 1.2.3-alpha.1, or 1.2.3+build.4.
USAGE
}

version="${1:-}"

if [[ -z "${version}" ]]; then
  usage
  exit 2
fi

if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z][0-9A-Za-z.-]*)?(\+[0-9A-Za-z][0-9A-Za-z.-]*)?$ ]]; then
  echo "Invalid release version: ${version}" >&2
  usage
  exit 2
fi

if [[ ! -f Cargo.toml || ! -f Cargo.lock ]]; then
  echo "Run this script from the Gail repository root." >&2
  exit 2
fi

export GAIL_RELEASE_VERSION="${version}"

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

cargo metadata --locked --no-deps --format-version=1 >/dev/null

echo "Gail release version set to ${version}"
