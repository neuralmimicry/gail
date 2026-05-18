#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/release-common.sh
source "${SCRIPT_DIR}/release-common.sh"

usage() {
  cat <<'USAGE'
Usage: package-release.sh [options]

Build and package Gail release artifacts.

Options:
  --version VERSION           Version label for the packaged artifacts.
  --output-dir DIR            Directory to receive the packaged artifacts.
  --target-triple TRIPLE      Optional cargo target triple.
  --platform NAME             Platform suffix in output names. Default: derived from host.
  --archive-format FORMAT     Archive format: tar.gz or zip.
  --binary-suffix SUFFIX      Binary suffix in packaged filenames (for example .exe).
  --deb-arch ARCH             Also build a Debian package for linux using ARCH (amd64 or arm64).
  --skip-build                Reuse existing release binaries instead of building them.
  --skip-preflight            Skip scripts/preflight.sh before building.
  --sign-update               Generate GAIL.update(.meta.json/.sig) with GAIL_UPDATE_KEY.
  -h, --help                  Show this help text.
USAGE
}

VERSION=
OUTPUT_DIR=
TARGET_TRIPLE=
PLATFORM=
ARCHIVE_FORMAT=
BINARY_SUFFIX=
DEB_ARCH=
SKIP_BUILD=0
SKIP_PREFLIGHT=0
SIGN_UPDATE=0

while (($#)); do
  case "$1" in
    --version) shift; (($#)) || nm_die "--version requires a value"; VERSION="$1" ;;
    --output-dir) shift; (($#)) || nm_die "--output-dir requires a value"; OUTPUT_DIR="$1" ;;
    --target-triple) shift; (($#)) || nm_die "--target-triple requires a value"; TARGET_TRIPLE="$1" ;;
    --platform) shift; (($#)) || nm_die "--platform requires a value"; PLATFORM="$1" ;;
    --archive-format) shift; (($#)) || nm_die "--archive-format requires a value"; ARCHIVE_FORMAT="$1" ;;
    --binary-suffix) shift; (($#)) || nm_die "--binary-suffix requires a value"; BINARY_SUFFIX="$1" ;;
    --deb-arch) shift; (($#)) || nm_die "--deb-arch requires a value"; DEB_ARCH="$1" ;;
    --skip-build) SKIP_BUILD=1 ;;
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    --sign-update) SIGN_UPDATE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) nm_die "unknown option: $1" ;;
  esac
  shift
done

[[ -n "$VERSION" ]] || nm_die "--version is required"
[[ -n "$OUTPUT_DIR" ]] || nm_die "--output-dir is required"

REPO_ROOT=$(nm_repo_root)
PLATFORM="${PLATFORM:-$(nm_default_platform)}"
ARCHIVE_FORMAT="${ARCHIVE_FORMAT:-$(nm_default_archive_format "$PLATFORM") }"
ARCHIVE_FORMAT="$(nm_trim "$ARCHIVE_FORMAT")"
BINARY_SUFFIX="${BINARY_SUFFIX:-$(nm_default_binary_suffix "$PLATFORM") }"
BINARY_SUFFIX="$(nm_trim "$BINARY_SUFFIX")"
BUILD_AS_USER="$(nm_resolve_build_user || true)"
BIN_DIR="$(nm_binary_dir "$REPO_ROOT" "$TARGET_TRIPLE")"
GAIL_BIN="${BIN_DIR}/GAIL${BINARY_SUFFIX}"
LOADER_BIN="${BIN_DIR}/GAIL-loader${BINARY_SUFFIX}"
GAIL_BIN_NAME="GAIL${BINARY_SUFFIX}"
LOADER_BIN_NAME="GAIL-loader${BINARY_SUFFIX}"
artifacts=()
STAGE_ROOT=
SIGN_DIR=
DEB_STAGE_ROOT=

cleanup() {
  [[ -n "${STAGE_ROOT:-}" ]] && rm -rf "$STAGE_ROOT"
  [[ -n "${SIGN_DIR:-}" ]] && rm -rf "$SIGN_DIR"
  [[ -n "${DEB_STAGE_ROOT:-}" ]] && rm -rf "$DEB_STAGE_ROOT"
}
trap cleanup EXIT

case "$ARCHIVE_FORMAT" in
  tar.gz|zip) ;;
  *) nm_die "unsupported archive format: $ARCHIVE_FORMAT" ;;
esac

run_as_build_user() {
  local cmd="$1"
  if [[ -n "$BUILD_AS_USER" ]]; then
    sudo -u "$BUILD_AS_USER" -H bash -lc "$cmd"
  else
    bash -lc "$cmd"
  fi
}

run_preflight_checks() {
  if (( SKIP_PREFLIGHT )); then
    nm_log 'skipping preflight checks'
    return
  fi
  local cmd
  printf -v cmd 'cd %q && bash scripts/preflight.sh' "$REPO_ROOT"
  [[ -n "$BUILD_AS_USER" ]] && nm_log "running preflight checks as ${BUILD_AS_USER}" || nm_log 'running preflight checks'
  run_as_build_user "$cmd"
}

build_binaries() {
  run_preflight_checks
  local args cmd
  args=(cargo build --locked --release --bin GAIL --bin GAIL-loader)
  [[ -n "$TARGET_TRIPLE" ]] && args+=(--target "$TARGET_TRIPLE")
  printf -v cmd '%q ' "${args[@]}"
  cmd="cd $(printf '%q' "$REPO_ROOT") && ${cmd% }"
  [[ -n "$BUILD_AS_USER" ]] && nm_log "building release binaries as ${BUILD_AS_USER}" || nm_log 'building release binaries'
  run_as_build_user "$cmd"
}

create_zip_archive() {
  if command -v zip >/dev/null 2>&1; then
    (cd "$STAGE_ROOT" && zip -q -r "$ARCHIVE_PATH" "$ARCHIVE_BASENAME")
    return
  fi
  if command -v powershell.exe >/dev/null 2>&1 && command -v cygpath >/dev/null 2>&1; then
    local archive_win payload_win ps_cmd
    archive_win="$(cygpath -w "$ARCHIVE_PATH")"
    payload_win="$(cygpath -w "$PAYLOAD_DIR")"
    printf -v ps_cmd 'Compress-Archive -LiteralPath "%s" -DestinationPath "%s" -Force' "$payload_win" "$archive_win"
    powershell.exe -NoLogo -NoProfile -Command "$ps_cmd" >/dev/null
    return
  fi
  nm_die "zip archive creation requires 'zip' or powershell.exe with cygpath"
}

create_archive() {
  case "$ARCHIVE_FORMAT" in
    tar.gz) tar -C "$STAGE_ROOT" -czf "$ARCHIVE_PATH" "$ARCHIVE_BASENAME" ;;
    zip) create_zip_archive ;;
  esac
}

create_debian_package() {
  [[ "$PLATFORM" == linux* ]] || nm_die "--deb-arch is only supported for linux platforms"
  nm_validate_deb_arch "$DEB_ARCH"
  nm_require_command dpkg-deb

  local deb_version deb_root deb_path depends
  deb_version="$(nm_debian_package_version "$VERSION")"
  DEB_STAGE_ROOT="${OUTPUT_DIR}/.deb-stage"
  deb_root="${DEB_STAGE_ROOT}/root"
  deb_path="${OUTPUT_DIR}/GAIL_${deb_version}_${DEB_ARCH}.deb"

  rm -rf "$DEB_STAGE_ROOT"
  install -d -m 0755 "$deb_root/DEBIAN" "$deb_root/usr/bin" "$deb_root/usr/share/doc/GAIL"
  install -m 0755 "$GAIL_BIN" "$deb_root/usr/bin/GAIL"
  install -m 0755 "$LOADER_BIN" "$deb_root/usr/bin/GAIL-loader"
  install -m 0644 "$REPO_ROOT/README.md" "$deb_root/usr/share/doc/GAIL/README.md"
  install -m 0644 "$REPO_ROOT/docs/OPERATIONS.md" "$deb_root/usr/share/doc/GAIL/OPERATIONS.md"

  depends="$(nm_compute_deb_depends "$deb_root/usr/bin/GAIL" "$deb_root/usr/bin/GAIL-loader")"
  {
    printf 'Package: GAIL\n'
    printf 'Version: %s\n' "$deb_version"
    printf 'Section: admin\n'
    printf 'Priority: optional\n'
    printf 'Architecture: %s\n' "$DEB_ARCH"
    [[ -n "$depends" ]] && printf 'Depends: %s\n' "$depends"
    printf 'Maintainer: NeuralMimicry <opensource@neuralmimicry.ai>\n'
    printf 'Homepage: https://github.com/neuralmimicry/GAIL\n'
    printf 'Description: GAIL loader and anomaly detection runtime\n'
    printf ' GAIL packages the core agent and GAIL-loader for Linux hosts.\n'
  } >"$deb_root/DEBIAN/control"

  dpkg-deb --build --root-owner-group "$deb_root" "$deb_path" >/dev/null
  artifacts+=("$deb_path")
}

if (( ! SKIP_BUILD )) || [[ ! -x "$GAIL_BIN" || ! -x "$LOADER_BIN" ]]; then
  build_binaries
fi

[[ -x "$GAIL_BIN" ]] || nm_die "missing GAIL binary: $GAIL_BIN"
[[ -x "$LOADER_BIN" ]] || nm_die "missing GAIL-loader binary: $LOADER_BIN"
(( SIGN_UPDATE == 0 )) || [[ -n "${GAIL_UPDATE_KEY:-}" ]] || nm_die "GAIL_UPDATE_KEY must be set when --sign-update is used"

ARCHIVE_BASENAME="GAIL-${VERSION}-${PLATFORM}"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
STAGE_ROOT="${OUTPUT_DIR}/.stage"
PAYLOAD_DIR="${STAGE_ROOT}/${ARCHIVE_BASENAME}"
ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_BASENAME}.${ARCHIVE_FORMAT}"
CHECKSUM_PATH="${OUTPUT_DIR}/${ARCHIVE_BASENAME}.sha256.txt"

rm -rf "$STAGE_ROOT"
install -d -m 0755 "$PAYLOAD_DIR/docs"
install -m 0755 "$GAIL_BIN" "$PAYLOAD_DIR/$GAIL_BIN_NAME"
install -m 0755 "$LOADER_BIN" "$PAYLOAD_DIR/$LOADER_BIN_NAME"
[[ "$PLATFORM" == linux* ]] && install -m 0755 "$REPO_ROOT/scripts/install-service.sh" "$PAYLOAD_DIR/install-service.sh"
install -m 0644 "$REPO_ROOT/README.md" "$PAYLOAD_DIR/README.md"
install -m 0644 "$REPO_ROOT/docs/OPERATIONS.md" "$PAYLOAD_DIR/docs/OPERATIONS.md"

create_archive
artifacts=("$ARCHIVE_PATH")
[[ -n "$DEB_ARCH" ]] && create_debian_package

if (( SIGN_UPDATE )); then
  SIGN_DIR="${OUTPUT_DIR}/.sign"
  rm -rf "$SIGN_DIR"
  mkdir -p "$SIGN_DIR"
  (cd "$REPO_ROOT" && GAIL_UPDATE_KEY="$GAIL_UPDATE_KEY" "$GAIL_BIN" sign-update --bundle "$GAIL_BIN" --version "$VERSION" --channel production --out "$SIGN_DIR")
  for suffix in update update.meta.json update.sig; do
    mv "$SIGN_DIR/GAIL.${suffix}" "$OUTPUT_DIR/${ARCHIVE_BASENAME}.${suffix}"
    artifacts+=("$OUTPUT_DIR/${ARCHIVE_BASENAME}.${suffix}")
  done
fi

checksum_cmd="$(nm_checksum_tool)"
(
  cd "$OUTPUT_DIR"
  relative_artifacts=()
  for artifact in "${artifacts[@]}"; do
    relative_artifacts+=("$(basename "$artifact")")
  done
  # shellcheck disable=SC2086 # checksum_cmd can intentionally include arguments, e.g. shasum -a 256.
  $checksum_cmd "${relative_artifacts[@]}" >"$(basename "$CHECKSUM_PATH")"
)

nm_log ''
nm_log 'packaged GAIL release artifacts:'
for artifact in "${artifacts[@]}" "$CHECKSUM_PATH"; do
  nm_log "  $artifact"
done
