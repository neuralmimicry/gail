#!/usr/bin/env bash
set -euo pipefail

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

Examples:
  version="$(bash scripts/derive-version.sh --build-version)"
  ./scripts/package-release.sh --version "$version" --output-dir ./dist
  ./scripts/package-release.sh --version "$version" --output-dir ./dist --target-triple x86_64-pc-windows-msvc --platform windows-amd64 --archive-format zip --binary-suffix .exe
  ./scripts/package-release.sh --version "$version" --output-dir ./dist --target-triple x86_64-unknown-linux-gnu --platform linux-amd64 --deb-arch amd64
  GAIL_UPDATE_KEY=shared ./scripts/package-release.sh --version "$version" --output-dir ./dist --sign-update
USAGE
}

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

default_platform() {
  local os arch
  os=$(uname -s | tr '[:upper:]' '[:lower:]')
  arch=$(uname -m)
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

default_archive_format() {
  case "$PLATFORM" in
    windows*) printf 'zip\n' ;;
    *) printf 'tar.gz\n' ;;
  esac
}

default_binary_suffix() {
  case "$PLATFORM" in
    windows*) printf '.exe\n' ;;
    *) printf '\n' ;;
  esac
}

sha256_tool() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf 'sha256sum\n'
  elif command -v shasum >/dev/null 2>&1; then
    printf 'shasum -a 256\n'
  else
    die "sha256sum or shasum is required"
  fi
}

binary_dir() {
  if [[ -n "$TARGET_TRIPLE" ]]; then
    printf '%s/target/%s/release\n' "$REPO_ROOT" "$TARGET_TRIPLE"
  else
    printf '%s/target/release\n' "$REPO_ROOT"
  fi
}

resolve_build_user() {
  if [[ $(id -u) -eq 0 && -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    printf '%s\n' "$SUDO_USER"
  fi
}

run_preflight_checks() {
  local cmd
  printf -v cmd 'cd %q && bash scripts/preflight.sh' "$REPO_ROOT"
  if [[ -n "$BUILD_AS_USER" ]]; then
    log "running preflight checks as ${BUILD_AS_USER}"
    sudo -u "$BUILD_AS_USER" -H bash -lc "$cmd"
  else
    log "running preflight checks"
    (
      cd "$REPO_ROOT"
      bash scripts/preflight.sh
    )
  fi
}

build_binaries() {
  local args
  local cmd
  if (( ! SKIP_PREFLIGHT )); then
    run_preflight_checks
  else
    log "skipping preflight checks"
  fi
  args=(cargo build --locked --release --bin gail)
  if [[ -n "$TARGET_TRIPLE" ]]; then
    args+=(--target "$TARGET_TRIPLE")
  fi
  printf -v cmd '%q ' "${args[@]}"
  cmd=${cmd% }
  if [[ -n "$BUILD_AS_USER" ]]; then
    log "building release binaries as ${BUILD_AS_USER}"
    sudo -u "$BUILD_AS_USER" -H bash -lc "cd $(printf '%q' "$REPO_ROOT") && ${cmd}"
  else
    log "building release binaries"
    (
      cd "$REPO_ROOT"
      "${args[@]}"
    )
  fi
}

archive_extension() {
  case "$ARCHIVE_FORMAT" in
    tar.gz|zip)
      printf '%s\n' "$ARCHIVE_FORMAT"
      ;;
    *)
      die "unsupported archive format: $ARCHIVE_FORMAT"
      ;;
  esac
}

create_zip_archive() {
  if command -v zip >/dev/null 2>&1; then
    (
      cd "$STAGE_ROOT"
      zip -q -r "$ARCHIVE_PATH" "$ARCHIVE_BASENAME"
    )
    return
  fi

  if command -v powershell.exe >/dev/null 2>&1 && command -v cygpath >/dev/null 2>&1; then
    local archive_win payload_win ps_cmd
    archive_win=$(cygpath -w "$ARCHIVE_PATH")
    payload_win=$(cygpath -w "$PAYLOAD_DIR")
    printf -v ps_cmd \
      'Compress-Archive -LiteralPath "%s" -DestinationPath "%s" -Force' \
      "$payload_win" \
      "$archive_win"
    powershell.exe -NoLogo -NoProfile -Command "$ps_cmd" >/dev/null
    return
  fi

  die "zip archive creation requires 'zip' or powershell.exe with cygpath"
}

create_archive() {
  case "$ARCHIVE_FORMAT" in
    tar.gz)
      tar -C "$STAGE_ROOT" -czf "$ARCHIVE_PATH" "$ARCHIVE_BASENAME"
      ;;
    zip)
      create_zip_archive
      ;;
    *)
      die "unsupported archive format: $ARCHIVE_FORMAT"
      ;;
  esac
}

should_include_install_service() {
  case "$PLATFORM" in
    linux*) return 0 ;;
    *) return 1 ;;
  esac
}

validate_deb_arch() {
  case "$1" in
    amd64|arm64)
      ;;
    *)
      die "unsupported Debian architecture: $1"
      ;;
  esac
}

debian_package_version() {
  local version sanitized
  version="$1"
  [[ -n "$version" ]] || die "Debian package version is empty"
  sanitized=$(
    printf '%s' "$version" \
      | tr '-' '~' \
      | sed -E 's/[^A-Za-z0-9.+:~]+/./g; s/^[^A-Za-z0-9]+//; s/[^A-Za-z0-9]+$//'
  )
  [[ -n "$sanitized" ]] || die "unable to derive Debian package version from '$version'"
  printf '%s\n' "$sanitized"
}

compute_deb_depends() {
  if ! command -v dpkg-shlibdeps >/dev/null 2>&1; then
    printf '\n'
    return
  fi

  local work_dir output depends
  work_dir=$(mktemp -d)
  output=$(
    cd "$work_dir"
    dpkg-shlibdeps -O "$1" "$2" 2>/dev/null || true
  )
  rm -rf "$work_dir"
  depends=$(printf '%s\n' "$output" | sed -n 's/^shlibs:Depends=//p' | tail -n 1)
  printf '%s\n' "$depends"
}

create_debian_package() {
  local deb_version deb_stage_root deb_root deb_path depends

  validate_deb_arch "$DEB_ARCH"
  command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb is required when --deb-arch is used"

  deb_version=$(debian_package_version "$VERSION")
  deb_stage_root="$OUTPUT_DIR/.deb-stage"
  deb_root="$deb_stage_root/root"
  deb_path="$OUTPUT_DIR/GAIL_${deb_version}_${DEB_ARCH}.deb"

  rm -rf "$deb_stage_root"
  install -d -m 0755 \
    "$deb_root/DEBIAN" \
    "$deb_root/usr/bin" \
    "$deb_root/usr/share/doc/gail"

  install -m 0755 "$GAIL_BIN" "$deb_root/usr/bin/gail"
  install -m 0644 "$REPO_ROOT/README.md" "$deb_root/usr/share/doc/gail/README.md"
  install -m 0644 "$REPO_ROOT/docs/OPERATIONS.md" "$deb_root/usr/share/doc/gail/OPERATIONS.md"

  depends=$(compute_deb_depends "$deb_root/usr/bin/GAIL")

  {
    printf 'Package: GAIL\n'
    printf 'Version: %s\n' "$deb_version"
    printf 'Section: admin\n'
    printf 'Priority: optional\n'
    printf 'Architecture: %s\n' "$DEB_ARCH"
    if [[ -n "$depends" ]]; then
      printf 'Depends: %s\n' "$depends"
    fi
    printf 'Maintainer: NeuralMimicry <opensource@neuralmimicry.ai>\n'
    printf 'Homepage: https://github.com/neuralmimicry/GAIL\n'
    printf 'Description: GAIL loader and anomaly detection runtime\n'
    printf ' GAIL packages the core agent and GAIL-loader for Linux hosts.\n'
  } >"$deb_root/DEBIAN/control"

  dpkg-deb --build --root-owner-group "$deb_root" "$deb_path" >/dev/null
  artifacts+=("$deb_path")
  rm -rf "$deb_stage_root"
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
BUILD_AS_USER=

while (($#)); do
  case "$1" in
    --version)
      shift
      (($#)) || die "--version requires a value"
      VERSION="$1"
      ;;
    --output-dir)
      shift
      (($#)) || die "--output-dir requires a value"
      OUTPUT_DIR="$1"
      ;;
    --target-triple)
      shift
      (($#)) || die "--target-triple requires a value"
      TARGET_TRIPLE="$1"
      ;;
    --platform)
      shift
      (($#)) || die "--platform requires a value"
      PLATFORM="$1"
      ;;
    --archive-format)
      shift
      (($#)) || die "--archive-format requires a value"
      ARCHIVE_FORMAT="$1"
      ;;
    --binary-suffix)
      shift
      (($#)) || die "--binary-suffix requires a value"
      BINARY_SUFFIX="$1"
      ;;
    --deb-arch)
      shift
      (($#)) || die "--deb-arch requires a value"
      DEB_ARCH="$1"
      ;;
    --skip-build)
      SKIP_BUILD=1
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=1
      ;;
    --sign-update)
      SIGN_UPDATE=1
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

[[ -n "$VERSION" ]] || die "--version is required"
[[ -n "$OUTPUT_DIR" ]] || die "--output-dir is required"

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
PLATFORM="${PLATFORM:-$(default_platform)}"
ARCHIVE_FORMAT="${ARCHIVE_FORMAT:-$(default_archive_format)}"
BINARY_SUFFIX="${BINARY_SUFFIX:-$(default_binary_suffix)}"
BUILD_AS_USER=$(resolve_build_user || true)

BIN_DIR=$(binary_dir)
GAIL_BIN="$BIN_DIR/GAIL${BINARY_SUFFIX}"
LOADER_BIN="$BIN_DIR/GAIL-loader${BINARY_SUFFIX}"
GAIL_BIN_NAME="GAIL${BINARY_SUFFIX}"
LOADER_BIN_NAME="GAIL-loader${BINARY_SUFFIX}"

if (( ! SKIP_BUILD )) || [[ ! -x "$GAIL_BIN" || ! -x "$LOADER_BIN" ]]; then
  build_binaries
fi

[[ -x "$GAIL_BIN" ]] || die "missing GAIL binary: $GAIL_BIN"
[[ -x "$LOADER_BIN" ]] || die "missing GAIL-loader binary: $LOADER_BIN"

if (( SIGN_UPDATE )) && [[ -z "${GAIL_UPDATE_KEY:-}" ]]; then
  die "GAIL_UPDATE_KEY must be set when --sign-update is used"
fi

ARCHIVE_BASENAME="GAIL-${VERSION}-${PLATFORM}"
OUTPUT_DIR=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)
STAGE_ROOT="$OUTPUT_DIR/.stage"
PAYLOAD_DIR="$STAGE_ROOT/$ARCHIVE_BASENAME"
ARCHIVE_PATH="$OUTPUT_DIR/${ARCHIVE_BASENAME}.$(archive_extension)"
CHECKSUM_PATH="$OUTPUT_DIR/${ARCHIVE_BASENAME}.sha256.txt"

rm -rf "$STAGE_ROOT"
mkdir -p "$PAYLOAD_DIR"
mkdir -p "$PAYLOAD_DIR/docs"

install -m 0755 "$GAIL_BIN" "$PAYLOAD_DIR/$GAIL_BIN_NAME"
install -m 0755 "$LOADER_BIN" "$PAYLOAD_DIR/$LOADER_BIN_NAME"
if should_include_install_service; then
  install -m 0755 "$REPO_ROOT/scripts/install-service.sh" "$PAYLOAD_DIR/install-service.sh"
fi
install -m 0644 "$REPO_ROOT/README.md" "$PAYLOAD_DIR/README.md"
install -m 0644 "$REPO_ROOT/docs/OPERATIONS.md" "$PAYLOAD_DIR/docs/OPERATIONS.md"

create_archive

artifacts=("$ARCHIVE_PATH")

if [[ -n "$DEB_ARCH" ]]; then
  [[ "$PLATFORM" == linux* ]] || die "--deb-arch is only supported for linux platforms"
  create_debian_package
fi

if (( SIGN_UPDATE )); then
  SIGN_DIR="$OUTPUT_DIR/.sign"
  rm -rf "$SIGN_DIR"
  mkdir -p "$SIGN_DIR"
  (
    cd "$REPO_ROOT"
    GAIL_UPDATE_KEY="$GAIL_UPDATE_KEY" "$GAIL_BIN" \
      sign-update \
      --bundle "$GAIL_BIN" \
      --version "$VERSION" \
      --channel production \
      --out "$SIGN_DIR"
  )

  mv "$SIGN_DIR/GAIL.update" "$OUTPUT_DIR/${ARCHIVE_BASENAME}.update"
  mv "$SIGN_DIR/GAIL.update.meta.json" "$OUTPUT_DIR/${ARCHIVE_BASENAME}.update.meta.json"
  mv "$SIGN_DIR/GAIL.update.sig" "$OUTPUT_DIR/${ARCHIVE_BASENAME}.update.sig"
  artifacts+=(
    "$OUTPUT_DIR/${ARCHIVE_BASENAME}.update"
    "$OUTPUT_DIR/${ARCHIVE_BASENAME}.update.meta.json"
    "$OUTPUT_DIR/${ARCHIVE_BASENAME}.update.sig"
  )
  rm -rf "$SIGN_DIR"
fi

checksum_cmd=$(sha256_tool)
(
  cd "$OUTPUT_DIR"
  relative_artifacts=()
  for artifact in "${artifacts[@]}"; do
    relative_artifacts+=("$(basename "$artifact")")
  done
  $checksum_cmd "${relative_artifacts[@]}" >"$(basename "$CHECKSUM_PATH")"
)

log
log "packaged GAIL release artifacts:"
for artifact in "${artifacts[@]}" "$CHECKSUM_PATH"; do
  log "  $artifact"
done

rm -rf "$STAGE_ROOT"
