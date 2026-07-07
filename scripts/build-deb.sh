#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/release-common.sh
source "${SCRIPT_DIR}/release-common.sh"

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/build-deb.sh --version VERSION --arch amd64|arm64 --binary PATH [options]

Options:
  --deb-version VERSION  Debian package version. Defaults to --version.
  --out-dir PATH        Directory for the resulting .deb. Defaults to dist.
  --config PATH         Gail config to install. Defaults to gail.yaml.
  --routing PATH        Routing profiles JSON. Defaults to config/ai-routing-profiles.json.
  --trainer-binary PATH Rust trainer binary to install as /usr/bin/gail-qlora-sft.
                        Defaults to <dirname(--binary)>/gail-qlora-sft when present.
USAGE
}

REPO_ROOT=$(nm_repo_root)
version="${VERSION:-}"
deb_version="${DEB_VERSION:-}"
deb_arch="${DEB_ARCH:-}"
binary_path="${BINARY_PATH:-}"
trainer_binary_path="${TRAINER_BINARY_PATH:-}"
out_dir="${OUT_DIR:-dist}"
config_path="${CONFIG_PATH:-${REPO_ROOT}/gail.yaml}"
routing_path="${ROUTING_PROFILES_PATH:-${REPO_ROOT}/config/ai-routing-profiles.json}"

while (($#)); do
  case "$1" in
    --version) shift; (($#)) || nm_die "--version requires a value"; version="$1" ;;
    --deb-version) shift; (($#)) || nm_die "--deb-version requires a value"; deb_version="$1" ;;
    --arch) shift; (($#)) || nm_die "--arch requires a value"; deb_arch="$1" ;;
    --binary) shift; (($#)) || nm_die "--binary requires a value"; binary_path="$1" ;;
    --trainer-binary) shift; (($#)) || nm_die "--trainer-binary requires a value"; trainer_binary_path="$1" ;;
    --out-dir) shift; (($#)) || nm_die "--out-dir requires a value"; out_dir="$1" ;;
    --config) shift; (($#)) || nm_die "--config requires a value"; config_path="$1" ;;
    --routing) shift; (($#)) || nm_die "--routing requires a value"; routing_path="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) nm_die "unknown argument: $1" ;;
  esac
  shift
done

[[ -n "$version" && -n "$deb_arch" && -n "$binary_path" ]] || { usage; exit 2; }
deb_version="${deb_version:-$version}"
if [[ -z "$trainer_binary_path" ]]; then
  default_trainer_binary_path="$(cd "$(dirname "$binary_path")" && pwd)/gail-qlora-sft"
  if [[ -x "$default_trainer_binary_path" ]]; then
    trainer_binary_path="$default_trainer_binary_path"
  fi
fi
nm_validate_deb_arch "$deb_arch"
nm_validate_deb_version "$deb_version"
[[ -x "$binary_path" ]] || nm_die "binary is missing or not executable: $binary_path"
if [[ -n "$trainer_binary_path" ]]; then
  [[ -x "$trainer_binary_path" ]] || nm_die "trainer binary is missing or not executable: $trainer_binary_path"
fi
[[ -f "$config_path" ]] || nm_die "config file is missing: $config_path"
[[ -f "$routing_path" ]] || nm_die "routing profiles file is missing: $routing_path"
[[ -f "${REPO_ROOT}/packaging/deb/gail.service" ]] || nm_die "systemd service file is missing: ${REPO_ROOT}/packaging/deb/gail.service"
nm_require_command dpkg-deb

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

package_root="${tmp_dir}/gail"
control_dir="${package_root}/DEBIAN"
install -d -m 0755 "$control_dir"

install -D -m 0755 "$binary_path" "${package_root}/usr/bin/gail"
if [[ -n "$trainer_binary_path" ]]; then
  install -D -m 0755 "$trainer_binary_path" "${package_root}/usr/bin/gail-qlora-sft"
fi
install -D -m 0644 "$config_path" "${package_root}/etc/gail/gail.yaml"
install -D -m 0644 "$routing_path" "${package_root}/etc/gail/ai-routing-profiles.json"
install -D -m 0644 "${REPO_ROOT}/packaging/deb/gail.service" "${package_root}/lib/systemd/system/gail.service"
install -d -m 0750 "${package_root}/var/lib/gail/data"

cat > "${package_root}/etc/gail/gail.env" <<'ENV'
# Environment variables loaded by gail.service.
# Set provider credentials, service tokens, and runtime overrides here.
GAIL_PUBLIC_BASE_URL=

GAIL_REFINER_API_TOKEN=
GAIL_TRACEY_API_TOKEN=
GAIL_CONTINUUM_API_TOKEN=
GAIL_ADMIN_API_TOKEN=

OPENAI_API_KEY=
GEMINI_API_KEY=
GOOGLE_ACCESS_TOKEN=
NVIDIA_API_KEY=

GAIL_OPENAI_MODEL=
GAIL_GEMINI_MODEL=
GAIL_NVIDIA_BASE_URL=
GAIL_NVIDIA_KIMI_MODEL=
GAIL_NVIDIA_MINIMAX_MODEL=
GAIL_NVIDIA_DEEPSEEK_MODEL=
GAIL_NVIDIA_GLM_MODEL=

GAIL_OLLAMA_MODEL=
GAIL_OLLAMA_BASE_URL=

GAIL_AARNN_ENDPOINT=
GAIL_AARNN_SOCKET_PATH=
GAIL_AARNN_REPO_ROOT=
GAIL_AARNN_BRIDGE_ENDPOINT=
GAIL_AARNN_BRIDGE_ACCESS_TOKEN=
GAIL_AARNN_BRIDGE_NETWORK_ID=
GAIL_AARNN_BRIDGE_NODE_ID=

GAIL_TRADING_ENABLED=false
GAIL_TRADING_OCTOBOT_URL=
GAIL_TRADING_OCTOBOT_PASSWORD=
GAIL_TRADING_REFINER_URL=
GAIL_TRADING_REFINER_TOKEN=
GAIL_TRADING_EVAL_INTERVAL=60
GAIL_TRADING_MAX_USD=10
GAIL_TRADING_MIN_USD=1
GAIL_TRADING_MAX_POSITIONS=3
GAIL_TRADING_CONFIDENCE_THRESHOLD=0.6
ENV

cat > "${control_dir}/conffiles" <<'CONFFILES'
/etc/gail/gail.yaml
/etc/gail/gail.env
/etc/gail/ai-routing-profiles.json
CONFFILES

cat > "${control_dir}/postinst" <<'POSTINST'
#!/bin/sh
set -e

if ! getent group gail >/dev/null 2>&1; then
  groupadd --system gail
fi

if ! id -u gail >/dev/null 2>&1; then
  useradd --system --gid gail --home-dir /var/lib/gail --shell /usr/sbin/nologin --comment "Gail service" gail
fi

install -d -o gail -g gail -m 0750 /var/lib/gail /var/lib/gail/data

if command -v systemctl >/dev/null 2>&1; then
  systemctl daemon-reload || true
fi

exit 0
POSTINST

cat > "${control_dir}/prerm" <<'PRERM'
#!/bin/sh
set -e

if [ "$1" = "remove" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl stop gail.service >/dev/null 2>&1 || true
  systemctl disable gail.service >/dev/null 2>&1 || true
fi

exit 0
PRERM

cat > "${control_dir}/postrm" <<'POSTRM'
#!/bin/sh
set -e

if command -v systemctl >/dev/null 2>&1; then
  systemctl daemon-reload || true
fi

exit 0
POSTRM
chmod 0755 "${control_dir}/postinst" "${control_dir}/prerm" "${control_dir}/postrm"

installed_size="$(du -sk "$package_root" | awk '{print $1}')"
deb_binaries=("${package_root}/usr/bin/gail")
if [[ -x "${package_root}/usr/bin/gail-qlora-sft" ]]; then
  deb_binaries+=("${package_root}/usr/bin/gail-qlora-sft")
fi
depends="$(nm_compute_deb_depends "${deb_binaries[@]}")"
[[ -n "$depends" ]] || depends='ca-certificates, libc6, libgcc-s1'

cat > "${control_dir}/control" <<CONTROL
Package: gail
Version: ${deb_version}
Section: net
Priority: optional
Architecture: ${deb_arch}
Maintainer: NeuralMimicry <support@neuralmimicry.ai>
Depends: ${depends}
Installed-Size: ${installed_size}
Homepage: https://github.com/neuralmimicry/gail
Description: Gateway AI and neuromorphic middleware for NeuralMimicry services
 Gail consolidates LLM routing, provider orchestration, neuromorphic
 specialist access, AER translation, transcription, and trading bridge
 integration behind one Rust HTTP service.
CONTROL

mkdir -p "$out_dir"
package_file="${out_dir}/gail_${deb_version}_${deb_arch}.deb"
dpkg-deb --build --root-owner-group "$package_root" "$package_file" >/dev/null
printf '%s\n' "$package_file"
