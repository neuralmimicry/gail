#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/build-deb.sh --version VERSION --arch amd64|arm64 --binary PATH [options]

Options:
  --deb-version VERSION  Debian package version. Defaults to --version.
  --out-dir PATH        Directory for the resulting .deb. Defaults to dist.
  --config PATH         Gail config to install. Defaults to gail.yaml.
  --routing PATH        Routing profiles JSON. Defaults to config/ai-routing-profiles.json.
USAGE
}

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="${VERSION:-}"
deb_version="${DEB_VERSION:-}"
deb_arch="${DEB_ARCH:-}"
binary_path="${BINARY_PATH:-}"
out_dir="${OUT_DIR:-dist}"
config_path="${CONFIG_PATH:-${root_dir}/gail.yaml}"
routing_path="${ROUTING_PROFILES_PATH:-${root_dir}/config/ai-routing-profiles.json}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      version="$2"
      shift 2
      ;;
    --deb-version)
      deb_version="$2"
      shift 2
      ;;
    --arch)
      deb_arch="$2"
      shift 2
      ;;
    --binary)
      binary_path="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --config)
      config_path="$2"
      shift 2
      ;;
    --routing)
      routing_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${version}" || -z "${deb_arch}" || -z "${binary_path}" ]]; then
  usage
  exit 2
fi

if [[ -z "${deb_version}" ]]; then
  deb_version="${version}"
fi

case "${deb_arch}" in
  amd64|arm64)
    ;;
  *)
    echo "Unsupported Debian architecture: ${deb_arch}" >&2
    exit 2
    ;;
esac

if [[ ! "${deb_version}" =~ ^[0-9][0-9A-Za-z.+~-]*$ ]]; then
  echo "Invalid Debian package version: ${deb_version}" >&2
  exit 2
fi

if [[ ! -x "${binary_path}" ]]; then
  echo "Binary is missing or not executable: ${binary_path}" >&2
  exit 2
fi

if [[ ! -f "${config_path}" ]]; then
  echo "Config file is missing: ${config_path}" >&2
  exit 2
fi

if [[ ! -f "${routing_path}" ]]; then
  echo "Routing profiles file is missing: ${routing_path}" >&2
  exit 2
fi

if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "dpkg-deb is required to build Debian packages." >&2
  exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

package_root="${tmp_dir}/gail"
control_dir="${package_root}/DEBIAN"
mkdir -p "${control_dir}"

install -D -m 0755 "${binary_path}" "${package_root}/usr/bin/gail"
install -D -m 0644 "${config_path}" "${package_root}/etc/gail/gail.yaml"
install -D -m 0644 "${routing_path}" "${package_root}/etc/gail/ai-routing-profiles.json"
install -D -m 0644 "${root_dir}/packaging/deb/gail.service" "${package_root}/lib/systemd/system/gail.service"
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

installed_size="$(du -sk "${package_root}" | awk '{print $1}')"

cat > "${control_dir}/control" <<CONTROL
Package: gail
Version: ${deb_version}
Section: net
Priority: optional
Architecture: ${deb_arch}
Maintainer: NeuralMimicry <support@neuralmimicry.ai>
Depends: ca-certificates, libc6, libgcc-s1
Installed-Size: ${installed_size}
Homepage: https://github.com/neuralmimicry/gail
Description: Gateway AI and neuromorphic middleware for NeuralMimicry services
 Gail consolidates LLM routing, provider orchestration, neuromorphic
 specialist access, AER translation, transcription, and trading bridge
 integration behind one Rust HTTP service.
CONTROL

mkdir -p "${out_dir}"
package_file="${out_dir}/gail_${deb_version}_${deb_arch}.deb"
dpkg-deb --build --root-owner-group "${package_root}" "${package_file}" >/dev/null

echo "${package_file}"
