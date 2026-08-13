#!/usr/bin/env bash
set -euo pipefail

# The NVIDIA 64K HWE kernel is a host package.  It must not be installed in
# an image layer, but it is required before a CUDA/libtorch container is used
# on Ubuntu 24.04 ARM64 hosts configured with 64K pages.
gail_ensure_64k_hwe_nvidia() {
  local accelerator="${1:-${LIBTORCH_ACCELERATOR:-cpu}}"
  local page_size
  page_size="$(getconf PAGESIZE 2>/dev/null || printf '0')"
  if [[ "${page_size}" != "65536" ]] || [[ ! -r /etc/os-release ]]; then
    return 0
  fi
  # shellcheck disable=SC1091
  . /etc/os-release
  [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] || return 0

  case "${accelerator,,}" in
    cpu|none|no-gpu|nogpu|auto) return 0 ;;
  esac
  if dpkg-query -W -f='${Status}' linux-nvidia-64k-hwe-24.04 2>/dev/null | grep -q 'install ok installed'; then
    return 0
  fi

  echo "Installing linux-nvidia-64k-hwe-24.04 for the 64K-page NVIDIA host"
  if [[ "${EUID}" -eq 0 ]]; then
    apt-get update
    apt-get install -y linux-nvidia-64k-hwe-24.04
  else
    sudo apt-get update
    sudo apt-get install -y linux-nvidia-64k-hwe-24.04
  fi
}

