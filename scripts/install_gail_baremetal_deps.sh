#!/usr/bin/env bash
#
# install_gail_baremetal_deps.sh
#
# Bare-metal Ubuntu dependency installer equivalent to the Gail Containerfile.
#
# Installs:
#   - Rust/Cargo build dependencies
#   - C/C++/CMake/Ninja toolchain
#   - OpenCL headers, ICD loader, clinfo and CPU OpenCL fallback
#   - libtorch for Rust tch/torch-sys, via prebuilt archive where available
#     or PyTorch source build fallback
#   - Python virtual environment and trainer dependencies
#   - Gail runtime directories and GPU/OpenCL detection helper
#
# Designed for Ubuntu bare metal on amd64/x86_64 and arm64/aarch64.
# Safe defaults favour CPU/no-GPU builds and OpenCL compatibility. GPU builds are
# attempted when requested, but unsupported accelerator combinations can fall
# back to CPU unless strict mode is enabled.
#
# Usage examples:
#
#   chmod +x install_gail_baremetal_deps.sh
#   sudo ./install_gail_baremetal_deps.sh
#
#   sudo BUILD_JOBS="$(nproc)" CARGO_BUILD_JOBS="$(nproc)" \
#     PYTORCH_BUILD_PARALLEL_LEVEL=2 ./install_gail_baremetal_deps.sh
#
#   sudo LIBTORCH_ACCELERATOR=cu124 ./install_gail_baremetal_deps.sh
#
#   sudo LIBTORCH_BUILD_FROM_SOURCE=true PYTORCH_BUILD_PARALLEL_LEVEL=2 \
#     ./install_gail_baremetal_deps.sh
#
#   BUILD_GAIL=true ./install_gail_baremetal_deps.sh
#
set -Eeuo pipefail

trap 'echo "ERROR: line ${LINENO}: command failed: ${BASH_COMMAND}" >&2' ERR

: "${INSTALL_PREFIX:=/opt}"
: "${LIBTORCH_DIR:=${INSTALL_PREFIX}/libtorch}"
: "${GAIL_PYTHON_DIR:=${INSTALL_PREFIX}/gail-python}"
: "${GAIL_APP_DIR:=/app}"
: "${GAIL_STATE_DIR:=/var/lib/gail}"
: "${GAIL_USER:=gail}"
: "${GAIL_UID:=10001}"
: "${GAIL_GID:=10001}"

# Keep aligned with the Rust tch/torch-sys version in Cargo.lock.
# The earlier build log showed torch-sys v0.24.0 expecting PyTorch/libtorch 2.11.0.
: "${LIBTORCH_VERSION:=2.11.0}"
: "${LIBTORCH_ACCELERATOR:=cpu}"       # cpu, cuda, cu118, cu121, cu124, cu126, cu128, auto, none, no-gpu
: "${LIBTORCH_URL:=}"
: "${LIBTORCH_BUILD_FROM_SOURCE:=auto}" # true, false, auto
: "${LIBTORCH_AMD64_BUILD_FROM_SOURCE:=false}"
: "${LIBTORCH_ARM64_BUILD_FROM_SOURCE:=auto}"
: "${LIBTORCH_ALLOW_CPU_FALLBACK:=true}"
: "${LIBTORCH_STRICT_ACCELERATOR:=false}"
: "${LIBTORCH_DOWNLOAD_FALLBACK_TO_SOURCE:=true}"
: "${LIBTORCH_BUILD_TYPE:=Release}"

: "${PYTORCH_GIT_REPOSITORY:=https://github.com/pytorch/pytorch.git}"
: "${PYTORCH_GIT_TAG:=auto}"
: "${PYTORCH_GIT_TAG_STRICT:=false}"
: "${PYTORCH_SOURCE_CMAKE_MIN_VERSION:=3.27}"

: "${BUILD_JOBS:=auto}"
: "${CARGO_BUILD_JOBS:=auto}"
: "${CMAKE_BUILD_PARALLEL_LEVEL:=auto}"
: "${PYTORCH_BUILD_PARALLEL_LEVEL:=auto}"

: "${INSTALL_RUSTUP:=true}"
: "${INSTALL_PYTHON_TRAINING_DEPS:=true}"
: "${INSTALL_PYTHON_TORCH:=true}"
: "${PYTHON_TORCH_REQUIRED:=false}"
: "${PYTORCH_PIP_INDEX_URL:=}"
: "${INSTALL_BITSANDBYTES:=auto}"
: "${BUILD_GAIL:=false}"
: "${CREATE_GAIL_USER:=true}"
: "${WRITE_PROFILE:=true}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
fatal() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

as_root() {
  if [[ ${EUID} -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

as_login_user() {
  local target_user="${SUDO_USER:-}"
  if [[ ${EUID} -eq 0 && -n "${target_user}" && "${target_user}" != "root" ]]; then
    sudo -u "${target_user}" -H "$@"
  else
    "$@"
  fi
}

normalise_arch() {
  case "$(uname -m)" in
    x86_64|amd64) echo "amd64" ;;
    aarch64|arm64) echo "arm64" ;;
    *) uname -m ;;
  esac
}

resolve_jobs() {
  local host_jobs="$(nproc)"
  [[ -z "${BUILD_JOBS}" || "${BUILD_JOBS}" == "auto" ]] && BUILD_JOBS="${host_jobs}"
  [[ -z "${CARGO_BUILD_JOBS}" || "${CARGO_BUILD_JOBS}" == "auto" ]] && CARGO_BUILD_JOBS="${BUILD_JOBS}"
  [[ -z "${CMAKE_BUILD_PARALLEL_LEVEL}" || "${CMAKE_BUILD_PARALLEL_LEVEL}" == "auto" ]] && CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"
  [[ -z "${PYTORCH_BUILD_PARALLEL_LEVEL}" || "${PYTORCH_BUILD_PARALLEL_LEVEL}" == "auto" ]] && PYTORCH_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL}"
  export BUILD_JOBS CARGO_BUILD_JOBS CMAKE_BUILD_PARALLEL_LEVEL PYTORCH_BUILD_PARALLEL_LEVEL
  export MAKEFLAGS="-j${BUILD_JOBS}"
  export MAX_JOBS="${PYTORCH_BUILD_PARALLEL_LEVEL}"
  log "Parallelism: BUILD_JOBS=${BUILD_JOBS}, CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}, CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}, PYTORCH_BUILD_PARALLEL_LEVEL=${PYTORCH_BUILD_PARALLEL_LEVEL}"
}

install_system_packages() {
  log "Installing Ubuntu system dependencies"
  as_root apt-get update
  as_root apt-get install -y --no-install-recommends \
    apt-transport-https build-essential ca-certificates ccache clang cmake curl \
    dpkg-dev git jq kmod libblas-dev libffi-dev libgomp1 libjpeg-dev \
    liblapack-dev libopenblas-dev libpng-dev libssl-dev libssl3 ninja-build \
    ocl-icd-libopencl1 ocl-icd-opencl-dev opencl-headers pciutils pkg-config \
    pocl-opencl-icd python3 python3-dev python3-pip python3-venv unzip zlib1g-dev
}

install_rust() {
  [[ "${INSTALL_RUSTUP}" != "true" ]] && { log "Skipping rustup installation"; return 0; }
  if command -v rustup >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
    log "Rust toolchain already present: $(rustc --version), $(cargo --version)"
    return 0
  fi
  log "Installing Rust toolchain with rustup"
  as_login_user sh -lc 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
  warn "Rust installed. Open a new shell or source ~/.cargo/env before building manually."
}

resolve_libtorch_plan() {
  NORMALISED_ARCH="$(normalise_arch)"
  REQUESTED_ACCELERATOR="${LIBTORCH_ACCELERATOR:-auto}"
  case "${REQUESTED_ACCELERATOR}" in
    auto|none|no-gpu|nogpu|cpu) EFFECTIVE_ACCELERATOR="cpu" ;;
    cuda) EFFECTIVE_ACCELERATOR="cu124" ;;
    cu118|cu121|cu124|cu126|cu128) EFFECTIVE_ACCELERATOR="${REQUESTED_ACCELERATOR}" ;;
    *)
      [[ "${LIBTORCH_STRICT_ACCELERATOR}" == "true" ]] && fatal "Unsupported LIBTORCH_ACCELERATOR=${LIBTORCH_ACCELERATOR}"
      warn "Unsupported LIBTORCH_ACCELERATOR=${LIBTORCH_ACCELERATOR}; falling back to CPU libtorch"
      EFFECTIVE_ACCELERATOR="cpu"
      ;;
  esac

  BUILD_LIBTORCH_FROM_SOURCE="false"
  if [[ "${LIBTORCH_BUILD_FROM_SOURCE}" == "true" ]]; then
    BUILD_LIBTORCH_FROM_SOURCE="true"
  elif [[ "${LIBTORCH_BUILD_FROM_SOURCE}" == "false" ]]; then
    BUILD_LIBTORCH_FROM_SOURCE="false"
  elif [[ "${NORMALISED_ARCH}" == "arm64" ]]; then
    if [[ "${LIBTORCH_ARM64_BUILD_FROM_SOURCE}" == "false" && -n "${LIBTORCH_URL}" ]]; then
      BUILD_LIBTORCH_FROM_SOURCE="false"
    else
      BUILD_LIBTORCH_FROM_SOURCE="true"
    fi
  elif [[ "${NORMALISED_ARCH}" == "amd64" && "${LIBTORCH_AMD64_BUILD_FROM_SOURCE}" == "true" ]]; then
    BUILD_LIBTORCH_FROM_SOURCE="true"
  fi

  if [[ "${NORMALISED_ARCH}" == "arm64" && "${EFFECTIVE_ACCELERATOR}" != "cpu" && -z "${LIBTORCH_URL}" ]]; then
    if [[ "${LIBTORCH_STRICT_ACCELERATOR}" == "true" || "${LIBTORCH_ALLOW_CPU_FALLBACK}" != "true" ]]; then
      fatal "No generic official arm64 CUDA libtorch archive is assumed. Provide LIBTORCH_URL or allow CPU fallback."
    fi
    warn "Requested ${EFFECTIVE_ACCELERATOR} on arm64 without LIBTORCH_URL; falling back to CPU source-built libtorch"
    EFFECTIVE_ACCELERATOR="cpu"
    BUILD_LIBTORCH_FROM_SOURCE="true"
  fi

  if [[ "${BUILD_LIBTORCH_FROM_SOURCE}" == "true" && "${EFFECTIVE_ACCELERATOR}" != "cpu" ]]; then
    if [[ "${LIBTORCH_STRICT_ACCELERATOR}" == "true" || "${LIBTORCH_ALLOW_CPU_FALLBACK}" != "true" ]]; then
      fatal "Source-building GPU libtorch requires vendor CUDA/ROCm libraries. Provide LIBTORCH_URL or use amd64 official CUDA archive."
    fi
    warn "Source-build path is CPU-only in this script; falling back to CPU libtorch"
    EFFECTIVE_ACCELERATOR="cpu"
  fi

  log "libtorch plan: arch=${NORMALISED_ARCH}, requested=${REQUESTED_ACCELERATOR}, effective=${EFFECTIVE_ACCELERATOR}, version=${LIBTORCH_VERSION}, source_build=${BUILD_LIBTORCH_FROM_SOURCE}"
}

download_libtorch_archive() {
  local libtorch_url archive_suffix archive_dir encoded_version tmp_zip
  if [[ -n "${LIBTORCH_URL}" ]]; then
    libtorch_url="${LIBTORCH_URL}"
  elif [[ "${NORMALISED_ARCH}" == "amd64" ]]; then
    case "${EFFECTIVE_ACCELERATOR}" in
      cpu) archive_suffix="cpu"; archive_dir="cpu" ;;
      cu118|cu121|cu124|cu126|cu128) archive_suffix="${EFFECTIVE_ACCELERATOR}"; archive_dir="${EFFECTIVE_ACCELERATOR}" ;;
      *) fatal "Unsupported effective accelerator ${EFFECTIVE_ACCELERATOR}" ;;
    esac
    encoded_version="$(printf '%s' "${LIBTORCH_VERSION}+${archive_suffix}" | sed 's/+/%2B/g')"
    libtorch_url="https://download.pytorch.org/libtorch/${archive_dir}/libtorch-cxx11-abi-shared-with-deps-${encoded_version}.zip"
  else
    return 1
  fi
  log "Downloading libtorch from ${libtorch_url}"
  tmp_zip="$(mktemp /tmp/libtorch.XXXXXX.zip)"
  if curl -fL --retry 5 --retry-delay 2 "${libtorch_url}" -o "${tmp_zip}"; then
    as_root rm -rf "${LIBTORCH_DIR}"
    as_root mkdir -p "${INSTALL_PREFIX}"
    as_root unzip -q "${tmp_zip}" -d "${INSTALL_PREFIX}"
    rm -f "${tmp_zip}"
    return 0
  fi
  rm -f "${tmp_zip}"
  return 1
}

build_libtorch_from_source() {
  local pytorch_tag expected_tag build_root venv cmake_bin
  pytorch_tag="${PYTORCH_GIT_TAG}"
  expected_tag="v${LIBTORCH_VERSION}"
  if [[ -z "${pytorch_tag}" || "${pytorch_tag}" == "auto" ]]; then
    pytorch_tag="${expected_tag}"
  elif [[ "${pytorch_tag}" != "${expected_tag}" && "${PYTORCH_GIT_TAG_STRICT}" != "true" ]]; then
    warn "PYTORCH_GIT_TAG=${pytorch_tag} does not match LIBTORCH_VERSION=${LIBTORCH_VERSION}; using ${expected_tag} to satisfy torch-sys"
    pytorch_tag="${expected_tag}"
  fi

  log "Building CPU libtorch from PyTorch source tag ${pytorch_tag}; this can take a long time"
  build_root="$(mktemp -d /tmp/gail-pytorch-build.XXXXXX)"
  venv="${build_root}/pytorch-venv"
  python3 -m venv "${venv}"
  "${venv}/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging
  "${venv}/bin/python" -m pip install --no-cache-dir \
    "cmake>=${PYTORCH_SOURCE_CMAKE_MIN_VERSION},<4" astunparse cffi filelock \
    fsspec future hypothesis jinja2 networkx ninja numpy protobuf psutil \
    PyYAML requests six sympy typing_extensions
  cmake_bin="${venv}/bin/cmake"
  "${cmake_bin}" --version

  git clone --branch "${pytorch_tag}" --recurse-submodules "${PYTORCH_GIT_REPOSITORY}" "${build_root}/pytorch"
  export PYTHONPATH="${build_root}/pytorch:${PYTHONPATH:-}"
  "${venv}/bin/python" -c 'import sys, yaml, torchgen; print("Using PyTorch build Python:", sys.executable); print("PyYAML:", yaml.__file__); print("torchgen:", torchgen.__file__)'

  mkdir -p "${build_root}/pytorch-build"
  as_root rm -rf "${LIBTORCH_DIR}"
  as_root mkdir -p "${LIBTORCH_DIR}"
  pushd "${build_root}/pytorch-build" >/dev/null
  export BUILD_BINARY=0 BUILD_CUSTOM_PROTOBUF=ON BUILD_PYTHON=0 BUILD_TEST=0 BUILD_TORCH=ON
  export MAX_JOBS="${PYTORCH_BUILD_PARALLEL_LEVEL}"
  export USE_CUDA=0 USE_CUDNN=0 USE_DISTRIBUTED=0 USE_FBGEMM=0 USE_MKLDNN=0 USE_NCCL=0 USE_NUMA=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 USE_ROCM=0 USE_XNNPACK=1
  "${cmake_bin}" \
    -G Ninja \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DBUILD_TEST:BOOL=OFF \
    -DBUILD_PYTHON:BOOL=OFF \
    -DBUILD_CAFFE2_OPS:BOOL=ON \
    -DCMAKE_BUILD_TYPE:STRING="${LIBTORCH_BUILD_TYPE}" \
    -DPYTHON_EXECUTABLE:PATH="${venv}/bin/python" \
    -DPython_EXECUTABLE:PATH="${venv}/bin/python" \
    -DPython3_EXECUTABLE:PATH="${venv}/bin/python" \
    -DCMAKE_INSTALL_PREFIX:PATH="${LIBTORCH_DIR}" \
    -DCMAKE_PREFIX_PATH:PATH="${venv}" \
    -DUSE_CUDA:BOOL=OFF \
    -DUSE_CUDNN:BOOL=OFF \
    -DUSE_DISTRIBUTED:BOOL=OFF \
    -DUSE_FBGEMM:BOOL=OFF \
    -DUSE_MKLDNN:BOOL=OFF \
    -DUSE_NCCL:BOOL=OFF \
    -DUSE_NUMA:BOOL=OFF \
    -DUSE_QNNPACK:BOOL=OFF \
    -DUSE_PYTORCH_QNNPACK:BOOL=OFF \
    -DUSE_ROCM:BOOL=OFF \
    -DUSE_XNNPACK:BOOL=ON \
    "${build_root}/pytorch"
  "${cmake_bin}" --build . --target install --parallel "${PYTORCH_BUILD_PARALLEL_LEVEL}"
  popd >/dev/null
  rm -rf "${build_root}"
}

install_libtorch() {
  resolve_libtorch_plan
  if [[ -f "${LIBTORCH_DIR}/lib/libtorch.so" && -f "${LIBTORCH_DIR}/lib/libtorch_cpu.so" ]]; then
    log "libtorch already exists at ${LIBTORCH_DIR}"
  else
    if [[ "${BUILD_LIBTORCH_FROM_SOURCE}" != "true" ]]; then
      if ! download_libtorch_archive; then
        if [[ "${LIBTORCH_DOWNLOAD_FALLBACK_TO_SOURCE}" == "true" && "${EFFECTIVE_ACCELERATOR}" == "cpu" ]]; then
          warn "libtorch download failed; falling back to CPU source build"
          BUILD_LIBTORCH_FROM_SOURCE="true"
        elif [[ "${LIBTORCH_ALLOW_CPU_FALLBACK}" == "true" && "${LIBTORCH_STRICT_ACCELERATOR}" != "true" ]]; then
          warn "Accelerated libtorch download failed; falling back to CPU source build"
          EFFECTIVE_ACCELERATOR="cpu"
          BUILD_LIBTORCH_FROM_SOURCE="true"
        else
          fatal "libtorch download failed and no safe fallback is enabled"
        fi
      fi
    fi
    if [[ "${BUILD_LIBTORCH_FROM_SOURCE}" == "true" && ! -f "${LIBTORCH_DIR}/lib/libtorch.so" ]]; then
      build_libtorch_from_source
    fi
  fi
  [[ -f "${LIBTORCH_DIR}/lib/libtorch.so" ]] || fatal "Missing ${LIBTORCH_DIR}/lib/libtorch.so"
  [[ -f "${LIBTORCH_DIR}/lib/libtorch_cpu.so" ]] || fatal "Missing ${LIBTORCH_DIR}/lib/libtorch_cpu.so"
  as_root find "${LIBTORCH_DIR}" -type f -name '*.a' -delete || true
  as_root find "${LIBTORCH_DIR}" -type f -name '*.debug' -delete || true
  as_root find "${LIBTORCH_DIR}" -type f -name '*.pyc' -delete || true
  printf 'LIBTORCH_VERSION=%s\nLIBTORCH_ACCELERATOR=%s\nLIBTORCH_ARCH=%s\n' "${LIBTORCH_VERSION}" "${EFFECTIVE_ACCELERATOR}" "${NORMALISED_ARCH}" | as_root tee "${LIBTORCH_DIR}/gail-libtorch-build.env" >/dev/null
  echo "${LIBTORCH_DIR}/lib" | as_root tee /etc/ld.so.conf.d/gail-libtorch.conf >/dev/null
  as_root ldconfig
}

install_python_training_env() {
  [[ "${INSTALL_PYTHON_TRAINING_DEPS}" != "true" ]] && { log "Skipping Python training dependencies"; return 0; }
  log "Installing Python training environment at ${GAIL_PYTHON_DIR}"
  as_root rm -rf "${GAIL_PYTHON_DIR}"
  as_root python3 -m venv "${GAIL_PYTHON_DIR}"
  as_root "${GAIL_PYTHON_DIR}/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel
  local pip_arch pytorch_index_url
  pip_arch="$(normalise_arch)"
  if [[ "${INSTALL_PYTHON_TORCH}" == "true" ]]; then
    pytorch_index_url="${PYTORCH_PIP_INDEX_URL}"
    if [[ -z "${pytorch_index_url}" ]]; then
      case "${pip_arch}:${EFFECTIVE_ACCELERATOR:-${LIBTORCH_ACCELERATOR}}" in
        amd64:cpu) pytorch_index_url="https://download.pytorch.org/whl/cpu" ;;
        amd64:cu118|amd64:cu121|amd64:cu124|amd64:cu126|amd64:cu128) pytorch_index_url="https://download.pytorch.org/whl/${EFFECTIVE_ACCELERATOR:-${LIBTORCH_ACCELERATOR}}" ;;
        *) pytorch_index_url="" ;;
      esac
    fi
    if [[ -n "${pytorch_index_url}" ]]; then
      as_root "${GAIL_PYTHON_DIR}/bin/python" -m pip install --no-cache-dir --index-url "${pytorch_index_url}" torch || { [[ "${PYTHON_TORCH_REQUIRED}" == "true" ]] && fatal "Python torch install failed"; warn "Python torch install failed; continuing because native Rust tch uses ${LIBTORCH_DIR}"; }
    else
      as_root "${GAIL_PYTHON_DIR}/bin/python" -m pip install --no-cache-dir torch || { [[ "${PYTHON_TORCH_REQUIRED}" == "true" ]] && fatal "Python torch install failed"; warn "Python torch install failed; continuing because native Rust tch uses ${LIBTORCH_DIR}"; }
    fi
  fi
  as_root tee "${GAIL_PYTHON_DIR}/requirements-trainer.txt" >/dev/null <<'REQS_EOF'
transformers>=4.46,<5
accelerate>=1,<2
datasets>=3,<4
peft>=0.13,<1
trl>=0.11,<1
tokenizers>=0.20,<1
safetensors>=0.4,<1
sentencepiece>=0.2,<1
protobuf>=4,<6
REQS_EOF
  as_root "${GAIL_PYTHON_DIR}/bin/python" -m pip install --no-cache-dir -r "${GAIL_PYTHON_DIR}/requirements-trainer.txt"
  if [[ "${INSTALL_BITSANDBYTES}" == "true" || ( "${INSTALL_BITSANDBYTES}" == "auto" && "${pip_arch}" == "amd64" ) ]]; then
    as_root "${GAIL_PYTHON_DIR}/bin/python" -m pip install --no-cache-dir 'bitsandbytes>=0.44,<1' || warn "bitsandbytes could not be installed; native tch training can still run without it"
  else
    log "Skipping bitsandbytes for arch=${pip_arch}"
  fi
}

create_runtime_layout() {
  log "Creating Gail runtime directories and optional service user"
  if [[ "${CREATE_GAIL_USER}" == "true" ]]; then
    getent group "${GAIL_USER}" >/dev/null 2>&1 || as_root groupadd --system --gid "${GAIL_GID}" "${GAIL_USER}" || as_root groupadd --system "${GAIL_USER}"
    id "${GAIL_USER}" >/dev/null 2>&1 || as_root useradd --system --uid "${GAIL_UID}" --gid "${GAIL_USER}" --home-dir "${GAIL_APP_DIR}" --shell /usr/sbin/nologin "${GAIL_USER}" || as_root useradd --system --gid "${GAIL_USER}" --home-dir "${GAIL_APP_DIR}" --shell /usr/sbin/nologin "${GAIL_USER}"
  fi
  as_root mkdir -p "${GAIL_APP_DIR}/config" "${GAIL_APP_DIR}/data" "${GAIL_APP_DIR}/scripts" "${GAIL_STATE_DIR}"
  id "${GAIL_USER}" >/dev/null 2>&1 && as_root chown -R "${GAIL_USER}:${GAIL_USER}" "${GAIL_APP_DIR}" "${GAIL_STATE_DIR}" "${GAIL_PYTHON_DIR}" || true
}

install_detection_helper() {
  log "Installing GPU/OpenCL detection helper"
  as_root tee /usr/local/bin/gail-detect-gpu-opencl >/dev/null <<'DETECT_EOF'
#!/usr/bin/env bash
set -euo pipefail
export OCL_ICD_VENDORS="${OCL_ICD_VENDORS:-/etc/OpenCL/vendors}"
export OPENCL_VENDOR_PATH="${OPENCL_VENDOR_PATH:-/etc/OpenCL/vendors}"
opencl_device_count=0
if command -v clinfo >/dev/null 2>&1; then
  opencl_device_count="$(clinfo 2>/dev/null | awk -F: '/Number of devices/ {gsub(/^[[:space:]]+/, "", $2); total += $2} END {print total + 0}')"
fi
backend="none"; gpu_available="false"; opencl_available="false"
if [[ "${opencl_device_count:-0}" -gt 0 ]]; then opencl_available="true"; backend="opencl"; fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  gpu_available="true"; [[ "${backend}" == "opencl" ]] && backend="cuda+opencl" || backend="cuda"
elif compgen -G '/dev/nvidia*' >/dev/null 2>&1; then
  gpu_available="true"; [[ "${backend}" == "opencl" ]] && backend="nvidia+opencl" || backend="nvidia"
elif [[ -e /dev/kfd ]] || compgen -G '/dev/dri/renderD*' >/dev/null 2>&1; then
  gpu_available="true"; [[ "${backend}" == "opencl" ]] && backend="drm+opencl" || backend="drm"
fi
cat <<ENV_EOF
export GAIL_OPENCL_AVAILABLE=${opencl_available}
export GAIL_OPENCL_DEVICE_COUNT=${opencl_device_count:-0}
export GAIL_GPU_AVAILABLE=${gpu_available}
export GAIL_GPU_BACKEND=${backend}
export OCL_ICD_VENDORS=${OCL_ICD_VENDORS}
export OPENCL_VENDOR_PATH=${OPENCL_VENDOR_PATH}
ENV_EOF
DETECT_EOF
  as_root chmod 0755 /usr/local/bin/gail-detect-gpu-opencl
}

write_profile() {
  [[ "${WRITE_PROFILE}" != "true" ]] && return 0
  log "Writing /etc/profile.d/gail-libtorch.sh"
  as_root tee /etc/profile.d/gail-libtorch.sh >/dev/null <<PROFILE_EOF
# Gail/libtorch/OpenCL environment
export LIBTORCH=${LIBTORCH_DIR}
export LD_LIBRARY_PATH=${LIBTORCH_DIR}/lib:\${LD_LIBRARY_PATH:-}
export LIBTORCH_CXX11_ABI=1
export GAIL_PYTHON=${GAIL_PYTHON_DIR}/bin/python
export PATH=${GAIL_PYTHON_DIR}/bin:\${PATH}
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors
export GAIL_RUST_QLORA_SFT_BIN=/usr/bin/gail-qlora-sft
PROFILE_EOF
}

build_gail_if_requested() {
  [[ "${BUILD_GAIL}" != "true" ]] && return 0
  log "Building Gail from current source tree"
  [[ -f Cargo.toml ]] || fatal "BUILD_GAIL=true requested, but no Cargo.toml found in $(pwd)"
  export LIBTORCH="${LIBTORCH_DIR}"
  export LD_LIBRARY_PATH="${LIBTORCH_DIR}/lib:${LD_LIBRARY_PATH:-}"
  export LIBTORCH_CXX11_ABI=1
  export CMAKE_BUILD_PARALLEL_LEVEL MAKEFLAGS="-j${BUILD_JOBS}"
  if command -v cargo >/dev/null 2>&1; then
    cargo build --locked --release -j "${CARGO_BUILD_JOBS}"
  elif [[ -n "${SUDO_USER:-}" && -x "/home/${SUDO_USER}/.cargo/bin/cargo" ]]; then
    sudo -u "${SUDO_USER}" -H env LIBTORCH="${LIBTORCH}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LIBTORCH_CXX11_ABI=1 CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL}" MAKEFLAGS="${MAKEFLAGS}" /home/${SUDO_USER}/.cargo/bin/cargo build --locked --release -j "${CARGO_BUILD_JOBS}"
  else
    fatal "cargo not found. Install Rust or set INSTALL_RUSTUP=true."
  fi
}

print_summary() {
  log "Installation summary"
  cat <<SUMMARY_EOF
Architecture:                  $(normalise_arch)
libtorch:                      ${LIBTORCH_DIR}
libtorch version:              ${LIBTORCH_VERSION}
libtorch accelerator:          ${EFFECTIVE_ACCELERATOR:-unknown}
Python venv:                   ${GAIL_PYTHON_DIR}
Build jobs:                    ${BUILD_JOBS}
Cargo jobs:                    ${CARGO_BUILD_JOBS}
CMake jobs:                    ${CMAKE_BUILD_PARALLEL_LEVEL}
PyTorch/libtorch source jobs:  ${PYTORCH_BUILD_PARALLEL_LEVEL}

Load environment in the current shell:
  source /etc/profile.d/gail-libtorch.sh

Inspect GPU/OpenCL availability:
  gail-detect-gpu-opencl

Build Gail manually from the repository root:
  source /etc/profile.d/gail-libtorch.sh
  cargo build --locked --release -j "${CARGO_BUILD_JOBS}"
SUMMARY_EOF
}

main() {
  resolve_jobs
  install_system_packages
  install_rust
  install_libtorch
  install_python_training_env
  create_runtime_layout
  install_detection_helper
  write_profile
  build_gail_if_requested
  print_summary
}

main "$@"
