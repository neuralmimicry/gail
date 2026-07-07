# syntax=docker/dockerfile:1
#
# Gail combined runtime image.
#
# Goals:
#   - Build consistently for linux/amd64 and linux/arm64.
#   - Support CPU/no-GPU builds without error.
#   - Support amd64 CUDA libtorch builds when an official PyTorch archive exists.
#   - Fall back safely to CPU libtorch when a requested GPU/accelerator build is
#     not available, unless LIBTORCH_STRICT_ACCELERATOR=true.
#   - Provide OpenCL build/runtime compatibility for both amd64 and arm64.
#   - Keep libtorch aligned with the Rust tch/torch-sys crate expectation.
#
# Recommended CPU builds:
#   JOBS="$(nproc)"
#   podman build --platform linux/amd64 --build-arg GAIL_VERSION=source --build-arg LIBTORCH_ACCELERATOR=cpu --build-arg BUILD_JOBS="${JOBS}" --build-arg CARGO_BUILD_JOBS="${JOBS}" -f Containerfile -t gail:amd64 .
#   podman build --platform linux/arm64 --build-arg GAIL_VERSION=source --build-arg LIBTORCH_ACCELERATOR=cpu --build-arg BUILD_JOBS="${JOBS}" --build-arg PYTORCH_BUILD_PARALLEL_LEVEL=2 -f Containerfile -t gail:arm64 .
#
# Parallelism:
#   BUILD_JOBS=auto uses all CPUs visible to the build container.
#   CARGO_BUILD_JOBS controls Rust compilation parallelism.
#   CMAKE_BUILD_PARALLEL_LEVEL controls CMake/Ninja parallelism.
#   PYTORCH_BUILD_PARALLEL_LEVEL controls libtorch/PyTorch source-build parallelism.
#   For libtorch source builds, set PYTORCH_BUILD_PARALLEL_LEVEL lower than
#   nproc if memory pressure or OOM kills occur.
#
# Recommended amd64 CUDA build:
#   podman build --platform linux/amd64 --build-arg GAIL_VERSION=source --build-arg LIBTORCH_ACCELERATOR=cu124 -f Containerfile -t gail:amd64-cu124 .
#
# Reusing a self-built local libtorch seed image:
#   podman build --platform linux/arm64 --target libtorch-export -f Containerfile -t gail-libtorch:arm64 .
#   podman build --platform linux/arm64 --build-arg LIBTORCH_SEED_IMAGE=gail-libtorch:arm64 --build-arg GAIL_VERSION=source -f Containerfile -t gail:arm64 .
#
# GitHub Actions/self-hosted runner notes:
#   - Build amd64 images on a self-hosted Linux X64 runner.
#   - Build arm64 images on a self-hosted Linux ARM64 runner.
#   - Avoid QEMU emulation for libtorch/PyTorch source builds.
#   - Pass BUILD_JOBS/CARGO_BUILD_JOBS/PYTORCH_BUILD_PARALLEL_LEVEL from the workflow.
#
# Strict mode:
#   --build-arg LIBTORCH_STRICT_ACCELERATOR=true
#

ARG LIBTORCH_SEED_IMAGE=docker.io/library/debian:bookworm-slim
FROM ${LIBTORCH_SEED_IMAGE} AS libtorch-seed

FROM docker.io/library/debian:bookworm-slim AS libtorch

ARG TARGETARCH
ARG LIBTORCH_SEED_IMAGE=docker.io/library/debian:bookworm-slim
ARG LIBTORCH_VERSION=2.11.0
ARG LIBTORCH_ACCELERATOR=cpu
ARG LIBTORCH_URL=
ARG LIBTORCH_BUILD_FROM_SOURCE=auto
ARG LIBTORCH_AMD64_BUILD_FROM_SOURCE=false
ARG LIBTORCH_ARM64_BUILD_FROM_SOURCE=auto
ARG LIBTORCH_ALLOW_CPU_FALLBACK=true
ARG LIBTORCH_STRICT_ACCELERATOR=false
ARG LIBTORCH_DOWNLOAD_FALLBACK_TO_SOURCE=true
ARG LIBTORCH_ARM64_SVE=auto
ARG LIBTORCH_ARM64_XNNPACK=auto
ARG LIBTORCH_ARM64_KLEIDIAI=false
ARG PYTORCH_GIT_REPOSITORY=https://github.com/pytorch/pytorch.git
ARG PYTORCH_GIT_TAG=auto
ARG PYTORCH_GIT_TAG_STRICT=false
ARG BUILD_JOBS=auto
ARG CMAKE_BUILD_PARALLEL_LEVEL=auto
ARG PYTORCH_BUILD_PARALLEL_LEVEL=auto
ARG LIBTORCH_BUILD_TYPE=Release
ARG PYTORCH_SOURCE_CMAKE_MIN_VERSION=3.27

ENV DEBIAN_FRONTEND=noninteractive \
    BUILD_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} \
    PYTORCH_BUILD_PARALLEL_LEVEL=${PYTORCH_BUILD_PARALLEL_LEVEL} \
    MAKEFLAGS=-j${BUILD_JOBS}

RUN mkdir -p /opt/libtorch
COPY --from=libtorch-seed /opt /opt

RUN set -eu; \
    host_jobs="$(nproc)"; \
    if [ "${BUILD_JOBS:-auto}" = "auto" ] || [ -z "${BUILD_JOBS:-}" ]; then BUILD_JOBS="${host_jobs}"; fi; \
    if [ "${CMAKE_BUILD_PARALLEL_LEVEL:-auto}" = "auto" ] || [ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"; fi; \
    if [ "${PYTORCH_BUILD_PARALLEL_LEVEL:-auto}" = "auto" ] || [ -z "${PYTORCH_BUILD_PARALLEL_LEVEL:-}" ]; then PYTORCH_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL}"; fi; \
    export BUILD_JOBS CMAKE_BUILD_PARALLEL_LEVEL PYTORCH_BUILD_PARALLEL_LEVEL; \
    export MAKEFLAGS="-j${BUILD_JOBS}"; \
    echo "Build parallelism: BUILD_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} PYTORCH_BUILD_PARALLEL_LEVEL=${PYTORCH_BUILD_PARALLEL_LEVEL}"; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        unzip; \
    rm -rf /var/lib/apt/lists/*; \
    detected_arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${detected_arch}" in \
        amd64|x86_64) norm_arch="amd64" ;; \
        arm64|aarch64) norm_arch="arm64" ;; \
        *) norm_arch="${detected_arch}" ;; \
    esac; \
    arm64_sve="false"; \
    if [ "${norm_arch}" = "arm64" ]; then \
        case "${LIBTORCH_ARM64_SVE:-auto}" in \
            auto|"") \
                if grep -m1 -E '^Features[[:space:]]*:.*(^|[[:space:]])sve([[:space:]]|$)' /proc/cpuinfo >/dev/null 2>&1; then \
                    arm64_sve="true"; \
                else \
                    arm64_sve="false"; \
                fi; \
                ;; \
            true|1|yes|on) arm64_sve="true" ;; \
            false|0|no|off) arm64_sve="false" ;; \
            *) \
                echo "Unsupported LIBTORCH_ARM64_SVE=${LIBTORCH_ARM64_SVE}; use auto, true or false" >&2; \
                exit 2; \
                ;; \
        esac; \
    fi; \
    echo "ARM64 SVE detection: arch=${norm_arch} sve=${arm64_sve}"; \
    requested_accelerator="${LIBTORCH_ACCELERATOR:-auto}"; \
    case "${requested_accelerator}" in \
        auto|none|no-gpu|nogpu|cpu) effective_accelerator="cpu" ;; \
        cuda) effective_accelerator="cu124" ;; \
        cu118|cu121|cu124|cu126|cu128) effective_accelerator="${requested_accelerator}" ;; \
        *) \
            if [ "${LIBTORCH_STRICT_ACCELERATOR}" = "true" ]; then \
                echo "Unsupported LIBTORCH_ACCELERATOR=${LIBTORCH_ACCELERATOR}" >&2; \
                exit 2; \
            fi; \
            echo "Unsupported LIBTORCH_ACCELERATOR=${LIBTORCH_ACCELERATOR}; falling back to CPU libtorch" >&2; \
            effective_accelerator="cpu"; \
            ;; \
    esac; \
    build_from_source="false"; \
    if [ "${LIBTORCH_BUILD_FROM_SOURCE}" = "true" ]; then \
        build_from_source="true"; \
    elif [ "${LIBTORCH_BUILD_FROM_SOURCE}" = "false" ]; then \
        build_from_source="false"; \
    elif [ "${norm_arch}" = "arm64" ]; then \
        if [ "${LIBTORCH_ARM64_BUILD_FROM_SOURCE}" = "false" ] && [ -n "${LIBTORCH_URL}" ]; then \
            build_from_source="false"; \
        else \
            build_from_source="true"; \
        fi; \
    elif [ "${norm_arch}" = "amd64" ] && [ "${LIBTORCH_AMD64_BUILD_FROM_SOURCE}" = "true" ]; then \
        build_from_source="true"; \
    fi; \
    if [ "${norm_arch}" = "arm64" ] && [ "${effective_accelerator}" != "cpu" ] && [ -z "${LIBTORCH_URL}" ]; then \
        if [ "${LIBTORCH_STRICT_ACCELERATOR}" = "true" ] || [ "${LIBTORCH_ALLOW_CPU_FALLBACK}" != "true" ]; then \
            echo "No generic official arm64 CUDA libtorch archive is assumed. Provide LIBTORCH_URL or allow CPU fallback." >&2; \
            exit 2; \
        fi; \
        echo "Requested ${effective_accelerator} on arm64 without LIBTORCH_URL; falling back to CPU source-built libtorch" >&2; \
        effective_accelerator="cpu"; \
        build_from_source="true"; \
    fi; \
    if [ "${build_from_source}" = "true" ] && [ "${effective_accelerator}" != "cpu" ]; then \
        if [ "${LIBTORCH_STRICT_ACCELERATOR}" = "true" ] || [ "${LIBTORCH_ALLOW_CPU_FALLBACK}" != "true" ]; then \
            echo "Source-building GPU libtorch requires a CUDA/ROCm-enabled base image and extra vendor libraries; provide LIBTORCH_URL or use an official amd64 CUDA archive." >&2; \
            exit 2; \
        fi; \
        echo "Source-build path is CPU-only in this Containerfile; falling back to CPU libtorch" >&2; \
        effective_accelerator="cpu"; \
    fi; \
    echo "libtorch configuration: arch=${norm_arch} requested=${requested_accelerator} effective=${effective_accelerator} version=${LIBTORCH_VERSION} source_build=${build_from_source}"; \
    reuse_cached_libtorch="false"; \
    if [ -f /opt/libtorch/lib/libtorch.so ] || [ -f /opt/libtorch/lib/libtorch_cpu.so ]; then \
        if [ -f /opt/libtorch/gail-libtorch-build.env ]; then \
            cached_version="$(awk -F= '/^LIBTORCH_VERSION=/{print $2; exit}' /opt/libtorch/gail-libtorch-build.env)"; \
            cached_accelerator="$(awk -F= '/^LIBTORCH_ACCELERATOR=/{print $2; exit}' /opt/libtorch/gail-libtorch-build.env)"; \
            cached_arch="$(awk -F= '/^LIBTORCH_ARCH=/{print $2; exit}' /opt/libtorch/gail-libtorch-build.env)"; \
            if [ "${cached_version}" = "${LIBTORCH_VERSION}" ] && [ "${cached_accelerator}" = "${effective_accelerator}" ] && [ "${cached_arch}" = "${norm_arch}" ]; then \
                reuse_cached_libtorch="true"; \
                build_from_source="false"; \
                echo "Reusing seeded /opt/libtorch cache for arch=${norm_arch} accelerator=${effective_accelerator} version=${LIBTORCH_VERSION}"; \
            else \
                echo "Seeded /opt/libtorch cache mismatch (version=${cached_version} accelerator=${cached_accelerator} arch=${cached_arch}); rebuilding"; \
                rm -rf /opt/libtorch; \
                mkdir -p /opt/libtorch; \
            fi; \
        else \
            echo "Seeded /opt/libtorch missing gail-libtorch-build.env; rebuilding"; \
            rm -rf /opt/libtorch; \
            mkdir -p /opt/libtorch; \
        fi; \
    fi; \
    if [ "${reuse_cached_libtorch}" != "true" ]; then \
        if [ -n "${LIBTORCH_URL}" ] || [ "${build_from_source}" != "true" ]; then \
            if [ -n "${LIBTORCH_URL}" ]; then \
                libtorch_url="${LIBTORCH_URL}"; \
            elif [ "${norm_arch}" = "amd64" ]; then \
                case "${effective_accelerator}" in \
                    cpu) archive_suffix="cpu"; archive_dir="cpu" ;; \
                    cu118|cu121|cu124|cu126|cu128) archive_suffix="${effective_accelerator}"; archive_dir="${effective_accelerator}" ;; \
                    *) echo "Unsupported effective accelerator ${effective_accelerator}" >&2; exit 2 ;; \
                esac; \
                encoded_version="$(printf '%s' "${LIBTORCH_VERSION}+${archive_suffix}" | sed 's/+/%2B/g')"; \
                libtorch_url="https://download.pytorch.org/libtorch/${archive_dir}/libtorch-cxx11-abi-shared-with-deps-${encoded_version}.zip"; \
            else \
                libtorch_url=""; \
                build_from_source="true"; \
            fi; \
            if [ "${build_from_source}" != "true" ]; then \
                echo "Downloading libtorch from ${libtorch_url}"; \
                if curl -fL --retry 5 --retry-delay 2 "${libtorch_url}" -o /tmp/libtorch.zip; then \
                    unzip -q /tmp/libtorch.zip -d /opt; \
                    build_from_source="false"; \
                elif [ "${LIBTORCH_DOWNLOAD_FALLBACK_TO_SOURCE}" = "true" ] && [ "${effective_accelerator}" = "cpu" ]; then \
                    echo "libtorch download failed; falling back to CPU source build" >&2; \
                    rm -f /tmp/libtorch.zip; \
                    build_from_source="true"; \
                elif [ "${LIBTORCH_ALLOW_CPU_FALLBACK}" = "true" ] && [ "${LIBTORCH_STRICT_ACCELERATOR}" != "true" ]; then \
                    echo "Accelerated libtorch download failed; falling back to CPU source build" >&2; \
                    rm -f /tmp/libtorch.zip; \
                    effective_accelerator="cpu"; \
                    build_from_source="true"; \
                else \
                    echo "libtorch download failed and no safe fallback is enabled" >&2; \
                    exit 2; \
                fi; \
            fi; \
        fi; \
        if [ "${build_from_source}" = "true" ]; then \
            pytorch_tag="${PYTORCH_GIT_TAG}"; \
            expected_pytorch_tag="v${LIBTORCH_VERSION}"; \
            if [ "${pytorch_tag}" = "auto" ] || [ -z "${pytorch_tag}" ]; then \
                pytorch_tag="${expected_pytorch_tag}"; \
            elif [ "${pytorch_tag}" != "${expected_pytorch_tag}" ] && [ "${PYTORCH_GIT_TAG_STRICT}" != "true" ]; then \
                echo "PYTORCH_GIT_TAG=${pytorch_tag} does not match LIBTORCH_VERSION=${LIBTORCH_VERSION}; using ${expected_pytorch_tag} to satisfy torch-sys" >&2; \
                pytorch_tag="${expected_pytorch_tag}"; \
            fi; \
            echo "Building CPU libtorch directly with CMake from PyTorch source tag ${pytorch_tag} for ${norm_arch}"; \
            apt-get update; \
            apt-get install -y --no-install-recommends \
                build-essential \
                ccache \
                cmake \
                git \
                libblas-dev \
                libffi-dev \
                libjpeg-dev \
                liblapack-dev \
                libopenblas-dev \
                libpng-dev \
                libssl-dev \
                ninja-build \
                ocl-icd-opencl-dev \
                opencl-headers \
                pkg-config \
                python3 \
                python3-dev \
                python3-venv \
                zlib1g-dev; \
            rm -rf /var/lib/apt/lists/*; \
            python3 -m venv /tmp/pytorch-venv; \
            /tmp/pytorch-venv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging; \
            /tmp/pytorch-venv/bin/python -m pip install --no-cache-dir \
                "cmake>=${PYTORCH_SOURCE_CMAKE_MIN_VERSION},<4" \
                astunparse \
                cffi \
                filelock \
                fsspec \
                future \
                hypothesis \
                jinja2 \
                networkx \
                ninja \
                numpy \
                protobuf \
                psutil \
                PyYAML \
                requests \
                six \
                sympy \
                typing_extensions; \
            cmake_bin="/tmp/pytorch-venv/bin/cmake"; \
            "${cmake_bin}" --version; \
            git clone --branch "${pytorch_tag}" --recurse-submodules "${PYTORCH_GIT_REPOSITORY}" /tmp/pytorch; \
            export PYTHONPATH="/tmp/pytorch:${PYTHONPATH:-}"; \
            /tmp/pytorch-venv/bin/python -c 'import sys, yaml, torchgen; print("Using PyTorch build Python: %s" % sys.executable); print("PyYAML module: %s" % yaml.__file__); print("torchgen module: %s" % torchgen.__file__)'; \
            mkdir -p /tmp/pytorch-build /opt/libtorch; \
            cd /tmp/pytorch-build; \
            export BUILD_BINARY=0; \
            export BUILD_CUSTOM_PROTOBUF=ON; \
            export BUILD_PYTHON=0; \
            export BUILD_TEST=0; \
            export BUILD_TORCH=ON; \
            export MAX_JOBS="${PYTORCH_BUILD_PARALLEL_LEVEL}"; \
            export USE_CUDA=0; \
            export USE_CUDNN=0; \
            export USE_DISTRIBUTED=0; \
            export USE_FBGEMM=0; \
            export USE_MKLDNN=0; \
            export USE_NCCL=0; \
            export USE_NUMA=0; \
            export USE_QNNPACK=0; \
            export USE_PYTORCH_QNNPACK=0; \
            export USE_ROCM=0; \
            caffe2_perf_with_sve="OFF"; \
            caffe2_perf_with_sve256="OFF"; \
            pytorch_use_xnnpack="ON"; \
            pytorch_use_xnnpack="OFF"; \
            pytorch_use_kleidiai="OFF"; \
            if [ "${norm_arch}" = "arm64" ]; then \
                case "${LIBTORCH_ARM64_SVE:-false}" in \
                    true|1|yes|on) \
                        if [ "${arm64_sve}" = "true" ]; then \
                            caffe2_perf_with_sve="ON"; \
                        else \
                            echo "LIBTORCH_ARM64_SVE=true requested, but SVE was not detected; keeping SVE OFF" >&2; \
                        fi; \
                        ;; \
                    auto|"") \
                        echo "LIBTORCH_ARM64_SVE=auto detected SVE=${arm64_sve}, but release builds keep SVE OFF unless explicitly enabled"; \
                        ;; \
                    false|0|no|off) \
                        caffe2_perf_with_sve="OFF"; \
                        ;; \
                    *) \
                        echo "Unsupported LIBTORCH_ARM64_SVE=${LIBTORCH_ARM64_SVE}; use auto, true or false" >&2; \
                        exit 2; \
                        ;; \
                esac; \
                case "${LIBTORCH_ARM64_XNNPACK:-auto}" in \
                    auto|"") \
                        if [ "${arm64_sve}" = "true" ]; then \
                            pytorch_use_xnnpack="OFF"; \
                        else \
                            pytorch_use_xnnpack="OFF"; \
                        fi; \
                        ;; \
                    true|1|yes|on) pytorch_use_xnnpack="ON" ;; \
                    false|0|no|off) pytorch_use_xnnpack="OFF" ;; \
                    *) \
                        echo "Unsupported LIBTORCH_ARM64_XNNPACK=${LIBTORCH_ARM64_XNNPACK}; use auto, true or false" >&2; \
                        exit 2; \
                        ;; \
                esac; \
            fi; \
            echo "ARM64 PyTorch CPU features: CAFFE2_PERF_WITH_SVE=${caffe2_perf_with_sve} CAFFE2_PERF_WITH_SVE256=${caffe2_perf_with_sve256} USE_XNNPACK=${pytorch_use_xnnpack} USE_KLEIDIAI=${pytorch_use_kleidiai}"; \
            if [ "${pytorch_use_xnnpack}" = "ON" ]; then \
                export USE_XNNPACK=1; \
            else \
                export USE_XNNPACK=0; \
            fi; \
            if [ "${pytorch_use_kleidiai}" = "ON" ]; then \
                export USE_KLEIDIAI=1; \
            else \
                export USE_KLEIDIAI=0; \
            fi; \
            "${cmake_bin}" \
                -G Ninja \
                -DBUILD_SHARED_LIBS:BOOL=ON \
                -DBUILD_TEST:BOOL=OFF \
                -DBUILD_PYTHON:BOOL=OFF \
                -DBUILD_CAFFE2_OPS:BOOL=ON \
                -DCMAKE_BUILD_TYPE:STRING="${LIBTORCH_BUILD_TYPE}" \
                -DPYTHON_EXECUTABLE:PATH=/tmp/pytorch-venv/bin/python \
                -DPython_EXECUTABLE:PATH=/tmp/pytorch-venv/bin/python \
                -DPython3_EXECUTABLE:PATH=/tmp/pytorch-venv/bin/python \
                -DCMAKE_INSTALL_PREFIX:PATH=/opt/libtorch \
                -DCMAKE_PREFIX_PATH:PATH=/tmp/pytorch-venv \
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
                -DUSE_KLEIDIAI:BOOL="${pytorch_use_kleidiai}" \
                -DUSE_XNNPACK:BOOL="${pytorch_use_xnnpack}" \
                -DCAFFE2_PERF_WITH_SVE:BOOL="${caffe2_perf_with_sve}" \
                -DCAFFE2_PERF_WITH_SVE256:BOOL="${caffe2_perf_with_sve256}" \
                ../pytorch; \
            "${cmake_bin}" --build . --target install --parallel "${PYTORCH_BUILD_PARALLEL_LEVEL}"; \
            rm -rf /tmp/pytorch /tmp/pytorch-build /tmp/pytorch-venv /root/.cache; \
        fi; \
    fi; \
    test -f /opt/libtorch/lib/libtorch.so; \
    test -f /opt/libtorch/lib/libtorch_cpu.so; \
    rm -f /tmp/libtorch.zip; \
    find /opt/libtorch -type f -name '*.a' -delete; \
    find /opt/libtorch -type f -name '*.debug' -delete; \
    find /opt/libtorch -type f -name '*.pyc' -delete; \
    printf 'LIBTORCH_VERSION=%s\nLIBTORCH_ACCELERATOR=%s\nLIBTORCH_ARCH=%s\n' "${LIBTORCH_VERSION}" "${effective_accelerator}" "${norm_arch}" > /opt/libtorch/gail-libtorch-build.env

FROM scratch AS libtorch-export
COPY --from=libtorch /opt/libtorch /opt/libtorch

FROM docker.io/library/rust:1-bookworm AS source-deb

ARG GAIL_VERSION=latest
ARG LIBTORCH_VERSION=2.11.0
ARG LIBTORCH_ACCELERATOR=cpu
ARG LIBTORCH_URL=
ARG BUILD_JOBS=auto
ARG CARGO_BUILD_JOBS=auto
ARG CMAKE_BUILD_PARALLEL_LEVEL=auto

ENV DEBIAN_FRONTEND=noninteractive \
    BUILD_JOBS=${BUILD_JOBS} \
    CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        dpkg-dev \
        libblas-dev \
        liblapack-dev \
        libopenblas-dev \
        libgomp1 \
        libssl-dev \
        ocl-icd-opencl-dev \
        opencl-headers \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv; \
    rm -rf /var/lib/apt/lists/*; \
    ldconfig

WORKDIR /src

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY config ./config
COPY gail.yaml .
COPY packaging ./packaging
COPY scripts ./scripts

RUN set -eu; \
    host_jobs="$(nproc)"; \
    if [ "${BUILD_JOBS:-auto}" = "auto" ] || [ -z "${BUILD_JOBS:-}" ]; then BUILD_JOBS="${host_jobs}"; fi; \
    if [ "${CARGO_BUILD_JOBS:-auto}" = "auto" ] || [ -z "${CARGO_BUILD_JOBS:-}" ]; then CARGO_BUILD_JOBS="${BUILD_JOBS}"; fi; \
    if [ "${CMAKE_BUILD_PARALLEL_LEVEL:-auto}" = "auto" ] || [ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]; then CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"; fi; \
    export BUILD_JOBS CARGO_BUILD_JOBS CMAKE_BUILD_PARALLEL_LEVEL; \
    export MAKEFLAGS="-j${BUILD_JOBS}"; \
    echo "Source build parallelism: BUILD_JOBS=${BUILD_JOBS} CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}"; \
    if [ "${GAIL_VERSION}" != "source" ] && [ "${GAIL_VERSION}" != "latest" ]; then \
        release_version="${GAIL_VERSION#v}"; \
        if [ -x scripts/set-release-version.sh ]; then \
            bash scripts/set-release-version.sh "${release_version}"; \
        fi; \
    fi; \
    cargo build --locked --release --bin gail --no-default-features -j "${CARGO_BUILD_JOBS}"; \
    package_version="$(sed -nE 's/^version = "([^"]+)"/\1/p' Cargo.toml | head -n 1)"; \
    if [ -z "${package_version}" ]; then \
        echo "Could not determine Gail package version from Cargo.toml" >&2; \
        exit 2; \
    fi; \
    deb_version="$(printf '%s' "${package_version}" | sed 's/-/~/g')"; \
    deb_arch="$(dpkg --print-architecture)"; \
    bash scripts/build-deb.sh \
        --version "${package_version}" \
        --deb-version "${deb_version}" \
        --arch "${deb_arch}" \
        --binary target/release/gail \
        --out-dir /out

FROM docker.io/library/debian:bookworm-slim

ARG TARGETARCH
ARG GAIL_VERSION=latest
ARG GAIL_DEB_URL=
ARG GAIL_RELEASE_REPOSITORY=neuralmimicry/gail
ARG GAIL_RELEASE_BASE_URL=
ARG GAIL_RELEASE_TOKEN=
ARG APP_USER=gail
ARG APP_UID=10001
ARG APP_GID=10001
ARG GAIL_INSTALL_PYTHON_TRAINING_DEPS=true
ARG GAIL_PYTHON_TORCH_INSTALL=true
ARG GAIL_PYTHON_TORCH_REQUIRED=false
ARG PYTORCH_PIP_INDEX_URL=
ARG LIBTORCH_ACCELERATOR=cpu
ARG BUILD_VERSION=dev
ARG VCS_REF=unknown
ARG GITHUB_REPOSITORY=neuralmimicry/gail
ARG IMAGE_CREATED=unknown

ENV DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.source="https://github.com/${GITHUB_REPOSITORY}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.description="Gail runtime image with native libtorch/tch training, Python training tooling and OpenCL runtime detection"

COPY --from=source-deb /out/*.deb /tmp/source-gail.deb
COPY gail.yaml /tmp/gail-defaults/gail.yaml
COPY config/ai-routing-profiles.json /tmp/gail-defaults/ai-routing-profiles.json

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        clinfo \
        curl \
        jq \
        kmod \
        libgomp1 \
        libblas-dev \
        liblapack-dev \
        libopenblas-dev \
        libssl3 \
        ocl-icd-libopencl1 \
        pciutils \
        pocl-opencl-icd \
        python3 \
        python3-pip \
        python3-venv \
        tini; \
    groupadd --system --gid "${APP_GID}" "${APP_USER}"; \
    useradd --system --uid "${APP_UID}" --gid "${APP_GID}" --home-dir /app --shell /usr/sbin/nologin "${APP_USER}"; \
    deb_arch="$(dpkg --print-architecture)"; \
    github_api_get() { \
        url="$1"; \
        if [ -n "${GAIL_RELEASE_TOKEN}" ]; then \
            curl -fsSL -H "Authorization: Bearer ${GAIL_RELEASE_TOKEN}" -H "Accept: application/vnd.github+json" "${url}"; \
        else \
            curl -fsSL -H "Accept: application/vnd.github+json" "${url}"; \
        fi; \
    }; \
    download_release_asset() { \
        release_json="$1"; \
        selector_description="$2"; \
        asset_url="$(printf '%s' "${release_json}" | jq -r --arg arch "${deb_arch}" '.assets[] | select(.name | test("^(gail|GAIL)_.*_" + $arch + "\\.deb$"; "i")) | .browser_download_url' | head -n 1)"; \
        if [ -z "${asset_url}" ] || [ "${asset_url}" = "null" ]; then \
            echo "Could not resolve ${selector_description} Gail ${deb_arch} .deb release asset URL" >&2; \
            echo "If the repository is private, provide GAIL_RELEASE_TOKEN." >&2; \
            exit 2; \
        fi; \
        echo "Installing Gail ${deb_arch} package from ${asset_url}"; \
        curl -fsSL "${asset_url}" -o /tmp/gail.deb; \
    }; \
    if [ -n "${GAIL_DEB_URL}" ]; then \
        echo "Installing Gail ${deb_arch} package from explicit GAIL_DEB_URL"; \
        if [ -n "${GAIL_RELEASE_TOKEN}" ]; then \
            curl -fsSL -H "Authorization: Bearer ${GAIL_RELEASE_TOKEN}" "${GAIL_DEB_URL}" -o /tmp/gail.deb; \
        else \
            curl -fsSL "${GAIL_DEB_URL}" -o /tmp/gail.deb; \
        fi; \
    elif [ "${GAIL_VERSION}" = "source" ]; then \
        echo "Installing Gail ${deb_arch} package from source-built Debian package"; \
        source_deb_arch="$(dpkg-deb -f /tmp/source-gail.deb Architecture)"; \
        if [ "${source_deb_arch}" != "${deb_arch}" ]; then \
            echo "Source-built Gail package architecture ${source_deb_arch} does not match target ${deb_arch}" >&2; \
            exit 2; \
        fi; \
        cp /tmp/source-gail.deb /tmp/gail.deb; \
    elif [ "${GAIL_VERSION}" = "latest" ]; then \
        release_json="$(github_api_get "https://api.github.com/repos/${GAIL_RELEASE_REPOSITORY}/releases/latest")"; \
        download_release_asset "${release_json}" "latest"; \
    else \
        release_version="${GAIL_VERSION#v}"; \
        release_json="$(github_api_get "https://api.github.com/repos/${GAIL_RELEASE_REPOSITORY}/releases/tags/v${release_version}" || github_api_get "https://api.github.com/repos/${GAIL_RELEASE_REPOSITORY}/releases/tags/${release_version}" || true)"; \
        if [ -n "${release_json}" ]; then \
            download_release_asset "${release_json}" "Gail ${release_version}"; \
        else \
            release_base_url="${GAIL_RELEASE_BASE_URL:-https://github.com/${GAIL_RELEASE_REPOSITORY}/releases/download}"; \
            deb_version="$(printf '%s' "${release_version}" | sed 's/-/~/g')"; \
            deb_url="${release_base_url}/v${release_version}/gail_${deb_version}_${deb_arch}.deb"; \
            echo "Installing Gail ${deb_arch} package from fallback URL ${deb_url}"; \
            curl -fsSL "${deb_url}" -o /tmp/gail.deb; \
        fi; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends /tmp/gail.deb; \
    rm -f /tmp/gail.deb /tmp/source-gail.deb; \
    mkdir -p /app/config /app/data /app/scripts /var/lib/gail; \
    if [ -f /tmp/gail-defaults/gail.yaml ]; then \
        cp /tmp/gail-defaults/gail.yaml /app/config/gail.yaml; \
    elif [ -f /etc/gail/gail.yaml ]; then \
        cp /etc/gail/gail.yaml /app/config/gail.yaml; \
    else \
        echo "Missing Gail runtime config: expected gail.yaml in build context or /etc/gail/gail.yaml from package" >&2; \
        exit 2; \
    fi; \
    if [ -f /tmp/gail-defaults/ai-routing-profiles.json ]; then \
        cp /tmp/gail-defaults/ai-routing-profiles.json /app/config/ai-routing-profiles.json; \
    elif [ -f /etc/gail/ai-routing-profiles.json ]; then \
        cp /etc/gail/ai-routing-profiles.json /app/config/ai-routing-profiles.json; \
    else \
        echo "Missing Gail routing profiles: expected config/ai-routing-profiles.json in build context or /etc/gail/ai-routing-profiles.json from package" >&2; \
        exit 2; \
    fi; \
    mkdir -p /opt/gail-python; \
    python3 -m venv /opt/gail-python; \
    /opt/gail-python/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    { \
        printf '%s\n' 'transformers>=4.46,<5'; \
        printf '%s\n' 'accelerate>=1,<2'; \
        printf '%s\n' 'datasets>=3,<4'; \
        printf '%s\n' 'peft>=0.13,<1'; \
        printf '%s\n' 'trl>=0.11,<1'; \
        printf '%s\n' 'tokenizers>=0.20,<1'; \
        printf '%s\n' 'safetensors>=0.4,<1'; \
        printf '%s\n' 'sentencepiece>=0.2,<1'; \
        printf '%s\n' 'protobuf>=4,<6'; \
    } > /opt/gail-python/requirements-trainer.txt; \
    if [ "${GAIL_INSTALL_PYTHON_TRAINING_DEPS}" = "true" ]; then \
        detected_arch_for_pip="${TARGETARCH:-$(dpkg --print-architecture)}"; \
        case "${detected_arch_for_pip}" in amd64|x86_64) pip_arch="amd64" ;; arm64|aarch64) pip_arch="arm64" ;; *) pip_arch="${detected_arch_for_pip}" ;; esac; \
        if [ "${GAIL_PYTHON_TORCH_INSTALL}" = "true" ]; then \
            pytorch_index_url="${PYTORCH_PIP_INDEX_URL}"; \
            if [ -z "${pytorch_index_url}" ]; then \
                case "${pip_arch}:${LIBTORCH_ACCELERATOR}" in \
                    amd64:cpu) pytorch_index_url="https://download.pytorch.org/whl/cpu" ;; \
                    amd64:cu118|amd64:cu121|amd64:cu124|amd64:cu126|amd64:cu128) pytorch_index_url="https://download.pytorch.org/whl/${LIBTORCH_ACCELERATOR}" ;; \
                    *) pytorch_index_url="" ;; \
                esac; \
            fi; \
            if [ -n "${pytorch_index_url}" ]; then \
                /opt/gail-python/bin/python -m pip install --no-cache-dir --index-url "${pytorch_index_url}" torch || { \
                    if [ "${GAIL_PYTHON_TORCH_REQUIRED}" = "true" ]; then exit 2; fi; \
                    echo "Python torch install failed; continuing because native Rust tch uses /opt/libtorch" >&2; \
                }; \
            else \
                /opt/gail-python/bin/python -m pip install --no-cache-dir torch || { \
                    if [ "${GAIL_PYTHON_TORCH_REQUIRED}" = "true" ]; then exit 2; fi; \
                    echo "Python torch install failed; continuing because native Rust tch uses /opt/libtorch" >&2; \
                }; \
            fi; \
        fi; \
        /opt/gail-python/bin/python -m pip install --no-cache-dir -r /opt/gail-python/requirements-trainer.txt; \
        if [ "${pip_arch}" = "amd64" ]; then \
            /opt/gail-python/bin/python -m pip install --no-cache-dir 'bitsandbytes>=0.44,<1' || echo "bitsandbytes could not be installed; native tch training can still run without it" >&2; \
        else \
            echo "Skipping bitsandbytes on ${pip_arch}; no reliable pre-built wheel is assumed" >&2; \
        fi; \
    fi; \
    ldconfig; \
    chown -R "${APP_UID}:${APP_GID}" /app /var/lib/gail /opt/gail-python; \
    rm -rf /tmp/gail-defaults; \
    apt-get purge -y --auto-remove jq; \
    rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    { \
        printf '%s\n' '#!/bin/sh'; \
        printf '%s\n' 'set -eu'; \
        printf '%s\n' ''; \
        printf '%s\n' 'export OCL_ICD_VENDORS="${OCL_ICD_VENDORS:-/etc/OpenCL/vendors}"'; \
        printf '%s\n' 'export OPENCL_VENDOR_PATH="${OPENCL_VENDOR_PATH:-/etc/OpenCL/vendors}"'; \
        printf '%s\n' ''; \
        printf '%s\n' 'opencl_device_count=0'; \
        printf '%s\n' 'if command -v clinfo >/dev/null 2>&1; then'; \
        printf '%s\n' '    opencl_device_count="$(clinfo 2>/dev/null | awk -F: '\''/Number of devices/ {gsub(/^[[:space:]]+/, "", $2); total += $2} END {print total + 0}'\'')"'; \
        printf '%s\n' 'fi'; \
        printf '%s\n' ''; \
        printf '%s\n' 'backend="none"'; \
        printf '%s\n' 'gpu_available="false"'; \
        printf '%s\n' 'opencl_available="false"'; \
        printf '%s\n' ''; \
        printf '%s\n' 'if [ "${opencl_device_count:-0}" -gt 0 ]; then'; \
        printf '%s\n' '    opencl_available="true"'; \
        printf '%s\n' '    backend="opencl"'; \
        printf '%s\n' 'fi'; \
        printf '%s\n' ''; \
        printf '%s\n' 'if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then'; \
        printf '%s\n' '    gpu_available="true"'; \
        printf '%s\n' '    if [ "${backend}" = "opencl" ]; then backend="cuda+opencl"; else backend="cuda"; fi'; \
        printf '%s\n' 'elif ls /dev/nvidia* >/dev/null 2>&1; then'; \
        printf '%s\n' '    gpu_available="true"'; \
        printf '%s\n' '    if [ "${backend}" = "opencl" ]; then backend="nvidia+opencl"; else backend="nvidia"; fi'; \
        printf '%s\n' 'elif [ -e /dev/kfd ] || ls /dev/dri/renderD* >/dev/null 2>&1; then'; \
        printf '%s\n' '    gpu_available="true"'; \
        printf '%s\n' '    if [ "${backend}" = "opencl" ]; then backend="drm+opencl"; else backend="drm"; fi'; \
        printf '%s\n' 'fi'; \
        printf '%s\n' ''; \
        printf '%s\n' 'export GAIL_OPENCL_AVAILABLE="${GAIL_OPENCL_AVAILABLE:-${opencl_available}}"'; \
        printf '%s\n' 'export GAIL_OPENCL_DEVICE_COUNT="${GAIL_OPENCL_DEVICE_COUNT:-${opencl_device_count:-0}}"'; \
        printf '%s\n' 'export GAIL_GPU_AVAILABLE="${GAIL_GPU_AVAILABLE:-${gpu_available}}"'; \
        printf '%s\n' 'export GAIL_GPU_BACKEND="${GAIL_GPU_BACKEND:-${backend}}"'; \
        printf '%s\n' ''; \
        printf '%s\n' 'exec /usr/bin/tini -- /usr/bin/gail "$@"'; \
    } > /usr/local/bin/gail-entrypoint.sh; \
    chmod 0755 /usr/local/bin/gail-entrypoint.sh

WORKDIR /app

ENV GAIL_CONFIG=/app/config/gail.yaml \
    GAIL_ROUTING_PROFILES_PATH=/app/config/ai-routing-profiles.json \
    GAIL_HEALTHCHECK_TOKEN= \
    LIBTORCH=/opt/libtorch \
    LD_LIBRARY_PATH=/opt/libtorch/lib \
    LIBTORCH_CXX11_ABI=1 \
    OCL_ICD_VENDORS=/etc/OpenCL/vendors \
    OPENCL_VENDOR_PATH=/etc/OpenCL/vendors \
    GAIL_OPENCL_AVAILABLE=false \
    GAIL_OPENCL_DEVICE_COUNT=0 \
    GAIL_GPU_AVAILABLE=false \
    GAIL_GPU_BACKEND=none \
    GAIL_RUST_QLORA_SFT_BIN=/usr/bin/gail-qlora-sft \
    GAIL_PYTHON=/opt/gail-python/bin/python \
    PATH=/opt/gail-python/bin:${PATH} \
    RUST_LOG=info

EXPOSE 8080

VOLUME ["/app/config", "/app/data"]

USER ${APP_UID}:${APP_GID}

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD sh -c 'if [ -n "${GAIL_HEALTHCHECK_TOKEN}" ]; then curl -fsS -H "Authorization: Bearer ${GAIL_HEALTHCHECK_TOKEN}" http://127.0.0.1:8080/healthz >/dev/null; else curl -fsS http://127.0.0.1:8080/healthz >/dev/null; fi || exit 1'

ENTRYPOINT ["/usr/local/bin/gail-entrypoint.sh"]

CMD ["--config", "/app/config/gail.yaml"]
