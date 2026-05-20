# syntax=docker/dockerfile:1
#
# Gail combined runtime image.
#
# Supports two installation modes:
#
#   1. Release mode:
#      Installs an architecture-matching Gail Debian package from GitHub Releases
#      or from an explicit GAIL_DEB_URL.
#
#   2. Source mode:
#      Builds Gail from the local source tree into a Debian package, then installs
#      that package into the final runtime image.
#
# Recommended production default:
#
#   --build-arg GAIL_VERSION=latest
#
# Developer/source build:
#
#   --build-arg GAIL_VERSION=source
#

FROM docker.io/library/rust:1-bookworm AS source-deb

ARG GAIL_VERSION=latest

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        dpkg-dev \
        pkg-config; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY config ./config
COPY gail.yaml .
COPY packaging ./packaging
COPY scripts ./scripts

RUN set -eu; \
    if [ "${GAIL_VERSION}" != "source" ] && [ "${GAIL_VERSION}" != "latest" ]; then \
        release_version="${GAIL_VERSION#v}"; \
        if [ -x scripts/set-release-version.sh ]; then \
            bash scripts/set-release-version.sh "${release_version}"; \
        fi; \
    fi; \
    cargo build --locked --release -j 1; \
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

LABEL org.opencontainers.image.source="https://github.com/neuralmimicry/gail" \
      org.opencontainers.image.description="Gail runtime image installed from release package or source-built Debian package"

COPY --from=source-deb /out/gail_*.deb /tmp/source-gail.deb

# Runtime defaults from the build context.
# These are copied independently of the Debian package so the image works even
# when the package only contains the binary/service artefacts.
COPY gail.yaml /tmp/gail-defaults/gail.yaml
COPY config/ai-routing-profiles.json /tmp/gail-defaults/ai-routing-profiles.json
COPY scripts/trainer /app/scripts/trainer

RUN set -eu; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        jq \
        tini; \
    rm -rf /var/lib/apt/lists/*; \
    detected_arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${detected_arch}" in \
        amd64|x86_64) deb_arch="amd64" ;; \
        arm64|aarch64) deb_arch="arm64" ;; \
        *) echo "Unsupported container architecture: ${detected_arch}" >&2; exit 2 ;; \
    esac; \
    if ! getent group "${APP_USER}" >/dev/null 2>&1; then \
        groupadd --gid "${APP_GID}" "${APP_USER}"; \
    fi; \
    if ! id -u "${APP_USER}" >/dev/null 2>&1; then \
        useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /bin/sh "${APP_USER}"; \
    fi; \
    github_api_get() { \
        api_url="$1"; \
        if [ -n "${GAIL_RELEASE_TOKEN}" ]; then \
            curl -fsSL \
                -H "Authorization: Bearer ${GAIL_RELEASE_TOKEN}" \
                -H "Accept: application/vnd.github+json" \
                "${api_url}"; \
        else \
            curl -fsSL \
                -H "Accept: application/vnd.github+json" \
                "${api_url}"; \
        fi; \
    }; \
    download_release_asset() { \
        release_json="$1"; \
        selector_description="$2"; \
        if [ -n "${GAIL_RELEASE_TOKEN}" ]; then \
            asset_url="$(printf '%s' "${release_json}" \
                | jq -r --arg arch "${deb_arch}" \
                    '.assets[] | select(.name | test("^(gail|GAIL)_.*_" + $arch + "\\.deb$"; "i")) | .url' \
                | head -n 1)"; \
            if [ -z "${asset_url}" ] || [ "${asset_url}" = "null" ]; then \
                echo "Could not resolve ${selector_description} Gail ${deb_arch} .deb release asset API URL" >&2; \
                exit 2; \
            fi; \
            echo "Installing Gail ${deb_arch} package from authenticated GitHub release asset"; \
            curl -fsSL \
                -H "Authorization: Bearer ${GAIL_RELEASE_TOKEN}" \
                -H "Accept: application/octet-stream" \
                "${asset_url}" \
                -o /tmp/gail.deb; \
        else \
            asset_url="$(printf '%s' "${release_json}" \
                | jq -r --arg arch "${deb_arch}" \
                    '.assets[] | select(.name | test("^(gail|GAIL)_.*_" + $arch + "\\.deb$"; "i")) | .browser_download_url' \
                | head -n 1)"; \
            if [ -z "${asset_url}" ] || [ "${asset_url}" = "null" ]; then \
                echo "Could not resolve ${selector_description} Gail ${deb_arch} .deb release asset URL" >&2; \
                echo "If the repository is private, provide GAIL_RELEASE_TOKEN." >&2; \
                exit 2; \
            fi; \
            echo "Installing Gail ${deb_arch} package from ${asset_url}"; \
            curl -fsSL "${asset_url}" -o /tmp/gail.deb; \
        fi; \
    }; \
    if [ -n "${GAIL_DEB_URL}" ]; then \
        echo "Installing Gail ${deb_arch} package from explicit GAIL_DEB_URL"; \
        if [ -n "${GAIL_RELEASE_TOKEN}" ]; then \
            curl -fsSL \
                -H "Authorization: Bearer ${GAIL_RELEASE_TOKEN}" \
                "${GAIL_DEB_URL}" \
                -o /tmp/gail.deb; \
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
        release_json="$(github_api_get "https://api.github.com/repos/${GAIL_RELEASE_REPOSITORY}/releases/tags/v${release_version}" \
            || github_api_get "https://api.github.com/repos/${GAIL_RELEASE_REPOSITORY}/releases/tags/${release_version}" \
            || true)"; \
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
    chown -R "${APP_UID}:${APP_GID}" /app /var/lib/gail; \
    rm -rf /tmp/gail-defaults; \
    apt-get purge -y --auto-remove jq; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV GAIL_CONFIG=/app/config/gail.yaml \
    GAIL_ROUTING_PROFILES_PATH=/app/config/ai-routing-profiles.json \
    GAIL_HEALTHCHECK_TOKEN= \
    RUST_LOG=info

EXPOSE 8080

VOLUME ["/app/config", "/app/data"]

USER ${APP_UID}:${APP_GID}

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD sh -c 'if [ -n "${GAIL_HEALTHCHECK_TOKEN}" ]; then curl -fsS -H "Authorization: Bearer ${GAIL_HEALTHCHECK_TOKEN}" http://127.0.0.1:8080/healthz >/dev/null; else curl -fsS http://127.0.0.1:8080/healthz >/dev/null; fi || exit 1'

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/bin/gail"]

CMD ["--config", "/app/config/gail.yaml"]