# syntax=docker/dockerfile:1
#
# Gail runtime image.
#
# This image intentionally does not compile Gail from source. It installs the
# architecture-matching Debian package published on github.com/neuralmimicry/gail
# releases, defaulting to the latest release asset.
#
# Runtime configuration is copied from the repository build context rather than
# assuming the Debian package ships /etc/gail/gail.yaml. This keeps the image
# compatible with both binary-only .deb assets and fuller system packages.

FROM debian:bookworm-slim

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
      org.opencontainers.image.description="Gail runtime image installed from the published Debian release package"

# Copy runtime defaults from the source checkout/build context. These files are
# runtime inputs, not source-build inputs, so this does not reintroduce an
# in-container Gail compilation path.
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
            curl -fsSL -H "Authorization: Bearer ${GAIL_RELEASE_TOKEN}" "${GAIL_DEB_URL}" -o /tmp/gail.deb; \
        else \
            curl -fsSL "${GAIL_DEB_URL}" -o /tmp/gail.deb; \
        fi; \
    elif [ "${GAIL_VERSION}" = "latest" ]; then \
        release_json="$(github_api_get "https://api.github.com/repos/${GAIL_RELEASE_REPOSITORY}/releases/latest")"; \
        download_release_asset "${release_json}" "latest"; \
    elif [ "${GAIL_VERSION}" = "source" ]; then \
        echo "GAIL_VERSION=source is no longer supported by this Containerfile." >&2; \
        echo "Publish a Gail Debian package to GitHub Releases, then build with GAIL_VERSION=latest or a specific release tag." >&2; \
        exit 2; \
    else \
        release_version="${GAIL_VERSION#v}"; \
        release_json="$(github_api_get "https://api.github.com/repos/${GAIL_RELEASE_REPOSITORY}/releases/tags/v${release_version}" \
            || github_api_get "https://api.github.com/repos/${GAIL_RELEASE_REPOSITORY}/releases/tags/${release_version}")"; \
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
    rm -f /tmp/gail.deb; \
    mkdir -p /app/config /app/data /var/lib/gail /app/scripts; \
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
