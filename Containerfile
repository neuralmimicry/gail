FROM rust:1.88-bookworm AS source-deb

ARG GAIL_VERSION=source

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        dpkg-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY config ./config
COPY gail.yaml .
COPY packaging ./packaging
COPY scripts ./scripts
RUN set -eux; \
    if [ "${GAIL_VERSION}" != "source" ] && [ "${GAIL_VERSION}" != "latest" ]; then \
        release_version="${GAIL_VERSION#v}"; \
        bash scripts/set-release-version.sh "${release_version}"; \
    fi; \
    cargo build --locked --release -j 1; \
    package_version="$(sed -nE 's/^version = "([^"]+)"/\1/p' Cargo.toml | head -n 1)"; \
    deb_version="$(printf '%s' "${package_version}" | sed 's/-/~/g')"; \
    deb_arch="$(dpkg --print-architecture)"; \
    bash scripts/build-deb.sh \
        --version "${package_version}" \
        --deb-version "${deb_version}" \
        --arch "${deb_arch}" \
        --binary target/release/gail \
        --out-dir /out

FROM debian:bookworm-slim

ARG TARGETARCH
ARG GAIL_VERSION=source
ARG GAIL_DEB_URL=
ARG GAIL_RELEASE_REPOSITORY=neuralmimicry/gail
ARG GAIL_RELEASE_BASE_URL=
ARG APP_USER=gail
ARG APP_UID=10001
ARG APP_GID=10001

COPY --from=source-deb /out/gail_*.deb /tmp/source-gail.deb

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini; \
    groupadd --gid "${APP_GID}" "${APP_USER}"; \
    useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /bin/sh "${APP_USER}"; \
    detected_arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${detected_arch}" in \
        amd64|x86_64) deb_arch="amd64" ;; \
        arm64|aarch64) deb_arch="arm64" ;; \
        *) echo "Unsupported container architecture: ${detected_arch}" >&2; exit 2 ;; \
    esac; \
    release_base_url="${GAIL_RELEASE_BASE_URL:-https://github.com/${GAIL_RELEASE_REPOSITORY}/releases/download}"; \
    if [ -n "${GAIL_DEB_URL}" ]; then \
        echo "Installing Gail package for ${deb_arch} from ${GAIL_DEB_URL}"; \
        curl -fsSL "${GAIL_DEB_URL}" -o /tmp/gail.deb; \
    elif [ "${GAIL_VERSION}" = "source" ]; then \
        echo "Installing Gail package for ${deb_arch} from source-built .deb"; \
        source_deb_arch="$(dpkg-deb -f /tmp/source-gail.deb Architecture)"; \
        if [ "${source_deb_arch}" != "${deb_arch}" ]; then \
            echo "Source-built Gail package architecture ${source_deb_arch} does not match target ${deb_arch}" >&2; \
            exit 2; \
        fi; \
        cp /tmp/source-gail.deb /tmp/gail.deb; \
    elif [ "${GAIL_VERSION}" = "latest" ]; then \
        deb_url="$(curl -fsSL "https://api.github.com/repos/${GAIL_RELEASE_REPOSITORY}/releases/latest" \
            | sed -nE "s/.*\"browser_download_url\": \"([^\"]*gail_[^\"]*_${deb_arch}\\.deb)\".*/\\1/p" \
            | head -n 1)"; \
        if [ -z "${deb_url}" ]; then \
            echo "Could not resolve latest Gail ${deb_arch} .deb release asset" >&2; \
            exit 2; \
        fi; \
        echo "Installing Gail package for ${deb_arch} from ${deb_url}"; \
        curl -fsSL "${deb_url}" -o /tmp/gail.deb; \
    else \
        release_version="${GAIL_VERSION#v}"; \
        deb_version="$(printf '%s' "${release_version}" | sed 's/-/~/g')"; \
        deb_url="${release_base_url}/v${release_version}/gail_${deb_version}_${deb_arch}.deb"; \
        echo "Installing Gail package for ${deb_arch} from ${deb_url}"; \
        curl -fsSL "${deb_url}" -o /tmp/gail.deb; \
    fi; \
    apt-get install -y --no-install-recommends /tmp/gail.deb; \
    rm -f /tmp/gail.deb /tmp/source-gail.deb; \
    mkdir -p /app/config /app/data; \
    cp /etc/gail/gail.yaml /app/config/gail.yaml; \
    cp /etc/gail/ai-routing-profiles.json /app/config/ai-routing-profiles.json; \
    chown -R "${APP_UID}:${APP_GID}" /app /var/lib/gail; \
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
