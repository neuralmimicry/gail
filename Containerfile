FROM rust:1.88-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# Build with one Cargo job to keep memory usage predictable on smaller arm64 nodes.
RUN cargo build --locked --release -j 1

FROM debian:bookworm-slim
ARG APP_USER=gail
ARG APP_UID=10001
ARG APP_GID=10001
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" "${APP_USER}" \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /bin/sh "${APP_USER}" \
    && mkdir -p /app/config /app/data \
    && chown -R "${APP_UID}:${APP_GID}" /app
WORKDIR /app
COPY --from=builder /src/target/release/gail /usr/local/bin/gail
COPY gail.yaml /app/config/gail.yaml
ENV GAIL_CONFIG=/app/config/gail.yaml \
    GAIL_HEALTHCHECK_TOKEN= \
    RUST_LOG=info
EXPOSE 8080
VOLUME ["/app/config", "/app/data"]
USER ${APP_UID}:${APP_GID}
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD sh -c 'if [ -n "${GAIL_HEALTHCHECK_TOKEN}" ]; then curl -fsS -H "Authorization: Bearer ${GAIL_HEALTHCHECK_TOKEN}" http://127.0.0.1:8080/healthz >/dev/null; else curl -fsS http://127.0.0.1:8080/healthz >/dev/null; fi || exit 1'
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/gail"]
CMD ["--config", "/app/config/gail.yaml"]
