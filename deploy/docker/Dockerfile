# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# hms-server: one static binary on distroless, non-root, read-only-root friendly.
#
#   docker build -f deploy/docker/Dockerfile -t hms-server .
#   docker run --rm -p 8080:8080 -e HMS_API_KEY=... -v hms-data:/data hms-server
#
# The build stage is Alpine, so the toolchain targets musl natively and the binary links
# statically; nothing but the binary and the certificate-free runtime base ships.

FROM rust:1.89-alpine AS build
RUN apk add --no-cache build-base
WORKDIR /src
COPY . .
# Only the server feature set: no model code, no research binaries.
RUN cargo build --release --locked --features server --bin hms-server \
 && strip target/release/hms-server \
 && mkdir -p /seed/data && chown -R 65532:65532 /seed

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /src/target/release/hms-server /hms-server
COPY --from=build --chown=65532:65532 /seed/data /data
ENV HMS_DATA_DIR=/data \
    HMS_BIND=0.0.0.0:8080 \
    HMS_LOG_FORMAT=json
USER 65532:65532
VOLUME ["/data"]
EXPOSE 8080
# No shell or curl in the image: the binary probes itself.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 CMD ["/hms-server", "healthcheck"]
STOPSIGNAL SIGTERM
ENTRYPOINT ["/hms-server"]
