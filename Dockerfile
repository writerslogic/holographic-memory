FROM rust:1.89-alpine@sha256:4b800f2e72e04be908e5f634c504c741bd943b763d1d8ad7b096cc340e1b5b46 AS build
RUN apk add --no-cache build-base
WORKDIR /src
COPY . .
ENV CARGO_PROFILE_RELEASE_LTO=false
RUN cargo build --release --locked --features mcp,security --bin warm-mcp \
 && strip target/release/warm-mcp \
 && mkdir -p /seed/data && chown -R 65532:65532 /seed

FROM gcr.io/distroless/static-debian12:nonroot@sha256:afa5c872c891853ca7fcf1f12c3edb23f7eeef36189728842dd51042ff57f7ab
COPY --from=build /src/target/release/warm-mcp /warm-mcp
COPY --from=build --chown=65532:65532 /seed/data /data
ENV WARM_MCP_DATA_DIR=/data
USER 65532:65532
WORKDIR /data
VOLUME ["/data"]
ENTRYPOINT ["/warm-mcp"]
