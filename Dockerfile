# syntax=docker/dockerfile:1.7
# lancor — end-to-end llama.cpp toolkit (API client, HF Hub, server orchestration).
#
# Single-binary Rust tool. reqwest uses native-tls (openssl-sys), so the builder
# needs libssl-dev/pkg-config and the runtime needs libssl3 + ca-certificates.
# Build:
#   docker build -t lancor:local .
# Run:
#   docker run --rm lancor:local --help

ARG RUST_IMAGE=rust:1.99-bookworm
ARG RUNTIME_IMAGE=debian:bookworm-slim

FROM ${RUST_IMAGE} AS chef
RUN cargo install cargo-chef --locked
WORKDIR /workspace/lancor

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
ENV CARGO_TERM_COLOR=always
ENV RUSTFLAGS="-C strip=symbols"
# native-tls (openssl-sys) builds against system OpenSSL.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libssl-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*
COPY --from=planner /workspace/lancor/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/workspace/lancor/target \
    cargo chef cook --release --recipe-path recipe.json

COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/workspace/lancor/target \
    cargo build --release --bin lancor \
 && cp target/release/lancor /usr/local/bin/lancor

FROM ${RUNTIME_IMAGE} AS runtime
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates libssl3 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin lancor

COPY --from=builder /usr/local/bin/lancor /usr/local/bin/lancor

USER lancor
WORKDIR /home/lancor
ENV HOME=/home/lancor

ENTRYPOINT ["lancor"]
CMD ["--help"]
