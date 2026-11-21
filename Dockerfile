# syntax=docker/dockerfile:1
FROM rust:1.84.0-slim-bookworm AS builder
WORKDIR /build

COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -rf src

COPY src ./src
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim
WORKDIR /bot
COPY --from=builder /build/target/release/apex-mech /bot/apex-mech
USER 65534:65534
ENTRYPOINT ["/bot/apex-mech"]