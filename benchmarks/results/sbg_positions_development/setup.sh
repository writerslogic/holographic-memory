#!/bin/bash
set -euo pipefail
cd ~/hms-sbg-positions
for variant in base candidate; do
  mkdir -p "$variant" "driver-$variant/src"
  tar -xzf "$variant-src.tar.gz" -C "$variant"
  cp query_kernel.rs "driver-$variant/src/main.rs"
  cat > "driver-$variant/Cargo.toml" <<MANIFEST
[package]
name = "posting-query-kernel"
version = "0.1.0"
edition = "2021"
[dependencies]
holographic-memory = { path = "../$variant" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
MANIFEST
  cp "$variant/Cargo.lock" "driver-$variant/Cargo.lock"
  (cd "driver-$variant"; CARGO_BUILD_JOBS=2 RUSTFLAGS='-C target-cpu=native' nice -n 19 cargo build --release) > "build-$variant.log" 2>&1
done
cp -a --reflink=auto ~/sbg-hms/dev/idx600 ./index600
# The independent protocol oracle uses the existing adapter, linked to this candidate.
mkdir -p driver-candidate/src/bin
cp adapter/build_index.rs adapter/do_query.rs verify_queries.rs driver-candidate/src/bin/
cp adapter/check_queries.py ./check_queries.py
(cd driver-candidate; CARGO_BUILD_JOBS=2 RUSTFLAGS='-C target-cpu=native' nice -n 19 cargo build --release --bins) > build-adapter.log 2>&1
touch verifier-ready
