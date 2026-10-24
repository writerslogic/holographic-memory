# Posting decode development measurement

These sources produced `../posting_decode_avx2_v1.json`. They are development
query kernels, not the leaderboard client. The fixture is the existing 600k-line
Wikipedia positional index and the unmodified search-benchmark-game query file.

Use Rust 1.99.1 on an AVX2 CPU. Create isolated parent/candidate source checkouts
at the commits in the result JSON. Under `~/hms-sbg-codec`, create
`driver-base/src/main.rs` and `driver-candidate/src/main.rs` from `query_kernel.rs`.
Each driver uses this manifest, with `VARIANT` replaced by `base` or `candidate`:

```toml
[package]
name = "posting-query-kernel"
version = "0.1.0"
edition = "2021"
[dependencies]
holographic-memory = { path = "../VARIANT" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

Build each driver with `RUSTFLAGS='-C target-cpu=native' cargo build --release`.
The scripts expect the upstream query file at `~/sbg-hms/sbg/queries.txt` and
the immutable 600k-line index at `~/sbg-hms/dev/idx600`. Copy the latter into
`~/hms-sbg-codec/index600` before running `paired.py`. Run Python via
`uv run --no-project python paired.py`.

The initial run opens each arm independently and writes `sample.json` plus
six records under `paired-results`. For the confirmation, place `persistent.py`
in the same directory and run it there. It copies two private indexes, keeps
both processes open, warms and calibrates the same query sample, and interleaves
each query across both arms for five rounds. Its repetition count targets
50 ms using the parent only, with a minimum of 5 and maximum of 500000.
The output is `persistent-results.json`.

Both scripts acquire `~/hms-timing.lock` around timing and pin the single query
thread to CPU 0. They record load averages and every returned result fingerprint.
The initial results, including losses, remain in the result JSON. Background
load and outliers preclude a blanket no-regression claim or a phrase-throughput
claim. Full-corpus standings remain unmeasured.

`correctness.py` uses the same driver to compare all 962 queries against exhaustive
evaluation on the 600k fixture for TOP_10, TOP_100_COUNT and COUNT. Run it with
`uv run --no-project python correctness.py`. With the `full` argument, it copies
the existing full index from `~/sbg-hms/sbg/engines/hms-before/idx` and checks all
962 COUNT answers against `~/sbg-hms/res/COUNT-tantivy-0.26.json`. These untimed
checks passed with zero mismatches; their result hashes are in the measurement JSON.
