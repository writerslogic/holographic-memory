<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Posting allocation diagnostics

`profile.rs` compiles the actual posting implementation into a separate executable
with a counting `System` allocator. Its `StoredChunk` stub only satisfies the unused
bulk scanner's type requirements. The probe constructs 1,000 lists per case, with
0/1/127/128/129/256 postings, batches of 1 or 128, and either one dominating impact
or a full nondominated frontier. Input runs and subsequent decoding checks are
outside the counters; the list-owner vector is included. Calls include successful
allocations and reallocations. Requested bytes exclude allocator rounding and
metadata. These synthetic allocation counts do not measure whole-engine traffic.

Build the library once, then use its dependency directory and matching Rust compiler:

```sh
cargo build --locked --release --lib
sh benchmarks/research/postings/profile.sh target/release/deps /absolute/path/to/probe
```

The script compares the pinned `d6a38eb11ab0ff320929249d0ec9e26f16e46dad`
implementation with the working tree, saves both source snapshots and compiler
metadata, checks decoded postings, and writes JSONL receipts. It only changes inner
doc comments to regular comments to permit `include!`.

The ignored library test `reopened_index_memory_census` opens an existing index
through `HmsCore`, walks the retained owners, and emits a joint histogram of posting
count, block length and block capacity. It counts shared dictionaries once and
reports all posting-data and block spare capacity. The census is test-only and does
not change the public API or production allocation path.

```sh
HMS_MEMORY_INDEX=/absolute/path/to/index \
HMS_MEMORY_CENSUS_OUT=/absolute/path/to/census.json \
RAYON_NUM_THREADS=4 cargo test --locked --release --lib \
  reopened_index_memory_census -- --ignored --nocapture --test-threads=1
```

For process RSS, build first and invoke the resulting test executable directly
under `/usr/bin/time`; including Cargo would include build memory. Supply
`HMS_MEMORY_QUERIES=/absolute/path/to/queries.json` to compare complete DAAT and WAND
results against the exhaustive evaluator at k=0/10/1000. This JSON file contains an
array of query strings. Equality checks include counts, IDs, scores and ordering.

The census measures requested vector/box capacities and map payloads. It excludes
tree node overhead, allocator metadata/rounding, other engine components and its
own scratch allocations. Metadata/filter/vector ownership outside the measured
fields is explicitly counted as unmeasured. It is not a complete heap profiler.

The [paired 600k receipts](../../results/lexical_posting_allocations_2026_10_22/)
include the two-function repair patch. To reproduce the original allocation policy
with the same census in an isolated checkout, reverse that patch, build and save the
test executable, then reapply it and build the candidate. The capacity regression
test deliberately fails with the old policy; run only the ignored census there.
