# Ordered payload rank prototype

This preserves synthetic correctness and logical allocation evidence for compacting
base vector payloads while retaining the existing navigation graph and delta.
Only source snapshots and reproducible patches are published. Production code,
formats and defaults are unchanged; this is not an adopted consolidation method.

The rank map reuses native tombstone words and stores one cumulative deleted count
per 64 slots. It replaces the dense 4N-byte map with 12×ceil(N/64) buffer bytes,
including 8×ceil(N/64) bytes already present in the native bitmap. Payload order
must remain unchanged. For N original slots and L surviving rows, net extra heap
metadata is `4*ceil(N/64) - 4*(N-L)`, including the old/new ID arrays and bitmap.

On the proposed 240k-row, 100D, 1.1%-tombstone fixture, arithmetic predicts 919,440 → 4,440
extra heap bytes, plus 32 additional inline bytes for this prototype's map. That
fixture was not run. Actual low-dimensional cases still lose space:

| Original rows | Dimensions | Potential row-file reclaim | Extra heap metadata | Difference before inline overhead |
| ---: | ---: | ---: | ---: | ---: |
| 4095 | 3 | 80 B | 236 B | −156 B |
| 4096 | 3 | 96 B | 232 B | −136 B |
| 4097 | 3 | 176 B | 232 B | −56 B |

All 25 footprint records and 20 dense/rank comparisons remain in `rank/summary.json`.
These are Vec-capacity and file-length accounts, excluding allocator overhead,
page rounding and residency. Original durable files remain intact, so the listed
reclaim is potential. There is no timing, RSS, durability or readiness claim.
The original full-precision mapping is dropped, while graph and delta ownership
move unchanged. No delta absorption, routing-node garbage collection, durable map,
root publication or recovery protocol is implemented.

Validation: 7 focused tests, 6,084 native/dense/rank query triplets and 62,405 exhaustive
rank lookups pass; prototype clippy passes. Cases cover 0/1, 63/64/65, 4095/4096/4097,
tail bits, all-live/all-dead layouts, deletion, replacement, committed ID reuse,
sparse IDs, reopen, both metrics and score bits. Dense baseline source/results
remain under `dense/`. Current-base publication checks are recorded separately in
`provenance.json`; prototype validation used the sealed source base below.

## Reproduce the prototype

From the repository root, verify the package, reconstruct its sealed source in a
separate checkout, then run only its synthetic tests:

```sh
cd benchmarks/results/streaming_payload_rank_v1
shasum -a 256 -c SHA256SUMS
cd ../../..
git worktree add --detach ../payload-rank-replay aa734fe7941c75dd3fb175871db24f6622cade61
gzip -dc benchmarks/results/streaming_payload_rank_v1/rank/prototype.patch.gz |
  git -C ../payload-rank-replay apply -
CARGO_TARGET_DIR="$PWD/../payload-rank-replay-target" cargo test --locked \
  --manifest-path ../payload-rank-replay/Cargo.toml --lib payload_probe -- --nocapture
```

Use a separate fresh checkout and `dense/prototype.patch.gz` to reproduce the
dense baseline. Compressed patches preserve their original bytes and hashes:
rank `f9b74ffd3e82fdaaae0d69214705f064db4045033250a3145e479daf7066ed6c`,
dense `eba534ba2177a2fa5d1c125a52c63bf63ac9bf7e5baf41f2e8df20e8fb036386`.
No official vectors, heldout queries, Linux runs or live benchmarks were used.
