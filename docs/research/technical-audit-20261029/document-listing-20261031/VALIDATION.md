# Metadata listing allocation and checkpoint validation

Base: `30cd5d46f3758c9f62ae0d49708deb606a5a5aa7`. Local macOS arm64,
Rust 1.99.0, debug test profile; no paid services. The existing thread-local
counting allocator records requested allocation bytes/calls, not RSS or retained
memory. Elapsed times in logs are observations, not controlled latency results.

18 lexical documents each hold a 64 KiB single-word text and small metadata.
The old full-record API discarded text after copying it: 15 summaries allocated
998,897 bytes / 94 calls. Initial live projection: 13,057 / 78. An expanded
cold-store check caught an incomplete fix: the first reopened listing allocated
2,634,098 bytes by hydrating the whole checkpoint (`final-core.log.gz`).

The final path merges sorted live documents with live checkpoint ordinals,
projects selected records, and counts live/tombstoned documents without hydration.
15 / 16 / 17 rows allocate 13,057 / 13,719 / 17,965 bytes live and
13,777 / 14,487 / 18,781 bytes reopened. Tests require <64 KiB for each page
and <4 KiB for a single summary and document count. Payloads, metadata and
chunk counts are checked, and stored text remains retrievable. Checkpoint decoding
still validates borrowed text and allocates per-chunk descriptors; this is not a
constant-time read or a guarantee independent of metadata/chunk count. Runs do
not establish HTTP throughput, RSS, tail latency or a competitor advantage.

Deep offline verification explicitly hydrates/validates the checkpoint before
reporting its count, rather than relying on an incidental count side effect.
A regression changes an occupied slot's chunk index and recomputes CRC; deep
verification must report the semantic corruption. Overlay tests cover deletions,
replacements, newly inserted IDs, Unicode/prefix cursors and journal replay.

Reproduce against base by decompressing `baseline-repro.patch.gz`, applying it,
and running `cargo +1.99.0 test --locked --test allocation_bounds metadata_listing
-- --nocapture`. This expanded reproduction uses the current fixture and fails
on the old live-page allocation; the original log predates the added cold phases.
Candidate command is identical without that patch. Local checks pass: nine
checkpoint tests, 19 serial HTTP API tests, two Node tests with the freshly built
native module, strict Clippy (`server,node-api,security`, including tests), scoped
rustfmt, and generated TypeScript/JSON schema comparison. Exact-head CI remains
required before merge.
