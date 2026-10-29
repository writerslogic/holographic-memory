<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compatibility notes for the October 2026 audit fixes

## Model identity

Managed local models now identify the effective embedding function: actual pack or model-file
bytes, tokenizer/configuration and effective preprocessing, pooling, prefixes, truncation,
normalization and task behavior. `EmbeddingSpace.revision` contains the versioned fingerprint;
its public shape stays the same. An explicit external embedding space remains caller-owned.
HTTP and MCP derive their space from an installed model instead of assigning a nominal label.

A managed store with an old nominal identity, or one whose effective model changed, fails
compatibility checks before new writes. Back up the store and export its retained source text,
IDs, versions, metadata and URIs with the compatible earlier binary. Create a separate store
using the intended model, regenerate embeddings, and compare counts, source hashes and query
fixtures before switching paths. When text was not retained, use the original sources. Do not
edit `store.json` to relabel vectors produced by a different function.

## Imported identities and checkpoints

Importers retain v1 destination IDs by default. They now stamp a typed source identity and
refuse a conflicting destination owner, including ambiguous v1 namespace/part IDs. Use
`--id-mapping v2` for unambiguous typed hash IDs. Mapping-version changes can create a second
set of destination IDs; choose a fresh destination or explicitly reconcile the old records.

Existing unstamped records require `--adopt-legacy` and an exact source/namespace/source-ID/
part-layout match. An ID match alone does not establish ownership. Changed split layouts fail
by default; `--split-policy replace` writes the complete replacement before conditionally
deleting obsolete owned parts. Checkpoints advance after that cleanup and verify the committed
source content on resume. This is migration/resume support, not general source-deletion sync.

## Retrieval and resource errors

Normal sparse retrieval consistently reports Jaccard similarity; Hopfield attention remains an
explicit separate operation. Exact dense selection uses canonical f64 scores. Approximate
indexes still have candidate-recall tradeoffs; reranking does not make their recall exact.

Rust reasoning callers should use `multi_hop_with_limits` and `synthesize_concepts_with_limits`.
Their legacy `Vec` wrappers panic on budget exhaustion. Node propagates checked failures.
Filtered SDK queries select eligible IDs before ranking. Exact cosine SDK paths merge bounded
ID groups; fused/model-text paths reject oversized selections when their scores cannot be
merged correctly. Reduce the selection or use a suitable server-side search contract.

## Persistence and format

Failed WASM saves retain the in-memory store for retry. Mobile snapshots reject destinations
within the live store after canonical path resolution. The server restores a tenant through a
recoverable intent rather than relying on in-process rename rollback.

The original [audit](../research/technical-audit-20261029/REPORT.md) is a historical baseline.
Its scores do not describe the fixed implementation; [remediation evidence](../research/technical-audit-20261029/REMEDIATION.md)
records what was implemented and validated, with remaining verification limits.
