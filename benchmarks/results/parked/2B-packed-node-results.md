# Parked: packed Node result transport (2A/2B lever, 2026-10-19)

Patch: `2B-packed-node-results.patch` (src/lib.rs searchDocumentsPacked + ResultStrings, index.js wrapper,
index.d.ts), on top of the V40 tree. Mechanism: the native side packs each result's numbers into a caller-owned
Float64Array (ten per result) and interns the strings that repeat across queries in a per-instance table mirrored
on the JavaScript side; the wrapper rebuilds the DocumentResult objects in the native property order. Output is
bit-identical to the napi path over 600 query/option pairs (JSON, deepEqual, key order); Node api tests pass.
Measured: V38 campaign (serving_memory_v38.json) QPS 2.40x the accepted build and 10.12x sqlite-full, p99 0.66x,
but serving max RSS 1.0027x the accepted build (V37 line was 0.970x), 1.117x sqlite against the accepted 1.092.
Linux A/B (writerslogic-02 ab-v38-v1, ab-v39-v1, ab-v39x-v1, ab-v39y-v1): the extra 8-9 MB of RSS appears only
when the JavaScript wrapper rebuilds the result objects (skipping the rebuild: 118 MB; async napi path of the same
binary: 121-124; full: 130-131); it is independent of the search thread (sync or tokio pool), of string interning,
of JSON.parse, and of object shape (constructor with fixed fields), and it is invisible to V8 heap, external,
array-buffer and malloced statistics (peak_malloced_memory 1.7-2.0 MB vs 0.25 MB is the only V8 signal).
Why parked: 2A acceptance forbids an RSS regression and the cause is unresolved. Revisit by isolating V8's page
pool / semi-space growth (node --trace-gc-nvp, v8.getHeapStatistics over time) or by returning the objects
from a napi-built array with cached string references (V30's node_result_strings.rs form).
