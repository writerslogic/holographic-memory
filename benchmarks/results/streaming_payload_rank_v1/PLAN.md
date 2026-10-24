# Bounded comparison proposal; unexecuted

Freeze source, fixture and driver before any future measurement. The prototype
is currently an immutable, in-memory base-payload adapter, so this plan cannot
establish a durable consolidation replacement.

Use deterministic Euclidean 100D data: 240000 base rows plus 2880 source rows,
SplitMix seed 7654, coordinates `(next_u64() >> 40) / 2^24` as f32. Build with
R32 / build-ef64 / default encoding. Generate 128 queries using seed 4321; fix k=10 and
ef=256 / rerank=256. No parameter grid or official heldout queries.

Commit deletion of IDs 0..2399; replace IDs 2400..2639 using source rows 242400..242639;
insert IDs 240000..242399 from their same-numbered source rows; commit before
reusing IDs 0..239 with source rows 242640..242879. Preserve one immutable fixture
and its exact live-ID mapping.

Compare native and rank-indirected ID-only query interfaces in three alternating
pairs, fresh processes and fixture copies. Keep score-bit checks outside timers.
Require identical IDs/order and authoritative scores; retain every raw answer,
sample, memory value and loss. Use one warmup and five 128-query batches.

Separately record payload-only rewriting and native whole-generation merge.
These perform different work: only native merge integrates topology, publishes
the root and retires delta state. A faster partial rewrite is not a consolidation
speedup. Record graph/delta retention, original/new row lengths, bitmap/prefix/ID
capacities, file hashes and omitted publication costs.

Use one shared timing lock for every heavy build, copy and measurement; private
targets; nice 19 / jobs 2 builds; eight query threads on pinned CPUs; ≥6 GiB available;
4 GiB/no-swap scopes; ≤800-second holds including cleanup. No overlapping build or
measurement. Stop on failed receipts, resource failures or any mismatch, preserve
the failed evidence, and do not retry or retune after outcomes.

No production retention follows automatically. Dynamic updates to rank metadata,
durable indirection, atomic root publication/recovery and bounded graph/delta
maintenance remain separate unsolved requirements. Arbitrary BFS payload order
also requires additional mapping information and is outside this proposal.
