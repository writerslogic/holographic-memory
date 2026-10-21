# Exact deletion from superposition memory

Prototype: `benchmarks/research/forget/`.

## The guarantee

A counter bundle stores, per dimension, the integer number of live items
active there. Insert adds one at each of the item's `k` active indices; delete
subtracts one. Counts form a group under addition, so after any sequence of
inserts and deletes the state is bit-for-bit the fold of the items still
present, in any order. A deleted item leaves no residue. Thresholded bundles (majority vote or
union) cannot offer this because thresholding discards the per-item
contribution.

The recall view `{ i : count[i] >= tau }` is derived on demand. At `tau = 1`
it equals `bundle_bloom` of the survivors, so containment recall is unchanged;
the unthresholded readout `sum(count[i])` over an item's indices stays
informative past the point where the binary view saturates.

## What a certificate proves

Each delete can issue a certificate: SHA-256 digests of the counter state
before, the state after, and the deleted item, plus item counts and a residual
record. A verifier holding the post-delete counts and the item needs no
pre-state: it recomputes `state_after`, digests `after + item` in one
streaming pass and compares with `state_before`. Equality proves the
transition was exactly the subtraction of that item. Certificates chain: each
`state_after` is the next `state_before`.

The residual record places the item's post-delete similarity against the exact
null for a random item of the same weight (hypergeometric for the view,
finite-population variance for the counts) and reports the z-score and whether
it lies within `mean + 3 sd`. It bounds what the memory still says about the
item. It is not a near-duplicate guarantee: a surviving look-alike
legitimately keeps the score high; the hash chain, not the residual, is the
exactness proof.

## Limits

Counts are `i16` or `i32`; an insert that would overflow is refused, never
saturated, and every refusal leaves the state untouched. A ledger of item
digests makes deleting a never-inserted item an error. Capacity is bounded by
a policy on live items and view density. Memory is `dim x 2` (or `x 4`) bytes
regardless of item count, plus 12 bytes per distinct live item; at `D = 16384,
k = 64` the bundle is smaller than per-document storage above 128 items.
Certificates cost two `O(dim)` digests per delete.

Document-store shards hold one vector per id and already delete exactly; this
applies to the bloom, counting-bloom and composite stores.
