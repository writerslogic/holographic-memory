# Bounded consolidation development protocol

Registered before query outcomes, 2026-10-24 UTC. Production parent: `b228cae0`;
test harness: `468370e`. This compares one native `UpdatableVGraph` with a
composition of independent native graphs over contiguous ID ranges. It is an
experiment, not a persistent multi-segment API or a standings replay.

Inputs are the first 1M and 3M rows of the official 100D clustered MSTuring
dataset. The 3M fbin has its header row count adjusted to match the prefix:
SHA256 `721a8ed1bc372bcc6ca628aeab0a631dc54bea17e2740ae937680b71eb6bc754`.
The official 10k-query fbin SHA256 is
`4785fe3c90c94c5f004f9cd84dca4cc398fc83280517a8d3acb6bc693d6ccd29`.
Only query IDs 512–639 select settings. IDs 640–767 are reserved for one final
configuration check at each size. No queries or oracle answers train the index.

For prefix size `n`, let `N=9n/10`, `c=n/100`, and `s=c/10`. Prepare the first
`N` vectors once per layout, using R32, build-ef64, alpha1.0 and native encoding
defaults. For every measured arm, copy that immutable fixture and perform:

1. Measure queries before changes.
2. Delete IDs `[0,c)`, append IDs `[N,N+c)`, replace IDs `[c,c+s)` with vectors
   from source rows `[N+c,N+c+s)`, and reuse IDs `[0,s)` with source rows
   `[N+c+s,N+c+2s)`. Commit all operations.
3. Measure queries after churn, consolidate only changed segments with
   `Merge { rebuild_after: 1.0 }`, then measure queries again and reopen.

Every segment is queried. Its returned ANN candidates are rescored from its
authoritative stored original rows using float64 L2; their global merge orders
by distance, then external ID. This merge is exact over those candidates; the
search remains approximate. It uses neither lossy routing nor oracle answers.
The one-segment control uses the same wrapper. Visibility, duplicate IDs and
reopen fingerprints are checked independently of recall.

The registered grid has eight configurations: global ef256/512, plus segment
caps 250k/500k each at ef128/256/512. At 1M, a candidate qualifies only when each
of the before/churn/after phases has recall at least global ef256. Among
qualifiers, minimize the mean of the three median query times, then
consolidation time. If none qualifies, retain the highest mean-recall candidate
for a negative-result check. Freeze that configuration before inspecting
heldout queries at 1M and 3M; do not search more parameters afterward.

Each cap has three paired alternating rounds: global/segments,
segments/global, global/segments. Query measurements use one warmup and five
identical 128-query batches. Time includes segment fanout and stored-row scoring.
Use eight Rayon threads on CPUs 0–7 of an Intel Core Ultra 9 185H, under one timing
lock across each pair. Build fixtures outside timing on two nice 19 cores with
the same eight-thread graph setting. Each arm has a fresh 8 GiB/no-swap cgroup;
each lock hold is bounded below 14 minutes. Record load, source/binary/input hashes,
every time sample, isolated consolidation VmHWM, reset memory.peak through the
same held-open FD, lifetime cgroup peak, and bytes written. Native creation
defaults to four-byte neighbor IDs; consolidation retains the existing engine's
automatic three-byte encoding rule for eligible graphs.

`oracle.py` computes float64 brute-force top10 using the exact live
ID-to-source mapping and distance/ID ties. Separately, it computes coordinate
min/max and centroid/max-radius summaries from base rows and measures their
potential pruning fraction against the true tenth-neighbor radius. This is an
oracle-assisted feasibility ceiling with a floating-point margin, not executable
routing, a formal floating-point exclusion proof, or a speed claim. Inspect both
caps while tuning, only the chosen cap on heldout queries. `score.py` validates
all returned IDs and retains paired query/consolidation ratios and losses.

A positive scaled result must precede any new versioned durable coordinator;
that implementation would still need atomic batch visibility, root publication
and journal recovery, old-format compatibility, corruption and restart tests.
A negative result retains this reproducible experiment without changing the
production default. Neither outcome establishes full 30M completion, 8 GiB/one-hour
compliance, or any leaderboard standing.
