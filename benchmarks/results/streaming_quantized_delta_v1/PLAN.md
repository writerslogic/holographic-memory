# Quantized delta search preregistration

Registered before any candidate measurement on 2026-10-24 UTC. Source parent
`b228cae0d0cb26d92f357cbec7a88c3dc44340a9` includes frontier rejection #289.
Branch `perf/streaming-quantized-delta`; no standings runs.

The graph stays R32/L64/alpha1.0 with the existing deterministic seed. Quantization
changes traversal estimates only. Every surviving final-beam entry is rescored
with the original float row before returning top10. This remains approximate
search: exact reranking does not restore vertices omitted by approximate routing.
No learned transform, graph training, test-answer feature or dataset branch.

Six candidate configurations, plus two exact controls:

| Representation | ef values | Encoding/kernel |
|---|---|---|
| exact control |256,512|current float delta search|
| i8 coordinates |256,512|existing symmetric quantize + VNNI/portable i8 dot|
| rotated i8 |256,512|existing three-round sign/FWHT Rotation, same i8 kernel|
| rotated binary |512,1024|existing Rotation, edge_factors from zero, QueryCode/raw_one|

Rotation length is the existing automatic power-of-two padding (128 for100D),
seed0x5eed. The binary arm considers distributed/holographic sign codes as a
coarse representation. It must beat ordinary quantization empirically; there is
no claim that structured FWHT rotation inherits Haar-rotation error guarantees.
Reference: https://arxiv.org/html/2405.12497v1 (Gao and Long, SIGMOD2024), with
existing repository kernels.rs and edge_factors as the implementation baseline.

Official msturing-30M-clustered prefix is read-only. Tune only on query IDs256..383;
holdout384..511 is opened once after choosing the final candidate. Initial
development fixture has1M rows. Evaluate both all-live and deletion of IDs
divisible by5 (routing vertices remain allocated), then measure append cost on
the next100k prefix rows. Exact top10 comes from exhaustive original-row scoring
over the live set. No other agents' reserved query slices are used.

Eligibility: recall@10 at most0.0025 below the corresponding exact control in
each tuning state, at least10% lower median paired query time at T1 and T8,
and every timed candidate/base pair below1.0. Both i8 representations compare
with the same-ef exact control; binary ef512/1024 compares with exact ef256/512
respectively (twice the routing beam). Choose the fastest eligible T8
candidate; break ties with fewer code bytes. Preserve all failed arms/results.
Additional code storage must be at most35% of raw delta float bytes; quantify
encoding/append time and process RSS separately, without claiming memory savings
while originals remain resident. Old defaults remain exact and any shipped
approximate mode is explicitly selected. If no candidate passes, reject this
lever with evidence rather than silently widening the grid.

Timing: Intel Core Ultra9 185H, CPUs0–7, T1/T8, three alternating pairs under
flock ~/hms-timing.lock; five timed batches per configuration after warmup.
Holds remain under14min. Limited nice19 builds use private targets and two jobs.
Record load, source/binary hashes, code bytes, raw timings and checksums. Confirm
the selected candidate and exact control on heldout queries at1M and3M without
further tuning; qualify any failed state. These are development checks only.

After a win, implement optional incremental derived codes using existing kernels,
keep exact rows for reranking/recovery, expose an explicit Rust/Python control,
and test insert/replace/delete, interrupted consolidation, reopen, empty and small
dimensions, score finiteness, shadow IDs, and unchanged exact-default results.
Derived caches need no durable-format change; if persistence changes become
necessary, version them and preserve old-load/golden/upgrade behavior. Finish
with one final required gate after production edits, signed ready PR, no merge.

Resource scheduling note (before any query outcome): the initial nonblocking
lock attempt found the lock busy; a second queued attempt was cancelled before
acquiring it because available memory was below the agreed6GiB threshold.
No candidate has been measured. Prototype a355be9ba34a93978a97142f49231ae1f13a72c3
changes only exhaustive-oracle result collection from consuming to borrowing the
10 retained tuples, so returned vectors cannot retain the million-row scratch
allocation. Candidate code, queries and grid are unchanged. The initial grid
is capped at225s; any incomplete grid remains incomplete, with no inferred win.
Any process peak includes graph construction and oracle scratch, not serving RSS.
