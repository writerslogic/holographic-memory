# Query overhead investigation

The frozen binary remains unchanged. This assessment separates recorded tuning
measurements from source-derived costs; it does not tune against confirmation
queries or claim an optimization result.

For cap250k/ef128 against global ef256, the three tuning pairs' median query-time
ratios are 1.7115 before churn, 1.4167 after churn, and 1.8879 after consolidation.
These values come from the existing `summary-tune.json` pairs for capacity250k,
candidate ef128, control ef256. The mean-of-phase aggregate used during selection
was about1.67x. All show query regressions.

Verified source mechanisms at `06101dfd`:

- `partitioned_probe.rs::query_batch` already parallelizes the128 queries with
  Rayon. `Segments::search` visits children serially for each query. At1M this
  means four nonempty graph searches; at3M it calls12 children, of which11 have
  live rows. Each graph receives ef128, whereas the control receives ef256 once.
  Summing ef gives2x or5.5x the control width; ef is a search-width setting, so
  these sums are not measured distance-evaluation ratios.
- Each child returns at most10 IDs. The wrapper reacquires that child's read lock,
  fetches authoritative original rows, computes float64 L2 for those candidates,
  sorts up to40 or110 distance/ID pairs, and truncates to10. The one-child control
  performs the same wrapper work for10 IDs. Native search has already reranked
  its own candidates. Query validation and query encoding also repeat per child.
- `update.rs` owns one thread-local `Option<SearchScratch>`. In
  `vertex.rs::search_reusing`, reuse requires `scratch.visited.n == graph.n`.
  Unequal child sizes therefore replace the entire scratch; this includes a
  newly zeroed visited bitmap and initially empty search/result buffers. Before
  and after churn, the1M layout visits250k,250k,250k,150k base graphs. In steady
  query-major traversal, the150k→250k and250k→150k transitions each recreate
  scratch. After consolidation the sizes are241k,250k,250k,160k, giving three
  incompatible transitions per successive query. At3M the nonempty tail starts
  at200k and ends at230k, while the first child ends at223k; the same two/three
  transition pattern applies. The empty twelfth child returns before using
  graph scratch.

The last mechanism is a concrete source-level allocation cost that the statement
“native search reuses thread-local scratch” alone misses. The elapsed-time share
of each mechanism remains unmeasured. All-segment traversal and the graph's
recall/work tradeoff may still dominate after scratch reuse is improved.

A separate future diagnostic could process the existing query batch one segment
at a time, retaining each query's exact float64 distance/ID merge. That could
reuse same-shape scratch and improve graph locality; it would still search every
segment and incur additional scheduling/storage. Another option is reusable
per-shape scratch, which needs an explicit memory bound and compatibility tests.
Neither is implemented here. Confirmation uses the sealed original query order,
binary and settings; no new heldout search is authorized by this note.
