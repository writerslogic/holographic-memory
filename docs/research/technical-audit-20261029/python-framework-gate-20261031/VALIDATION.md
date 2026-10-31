# Python framework final gate

PR #362 tested head `e8dc40dcab99225173027689c57eb906bdf6884c` merged as
`08776a83780e670056ec3881acbf98e1a95dd560`; tested and merged trees both
`020f7ca80900ff861ecd2ce269e52b467c519dd4`. Merge signature verified by GitHub.
49 successful checks / four skipped, none pending or failed. Required aggregate
`check` passed; no unresolved review threads or open code-scanning alerts.
Admin bypass applied to owner review only after checks passed.

CI 41219760951: Windows baseline 807 pass / 14 skip, service 170 pass / zero skip.
Real-server Python job: LangChain 17 tests each at 0.3.86 and current 1.x;
LlamaIndex 42 at 0.10.0 and 48 at 0.14.25. See retained logs for environment and
commands. Local evidence in the earlier framework directories remains qualified
as recorded; these are contract checks, not production-load benchmarks.

Dependency alerts queried after PR #360 remained 210 (10 critical / 72 high /
99 medium / 29 low), including four pyasn1 alerts for warm-sdk/uv.lock despite
the merged 0.6.5 lock update. Do not interpret zero code-scanning alerts as a
clean dependency inventory or dismiss alerts without investigating their status.
