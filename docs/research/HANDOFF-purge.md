# Purge handoff

PR #124 (feat/purge). Done: text segment zeroed on delete, fsync in forget and scrubbing compaction, replaced and orphan generations overwritten, server DELETE ?purge=true, tests/purge.rs, benchmarks/results/purge_v1.json.

Unfinished:
- CI result, review threads and bot findings not yet checked; merge with gh pr merge 124 --squash --admin --delete-branch.
- MSRV 1.89 compile not run.
- No encrypted-store end-to-end purge test (only unit and lib tests with security).
- Updated-but-not-deleted documents leave old text in documents.text until reopen.
- Windows scrub path untested.
