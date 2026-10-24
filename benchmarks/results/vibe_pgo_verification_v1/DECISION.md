# PGO verification decision

The 2026-10-24 UTC repeat passes the pre-registered development bar for the existing
older-source binaries. Production adoption remains **false**: the current engine,
wheel profile training, and image build were not validated. This k=10 comparison
does not establish VIBE k=100 performance or a leaderboard win.

The [protocol](PLAN.md) was fixed before the repeat. All 261 recovered Rust sources,
`Cargo.toml`, and `Cargo.lock` match commit
`99e84d92f72a42e6f54b2cc48fb30fc86cdfa99c`. The runner checked all three binary hashes,
dataset files, saved indexes and metadata, and the profile hash. The existing profile
used nytimes training holdouts (2,000 requested, 1,997 evaluated according to
[train.json](train.json)); no build, profile training, or fixture generation occurred
during this repeat.

The [lock receipt](receipt.log) records 04:49:35–04:50:45 UTC on writerslogic-02,
under a 238-second timeout with a two-second kill grace. All 18 arm invocations
finished. Maximum recorded 1-minute load was 0.70. Every invocation used one thread,
CPU 0, 10,000 queries, k=10, the same saved index per dataset, ef=64/128/256, and
rerank=100. Recall and mean exact evaluation counts were identical across all arms
and rounds, and match the earlier run.

Median paired QPS ratios (candidate / v3), rounded to six decimals:

| Dataset | ef | PGO / v3 | native / v3 |
|---|---:|---:|---:|
| glove-100-angular | 64 | 1.060390 | 0.961824 |
| glove-100-angular | 128 | 1.075176 | 0.966562 |
| glove-100-angular | 256 | 1.071080 | 0.969372 |
| sift-128-euclidean | 64 | 1.088922 | 0.996384 |
| sift-128-euclidean | 128 | 1.075363 | 1.004173 |
| sift-128-euclidean | 256 | 1.087446 | 1.007048 |

All six PGO median ratios exceed 1.03, and all 18 individual PGO pairs exceed 1.0.
Native loses 13 of 18 pairs: all nine glove pairs and four sift pairs. The earlier
contended run remains in [contended/](contended/), including its PGO losses at
glove ef=64 round 3 (ratio 0.9573728572017954) and sift ef=256 round 1
(ratio 0.9398472822442303). That earlier run is ineligible under the load gate.

Evidence: [paired ratios](run/summary.json), [commands and input hashes](run/manifest.json),
[source hashes](source-manifest.json), [provenance](provenance.json),
[host](host.txt), [processes before](processes-before.txt),
[processes after](processes-after.txt), and all 18 raw JSON/log pairs in [run/](run/).
The remote receipt directory is `writerslogic-02:~/vibe-image-ab/verification-v1/`.
The remote binaries, indexes, and profile remain under `~/vibe-image-ab/`; large
binary and dataset contents are not checked into git.

From the repository root, reproduce the local checks without building:

```sh
uv run --no-project python -m unittest discover -s benchmarks/results/vibe_pgo_verification_v1 -p test_paired.py -v
uv run --no-project python benchmarks/results/vibe_pgo_verification_v1/paired.py --summarize-only --out benchmarks/results/vibe_pgo_verification_v1/run
```

[run.sh](run.sh) records the exact locked remote invocation. It requires the prepared
remote files and a fresh output directory; the committed receipt directory must not
be overwritten. [validation.json](validation.json) records the final local checks.
