<!-- BENCH:START -->

_Generated from `benchmarks/results/competitive/` by `hms-bench table`; method and caveats in [docs/benchmarks/COMPARISON.md](docs/benchmarks/COMPARISON.md). W = HMS ahead (95% interval excludes parity), T = tie, L = a comparator is ahead._

| Dataset | Metric | HMS | Best other | Advantage | |
|---|---|---|---|---|---|
| memory | nDCG@10 | 0.4561 | Apache Lucene 9.12.3 (JVM) 0.4627 | 0.99x | T |
| memory | QPS (single thread) (QPS) | 2,225 | tantivy 2,399 | 0.93x | T |
| memory | p99 latency (us) | 786 | tantivy 733 | 0.93x | T |
| memory | Ingest/build time (s) | 2.52 | SQLite FTS5 2.06 | 0.82x | L |
| memory | Peak RSS, serving (MiB) | 66 | SQLite FTS5 26 | 0.39x | L |
| memory | On-disk size (MiB) | 9.8 | Apache Lucene 9.12.3 (JVM) 10.3 | 1.05x | W |
| beir:nfcorpus | nDCG@10 | 0.2916 | SQLite FTS5 0.2914 | 1.00x | T |
| beir:nfcorpus | QPS (single thread) (QPS) | 9,532 | tantivy 7,105 | 1.34x | T |
| beir:nfcorpus | p99 latency (us) | 298 | tantivy 300 | 1.01x | T |
| beir:nfcorpus | Ingest/build time (s) | 0.59 | SQLite FTS5 0.29 | 0.48x | L |
| beir:nfcorpus | Peak RSS, serving (MiB) | 57 | SQLite FTS5 30 | 0.53x | L |
| beir:nfcorpus | On-disk size (MiB) | 2.1 | Apache Lucene 9.12.3 (JVM) 1.5 | 0.72x | L |
| beir:scifact | nDCG@10 | 0.7027 | Apache Lucene 9.12.3 (JVM) 0.7022 | 1.00x | T |
| beir:scifact | QPS (single thread) (QPS) | 2,711 | tantivy 2,879 | 0.94x | L |
| beir:scifact | p99 latency (us) | 526 | tantivy 520 | 0.99x | T |
| beir:scifact | Ingest/build time (s) | 1.42 | SQLite FTS5 0.43 | 0.30x | L |
| beir:scifact | Peak RSS, serving (MiB) | 160 | SQLite FTS5 54 | 0.34x | L |
| beir:scifact | On-disk size (MiB) | 2.9 | Apache Lucene 9.12.3 (JVM) 2.0 | 0.68x | L |

<!-- BENCH:END -->
