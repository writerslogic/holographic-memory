# Uncontended PGO development verification

Repeat the existing comparison without changing its build, profile, datasets or query settings. The recovered 261 Rust source files match commit `99e84d92f72a42e6f54b2cc48fb30fc86cdfa99c`; the three binary hashes are fixed in `paired.py`. This older source is not the final integrated engine.

Use the existing saved glove-100-angular and sift-128-euclidean indexes with the v3, native and v3+PGO binaries. The PGO profile was trained on nytimes data, not these evaluation queries. Fix k=10, ef=64/128/256, rerank=100, 10,000 queries, one thread pinned to CPU 0, and three rotated arm orders. Hold the exclusive timing lock for the whole bounded comparison; allow no concurrent builds or other benchmark work. Require load below 3 at every recorded point. Abort rather than relax the load gate.

The existing development bar is at least 3% median paired PGO improvement at all six dataset/ef points, with equal recall and distance-evaluation counts. Preserve every pair and any loss. Passing this older-source repeat does not adopt PGO: the current engine, wheel profile training and image build remain unvalidated. This k=10 microbenchmark does not establish the VIBE k=100 target or a leaderboard win.

`paired.py` records dataset/index hashes, timestamps, CPU/pinning, loads, raw harness outputs and all paired ratios. `test_paired.py` checks paired ratios, the load boundary and recall mismatches. Run with the existing root and converted data directories under `flock ~/hms-timing.lock`, with a 240-second outer timeout; no network access or paid compute is required.
