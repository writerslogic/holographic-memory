The local wheel tool makes the candidate feature selection explicit and binds the resulting wheel/native libraries to source and the actual build command. The discovery example attributes missing true neighbors to existing traversal boundaries. Neither changes search behavior or establishes a performance gain.

`benchmarks/public/vector_candidate_wheel.py` prepares a local plan by default. `--execute` creates a fresh plan and executes that exact offline, locked build under the caller's timing and memory guard; it never publishes or deploys. Existing output directories are rejected. A failure leaves logs and a failure receipt and does not retry. The tool sets one Cargo job and a 600-second child-process-group timeout; the parent controller must enforce the 4-GiB sampled child-tree RSS guard and own `hms-timing.lock`.

The `disabled` arm keeps the project's ordinary Python features; `v3` adds `experimental-batch-pool`; `candidate` adds that feature and each explicitly requested `--feature` (`--query-feature` is an alias). Supported optional features are `experimental-neighbor-admission`, `experimental-query-x4`, and `experimental-build-search`; they must already be declared in the integration source, and the build feature needs its own ablation before selection. The feature list unions project and command choices explicitly, so it does not depend on whether Maturin merges or replaces project defaults. Candidate features already present in defaults, transitive mechanism contamination, and all-features configuration are rejected. Default project files are unchanged. Maturin exposes Cargo features and locked-build options through its [official configuration](https://www.maturin.rs/config).

Example preparation commands, after integrating the selected source:

```sh
uv run --no-project --offline --python 3.13 --with maturin==1.15.0 python benchmarks/public/vector_candidate_wheel.py --tree . --output /ABS/NEW/v3-plan --arm v3
uv run --no-project --offline --python 3.13 --with maturin==1.15.0 python benchmarks/public/vector_candidate_wheel.py --tree . --output /ABS/NEW/disabled-plan --arm disabled
uv run --no-project --offline --python 3.13 --with maturin==1.15.0 python benchmarks/public/vector_candidate_wheel.py --tree . --output /ABS/NEW/candidate-plan --arm candidate --query-feature experimental-neighbor-admission --query-feature experimental-query-x4
```

The interpreter path deliberately retains a virtual environment's symlink: resolving it to the base executable would lose installed Maturin. An executed receipt contains source file SHA256s, requested features, exact command, tool versions, wheel SHA256, individual native-member SHA256s, and log hashes. Zip members are inspected without extraction, with path/duplicate/encryption/size checks. This proves the recorded local build/artifact association; correctness and performance still require their own gates. A historical exact-S baseline is a separate source package, not the candidate-source disabled control.

`examples/vector_discovery_diagnosis.rs` reuses `VGraph::diagnose` and emits all existing evaluated/beam/retained/rerank/result IDs, independent cosine ground truth, ordinary-search distance-evaluation counts, and a mutually exclusive truth classification: never scored; scored but not retained; retained but not reranked; reranked but not returned; returned. It validates subset/range/uniqueness invariants and requires diagnostic ordered results to equal ordinary search. Unique evaluated IDs are not equated with the native evaluation counter, which includes repeated/descent/rerank work. No diagnostic times are reported as speed evidence.

The synthetic mode uses 257 construction rows and eight separate queries. The prepared mode is restricted to the frozen 65,536/256 train-only fixture and saved index: declaration, four input files, and index hashes are checked, source ID partitions are verified disjoint, and the rebuilt graph must match the recorded structure hash. It reads no official test arrays and rejects other data rather than silently widening the experiment. All diagnostic work is separate from timed measurements. The optional `sha2` dependency already exists; no dependency or lockfile update is required.

Parent-controlled native validation commands:

```sh
cargo test --offline --locked --example vector_discovery_diagnosis --features experimental,sha2
cargo build --offline --locked --release --example vector_discovery_diagnosis --features experimental-batch-pool,sha2
RAYON_NUM_THREADS=1 target/release/examples/vector_discovery_diagnosis synthetic /ABS/NEW/synthetic-discovery.json
RAYON_NUM_THREADS=1 target/release/examples/vector_discovery_diagnosis prepared /Volumes/A/hms-coordination/benchmark-improvement-20261024/vector-glove-local-diagnosis/input /Volumes/A/hms-coordination/benchmark-improvement-20261024/vector-glove-local-diagnosis-attempt5/plain.hms /ABS/NEW/prepared-discovery.json
```

Pure Python validation uses `uv run --no-project --offline python -m unittest discover -s benchmarks/public -p test_vector_candidate_wheel.py -v`. Native compilation, discovery execution, and actual wheel builds remain pending parent scheduling; the subagent ran only formatting, Python tests, and dry-plan preparation.
