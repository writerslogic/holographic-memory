<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retained ANN harness results

Three local harness comparisons completed on 2026-10-24: GloVe/ScaNN, SIFT/FAISS IVFPQFS, and Fashion-MNIST/Glass. The 702 native points include 36 SIFT HMS degree-32 points from an earlier container. Those points remain diagnostic; SIFT standings use the 72 HMS degree-48/64 points measured in the comparator container. These are not official AWS leaderboard results.

Regenerate a summary offline with the committed CSV, byte-preserved dataset manifest and raw-point metadata, for example:

```sh
uv run --no-project python benchmarks/results/ann_live_20261024/ann-live-export-20261024.py benchmarks/results/ann_live_20261024/glove-100-angular/results.csv benchmarks/results/ann_live_20261024/glove-100-angular/manifest.json glove-summary.json
```

Run the adjacent `test_ann_live_export_20261024.py` with pytest to check grid completeness, raw identity matching, aliased Glass labels and diagnostic-point exclusion. Compare generated JSON with the committed summary; input digests identify the precise source bytes. The generated definition files actually passed to the harness and the resolved requirements are retained unchanged. Original manifest config digests identify pre-generation source files at the pinned HMS/harness commits; per-degree filtering and YAML reserialization produce different definition bytes.

The source-package Glass attempt was diagnosed and cancelled; its compact failure record remains under `diagnostic-source-glass/`. Fashion results use the documented repaired wheel run. Cost estimates preserve that limitation and are not invoices.

Native HDF5 predictions and timings remain external, with SHA-256 inventories retained; offline summary regeneration does not independently validate recall. Historical absolute paths and container IDs are evidence identities, not required local dependencies.

The manifest pins the archive commit for omitted bulk logs, remote launchers and abandoned repair scripts. Retrieve originals with `git show <archive-commit>:<repository-path>`. Maintained Python scripts add license headers; original script digests remain identified separately.
