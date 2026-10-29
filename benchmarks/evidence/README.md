# Benchmark evidence and historical inputs

Result summaries and their measurement conditions stay under `benchmarks/results/`.
Each result describes its recorded source revision and environment; it is not a
measurement of every later commit. Source cleanup does not rewrite measured results.

The existing [release archive manifest](MANIFEST.json) records larger evidence bundles
restored by `scripts/fetch_evidence.py`. Git LFS inputs are identified by `.gitattributes`
and by their result guides.

The [retired-input manifest](retired-inputs.json) records frozen source copies,
copied build directories and oversized model-output caches removed from the active
tree. Their original bytes remain at the manifest's immutable Git revision. Maintained
summaries and reproduction code remain in the checkout.

From the repository root:

```sh
uv run scripts/restore-retired-inputs.py
uv run scripts/restore-retired-inputs.py --path benchmarks/results/query_aware_graph_v1/sources
uv run scripts/restore-retired-inputs.py --path benchmarks/results/longmemeval_dev_answers_v3.json
```

The first command lists available inputs. Restoration verifies the recorded size and
SHA-256, publishes complete files atomically and preserves differing local files.
Restored inputs are ignored by Git. A shallow clone must first fetch the recorded
revision with `git fetch --no-tags origin <revision>`.

Historical reader-suspension checks also need the external binaries identified by
their original manifest. Restoring archived source and logs does not establish a
completed or accepted timing result.
