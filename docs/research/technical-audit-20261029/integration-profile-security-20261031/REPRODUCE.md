# Reproduce the integration dependency profiles

`validation.json` records the baseline and final source commits. Use the baseline
commit to reproduce the initial LiteLLM dependency selection; it already includes
the Semantic Kernel security floor.

The two `baseline-*-project.toml.gz` archives preserve the package manifests used
for that resolution, independently of commit availability after rebasing. Restore
them into `integrations/python/pyproject.toml` and
`bindings/python/warm-sdk/pyproject.toml` in an isolated checkout when reconstructing
the baseline. Frozen advisory scans can be repeated directly from the archived
locks without rebuilding either package.

For each extra, decompress `<extra>-probe.toml.gz` to
`<scratch>/<extra>/pyproject.toml` and replace the two absolute local source paths
with the corresponding paths in that checkout. Decompress
`baseline-resolve.py.gz` into `<scratch>/resolve.py`, then run it with
`uv run --no-project --python 3.11 python <scratch>/resolve.py`. Inspect every
`exit_code` in its `resolution-summary.json`; a partial result is not full coverage.
The manifests constrain Python to `>=3.12,<3.13` and select exactly one extra.

Fresh resolution may change over time. Archived `<extra>-baseline.lock.gz` files
record the exact original versions; `litellm-final.lock.gz` records the changed
profile. The other profile package selections are unchanged. The synthetic
registry inventory deduplicates package/version pairs across those independent
locks; it is not an installable combined lock.

The existing `../maintained-integrations-20261031/scan_lock.py` queries the public
OSV API for an extracted lock or registry inventory. Archived responses retain
query timestamps and hashes. `final-summary.json` reuses unchanged baseline
responses and incorporates the independent final LiteLLM scan; it does not claim
to have rerun every framework's tests.

Runtime validation commands are in the Semantic Kernel and LiteLLM steps of
`.github/workflows/ci.yml`. They require a local `warm-server` binary and use
offline embedding/model fixtures. The Chroma probe requires the SDK and Chroma
1.5.9, a local temporary directory, and telemetry/model downloads disabled; its
recorded result covers only the tested persistent-store read path.
