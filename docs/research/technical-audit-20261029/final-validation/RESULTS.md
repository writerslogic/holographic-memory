<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Campaign integration evidence

The implemented source at `768e1bf93bf2c67fa07aec61df22f99955a2439d` passed the local checks and every applicable PR gate. [PR #342](https://github.com/writerslogic/holographic-memory/pull/342) is the authoritative final integration and merge record; its final evidence/workflow commit must also pass normal CI before merging. Production source is identified by tree `e80979db6477117092f3e684fa0f5bfb54c94c3c` and [file hashes](source-manifest.json). The historical 6.2 assessment in REPORT.md is unchanged.

## Retrieval comparison

Baseline is `a736af02171da7c8b757dcd83a46422c614c07d7`, the recorded R16 starting revision. The four variants are baseline, eligibility/postings alone, bounded update delta alone, and their combination. All use the same exhaustive ordered-result oracle, canonical scores, candidate limit, one runtime worker, counting allocator and synthetic 10,000-row/16-dimensional corpus. Initial implementation and repaired implementation observations are both retained in [measurements.json](measurements.json); [plan.json](plan.json) predates execution.

Values below are medians of three independent round statistics. Each query group has 32 samples per round; its p99 is the observed maximum, not an estimated population-tail guarantee.

| Observation | Baseline | Combined | Interpretation |
|---|---:|---:|---|
| Initial rare-term capture p95 | 2.271 ms | 1.185 ms | 47.8% lower; exceeds the 20% target |
| Updated rare-term capture p95 | 2.073 ms | 1.089 ms | 47.5% lower |
| Initial hot-query total p99 | 1.943 ms | 1.979 ms | 1.8% higher |
| Updated hot-query total p99 | 2.029 ms | 2.262 ms | 11.5% higher; investigated tradeoff |
| Equal rare/hot 128-query total | 237.66 ms | 177.12 ms | 25.5% lower on this declared mix |
| 32 mutations plus normal dense maintenance | 153.728 ms | 0.402 ms | 32 rebuilds become zero |
| Initial dense training | 4.90 ms | 5.18 ms | Construction cost remains |
| 1,000-row reopen | 16.60 ms | 16.55 ms | No meaningful demonstrated change |
| Worst rare-query requested working bytes | 43,709 | 327,689 | +283,980 bytes; below 12 MiB guard |
| Initial hot-query requested working bytes | 44,703 | 44,767 | +64 bytes after repair |
| Process peak RSS, macOS | 97.7 MiB | 98.0 MiB | Includes ingest, oracle and lifecycle work |

Each comparison asserts 1,536 full ordered result equalities with no differences, including score, source span and metadata. Initial plus repaired execution consumed 29.362 of the 600-second budget. Compilation is separate. No 100k expansion was needed.

The initial implementation violated the hot-query investigation threshold substantially. Its trace identified an unnecessary owner hash set for scan-planned queries and redundant changed-ID lookups when owner eligibility already proved liveness. The repair removes that allocation and duplicate work, using existing hashed membership for the bounded changed-ID set. It preserves exact selection and all delta overflow limits. Generation caching was rejected because it adds retained state and invalidation during churn.

The remaining updated hot p99 exceeds the investigation threshold by 1.5 percentage points. Its p50 is also 11.0% higher. The delta-only variant uses the same hot scan/liveness logic yet has lower observed times, so host noise and generated-code layout remain confounders. The first run also showed large variation between variants with identical clean-state code. Further parameter tuning would not establish a causal improvement. The combined path is retained for the declared mixed workload and the removed update rebuild cost; no universal hot-query speedup or <=10% tail bound is claimed.

The complementary mechanisms are distinct: eligibility alone cuts rare-query work but still rebuilds 32 times; delta alone removes rebuilding but leaves rare-term capture near baseline. Together they deliver both effects. Startup, training, mutation, maintenance, query phases, allocation peaks and retained lifecycle memory are included in the raw results. The combined lifecycle retains 200,224 additional requested bytes, including lexical scratch. High-dimensional capacity, arbitrary selectivities and production latency distributions remain unmeasured.

Reproduce the summaries without execution using `uv run --no-project --offline python docs/research/technical-audit-20261029/final-validation/summarize_probes.py`. The probe sources, build manifests and gzip-compressed initial source patch preserve the baseline/ablation method. The production hot-path source matches the repaired probe; subsequent changes to its regression assert owner eligibility directly instead of requiring a redundant changed-ID copy.

For a separately authorized fresh comparison, create a new sibling directory under this audit directory, copy only `retrieval_probe.py` and `retrieval_probe.rs` into it, then run the copied script with `build` followed by `run` using `uv run --no-project --offline python`. Keeping the same directory depth preserves repository resolution. This writes a new three-round ledger and enforces its 600-second ceiling; the script refuses to overwrite an existing execution ledger. Do not delete the retained original observations to rerun.

## Native evidence

The frozen corpus has three development and three evaluation scenarios, each with 12 steps. With unchanged `k=candidateLimit=1`, baseline retrieval fails 24 distinct steps of 72 (72 failed assertion pointers); the combined path passes all 72 in each of three rounds. The evaluation partition accounts for 12 of those baseline failures across its 36 steps. Expected abstentions remain 12 per round in both variants; they are reported separately from supported answers. Successful source-quote checks increase from 6 to 30 per round.

This establishes the temporal preselection repair on frozen source-grounded fixtures, including ambiguity, contradictory updates, event/ingestion time, deletion and restart. It does not establish learned extraction, representative reader inference, general native QA quality or superiority to LongMemEval reference ranking. No expected answer was indexed; learned-answer quality remains null. Fixture hashes and every scenario outcome are retained.

## Trust and lifecycle limits

The SCITT vectors are independently CBOR-encoded and Ed25519-signed in Python, then checked by the Rust verifier. They cover inclusion, issuer/service identity and trust, wrong proof/signature/root, detached payload, malformed bytes and checkpoint expiry. Regeneration exactly matches the committed fixture. The supported profile still requires an independently authenticated checkpoint and explicit issuer/service anchors.

MCP tests exercise the actual HTTPS client, plaintext and self-signed TLS rejection, redirects, timeout, cancellation, scope/revocation and namespace isolation. The official SDK also exercises stdio and HTTP, source operations, concurrent compare-and-set, authorization, protocol negotiation, persistence and encrypted stores. Bounded admission rejects uncommitted work explicitly. A 413 response now advertises HTTP/1 connection closure so pooled clients do not reuse a socket with unread request bytes.

Security tests cover immutable key/salt publication, concurrent creators, malformed and oversized files, process interruption, scrub ownership and stale dense-owner release. [The security review package](../security-assurance.md) supplies the failure-state model, cryptographic inputs, actual DP assumptions and precise hardware experiment. Process interruption is not hardware power loss. Independent cryptographic review, finite-precision DP composition, hostile-host protection and platform ACL assurance remain external limits. Durable sidecar reconciliation and source-deletion synchronization were not selected; typed committed outcomes remain explicit.

## Integration status

Normal workflow conditions have been restored from `554aec0c21ef84aaafee8feb52fd1553be791d2d`; temporary campaign policy files are removed. Intentional additions enforce native framework and official MCP SDK checks and include Windows in the aggregate gate. Performance execution remains local; CI checks the benchmark harness and correctness floors. CodeQL and Code Quality are configured with weekly schedules and passed on the integrated source. GitHub normalized the old language selection to its detected supported set, including Rust; the rejected legacy aliases and resulting settings are retained in the ledger.

[The source-revision rollup](source-pr-gate.json) contains 157 successful entries and 18 deliberate skips; the completed CLI snapshot reported 153 successful checks. Skips are publication/tag-only paths and the existing DCO owner exemption. The commits are signed and DCO-trailed. Windows ran 772 tests successfully; its 583-second torn-append recovery test explains the longer job. The final evidence/workflow commit repeats normal CI under the repository's existing PR triggers; successful prior results never authorize accepting a failed later gate. No packages or tags were published.

[Validation commands and outcomes](validation-results.json) include default/all-feature tests, Rust1.89 compatibility, Rust1.99 strict lint, rustdoc, schemas, bindings, packages, official SDKs, independent W3C/SCITT artifacts and the synthetic offline embedding addon. [The metric reassessment](assessment.json) is 7.5/10 using the original domain weights and explicit ordinal confidence; M6 remains separate at 8. These are engineering judgments about evidenced scope, not measured model quality or certification. Optional outbox/synchronization and strategic routing/calibration were not selected or represented as implemented.

The 107 raw logs, including failed initial checks and all temporal outcomes, are preserved byte-for-byte in [logs.tar.gz](logs.tar.gz), with sizes and SHA-256 hashes in [log-index.json](log-index.json). Log names in the ledger and command manifest identify archive entries. The summary script reads that archive directly; its output is byte-identical to the summary made from the original files.
