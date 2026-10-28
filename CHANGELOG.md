# Changelog

All notable changes to this project are generated from the commit history.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) +
[Conventional Commits](https://www.conventionalcommits.org/).

[v0.6.3 migration and integrity notes](https://docs.holographic-memory.com/latest/releasing/v0.6.3/).
## [0.6.4] - 2026-10-28

### Added
- Add optional AML provenance and evidence research tools

### Changed
- Organize bindings and research source modules

### Documentation
- Align landing page with retrieval architecture
- Add Glama score badge (#316)
- Explain crate recovery environment policy (#322)
- Explain npm registry propagation gate

### Fixed
- Preserve guarded final benchmark allocations
- Enforce provenance semantics and cached status expiry
- Preserve Mem0 recovery and completion integrity
- Clarify MCP tool behavior for Glama (#317)
- Reduce crate archive for crates.io (#321)
- Verify every npm platform before install (#320)
- Verify every npm platform before install
## [0.6.3] - 2026-10-26

### Added
- Add reproducible AML GCE installer
- Add persistent AML host launcher
- Add AML Cycle 2 adapter and query repair
- Make MCP server scorable by Glama (#313)
- Add a blocked learned-sparse inner-product index (big-ann sparse track) (#294)
- Scale query-aware projection to millions of learn queries (#295)
- Add label-filtered exact uint8 search with per-label inverted files (#291)
- Add direct IP serving and independent candidate retention (#207)
- Add compact adjacency and publish ANN cost results (#206)
- Assemble bounded conversation context with exact source provenance (#203)
- Add optional query-aware graph routing (#204)
- Improve memory recall within a small round budget (#202)
- Diagnose streaming recall and verify crash boundaries (#201)
- Add exact Boolean lexical search and blocked postings (#200)
- Batch inserts and a Python UpdatableIndex for the updatable vector graph (#190)
- Count matching documents and ingest unstemmed terms (#193)
- Persist and update mapped vector graphs (#181)
- Preserve original conversation evidence and audit memory quality (#184)
- Add VIBE diagnostics and rotation controls (#183)
- Euclidean and inner-product metrics in the Python VectorIndex (#186)
- Clustered private ranking of dense codes at n = 1280 (#177)
- Server-mode comparators and an hms-server row in the harness (#157)
- Raise HolographicError with the server's error code and message (#145)
- Add NGT, ScaNN, PyNNDescent, Voyager and DiskANN to the ANN harness (#156)
- Euclidean metric for the vertex index and SIFT in the harness (#150)
- Expose the vector index to Python and add ann-benchmarks and VIBE wrappers (#149)
- Framework adapters for agent and RAG stacks (Python and JS) (#106)
- Stable C ABI with Go, Java and .NET bindings (#109)
- Index-routed server vector queries, deferred ingest modes, memoized encoder, and round-2 research kill tests (#139)
- Local embeddings (pure-Rust engine, pinned model packs, LocalEmbedV1) (#132)
- Evidence manifest, fetch script and worktree audit record (#117)
- Format verifier, golden stores for every format, drift gate, crash-window fixes (#126)
- WebAssembly build of the document store (@holographic-memory/wasm) (#129)
- Production hms-server (scoped auth, observability, TLS, snapshots, deploy artifacts) (#127)
- UniFFI mobile bindings (Swift, Kotlin/Android) (#121)
- Hms-mcp MCP server, launchers, memory-tool backends and editor integrations (#110)
- Memory API with temporal validity, supersession, explain and opt-in policies (#114)
- Hms-import, one-command migration into HMS (#108)
- Platform kernels (AVX512-VNNI, NEON fallback) with kill-test results (#107)
- Integrate the parallel sessions' document store, compaction and server work (#81)
- Document store with compact postings and file-backed text, staged compaction, multi-tenant server and the evidence client
- Learned-sparse impact index served by HMS, measured against Anserini
- One series per patience value in the ANN comparison tables
- Stemmed BM25 and min-max score blend in the document API
- Patience stop rule, opt-in 1-bit screen and error-returning builds for the vertex graph index
- Competitor ANN harness for SymphonyQG, RaBitQ, NGT and Lucene HNSW
- Check the graph cache header and verify the merged qgraph defaults
- Add integer vector stores and per-vertex codes to qgraph
- Batch Qwen3 decoding and padded encoder batches on Metal and CUDA
- Add per-vertex 8-bit code graph index with residual re-rank
- Add held-out tuning mode to public-bench ann-qgraph
- Add on-device Qwen3 model stages behind local-models
- Add quantized graph ANN index and its public benchmark mode
- Add hms-server HTTP service for the Python SDK
- Subset-union bundle experiment and capacity sweep
- Add experimental query-private similarity scoring behind private-search

### Changed
- Clean up memory API, provenance, security and core modules (#214)
- Clean up the full-text storage engine (#213)
- Clean up bindings, server and MCP (#212)
- Clean up qgraph kernels, projection and update format (#211)
- Prune uncited experimental research code (#210)
- Consolidate audit entry serialization (#185)
- Drop rejected intersection kernel candidates

### Documentation
- Prepare npm listing and runtime release (#318)
- Surface holographic operations in search
- Note npm and runtime release
- Prepare npm listing and runtime release
- Document Glama release and score verification
- Polish MCP package pages and link website (#311)
- Make source quickstart accurate
- Refresh branding and improve integration and search guides (#308)
- Drop host names and local install notes, fix the open-test citation (#221)
- Replace private paths and hosts, route large result JSON to LFS (#217)
- Rewrite README with sourced benchmark badges and current architecture (#215)
- Serve the documentation at docs.holographic-memory.com (#189)
- List every Cargo feature and correct the server and MCP rows (#173)
- Build the rustdoc with warnings denied in CI (#148)
- Remove internal scoreboards, session records and process notes (#140)
- Remove session logs, handoffs and planning files (#137)
- Embeddings default-build check and handoff update (#135)
- Render OpenAPI boolean property schemas (#131)
- Passage text is encrypted at rest; document the text segment layout (#115)
- Redesigned site with landing page, generated API reference, search and browser checks (#125)
- Persisted file format specification and stability policy (#119)
- Versioned site, quickstarts, offline agent demo, README, contributor guides (#105)
- Ideas tree for October 2026 with its 10-minute proxies
- Preliminary nytimes VGraph comparison on the M4 and the next-session specs
- Record the green CI and the clippy toolchain for the gate
- Record that main is pushed and in sync
- Record that all work is on main and what was superseded
- Record the merged qgraph state and what is unverified
- Record the integration state
- Standardize README header (#71)
- Update continuation state
- Record the index and model-stage outcomes
- Record the holographic prototype outcome
- Record continuation state for the next session
- Plan the holographic fact memory and its tests
- Plan local model stages inside HMS
- Define the single-run protocol for LongMemEval_M
- Set the speed and accuracy targets and record the compute limits
- Plan the LongMemEval_M run with larger models
- Record the encoder test and the agent-memory benchmark plan
- Attribute semantic search to the document API
- Update changelog [skip ci]
- Record the outcomes of the algorithm work
- Give the SDK README the project header, badges and a quick start
- Use absolute logo URLs so the README renders on package registries
- Update changelog [skip ci]

### Fixed
- Keep Python source archive within PyPI limit (#314)
- Wait for MCP package indexes before registry publish (#312)
- Rename Python packages and polish PyPI pages (#310)
- Persist ANN benchmark indexes before serving
- Update benchmark harnesses for current Clippy
- Complete graph search results after component exhaustion (#301)
- Refuse web-page and rebound-host requests on an unauthenticated server (#285)
- Check checkpoint table sizes, stage model packs and semantic indexes privately, keep oversized fact turns out of range (#281)
- Reject damaged delta offsets, save vertex indexes under unique temp names and assert re-rank row lengths (#280)
- Domain-separate Sigstore signing and close KERI, CAWG, revocation and DID verification gaps (#279)
- Scope tenant metrics to the key, clean up abandoned restore uploads and stop insight growth (#278)
- Sync migrated and snapshot copies and the API-key directory before reporting success (#276)
- Normalize extreme angular rows, bound graph-cache reads and stage Python deletes all or none (#275)
- Persist revocations before applying them and add a strict status-list mode (#273)
- Keep documents and format scans consistent after failed rebuilds and malformed frames (#272)
- Keep tenant generations unique across reopen and harden tenant lifecycle (#265)
- Leave the journal unchanged when a commit sync or segment rotation fails (#263)
- Apply a journaled batch in full when one mutation fails (#264)
- Domain-separate batch provenance Merkle trees and version the signed root (#243)
- Decode sigstore base64 strictly and write the key event log atomically (#257)
- Bind provenance signatures to the record they verify (#242)
- Track VIBE job CPU per process and kill orphaned config processes in the Modal runner (#232)
- Update ctutils past RUSTSEC-2026-0385 (#234)
- Build the updatable index with alpha 1.0 by default (#229)
- Read vector result text under the store lock (#227)
- Mark the gzip LFS exception as binary (#225)
- Read search result text under the mutation gate (#223)
- Core review fixes for provenance, memory API and server (#219)
- Measure one durable ANN mutation per operation (#180)
- Make a purged store match one that never held the document (#179)
- Durability-fair server verdicts and harness adapter checks (#175)
- Match server durability on macOS and build ScaNN with the dataset's metric (#171)
- Score docs search on the answer to the full query (#170)
- Judge recall status from the index of the revision searched (#168)
- Modal shard runner, text harness defects and the x86 text results (#152)
- Make each journal commit durable on macOS with F_FULLFSYNC (#151)
- Name the passphrase when an encrypted store cannot be decrypted (#144)
- Strip debuginfo only so macOS 27 can load the native addon (#136)
- Clear code-scanning alerts and restore the server_requests fuzz target (#134)
- Zero replaced document text in the text segment (#128)
- Purge physically erases deleted text (#124)
- Encrypt the document text segment at rest (#111)
- Windows compaction rename, changelog drift, dependabot npm 404 (#112)
- Check tenant directories against the canonical tenants root
- Scrub 256 KiB of stack below the key derivation and locate surviving key copies in the scan test
- Gate scrub_stack and the AES cipher impl behind the security feature
- Fail open on arena corruption, checksum the manifests, zeroize the AES key schedule
- Satisfy clippy 1.99 lints in the qgraph kernels and graph cache
- Keep greedy decoding at batch 1 unless a batch is set
- Reserve Modal wave cost before launch so preemption cannot bypass the cap
- Run HMS model stages on L4 on Modal
- Decode benchmark data with as_chunks for the current clippy
- Serve SDK queries by exact cosine through the document API
- Resolve leftover merge markers in .bestpractices.json
- Return query results best-first on exact-scan, multi-shard and federated paths

### Performance
- Accelerate lexical query execution and decoding
- Improve phase memory and recover conversation premises
- Add opt-in vector search improvements and memory controls
- Reduce filtered-search allocation and scoring overhead
- Share immutable position sources by document slot (#305)
- Add an opt-in bound-gated exact re-rank to the vertex graph search (#296)
- Resolve phrase positions through posting ordinals (#290)
- Decode packed posting blocks with AVX2 (#293)
- Skip frontier searches for noncompetitive delta neighbours (#289)
- Spill delta rows during vector consolidation (#286)
- Release old graph pages before delta insertion during consolidation (#284)
- Score construction candidates four at a time and with VNNI on AVX-VNNI cores (#283)
- Avoid copying surviving vectors to replace a deleted graph entry (#282)
- Reduce maintenance allocations and fix retrieval regressions (#274)
- Skip inverted-index hits no better than the worst kept one (#277)
- Prune lexical top-k with block-max MaxScore and best-blocks-first term scans (#237)
- Merge the delta into the streaming base instead of rebuilding it (#271)
- Reuse warm ingest interners across rayon splits and batches (#268)
- Return inverted-index hits as slot numbers instead of decimal strings (#267)
- Score Hopfield queries against the stored patterns in place (#266)
- Cut Python SDK per-call overhead (#269)
- Fuse the first three Walsh-Hadamard stages per block of eight (#260)
- Sign batch provenance credentials in parallel (#258)
- Keep the pool's expanded flags in a bitmask (#252)
- Clear the streaming delete-runbook baseline and drop the insert row copy (#238)
- Count majority bundles in a per-thread histogram (#259)
- Count majority-vote encoder levels at the touched indices (#256)
- AVX-VNNI traversal kernels with the query sum computed once per query (#251)
- Huge-page advice on by default, with counters and a refusal fallback test (#241)
- Track visited vertices in a bitset cleared by touched word (#240)
- Reuse construction scratch buffers across insertions, merges and fills (#248)
- Prepare single document writes with the store's kept interner (#254)
- Build chunk ids at their exact size and check them without allocating (#253)
- Reuse one output buffer when recovery tests a damaged journal tail (#250)
- Rebuild a checkpoint without the postings of its dead documents (#249)
- Single-thread vertex search (prefetch terms, SIMD re-rank, 8-bit codes in the harness) (#226)
- Open text stores from a durable document checkpoint (#235)
- Persist the hms-server vector graph and open it at start (#228)
- Qgraph review fixes (#220)
- Full-text engine and storage review fixes (#218)
- Eliminate redundant graph construction and query work (#209)
- Reduce posting block allocation overhead (#208)
- Bound streaming consolidation memory with mapped rows (#205)
- Rank unfiltered lexical queries in the slot table at any store size (#197)
- Store construction adjacency in one fixed-slot array (#182)
- Faster single-threaded text ingest (#178)
- Lean BM25 search endpoint and in-place fact replacement in hms-server (#176)
- Build the vertex graph faster (#174)
- Serve the vertex index from a memory-mapped file (#172)
- Binary journal frames, in-place posting deletes and an ingest word interner (#155)
- Rank unfiltered lexical queries in a per-thread slot table (#154)
- Round-3 ingest, memory and ANN work with research results (#143)
- Quantized construction search for the graph build; cache header records it
- Quantized construction search and owned-input build for the proximity graph
- Byte visited marks for the unfused block path; close the neighbour-block mechanism
- Keep the default neighbour loop free of the delta decoder and the cap counter
- Neighbour cap for the per-row search, lint fixes, x86 paired rows
- Unbiasing scale for block codes, sorted candidate pool, delta-coded adjacency
- Neighbour-block fast-scan codes with evidence fusion and a certified exit for the vertex graph
- NEON per-row 1-bit screen kernel with its microbenchmark kill test (screen stays off)
- Screen exact distances in vertex graph construction with 8-bit bounds
- Add 4-bit traversal codes, aligned rows, BFS reorder and 3-byte ids to the vertex index
- Measure Metal batching and set per-stage defaults
- Prefetch rows ahead in qgraph fill, prune inputs and edge encoding
- Use a bitset visited set and prefetch adjacency in qgraph construction
- Scan both planes of 2-bit qgraph edge codes in one NEON pass
- Speed up qgraph construction without changing the built index
- Add FastScan edge kernel, batched pool merge and 2-bit qgraph edge codes
- Speed up from_dense with branchless term compaction and add encoder-eval
- Gallop before AVX2 on skewed sizes; add intersection kernel bench
- Route sparse queries to the exact inverted index

### Bench
- Sync the VIBE fork and current engine pins (#307)
- Record ordered payload rank prototype evidence (#306)
- Record frozen bounded-consolidation tradeoffs (#303)
- Record rejected consolidation scratch reuse (#302)
- Verify PGO gains with an uncontended paired repeat (#300)
- Record the rejected low-rank OOD experiment (#299)
- Add Gorder window locality diagnostics (#298)
- Record quantized delta heldout recall rejection (#297)
- Record the neighbour-major block kill test against the per-row path (#292)
- Build the VIBE image for x86-64-v3 and pre-register the OOD grid (#287)
- Attribute the streaming runs to the measured commit (#231)
- Add an HMS backend for Mem0's memory-benchmarks harness (#262)
- Measure memory API write costs and record host-layer findings (#245)
- VIBE wrapper with 8-bit codes, query-aware OOD builds and huge pages (#255)
- Add a LoCoMo captions variant measured against the matched protocol (#261)
- Profile vector-index build allocations and vertex-order locality (#247)
- Count heap allocations per write-side phase of the document store (#246)
- Add LoCoMo retrieval round 2 diagnosis and gated packing levers (#239)
- Add LoCoMo results (#236)
- Add a batch mode for the LongMemEval reader and judge (#233)
- Run llama-128-ip and yi-128-ip on Modal with mlann-rf and roargraph arms (#230)
- Add a crash-safe Modal runner for the VIBE float datasets (#224)
- VIBE adapter for euclidean and ip datasets (#194)
- Add single-core ann-benchmarks harness results on the Core Ultra 9 (#198)
- Run the Bemula forgetting benchmark against the memory API (#199)
- LoCoMo adapter and partial results (#196)
- Score LongMemEval dev under the official reader and judge protocol (#195)
- Pin the ann-benchmarks image and run euclidean datasets (#187)
- Add Oct 21 2026 Core Ultra 9 competitive results (SIFT-1M, BEIR text) (#158)
- Reproducible comparison harness and self-updating table (#122)
- Per-session quota on the tuned LongMemEval turn ranking fails on cached dev scores
- Add grid-matched vertex search pairs and correct the alignment claim
- Record paired held-out timing of the vertex search encodings
- Narrow the FAISS comparison and vendor the clean edge baselines
- Record paired held-out timing of the per-vertex code index
- Add paired-timing sync to ann-qgraph
- Record held-out recall and memory of the per-vertex code index
- Record timed held-out glove run of i8 vertex-code qgraph
- Correct reference eval counts in memory-first results
- Record held-out recall screening of qgraph memory layouts
- Record the s-dev-hms cost overrun and stop
- Record held-out qgraph tuning on nytimes
- Measure HMS fact extraction against the Python stage on S dev
- Add seeds and an equal-bytes index control to the holographic prototype
- Record quantized graph results against HNSW
- Prototype the holographic fact memory and measure it on LongMemEval_S dev
- Make the Modal cost cap cumulative across relaunches
- Key LongMemEval fact extraction by content and re-tune on dev
- Record the LongMemEval dev results and the Modal dry run
- Tune the LongMemEval pipeline on dev and record the ablation
- Add the LongMemEval multi-key pipeline, Modal job and dev sweep
- Freeze the LongMemEval 100 dev / 400 held-out split
- Add LongMemEval_S retrieval baseline (dense, lexical, hybrid)
- Add public BEIR and ann-benchmarks comparisons
- Report encoder candidate recall after exact rerank

### Deps
- Rand 0.10, rand_chacha 0.10, getrandom 0.4 for bindings/wasm (#120)

### Fuzz
- Targets for every untrusted-input parser (#116)

### Merge
- Include AML GCE installer on main
- Include persistent AML host launcher on main
- Include AML Cycle 2 adapter on main
- Integrate vector search improvements

### Research
- Harnesses, result files and notes from the parallel sessions
- X86 confirmation of the kernel fix and the build lever on Modal (#80)
- X86 confirmation of the kernel fix and the build lever on Modal
- Superiority session, October 16 to 17 (#76)
- Accept V40 serving row, record the V38 transport and 2C relevance measurements
- Record the V37 compact-postings serving row and the session handoff
- Security and privacy matrix harness and first matrix
- Durability matrix harness and first matrix against SQLite and Lucene
- Record native benchmark gate checkpoint
- Checkpoint native harness validation draft
- Establish checked LongMemEval dev answer results
- Checkpoint final local reader batch and faster validation
- Preserve dev reader progress and speed pilot audits
- Checkpoint local reader evidence and judge progress
- Checkpoint judge instability and local evidence
- Checkpoint dev answer scores and repeat study
- Freeze dev reader protocol and checkpoint quota blocker

### Style
- Address code quality findings in the integrated harness scripts
- Use as_chunks for the fixed four-byte float and id decoding in the ANN bins
- Read cpuinfo once and drop an unused import
- Address code quality findings in the benchmark scripts
- Close files with pathlib and unpack the tamper case explicitly
- Rustfmt
## [0.6.1] - 2026-10-05

### Added
- Introduce IdentityRegistry enforcing W3C DIDs, Verifiable Credentials, and POSME Receipts for queries
- Add AsyncHolographicClient, batch uploading, and metadata filtering
- Add Pydantic AI and SmolAgents tool wrappers
- Add Phidata integration and fix broken Rust examples
- Add Semantic Kernel and DSPy integrations to Holographic SDK
- Add LlamaIndex SDK adapter and CI pipeline
- Pivot to Zero-Trust Edge Architecture (FHE-Lite, AGPL-3.0, SIMD)
- Python bindings + PyPI packaging for holographic-vsa (maturin/pyo3)
- PhaseResonator reusable-index API + clean re-export + doctest (§20 capability)
- Counting membership store with Poisson z-score readout (4x Bloom capacity)
- PhaseHVec core type + real Frady/Kent resonator on quantized-phase substrate
- Real SHA-256 commitment for verifiable phasor mutation (lane 3)
- Multi-hop reasoning over the phasor memory (retrieve_path)
- PhaseGraph -- phasor relational memory with relation algebra + retrieval
- Presence-gate the observer effect (no confabulation-on-query)
- Persist ConnectionGraph via its event log; survives restart
- ConnectionGraph -- plastic event-sourced relation store, wired into engine
- Trust-anchored provenance verification and sharded ANN persistence
- Runnable cross-verification of the cogmem C2PA sample
- Add provenance system with COSE, VCs, C2PA, JUMBF, Sigstore, KERI, CAWG
- Multi-scale encoding, morphological decomposer, fault-tolerant federation
- Add Hopfield-Fenchel-Young energy-based associative retrieval
- Define HolographicAlgebra trait boundary for future geometric algebra
- Add research-grade benchmarks, scaling analysis, and visualizations
- Add sparse Clifford algebra multivector type (CliffordVec)
- Improve SOTA for analogy, planner, and concept synthesis
- Add cost/total_cost fields to planner N-API types
- Evidence-weighted multi-hop confidence, ranked results
- Expose cleanup_vector N-API method for standalone Hopfield denoising
- Add agency layer (GoalStore, Planner, QuestionGenerator, SelfModifier)
- Add cognition layer, distributional refiner, CI coverage
- N-API bindings, persistence, load_from_log, compact for meaning memory (G6)
- Integrate meaning memory into HmsCore, wire write/delete paths (G5)
- Fuzzy_structural_query + multi_hop_query pipelines (G4)
- AtomMemory, CompositeMemory, TripleStore, RuleStore, Decomposer (G3)
- Shared utilities + IndexedMemory with Hopfield attractor (G1-G2)
- Rename to holographic-memory, fix npm logo, update all URLs
- Graph engine with multi-hop traversal, inference, temporal, federated queries
- Complete security integration, DP wiring, docs, LF readiness
- Add security features behind feature flag for LF Decentralized Trust readiness
- Migrate to writerslogic org, add npm publish pipeline
- Eliminate remaining gaps, expose train_nsg/train_ivf via N-API
- Expose diffusion config through HmsConfig and N-API
- Configurable thresholds, N-API config constructor, status getters
- Add query_sequence, elevate test coverage for weak features
- Add multi-shard support with auto-sharding
- Initial release of Holographic Memory System (HMS)

### Changed
- Centralize SDK HTTP client and add CrewAI + Embedchain platforms
- Consolidate LangChain, LlamaIndex, and Haystack adapters into unified holographic-sdk
- Remove dead self-inverse resonator; move ResonatorConfig to phase_resonator
- Extract shared wire codec (write/read_lp_str, write/read_deltas)
- Extract meaning_ctx() helper shared by structural_query and multi_hop
- Gate experimental VSA modules behind experimental feature
- Migrate HMS agent identity from did.cose to CAWG ICA
- Deduplicate RNG pruning, unify patterns, add doc comments
- Eliminate dead code, add delete/compact with persistence
- Remove redb dependency, use in-memory FxHashMap with arena log persistence

### Documentation
- Update changelog [skip ci]
- State the AGPL-3.0-or-later license in the security policy
- Record algorithm findings and private-search threat models
- Update changelog [skip ci]
- Rewrite README to highlight Zero-Trust Enterprise Security and Python SDK
- State the testing policy explicitly in CONTRIBUTING.md (#66)
- Sync changelog (#54)
- Align guidance and track deferred production work (#52)
- Update changelog [skip ci]
- Update changelog [skip ci]
- Standardize badges (add OpenSSF Scorecard; consistent order/format)
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Retract 'info-theoretically impossible' framing; 0.1D Plate is a readout artifact, closing gap to ceiling is open (§22)
- Block codes also floor (§24); capacity gap is retrieval-vs-information, open not impossible
- Wire deterministic-resonator capability into RESEARCH.md (discoverability + correct citation)
- DETERMINISTIC-RESONATOR.md — §20 validation write-up from in-loop 24-seed run
- Correct resonator-factorize repro command — bin is std-only, no experimental feature
- §29 confirmed on unseen seeds — 4-bit matches float, 8-16x footprint win; log floor-exploration findings
- Overnight closeout — queue exhausted, 337 tests green, flag pre-existing §21 bin lint
- Reproducibility header for resonator-factorize (fixed seeds, correct repro command)
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Record why presence gate uses an absolute margin, not SNR
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Pre-register non-self-inverse binding + nonlinear readout experiment
- Update changelog [skip ci]
- Update changelog [skip ci]
- Replace logo with continuous-rotation animation
- Update changelog [skip ci]
- Update changelog [skip ci]
- Restructure README with collapsible sections
- Update changelog [skip ci]
- Update changelog [skip ci]
- Rewrite README — fix logo for npmjs, restructure with quick start first, improve clarity
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Add agent-provenance stack cross-reference to README
- Update changelog [skip ci]
- Update all documentation for v0.5.0 accuracy
- Add meaning memory architecture, structural queries, attractor cleanup
- Add meaning memory architecture, structural queries, attractor cleanup

### Fixed
- Remove unimplemented security claims and retired cloud code
- Add missing experimental feature gate to zero_trust_demo
- Add missing experimental feature gate to zero_trust_demo
- Resolve CodeQL alerts and README formatting
- Resolve clippy::chunks_exact_to_as_chunks lint regression on main (#61)
- Resolve workflow code scanning alerts (#53)
- Give relations a distinct magic byte (0xFA), accept legacy 0xFE on read
- Clippy needless_range_loop in resonator-bundle (§21 bin) — enumerate rewrite, outcome-neutral
- Close remaining provenance trust-anchor gaps (cawg, vc-only records)
- Use sort_by_key to satisfy clippy 1.96 unnecessary_sort_by
- Repair --all-features build (port SCITT to ureq 3 API, drop dead CliffordVec bench, gate provenance example)
- Allow CDLA-Permissive-2.0 license from webpki-roots via ureq
- Replace bare unwrap() with expect() or total_cmp in production code
- Remove dead CliffordVec code and fix clippy warnings
- Perf and safety improvements from audit
- 5 bugs from audit (deadlock, double-decay, over-count, double-start, stale-used)
- Cargo fmt
- Use sort_by_key for clippy on newer Rust
- Cargo fmt formatting
- Benchmark crate rename, cargo fmt
- Resolve all clippy warnings for CI
- Quality pass on G1-G4 code
- Use absolute URL for logo on npmjs, add project.yaml for orchestrator
- Resolve 15 audit findings (3 critical, 8 high, 4 medium)
- Crash-safety, error propagation, and compaction correctness
- Propagate errors from shard insert/remove, guard ShardManager invariant
- Use .clamp() instead of .max().min() to satisfy clippy

### Performance
- Optimize Zero-Trust FHE-Lite key generation and eliminate string cloning in hot loops
- Use posting-list candidate generation for synthesizeConcepts (>500 vectors)

### Security
- Remediate repository alerts and dependencies (#38)
- Harden signing-key file perms, zeroize copies, bound JUMBF recursion

### Assets
- Static transparent logo from first frame, text removed
- Simplify logo — strip floating decorative specks, keep the connected graph mark
- Replace logo with self-contrasting node-graph mark (animated gif + static svg), drop garbled-text variants

### Deps
- Land 10 Dependabot major bumps with API fixes (#26)

### Example
- Story_memory -- HMS reasoning memory serving a writer (scrivener-mcp)

### Research
- Fractional-power encoding gives the phasor memory continuous 'near'
- Phase memory retrieves under load where sparse collapses
- Rotation binding gives relation algebra the current substrate lacks
- Path-plasticity -- holographic generalization a cache can't do
- Living-connection-graph slice -- plasticity beats saturation, verifiably
- #2 strong baselines HRR + MAP vs sparse permutation
- #2 step-1 binding discriminator harness + result
- #2 density-matched control rules out the confound
- #2 hardening RETRACTS the sparse-wins claim (AUC artifact)
- #2 involution control disconfirms the self-inverse framing

### Style
- Apply rustfmt to experiment binaries

