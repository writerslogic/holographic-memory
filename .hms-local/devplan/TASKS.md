# HMS Meaning Memory Task List

## Rules

1. Tasks within the same group (G) may execute in parallel.
2. Run `cargo` ONCE per group milestone, not after every edit.
3. Read ARCHITECTURE.md, CONTEXT.md, and MODULES.md before starting any task.
4. Each task file must include the copyright header and follow existing conventions.

---

## Dependency Table

| Task | Title | Group | Depends On |
|------|-------|-------|------------|
| T01 | PostingShard | G1 | — |
| T02 | IdfWeights | G1 | — |
| T03 | TombstoneMap | G1 | — |
| T04 | AdmissionControl | G1 | — |
| T05 | RoleRegistry | G1 | — |
| T06 | IndexedMemory + hopfield_cleanup + sparse_softmax | G2 | T01, T02, T03 |
| T07 | AtomMemory | G3 | T06 |
| T08 | CompositeMemory | G3 | T06 |
| T09 | TripleStore | G3 | — |
| T10 | RuleStore | G3 | — |
| T11 | Decomposer | G3 | — |
| T12 | StructuralQuery (engine/structural.rs) | G4 | T04, T05, T07, T08, T09 |
| T13 | MultiHopQuery (engine/multi_hop.rs) | G4 | T10, T12 |
| T14 | MeaningConfig + HmsConfig integration | G4 | — |
| T15 | HmsCore integration (meaning fields, load_from_log dispatch, memorize_text) | G5 | T07, T08, T09, T10, T11, T12, T13, T14 |
| T16 | Persistence (serialize/deserialize for atoms, composites, triples, rules) | G5 | T07, T08, T09, T10 |
| T17 | N-API bindings (HmsConfigJs additions, result types, new methods) | G5 | T14, T15 |
| T18 | Phase 1 integration tests (8 tests, see G6 detail) | G6 | T15, T16, T17 |
| T19 | Clippy + doc cleanup | G6 | T18 |
| T20 | PatternScanner + AbstractionEngine + GapDetector + HypothesisEngine + AnalogyDetector | G7 | T15 |
| T21 | CognitionLoop + MemoryGovernor | G7 | T20 |
| T22 | GoalStore + Planner + QuestionGenerator + SelfModifier | G7 | T20, T21 |
| T23 | Release: version bump, CHANGELOG, npm publish | G8 | T22 |

---

## Group Execution Details

### G1: Shared Infrastructure (T01-T05)

All five modules are independent. Write them in parallel.

- **T01 PostingShard** (`src/core/meaning/posting.rs`): inverted index dim_index -> sorted vec of vec_ids. Insert maintains sorted order via `partition_point`. Remove via binary search. `overlap_counts` iterates query indices, merges posting lists, excludes tombstones.

- **T02 IdfWeights** (`src/core/meaning/idf.rs`): `log(N / (1 + df))` per dimension. `clip()` finds median weight, caps all at `3.0 * median`. Incremental update on insert/remove.

- **T03 TombstoneMap** (`src/core/meaning/tombstone.rs`): bit-packed `Vec<u64>`. `mark_deleted` sets bit. `is_deleted` checks bit. Auto-grows on access beyond current capacity.

- **T04 AdmissionControl** (`src/core/meaning/admission.rs`): trivial struct with `check(fan_out) -> AdmissionDecision` enum. Two variants: `Algebraic` and `MaterializedLookup`.

- **T05 RoleRegistry** (`src/core/meaning/role.rs`): FxHashMap<String, usize> for role->shift. `default_triple_roles()` pre-registers subject=0, relation=1, object=3. `compose` and `unbind` use `permute` and `bind` from EntangledHVec.

**Also:** Create `src/core/meaning/mod.rs` with `pub mod` declarations for all Phase 0 and Phase 1 modules. Add `pub(crate) mod meaning;` to `src/core/mod.rs`.

### G2: IndexedMemory (T06)

Depends on T01-T03. Single task but substantial.

- **T06 IndexedMemory** (`src/core/meaning/indexed_memory.rs`): wraps PostingShard + IdfWeights + TombstoneMap. Vectors stored in `Vec<(String, EntangledHVec)>` with `FxHashMap<String, usize>` for ID lookup. `overlap_scan` uses posting lists to compute IDF-weighted overlaps. `rebuild_indices` reconstructs postings and IDF from stored vectors.

  Standalone functions `hopfield_cleanup` and `sparse_softmax` live in this file.

  `hopfield_cleanup` algorithm:
  1. `overlap_scan(probe)` -> `Vec<(u32, f32)>`
  2. `sparse_softmax(overlaps, beta)` -> attention weights
  3. Weighted index frequency counting -> new probe indices
  4. Keep top-k indices by frequency
  5. Repeat until convergence or max_iter
  6. Return `CleanupResult` with nearest stored atom

### G3: Memory Stores + Decomposer (T07-T11)

All five are independent (each wraps IndexedMemory or is standalone).

- **T07 AtomMemory** (`src/core/meaning/atom_memory.rs`): wraps IndexedMemory. `get_or_insert` hashes atom string to seed, creates deterministic HVec if not present. Persistence: magic 0xFD, delta-encoded indices.

- **T08 CompositeMemory** (`src/core/meaning/composite_memory.rs`): wraps IndexedMemory. Same persistence pattern as AtomMemory but magic 0xFC.

- **T09 TripleStore** (`src/core/meaning/triple_store.rs`): FxHashMap indices by subject/relation/object. `query` accepts optional filters for each field. Magic 0xFB for persistence. Follow `graph.rs` serialize/deserialize pattern.

- **T10 RuleStore** (`src/core/meaning/rules.rs`): stores CompositionRule records. Indexed by (relation1, relation2) pair. Magic 0xFE (shared with relations, differentiated by field structure). `find_rule` is the hot path for multi-hop optimization.

- **T11 Decomposer** (`src/core/meaning/decompose.rs`): 40 prepositions, 30 verbs, determiners as FxHashSet<&'static str>. Tokenize by whitespace + lowercasing. Extract NPs, match against 4 patterns. Return `Vec<(String, String, String)>`. Include unit tests for each pattern.

### G4: Query Engine (T12-T14)

- **T12 StructuralQuery** (`src/core/engine/structural.rs`): implements the full structural query flow. Build query vector from known bindings via RoleRegistry. Overlap scan on CompositeMemory. Fan-out check via AdmissionControl. Algebraic path: unbind each composite, cleanup via AtomMemory. Materialized path: TripleStore.query with known fields.

- **T13 MultiHopQuery** (`src/core/engine/multi_hop.rs`): check RuleStore for chain shortcut. If rule found, rewrite to single structural_query. If not, chain TripleStore lookups step by step. Return MultiHopResult with per-hop details.

- **T14 MeaningConfig** (`src/core/config.rs`): add `MeaningConfig` struct with Default impl (enabled=false). Add `pub meaning: MeaningConfig` to HmsConfig. Update HmsConfig Default impl.

### G5: Integration (T15-T17)

- **T15 HmsCore Integration** (`src/core/engine/mod.rs`): add optional meaning memory fields (AtomMemory, CompositeMemory, TripleStore, RuleStore, RoleRegistry, Decomposer). Initialize when `config.meaning.enabled`. Add `memorize_text` method that runs decomposer, stores atoms, composes and stores composites, records triples. Add `structural_query` and `multi_hop_query` forwarding methods. Update `load_from_log` to dispatch 0xFD/0xFC/0xFB/0xFE frames to meaning stores. Update `compact` to include meaning data.

- **T16 Persistence** (`src/core/meaning/` atom/composite/triple/rule files): implement `serialize_*` and `deserialize_*` following the graph.rs magic-byte pattern. Delta-encode HVec indices for compact storage. Add round-trip tests for each format.

- **T17 N-API Bindings** (`src/lib.rs`): add meaning config fields to HmsConfigJs. Add `into_config` mapping. Add `StructuralResultJs` and `MultiHopResultJs` N-API objects. Add `memorize_text`, `structural_query`, `multi_hop_query` async methods to HolographicMemorySystem.

### G6: Testing + Polish (T18-T19)

- **T18 Phase 1 Integration Tests** (`tests/meaning_integration.rs` or `src/core/meaning/tests.rs`):

  **Test 1 — Basin Certification:**
  Insert 100K atoms into AtomMemory. For each of 10 random probe atoms, add 0%, 10%, and 25% noise (flip random indices). Run hopfield_cleanup. Assert recall >= 0.95 at 25% noise. Assert convergence in <= 3 iterations.

  **Test 2 — Fan-Out Sweep:**
  Create triples with fan-out counts {1, 5, 10, 20, 30, 40, 50}. Add 50K noise triples. Query with 25% noisy subject vector. For fan-out <= 40: assert algebraic recall >= 0.95. For fan-out > 40: assert materialized fallback is used and returns correct results.

  **Test 3 — Role Inversion:**
  Insert 100 triples. For each triple, query subject from (relation, object), relation from (subject, object), and object from (subject, relation). Assert all three unbinding directions recover the correct atom.

  **Test 4 — Poisoning Resistance:**
  Create 100 atoms, of which 10 share a 48-index "decoy core" (common indices injected). Insert into AtomMemory. Query with a probe similar to the decoy. Assert IDF clipping prevents the decoy indices from dominating results. Non-decoy atoms should not spuriously match.

  **Test 5 — Concurrent Access:**
  Spawn 8 writer threads and 4 reader threads. Each writer does 10K insert operations. Each reader does 10K overlap_scan operations. Assert zero deadlocks (test completes within timeout). Assert no data corruption (all inserted atoms retrievable after joins).

  **Test 6 — Multi-Hop with Rules:**
  Insert triples: father(John, Mark), father(Mark, Bob). Add rule: father + father = grandfather. Query: grandfather(John, ?). Assert result contains Bob. Verify hop_details show rule application.

  **Test 7 — Persistence Round-Trip:**
  Store atoms, composites, triples, and rules. Drop all in-memory state. Reload from arena. Verify all recovered data matches originals. Verify posting lists and IDF rebuilt correctly.

  **Test 8 — Backward Compatibility:**
  Set `meaning.enabled = false`. Run the full existing test suite (all 133+ tests). Assert all pass with zero modifications. Assert no meaning memory structs are allocated (check with a simple flag or counter).

- **T19 Clippy + Doc Cleanup:** Run `cargo clippy --all-targets --all-features`. Fix all warnings. Add `//!` module doc comments to every new file. Add `///` doc comments to every public function and type.

### G7: Cognition + Agency (T20-T22)

- **T20 Cognition Components** (`src/core/meaning/`): PatternScanner, AbstractionEngine, GapDetector, HypothesisEngine, AnalogyDetector. Each is a stateless struct with a single main method. All operate with READ-ONLY locks on meaning stores. Include unit tests per component.

- **T21 CognitionLoop + Governor** (`src/core/meaning/loop.rs`, `src/core/meaning/governor.rs`): background thread with stop flag. Read-only lock acquisition. Insights collected in separate Vec. Governor handles consolidate/forget/reorganize with write locks only when explicitly called.

- **T22 Agency Layer** (`src/core/meaning/goals.rs`, `planner.rs`, `questions.rs`, `self_modify.rs`): GoalStore separate from AtomMemory. Planner does backward-chaining. QuestionGenerator is template-based. SelfModifier requires approval for all changes, logs to audit.

### G8: Release (T23)

- **T23 Release:** Bump version in Cargo.toml. Update CHANGELOG. Run full quality gate. Publish to crates.io and npm (if applicable).

---

## Verification Milestones

| Group | Gate | Command | Pass Criteria |
|-------|------|---------|---------------|
| G1 | Compile check | `cargo check` | Zero errors |
| G2 | Compile + cleanup test | `cargo check` + unit tests for hopfield_cleanup | Zero errors, cleanup converges |
| G3 | Compile check | `cargo check` | Zero errors |
| G4 | Compile check | `cargo check` | Zero errors |
| G5 | Full test suite | `cargo test --lib` | ALL tests pass (133+ existing + new) |
| G6 | Tests + lint | `cargo test` + `cargo clippy --all-targets --all-features` | All 8 integration tests pass, zero clippy warnings |
| G7 | Full quality gate | `cargo test` + `cargo clippy --all-targets --all-features` + doc check | All tests pass, zero warnings, all pub items documented |
| G8 | Publish | `cargo publish --dry-run` | Clean publish, version bumped |
