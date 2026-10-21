# HMS Meaning Memory Module API Reference

Every new module, every public method, every error condition.

---

## Phase 0: Shared Infrastructure

### posting.rs — Posting Shard

Inverted index from dimension index to vector IDs.

```rust
pub struct PostingShard {
    lists: Vec<Vec<u32>>,  // lists[dim_index] = sorted vec of vector IDs
    num_shards: usize,
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `(dim: usize) -> Self` | None | IndexedMemory |
| `insert` | `(&mut self, vec_id: u32, indices: &[u32])` | None | IndexedMemory::insert |
| `remove` | `(&mut self, vec_id: u32, indices: &[u32])` | None | IndexedMemory::delete |
| `get_list` | `(&self, dim_index: u32) -> &[u32]` | None (empty slice for OOB) | overlap_scan |
| `overlap_counts` | `(&self, query: &[u32], tombstones: &TombstoneMap) -> Vec<(u32, u32)>` | None | IndexedMemory::overlap_scan |
| `rebuild` | `(&mut self, vectors: &[(u32, &[u32])])` | None | IndexedMemory::load |

**Invariant:** Every list is sorted by vector ID. Insert maintains sorted order via `partition_point`. Remove uses binary search.

---

### idf.rs — IDF Weights

Inverse document frequency weights per dimension index.

```rust
pub struct IdfWeights {
    weights: Vec<f32>,    // weights[dim_index]
    total_docs: u32,
    clip_factor: f64,     // default 3.0
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `(dim: usize, clip_factor: f64) -> Self` | None | IndexedMemory |
| `update_insert` | `(&mut self, indices: &[u32])` | None | IndexedMemory::insert |
| `update_remove` | `(&mut self, indices: &[u32])` | None | IndexedMemory::delete |
| `recompute` | `(&mut self, postings: &PostingShard)` | None | IndexedMemory::load |
| `weight` | `(&self, dim_index: u32) -> f32` | None (1.0 for OOB) | overlap_scan |
| `clip` | `(&mut self)` | None | Called after recompute and periodically |

**IDF formula:** `log(total_docs / (1 + doc_freq))`. Clipped at `clip_factor * median(weights)`.

---

### tombstone.rs — Tombstone Map

Bitset tracking deleted vector IDs.

```rust
pub struct TombstoneMap {
    bits: Vec<u64>,  // bit-packed, 64 IDs per u64
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `() -> Self` | None | IndexedMemory |
| `mark_deleted` | `(&mut self, vec_id: u32)` | None | IndexedMemory::delete |
| `is_deleted` | `(&self, vec_id: u32) -> bool` | None (false for OOB) | overlap_scan, queries |
| `count` | `(&self) -> usize` | None | diagnostics |
| `clear` | `(&mut self)` | None | compaction |

---

### admission.rs — Admission Control

Fan-out gating for structural queries.

```rust
pub struct AdmissionControl {
    fanout_limit: usize,  // default 40
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `(fanout_limit: usize) -> Self` | None | StructuralQuery |
| `check` | `(&self, fan_out: usize) -> AdmissionDecision` | None | StructuralQuery |

```rust
pub enum AdmissionDecision {
    Algebraic,          // fan_out <= limit: use unbind + cleanup
    MaterializedLookup, // fan_out > limit: use TripleStore
}
```

---

### indexed_memory.rs — Shared Memory Backbone

Both AtomMemory and CompositeMemory wrap this.

```rust
pub struct IndexedMemory {
    vectors: RwLock<Vec<(String, EntangledHVec)>>,
    id_to_idx: RwLock<FxHashMap<String, usize>>,
    postings: RwLock<PostingShard>,
    idf: RwLock<IdfWeights>,
    tombstones: RwLock<TombstoneMap>,
    dim: usize,
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `(dim: usize, idf_clip_factor: f64) -> Self` | None | AtomMemory, CompositeMemory |
| `insert` | `(&self, id: String, vec: EntangledHVec) -> u32` | None | AtomMemory, CompositeMemory |
| `get` | `(&self, id: &str) -> Option<EntangledHVec>` | None | queries |
| `get_by_idx` | `(&self, idx: u32) -> Option<(String, EntangledHVec)>` | None | cleanup results |
| `delete` | `(&self, id: &str) -> bool` | None (false if not found) | deletion |
| `overlap_scan` | `(&self, query: &EntangledHVec) -> Vec<(u32, f32)>` | None | structural_query, cleanup |
| `count` | `(&self) -> usize` | None | diagnostics |
| `rebuild_indices` | `(&self)` | None | after load |
| `all_vectors` | `(&self) -> Vec<(u32, String, EntangledHVec)>` | None | persistence snapshot |

### Standalone Functions

```rust
pub fn hopfield_cleanup(
    probe: &EntangledHVec,
    memory: &IndexedMemory,
    beta: f64,
    k: usize,
    max_iter: usize,
) -> CleanupResult
```

Modern Hopfield attractor dynamics. Returns the nearest stored vector.

**Algorithm:** overlap scan -> softmax(beta * overlaps) -> weighted superposition -> top-k projection -> iterate.

**Errors:** Returns `CleanupResult { found: false, .. }` if no vectors exceed threshold.

```rust
pub fn sparse_softmax(overlaps: &[(u32, f32)], beta: f64) -> Vec<(u32, f64)>
```

Numerically stable softmax over sparse overlap scores.

---

### role.rs — Role Algebra

```rust
pub struct RoleRegistry {
    shifts: FxHashMap<String, usize>,  // role_name -> shift amount
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `() -> Self` | None | HmsCore meaning init |
| `default_triple_roles` | `() -> Self` | None | convenience constructor |
| `register` | `(&mut self, role: &str, shift: usize)` | `Err` if role already registered with different shift | config |
| `compose` | `(&self, bindings: &[(&str, &EntangledHVec)]) -> EntangledHVec` | `Err` if unknown role | write path |
| `unbind` | `(&self, composite: &EntangledHVec, known: &[(&str, &EntangledHVec)]) -> EntangledHVec` | `Err` if unknown role | structural query |
| `shift_for` | `(&self, role: &str) -> Option<usize>` | None | internal |

**Default roles:** subject=0, relation=1, object=3.

**compose:** `T = XOR_i(permute(vec_i, shift_i))`
**unbind:** `result = composite XOR XOR_i(permute(known_i, shift_i))`

---

## Phase 1: Meaning Memory Core

### atom_memory.rs — Atom Memory

Thin wrapper around IndexedMemory for individual concept atoms.

```rust
pub struct AtomMemory {
    inner: IndexedMemory,
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `(dim: usize, idf_clip: f64) -> Self` | None | HmsCore |
| `get_or_insert` | `(&self, atom_str: &str) -> (u32, EntangledHVec)` | None | Decomposer, StructuralQuery |
| `get` | `(&self, id: &str) -> Option<EntangledHVec>` | None | queries |
| `get_by_idx` | `(&self, idx: u32) -> Option<(String, EntangledHVec)>` | None | cleanup results |
| `cleanup` | `(&self, noisy: &EntangledHVec, beta: f64, k: usize, max_iter: usize) -> CleanupResult` | None | structural query |
| `delete` | `(&self, id: &str) -> bool` | None | deletion |
| `serialize_atom` | `(id: &str, vec: &EntangledHVec) -> Vec<u8>` | None | persistence |
| `deserialize_atom` | `(data: &[u8]) -> Option<(String, EntangledHVec)>` | None (magic 0xFD check) | load |
| `load_atom` | `(&self, id: String, vec: EntangledHVec)` | None | log replay |
| `count` | `(&self) -> usize` | None | diagnostics |
| `rebuild_indices` | `(&self)` | None | after load |

---

### composite_memory.rs — Composite Memory

Thin wrapper around IndexedMemory for role-bound composites.

```rust
pub struct CompositeMemory {
    inner: IndexedMemory,
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `(dim: usize, idf_clip: f64) -> Self` | None | HmsCore |
| `insert` | `(&self, id: String, composite: EntangledHVec) -> u32` | None | write path |
| `get` | `(&self, id: &str) -> Option<EntangledHVec>` | None | queries |
| `get_by_idx` | `(&self, idx: u32) -> Option<(String, EntangledHVec)>` | None | unbinding |
| `overlap_scan` | `(&self, query: &EntangledHVec) -> Vec<(u32, f32)>` | None | structural query |
| `delete` | `(&self, id: &str) -> bool` | None | deletion |
| `serialize_composite` | `(id: &str, vec: &EntangledHVec) -> Vec<u8>` | None | persistence |
| `deserialize_composite` | `(data: &[u8]) -> Option<(String, EntangledHVec)>` | None (magic 0xFC check) | load |
| `load_composite` | `(&self, id: String, vec: EntangledHVec)` | None | log replay |
| `count` | `(&self) -> usize` | None | diagnostics |
| `rebuild_indices` | `(&self)` | None | after load |

---

### triple_store.rs — Triple Store

Materialized subject-relation-object-composite records.

```rust
pub struct TripleStore {
    triples: RwLock<Vec<TripleRecord>>,
    by_subject: RwLock<FxHashMap<String, Vec<usize>>>,
    by_relation: RwLock<FxHashMap<String, Vec<usize>>>,
    by_object: RwLock<FxHashMap<String, Vec<usize>>>,
}

pub struct TripleRecord {
    pub subject_id: String,
    pub relation_id: String,
    pub object_id: String,
    pub composite_id: String,
    pub deleted: bool,
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `() -> Self` | None | HmsCore |
| `add` | `(&self, subject: &str, relation: &str, object: &str, composite_id: &str) -> usize` | None | write path |
| `remove` | `(&self, subject: &str, relation: &str, object: &str) -> bool` | None | deletion |
| `query` | `(&self, subject: Option<&str>, relation: Option<&str>, object: Option<&str>) -> Vec<TripleRecord>` | None | materialized lookup, multi-hop |
| `serialize_triple` | `(record: &TripleRecord) -> Vec<u8>` | None | persistence |
| `deserialize_triple` | `(data: &[u8]) -> Option<TripleRecord>` | None (magic 0xFB check) | load |
| `load_triple` | `(&self, record: TripleRecord)` | None | log replay |
| `count` | `(&self) -> usize` | None | diagnostics |
| `snapshot` | `(&self) -> Vec<TripleRecord>` | None | compaction |
| `all_for_subject` | `(&self, subject: &str) -> Vec<TripleRecord>` | None | cognition layer |
| `all_for_relation` | `(&self, relation: &str) -> Vec<TripleRecord>` | None | pattern scanner |

---

### rules.rs — Rule Store

Composition rules for multi-hop inference.

```rust
pub struct RuleStore {
    rules: RwLock<Vec<CompositionRule>>,
    by_chain: RwLock<FxHashMap<(String, String), usize>>,  // (rel1, rel2) -> rule index
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `() -> Self` | None | HmsCore |
| `add_rule` | `(&self, rule: CompositionRule) -> usize` | None | config, learning |
| `find_rule` | `(&self, rel1: &str, rel2: &str) -> Option<CompositionRule>` | None | multi_hop |
| `all_rules` | `(&self) -> Vec<CompositionRule>` | None | persistence |
| `serialize_rule` | `(rule: &CompositionRule) -> Vec<u8>` | None | persistence |
| `deserialize_rule` | `(data: &[u8]) -> Option<CompositionRule>` | None (magic 0xFE check) | load |
| `load_rule` | `(&self, rule: CompositionRule)` | None | log replay |
| `count` | `(&self) -> usize` | None | diagnostics |

---

### decompose.rs — Text Decomposer

Rule-based triple extraction from English text.

```rust
pub struct Decomposer {
    prepositions: FxHashSet<&'static str>,
    verbs: FxHashSet<&'static str>,
    determiners: FxHashSet<&'static str>,
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `() -> Self` | None | HmsCore meaning init |
| `extract_triples` | `(&self, text: &str) -> Vec<(String, String, String)>` | None (empty vec for unparseable) | memorize_text |
| `extract_noun_phrases` | `(&self, tokens: &[&str]) -> Vec<NounPhrase>` | None | internal |
| `classify_token` | `(&self, token: &str) -> TokenClass` | None | internal |

**Error handling:** Never fails. Returns empty vec if no triples can be extracted. Precision over recall: ambiguous parses are dropped.

---

### engine/structural.rs — Structural Query

```rust
pub fn structural_query(
    atoms: &AtomMemory,
    composites: &CompositeMemory,
    roles: &RoleRegistry,
    triples: &TripleStore,
    admission: &AdmissionControl,
    known_bindings: &[(&str, &str)],  // (role_name, atom_id)
    config: &MeaningConfig,
) -> Result<StructuralResult>
```

| Error | Condition |
|-------|-----------|
| `Err("unknown atom")` | A known_binding references an atom not in AtomMemory |
| `Err("unknown role")` | A known_binding uses an unregistered role name |
| `Err("no bindings")` | Empty known_bindings |

**Used by:** HmsCore meaning query methods, multi_hop chaining.

---

### engine/multi_hop.rs — Multi-Hop Query

```rust
pub fn multi_hop_query(
    atoms: &AtomMemory,
    composites: &CompositeMemory,
    roles: &RoleRegistry,
    triples: &TripleStore,
    rules: &RuleStore,
    admission: &AdmissionControl,
    chain: &[(&str, &str)],  // [(relation1, start_entity), (relation2, ?), ...]
    config: &MeaningConfig,
) -> Result<MultiHopResult>
```

| Error | Condition |
|-------|-----------|
| `Err("empty chain")` | No hops specified |
| `Err("unknown atom")` | Start entity not in AtomMemory |
| `Err("chain too long")` | More than 10 hops (safety limit) |

**Used by:** HmsCore meaning query methods.

---

## New Types

All types in `core/types.rs` (or a new `core/meaning_types.rs`).

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuralResult {
    pub matches: Vec<(String, f64)>,   // (atom_id, score) ranked
    pub method: String,                // "algebraic" or "materialized"
    pub fan_out: usize,
    pub iterations: usize,            // Hopfield iterations used
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiHopResult {
    pub final_matches: Vec<(String, f64)>,
    pub hops: Vec<HopDetail>,
    pub rule_applied: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HopDetail {
    pub relation: String,
    pub intermediates: Vec<String>,
    pub method: String,
}

#[derive(Clone, Debug)]
pub struct MeaningUnit {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompositionRule {
    pub relation1: String,
    pub relation2: String,
    pub composed_relation: String,
}

#[derive(Clone, Debug)]
pub struct CleanupResult {
    pub found: bool,
    pub atom_id: Option<String>,
    pub atom_idx: Option<u32>,
    pub similarity: f64,
    pub iterations: usize,
}

#[derive(Clone, Debug)]
pub struct StructuralPattern {
    pub relation: String,
    pub subject_atoms: Vec<String>,
    pub object_atoms: Vec<String>,
    pub frequency: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeGap {
    pub entity: String,
    pub missing_relation: String,
    pub peer_entities: Vec<String>,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hypothesis {
    pub entity: String,
    pub relation: String,
    pub predicted_filler: String,
    pub confidence: f64,
    pub supporting_peers: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Analogy {
    pub source_entities: Vec<String>,
    pub target_entities: Vec<String>,
    pub mapping: Vec<(String, String)>,  // (source, target) pairs
    pub structural_similarity: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Insight {
    Pattern(StructuralPattern),
    Gap(KnowledgeGap),
    Hypothesis(Hypothesis),
    Analogy(Analogy),
}
```

### HmsConfigJs Additions (lib.rs)

```rust
// Add to HmsConfigJs struct:
pub meaning_enabled: Option<bool>,
pub meaning_dim: Option<u32>,
pub meaning_hopfield_beta: Option<f64>,
pub meaning_hopfield_k: Option<u32>,
pub meaning_hopfield_max_iter: Option<u32>,
pub meaning_fanout_limit: Option<u32>,
pub meaning_idf_clip_factor: Option<f64>,
```

### N-API Result Types

```rust
#[cfg(feature = "node-api")]
#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct StructuralResultJs { /* mirrors StructuralResult */ }

#[cfg(feature = "node-api")]
#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
pub struct MultiHopResultJs { /* mirrors MultiHopResult */ }
```

---

## Phase 2: Cognition Layer

### patterns.rs — Pattern Scanner

Finds recurring structural patterns across triples.

```rust
pub struct PatternScanner;
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `scan` | `(triples: &TripleStore, min_frequency: usize) -> Vec<StructuralPattern>` | None | CognitionLoop |

**Algorithm:**
1. Group triples by relation.
2. For each relation with >= min_frequency triples, collect subject and object atom sets.
3. Emit a StructuralPattern with frequency count.

---

### abstraction.rs — Abstraction Engine

Creates prototype atoms from recurring patterns.

```rust
pub struct AbstractionEngine;
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `maybe_abstract` | `(pattern: &StructuralPattern, atoms: &AtomMemory, min_members: usize) -> Option<(String, EntangledHVec)>` | None | CognitionLoop |

**Algorithm:** Collects HVecs for all subject atoms in the pattern, then calls `EntangledHVec::bundle()` (NOT XOR) to create a prototype. Returns None if fewer than min_members atoms found.

The resulting prototype represents the "average" entity in that pattern role. Named `"proto:{relation}:{role}"`.

---

### gaps.rs — Gap Detection

Detects missing relations by comparing entity profiles.

```rust
pub struct GapDetector;
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `detect_gaps` | `(entity: &str, triples: &TripleStore, min_peers: usize, min_confidence: f64) -> Vec<KnowledgeGap>` | None | CognitionLoop |

**Algorithm:**
1. Get all relations for entity.
2. Find peer entities (share at least one relation pattern).
3. For each relation that peers have but entity lacks, compute confidence = peer_count / total_peers.
4. Filter by min_confidence, return sorted by confidence descending.

---

### hypothesis.rs — Hypothesis Generation

Predicts fillers for detected gaps.

```rust
pub struct HypothesisEngine;
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `generate` | `(gap: &KnowledgeGap, triples: &TripleStore, atoms: &AtomMemory, config: &MeaningConfig) -> Option<Hypothesis>` | None | CognitionLoop |

**Algorithm:**
1. Collect filler atoms from peer entities for the missing relation.
2. Bundle peer fillers into a predicted filler prototype.
3. Run Hopfield cleanup on the prototype to find nearest stored atom.
4. Return Hypothesis with the cleaned-up prediction and confidence.

---

### analogy.rs — Analogy Detection

Finds structural isomorphisms between entity subgraphs.

```rust
pub struct AnalogyDetector;
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `find_analogies` | `(triples: &TripleStore, atoms: &AtomMemory, min_similarity: f64) -> Vec<Analogy>` | None | CognitionLoop |

**Algorithm:**
1. Build relation-adjacency graph.
2. Find connected components (groups of entities linked by shared relation patterns).
3. For each pair of components, compute greedy bipartite mapping by relation overlap.
4. Score each mapping by structural similarity (fraction of relations preserved).
5. Filter by min_similarity, return sorted descending.

---

### loop.rs — Cognition Loop

Background processing loop.

```rust
pub struct CognitionLoop {
    handle: Option<std::thread::JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
    insights: Arc<RwLock<Vec<Insight>>>,
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `start` | `(atoms: Arc<AtomMemory>, composites: Arc<CompositeMemory>, triples: Arc<TripleStore>, rules: Arc<RuleStore>, config: MeaningConfig) -> Self` | None | HmsCore |
| `stop` | `(&mut self)` | None | HmsCore::drop |
| `take_insights` | `(&self) -> Vec<Insight>` | None | HmsCore query methods |

**Threading model:**
- Background thread acquires READ-ONLY locks on AtomMemory, CompositeMemory, TripleStore.
- Discovered insights are pushed to `self.insights` (separate Vec, not main stores).
- No write locks on main stores ever held by cognition thread.
- `stop_flag` is checked between each phase to allow clean shutdown.
- Cycle frequency: configurable, default every 10 seconds if idle.

---

### governor.rs — Memory Governor

Consolidation, forgetting, and reorganization.

```rust
pub struct MemoryGovernor;
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `consolidate` | `(insights: &[Insight], atoms: &AtomMemory, composites: &CompositeMemory, triples: &TripleStore, rules: &RuleStore) -> Vec<MeaningUnit>` | None | explicit call |
| `forget` | `(triples: &TripleStore, atoms: &AtomMemory, max_age_ms: u64, min_access_count: u32) -> usize` | None | explicit call, returns count forgotten |
| `reorganize` | `(atoms: &AtomMemory, composites: &CompositeMemory) -> usize` | None | explicit call, returns count reorganized |

**consolidate:** Promotes high-confidence insights (hypotheses, patterns) into permanent triples and rules. Requires WRITE locks.

**forget:** Removes low-value triples (old, rarely accessed). Cascades to composite and atom deletion if no remaining references.

**reorganize:** Rebuilds posting lists and IDF weights. Compacts tombstones.

---

## Phase 3: Agency Layer

### goals.rs — Goal Store

Separate from AtomMemory. Stores goals with utility scores.

```rust
pub struct GoalStore {
    goals: RwLock<Vec<Goal>>,
    vectors: RwLock<FxHashMap<String, EntangledHVec>>,
}

pub struct Goal {
    pub id: String,
    pub description: String,
    pub relevance: f64,
    pub urgency: f64,
    pub cost: f64,
}
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `new` | `() -> Self` | None | HmsCore |
| `add_goal` | `(&self, goal: Goal, dim: usize) -> u32` | None | external API |
| `remove_goal` | `(&self, id: &str) -> bool` | None | external API |
| `active_goals` | `(&self) -> Vec<Goal>` | None | planner, questions |
| `utility` | `(goal: &Goal) -> f64` | None | ranking |
| `most_relevant` | `(&self, context: &EntangledHVec) -> Option<Goal>` | None | planner |

**Utility function:** `u(goal) = relevance * urgency * (1.0 - cost)`

---

### planner.rs — Backward-Chaining Planner

Plans to achieve goals by chaining causal triples.

```rust
pub struct Planner;
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `plan` | `(goal: &Goal, triples: &TripleStore, atoms: &AtomMemory, max_depth: usize) -> Option<Vec<MeaningUnit>>` | None | agency loop |

**Algorithm:**
1. Encode goal description to atom vector.
2. Find triples where object matches goal (backward search).
3. Recursively plan to achieve each subject (sub-goals).
4. Return ordered list of actions (triples to make true).
5. Max depth prevents infinite recursion (default 5).

---

### questions.rs — Question Generation

Template-based question generation from gaps and hypotheses.

```rust
pub struct QuestionGenerator;
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `generate` | `(gaps: &[KnowledgeGap], hypotheses: &[Hypothesis], goals: &GoalStore, atoms: &AtomMemory) -> Vec<RankedQuestion>` | None | agency loop |

```rust
pub struct RankedQuestion {
    pub text: String,
    pub source: String,        // "gap" or "hypothesis"
    pub goal_relevance: f64,
    pub priority: f64,
}
```

**Templates:**
- Gap: "What is the {relation} of {entity}?"
- Hypothesis: "Is it true that {entity} {relation} {predicted_filler}?"

**Ranking:** `priority = confidence * goal_relevance`. Goal relevance is computed as overlap between question vector and active goal vectors.

---

### self_modify.rs — Self-Modification

Proposals for changing rules, thresholds, or patterns.

```rust
pub struct SelfModifier;
```

| Method | Signature | Errors | Used By |
|--------|-----------|--------|---------|
| `propose_rule` | `(pattern: &StructuralPattern, triples: &TripleStore) -> Option<Proposal>` | None | agency loop |
| `propose_threshold_change` | `(metric: &str, current: f64, suggested: f64, evidence: &str) -> Proposal` | None | agency loop |
| `apply_proposal` | `(proposal: &Proposal, rules: &RuleStore, config: &mut MeaningConfig, audit: Option<&AuditLog>) -> Result<()>` | `Err("approval required")` if not approved | explicit call |

```rust
pub struct Proposal {
    pub id: String,
    pub description: String,
    pub proposal_type: ProposalType,
    pub approved: bool,
    pub timestamp_ms: u64,
}

pub enum ProposalType {
    NewRule(CompositionRule),
    ThresholdChange { metric: String, from: f64, to: f64 },
    DecompositionPattern { pattern: String },
}
```

**Critical constraint:** `apply_proposal` checks `proposal.approved == true`. Unapproved proposals return `Err`. All applications are logged in the audit trail. No autonomous self-modification without explicit approval.
