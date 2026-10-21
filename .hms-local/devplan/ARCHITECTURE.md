# HMS Meaning Memory Architecture

## 1. The Big Picture

Three new layers sit atop the existing VSA engine:

```
+---------------------------------------------+
| Phase 3  Agency                              |
|  GoalStore, Planner, Questions, SelfModify   |
+---------------------------------------------+
| Phase 2  Cognition                           |
|  PatternScanner, Abstraction, Gaps,          |
|  Hypothesis, Analogy, CognitionLoop          |
+---------------------------------------------+
| Phase 1  Meaning Memory                      |
|  AtomMemory, CompositeMemory, TripleStore,   |
|  RuleStore, RoleRegistry, Decomposer,        |
|  StructuralQuery, MultiHop                   |
+---------------------------------------------+
| Existing  VSA Engine                         |
|  EntangledHVec, PersistentArena, HmsCore,    |
|  RelationStore, IVF, NSG, Shards, Encoding   |
+---------------------------------------------+
```

Everything is gated behind `config.meaning.enabled` (default `false`).
When disabled, all 133+ existing tests pass unchanged, no new allocations
occur, and HmsCore behaves identically to the pre-meaning codebase.

---

## 2. Phase 1: Meaning Memory Data Flows

### Write Path

```
memorize_text(text)
  -> Decomposer extracts triples: Vec<(subject, relation, object)>
  -> For each unique atom string:
       AtomMemory.get_or_insert(atom_str)
       -> EntangledHVec::new_deterministic(dim, hash(atom_str))
       -> store in IndexedMemory (posting lists + IDF updated)
  -> For each triple (S, R, O):
       RoleAlgebra composes:
         T = S XOR permute(R, 1) XOR permute(O, 3)
       -> composite into CompositeMemory (IndexedMemory)
       -> triple record into TripleStore (subject_id, relation_id, object_id, composite_id)
  -> Persist: atom frames (magic 0xFD), composite frames (magic 0xFC),
     triple frames (magic 0xFB), rule frames (magic 0xFE reused)
```

### Structural Query (single-hop)

```
structural_query(known_roles)
  -> Build query vector Q from known roles via RoleAlgebra:
       e.g., Q = permute(R, 1) XOR permute(O, 3)  (to find subject)
  -> Hopfield overlap scan over CompositeMemory:
       overlaps[i] = |Q.indices intersect C_i.indices|
  -> Fan-out decision:
       fan_out = count of composites above threshold
       IF fan_out <= 40:
         Per-composite algebraic unbind:
           candidate = C_i XOR Q   (recovers missing role's atom)
         AtomMemory cleanup:
           Hopfield attractor dynamics on candidate -> nearest clean atom
         Return StructuralResult with ranked atoms
       IF fan_out > 40:
         Materialized TripleStore lookup:
           Filter triples by known role values
         Return StructuralResult from TripleStore rows
```

### Multi-Hop Query

```
multi_hop_query(chain)
  -> Check RuleStore for matching composition rule
  -> Rule exists:
       Rewrite chain to single structural_query using rule's composed relation
       Return result with hop_details showing rule application
  -> No rule:
       Chain TripleStore lookups:
         Step 1: query with initial known roles -> intermediate results
         Step 2: feed intermediates as known roles into next query
         ...
       Return MultiHopResult with per-hop details
```

---

## 3. Module Connection Map

```
                          +-----------+
                          |  HmsCore  |
                          +-----+-----+
                                |
          +----------+----------+----------+-----------+
          |          |          |          |           |
     +----v----+ +--v---+ +---v---+ +---v----+ +----v-----+
     |  Atom   | |Compos| |Triple | | Rule   | |  Role    |
     | Memory  | | ite  | | Store | | Store  | | Registry |
     +----+----+ |Memory| +---+---+ +---+----+ +----------+
          |      +--+---+     |         |
          |         |         |         |
     +----v---------v----+    |         |
     |  IndexedMemory    |    |         |
     |  (shared backbone)|    |         |
     +---+---+---+---+---+    |         |
         |   |   |   |        |         |
    +----v+  v  +v   v----+   |         |
    |Post | IDF |Tomb|Admis|  |         |
    |Shard|     |ston|sion |  |         |
    +-----+     +----+-----+  |         |
                               |         |
                          +----v---------v----+
                          |  PersistentArena  |
                          +-------------------+
```

- **AtomMemory** and **CompositeMemory** both wrap `IndexedMemory`.
- `IndexedMemory` implements posting lists, IDF weights, tombstones, and overlap scan ONCE.
- **TripleStore** holds materialized (subject_id, relation_id, object_id, composite_id) rows.
- **RuleStore** holds composition rules (e.g., father + father = grandfather).
- **RoleRegistry** provides the shift constants and unbinding logic.
- **AdmissionControl** gates fan-out to prevent combinatorial explosion.
- **PersistentArena** handles all disk I/O with CRC32+LZ4 framing.

---

## 4. XOR Commutativity Fix

Plain XOR is commutative: `S XOR R XOR O == O XOR R XOR S`. This means
"dog bites man" and "man bites dog" would produce the same composite.

**Fix: distinct permutation shifts per role.**

| Role     | Shift |
|----------|-------|
| Subject  | 0     |
| Relation | 1     |
| Object   | 3     |

**Triple composition formula:**

```
T = permute(S, 0) XOR permute(R, 1) XOR permute(O, 3)
  = S XOR permute(R, 1) XOR permute(O, 3)
```

**Unbinding formula (to recover Subject):**

```
S_noisy = T XOR permute(R, 1) XOR permute(O, 3)
```

Since `permute(X, n) XOR permute(X, n) = empty` and XOR is its own inverse,
this exactly recovers S when T was composed with those R and O values.
The shifts 0, 1, 3 are pairwise distinct, so role positions are unambiguous.

---

## 5. Attractor Dynamics (Hopfield Cleanup)

Retrieval uses Modern Hopfield Network dynamics to clean up noisy
probe vectors into stored clean atoms.

**Algorithm:**

```
1. Compute overlaps: overlap_i = sparse_intersection_count(probe, stored_i)
2. Attention: weights = softmax(beta * overlaps)
     where beta = 1 / sqrt(active_count)
3. Superposition: new_probe = weighted_sum(stored_i, weights[i])
     (implemented as weighted index frequency counting)
4. Project: keep top-k=64 indices by weight (maintains sparsity)
5. Iterate until convergence or max 3 iterations
```

**Energy function:**

```
E = -log(sum_i exp(beta * overlap_i))
```

Energy monotonically decreases on each iteration; convergence is guaranteed.

**Performance:**

- Basin of attraction: 96.9% recall at k=64, N=100K stored atoms
- Convergence: 1-3 iterations typical
- Sparse overlap computation: O(active_count) per stored vector via sorted merge

---

## 6. IndexedMemory: Shared Backbone

Both `AtomMemory` and `CompositeMemory` are thin wrappers around `IndexedMemory`.
This eliminates code duplication for:

- **Posting lists** (`PostingShard`): inverted index from dimension index to vector IDs.
  Sorted by vector ID within each shard. Enables fast overlap computation.
- **IDF weights** (`IdfWeights`): inverse document frequency per dimension index.
  Rare indices contribute more to overlap scores. Clipped at 3x median.
- **Tombstones** (`TombstoneMap`): bitset of deleted vector IDs. Excluded from
  all scans and overlap computations. Compacted lazily.
- **Overlap scan**: single implementation used by both atom cleanup and
  composite structural query.

`IndexedMemory` stores vectors in a `Vec<(String, EntangledHVec)>` with
FxHashMap<String, usize> for ID lookup. The posting lists and IDF are
rebuilt on load (NOT persisted) since they are derivable from the vectors.

---

## 7. Text Decomposition Rules

The decomposer extracts (subject, relation, object) triples from English text
using rule-based pattern matching. Precision over recall: it is better to
extract fewer correct triples than many incorrect ones.

### Vocabulary

**Prepositions (40):**
about, above, across, after, against, along, among, around, at, before,
behind, below, beneath, beside, between, beyond, by, despite, down, during,
except, for, from, in, inside, into, near, of, off, on, onto, out,
outside, over, past, through, to, toward, under, until, up, upon, with, without

**Verbs (30):**
is, are, was, were, has, have, had, does, do, did, can, could, may, might,
shall, should, will, would, must, makes, causes, enables, prevents, allows,
requires, includes, contains, produces, creates, supports

### NP Grammar

```
NP = det? adj* noun+
```

Where:
- `det` = a, an, the, this, that, these, those, some, any, every, each, no
- `adj` = word preceding a noun (heuristic: not in verb/prep/det lists)
- `noun+` = one or more consecutive nouns (compound nouns)

### Extraction Patterns

| Pattern                    | Triple                           |
|----------------------------|----------------------------------|
| NP + verb + NP             | (NP1, verb, NP2)                |
| NP + verb + prep + NP      | (NP1, verb_prep, NP2)           |
| NP + is + NP               | (NP1, is_a, NP2)                |
| NP + is + adj              | (NP1, has_property, adj)         |

**Scope:** English only. No coreference resolution, no clause embedding,
no passive voice transformation. These can be added in future phases.

---

## 8. Persistence

Each data type uses a distinct magic byte as the first byte of its
serialized frame (before the PersistentArena's CRC32+LZ4 envelope):

| Magic  | Data Type            | Format                                      |
|--------|----------------------|---------------------------------------------|
| `0xFD` | Atom                 | [magic][id_len:u16][id_str][dim:u32][deltas] |
| `0xFC` | Composite            | [magic][id_len:u16][id_str][dim:u32][deltas] |
| `0xFB` | Triple               | [magic][subj_id_len:u16][subj][rel_id_len:u16][rel][obj_id_len:u16][obj][composite_id_len:u16][comp] |
| `0xFE` | Rule / Relation      | Existing relation format (reused from graph.rs) |

**Posting lists are NOT persisted.** They are rebuilt on load by scanning
the atom/composite vectors. This keeps the persistence format simple and
avoids consistency issues between posting lists and vectors.

**Load sequence:**
1. Scan arena frames from offset 0
2. Dispatch on magic byte: atoms to AtomMemory, composites to CompositeMemory,
   triples to TripleStore, rules to RuleStore
3. Rebuild posting lists and IDF weights from loaded vectors

---

## 9. Production Safety

### IDF Clipping (3x Median)

Without clipping, a rare index that appears in only 1 vector gets
astronomically high IDF weight, allowing a single poisoned dimension
to dominate overlap scores.

**Fix:** After computing all IDF weights, find the median weight and
clip all weights to `min(weight, 3.0 * median)`. This bounds the
influence of any single dimension to 3x the typical dimension.

### Fan-Out Admission Control

A broad query might match thousands of composites. Unbinding each one
against AtomMemory would be O(matches * atoms), which is unbounded.

**Fix:** `AdmissionControl` counts the number of composites exceeding
the overlap threshold. If `fan_out > 40`, the algebraic path is
abandoned and the query falls back to materialized TripleStore lookup.
The threshold 40 is chosen so that the algebraic path stays under 50ms.

### Ascending Lock Order

All RwLocks are acquired in a fixed order to prevent deadlocks:

```
AtomMemory -> CompositeMemory -> TripleStore -> RuleStore
```

Within `IndexedMemory`: vectors lock -> posting lock -> idf lock.

The existing `HmsCore` lock ordering (`ShardSet -> Shard.vectors -> Shard.ivf -> Shard.nsg`)
is independent and does not interact with meaning memory locks.

### LSM Write Path

Writes go to the PersistentArena append log. The arena handles
CRC32 integrity, LZ4 compression, and segment rotation. Compaction
is triggered explicitly via `compact()`. No WAL; version counter
resets on restart (acceptable for meaning memory since posting
lists are rebuilt anyway).

---

## 10. Phase 2: Cognition Layer

A background loop that discovers implicit knowledge from stored triples.

**Components:**
- **PatternScanner**: finds recurring (?, R, ?) patterns across triples.
  Groups triples by relation, counts co-occurring subject/object atoms.
- **Abstraction**: when N entities share the same relation pattern,
  bundles their atoms (using `EntangledHVec::bundle`, NOT XOR) to create
  a prototype/category atom.
- **Gaps**: compares an entity's relation profile against peer profiles
  to detect missing relations (e.g., "most cities have a country, but
  city X doesn't").
- **Hypothesis**: bundles peer fillers for a detected gap to predict
  the most likely filler, then runs Hopfield cleanup to find the
  nearest stored atom.
- **Analogy**: finds structural isomorphisms between subgraphs.
  Connected components + greedy bipartite mapping by relation overlap.

**Threading model:** The cognition loop runs in a background thread.
It acquires READ-ONLY locks on AtomMemory, CompositeMemory, TripleStore.
Discovered insights (patterns, abstractions, hypotheses, analogies) are
collected into a separate `Vec<Insight>` that does not require write
locks on the main stores. Insights can be promoted to stored knowledge
by an explicit `promote_insights()` call that acquires write locks.

---

## 11. Phase 3: Agency Layer

Gives the system goal-directed behavior.

- **GoalStore**: separate from AtomMemory. Stores goal vectors with
  utility scores. Goals are hypervectors composed from desired state
  descriptions. Utility function: `u(goal) = relevance * urgency * (1 - cost)`.
- **Planner**: backward-chaining over causal triples. Starting from a
  goal state, finds triples where the object matches the goal and
  recursively plans to achieve the subject. Produces a plan as an
  ordered list of actions (triples to make true).
- **Questions**: template-based question generation. Given a gap or
  hypothesis, generates natural language questions. Ranked by
  goal-relevance (overlap between question vector and active goals).
- **Self-Modification**: the system can propose changes to its own
  rules, thresholds, or decomposition patterns. ALL proposals require
  explicit approval (no autonomous self-modification). Proposals are
  logged in the audit trail.

---

## 12. Key Invariants

1. **Sorted posting lists**: all posting shards maintain sorted vector ID
   order. Binary search and merge operations depend on this.

2. **Sorted indices**: all `EntangledHVec.indices` are sorted and deduplicated.
   Every constructor and operation maintains this invariant.

3. **Tombstone exclusion**: deleted vectors are excluded from ALL scan
   operations (overlap, cleanup, query). No tombstoned vector ever
   appears in results.

4. **Lock ordering**: AtomMemory -> CompositeMemory -> TripleStore -> RuleStore.
   Within IndexedMemory: vectors -> postings -> idf. Violation causes deadlock.

5. **Atomic writes**: each arena write is a single CRC32-framed entry.
   Partial writes are detected by CRC mismatch and skipped on recovery.

6. **meaning.enabled=false safety**: when the meaning config is disabled,
   no meaning memory structs are allocated, no new code paths execute,
   and all existing tests pass with identical behavior.

---

## 13. Performance Budgets

| Operation         | Budget   | Notes                                    |
|-------------------|----------|------------------------------------------|
| Atom insert       | < 1 us   | Hash + sorted insert into posting list   |
| Hopfield cleanup  | < 5 ms   | 1-3 iterations, k=64, sparse overlaps    |
| Overlap scan      | < 5 ms   | Posting list intersection, IDF-weighted  |
| Structural query  | < 50 ms  | Overlap scan + fan-out unbind + cleanup  |
| Multi-hop query   | < 50 ms  | Rule rewrite or chained TripleStore lookups |
| Full reconstruction | < 5 s  | Rebuild posting lists + IDF on load      |
| Cognition cycle   | < 1 s    | Pattern scan + abstraction + gaps        |
