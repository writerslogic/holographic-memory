# HMS Elevation Prompt: From Semantic Search to Invertible Compositional Memory

## Source
Synthesized from a 5-round, 36-turn debate across Claude Opus 4.8, DeepSeek V4-Pro, MiMo V2.5-Pro, Step 3.7 Flash, MiniMax M3, and Qwen 3.7 Max. Full transcript: `conductor/debate_output/full_transcript.txt`.

## What Survived the Debate (Consensus After 5 Rounds of Demolition)

The debate **killed** 60% of proposed features through honest scrutiny:
- DEAD: Dense FHRR layer + hand-rolled FFT (complexity, synchronization bugs)
- DEAD: Superposition KV plates (threshold-bundling breaks XOR-unbind at useful N)
- DEAD: "Algebraic multi-hop independent of branching" (frontier capacity wall)
- DEAD: Auto-discovery of relation compositions (O(R²K), research project not DB feature)
- DEAD: Parity bundling (mathematically incompatible with sparsity at scale)

What survived is smaller, buildable, and genuinely unprecedented.

## The Irreducible Capability: Fuzzy Structural Query

**The one thing no other system can do:** Given a partial, noisy structural pattern (some roles specified with corrupted vectors, one role unknown), return the ranked set of entities filling the unknown role — tolerating noise in specified roles AND inverting on any role from one index.

- Neo4j needs exact match keys; cannot tolerate noise in query subjects
- Pinecone can't bind/unbind roles; needs separate index per query shape
- SQL needs an index per query shape; cannot invert

HMS does it from ONE index via algebraic composition + attractor cleanup.

## Architecture: Three Components

### Component 1: AtomMemory (Sharded Inverted-Index Modern Hopfield Attractor)

Stores only atomic entity and relation vectors. Never composites.

```rust
pub struct AtomMemory {
    shards: [PostingShard; 64],          // 64 shards × 256 dims each = 16384
    arena: Vec<EntangledHVec>,           // lock-free append-only
    live: Vec<bool>,                     // tombstone bitmap (or RoaringBitmap)
    beta: f32,                           // inverse temperature (~24.0)
    k_target: usize,                     // 64 (rho * D)
}

struct PostingShard {
    lists: parking_lot::RwLock<Vec<Vec<u32>>>,  // len 256 per shard
}
```

**Cleanup algorithm (k-sparse Modern Hopfield):**
1. `overlaps[pid] = Σ_{d∈query} posting[d].contains(pid)` — inverted index scan
2. `attention = softmax(beta * overlaps)` over live atoms
3. `dim_score[d] = Σ_i attention[i] * indicator(d ∈ arena[i])` — weighted superposition
4. `new_query = top_k(dim_score, k_target=64)` — project to k-sparse
5. Repeat until convergence or max_iters (1-3 typically)

**Energy function:** `E = -lse(β, overlaps)` over finite k-sparse state space → monotone descent → guaranteed convergence (Ramsauer et al. 2020, Theorem 4 adapted to sparse overlap metric).

**Set retrieval (multi-extract):** Single-pass overlap scan, return ALL atoms above threshold. No iterative peeling (peeling is mathematically invalid on threshold-bundles).

### Component 2: CompositeMemory + RoleAlgebra

Stores canonical triple composites using permutation-based role binding.

**CRITICAL FIX: XOR commutativity bug.** Current `h ⊕ r ⊕ t` cannot distinguish subject from object ("John loves Mary" = "Mary loves John"). Fix: cyclic-shift role binding with odd shifts coprime to D=16384:

```
Triple T = Subject ⊕ ρ₁(Relation) ⊕ ρ₂(Object)
```

where `ρ_n` is cyclic shift by n positions, and shifts are chosen as odd numbers coprime with 16384 (e.g., shift_subject=0, shift_relation=1, shift_object=3).

Unbinding recovers any role:
```
Object_noisy = ρ₂⁻¹(T ⊕ Subject ⊕ ρ₁(Relation))
Object_clean = AtomMemory.cleanup(Object_noisy)
```

**Data structure:** Same sharded inverted-index pattern as AtomMemory, but storing composite vectors (|T| ≈ 192 active bits).

**Capacity:** Global N bounded by scan time (~1M composites per segment at 5ms budget). Per-query fan-out bounded by set-retrieval capacity (~40 matches on algebraic path).

### Component 3: TripleStore (Materialized Fallback)

Plain `(subject_id, relation_id, object_id)` columnar store for:
- Exact forward lookups (bypass attractor entirely)
- Fan-out > 40 fallback (materialized path, never worse than Neo4j)
- Persistence and crash recovery (WAL replays atoms + triples, rebuilds indices)

### The Query Pipeline: `fuzzy_structural_query`

```
fuzzy_structural_query(known: [(Role, NoisyVec)], target: Role) -> Vec<(EntityId, f32)>:

  1. Build query vector: Q = ⊕_{(role, vec) in known} ρ_{shift(role)}(vec)       // O(k·|known|)
  2. Overlap scan CompositeMemory: overlaps[pid] = Σ_{d∈Q} posting[d]            // O(|Q|·L̄)
  3. Threshold to match set: matches = { pid : overlaps[pid] > signal_floor }
  4. BRANCH on fan-out:
     if |matches| ≤ 40:
       // ALGEBRAIC PATH (the unprecedented capability)
       For each matched composite individually:
         residual = ρ_{target}⁻¹(composite ⊕ Q)    // unbind known roles
         cleaned = AtomMemory.cleanup(residual)     // snap to stored atom
         accumulate (entity_id, confidence)
       Aggregate per-entity: conf = 1 - (1 - mean_conf)^support_count
       Return results above confidence threshold
     else:
       // MATERIALIZED PATH (honest fallback)
       Return TripleStore.project(matches, target)
```

**Why per-composite unbind, NOT bundle-then-unbind:** Bundling the match set and unbinding once was proven invalid in Round 2. Bundle-threshold destroys the XOR structure. Per-composite unbind preserves exact algebra and sends each residual through cleanup independently.

## Embedding When Collapsing (The C_max Breakthrough)

For the algebraic path's set-retrieval step, when multiple matches produce a superposition, use **attention-weighted, relation-conditioned collapse:**

```
collapse(frontier_residuals, target_relation) =
    Σ_i softmax(β * overlap(residual_i, target_relation)) * residual_i
```

This focuses the bundle on residuals most relevant to the target role. Irrelevant residuals get near-zero weight and don't pollute. Effective frontier size becomes proportional to query selectivity, not total fan-out.

For continuous-valued processing: keep raw counts in `[u16; 16384]` accumulator (fits L1 cache at 32KB). Do NOT threshold to binary. Feed continuous counts directly to the Hopfield attractor. This preserves magnitude information and pushes effective capacity from ~100 (binary threshold) to ~1000+ (continuous).

## Tests That PROVE It Works (Or Expose Where It Breaks)

### Test 1: Attractor Convergence Under Noise
Insert 100K atoms. Query with 0/10/25% bit-flip noise (deterministic corruption function that flips random active bits while preserving k=64 sparsity). Assert: recall ≥ 0.95 at 25% noise. Assert: convergence in ≤ 3 iterations.

### Test 2: Fuzzy Structural Query Across Fan-Out Boundary
Build graph with controlled fan-outs {1, 5, 10, 20, 30, 40, 50, 80} plus 50K noise triples. Query with 25%-corrupted subject vectors. Assert:
- Algebraic path (fan-out ≤ 40): recall ≥ 0.95, precision ≥ 0.95
- Aggregated noise-hit confidence never exceeds 0.5
- Materialized path activates cleanly for fan-out > 40

### Test 3: Graceful Degradation Curve
Store 1000 atoms. For each, measure cleanup recall at 0/10/20/.../90% bit corruption. Assert: the curve is smooth and monotonically decreasing. Assert: at 50% corruption, recall > 0.5. Assert: no cliff edge (max drop between adjacent points < 0.3).

### Test 4: Role Inversion (Same Structure, Different Queries)
Store 100 triples. For each, verify that querying for subject, relation, AND object all succeed from the SAME stored composite. This proves the invertibility claim.

### Test 5: Noise Tolerance vs Exact Match
Query both noisy-algebraic path and exact-TripleStore path. Assert: algebraic path returns same entity set as exact path for fan-out ≤ 40 and noise ≤ 25%.

### Test 6: Concurrent Read/Write
Spawn 8 writer threads (10K inserts each) + 4 reader threads (10K queries each). Assert: zero deadlocks, zero corruption, p99 read latency < 5ms under maximum write load.

### Test 7: Cold-Page-Fault Performance
Insert 10M triples, compact to disk, evict page cache. Run 1000 cold queries. Assert: p50 < 0.5ms, p99 < 2ms.

## What This Makes HMS

Not "holographic" in the physics sense (the debate killed that honestly). Instead:

**An associative memory with an invertible compositional query algebra** — the first system where:
1. Queries tolerate noise (not just approximate similarity, but corrupted structural patterns)
2. One stored structure answers multiple query shapes (role inversion)
3. Relation composition is algebraically expressible (`Q = S ⊕ ρ₁(father) ⊕ ρ₁(father)` retrieves grandfathers)
4. Privacy is information-theoretic (superposition is NP-hard subset-sum to invert without keys)
5. The attractor provides exact reconstruction from noisy input (provable convergence)

No vector database, graph database, or relational database can express these operations. HMS is not a better Pinecone or a better Neo4j. It is a **new category**: a memory system with algebraic query closure over an attractor-cleaned compositional vector space.

## Implementation Priority

1. **AtomMemory** — the sharded inverted-index Modern Hopfield attractor (the foundation everything depends on)
2. **RoleAlgebra** — permutation-based role binding with odd-coprime shifts (fixes the commutativity bug)
3. **CompositeMemory** — second inverted index for canonical triples
4. **fuzzy_structural_query** — the pipeline connecting all three components
5. **TripleStore** — materialized fallback for high-fan-out
6. **Tests 1-7** — prove every claim or expose exactly where it fails

## Constraints
- Rust, no new external crates (build on existing EntangledHVec primitives)
- Must not break existing 133 tests or public API
- Additive: new modules alongside existing code
- Performance: cleanup < 5ms for 100K atoms, structural query < 50ms
- Concurrency: 64 sharded RwLocks with strict ascending lock ordering
