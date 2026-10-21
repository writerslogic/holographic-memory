# Key Decisions from Multi-Model Debates

Reference material. Read if you need to understand WHY a design decision was made.

Full transcripts: conductor/debate_output/ and conductor/debate_phase1/ and conductor/debate_phase2/

## Decisions That Shaped the Architecture

1. Dense FHRR layer was killed. Sparse BSC stays as only representation.
2. Superposition KV plates were killed. Threshold-bundling breaks XOR-unbind.
3. XOR commutativity is a correctness bug. Fixed with cyclic-shift role binding.
4. The attractor IS the holographic property. Basin width = 96.9% erasure tolerance.
5. father XOR father = 0 (self-annihilation). Composition must be stored as rule atoms.
6. Self-supervised embeddings deferred. 350M token corpus floor too high.
7. Multi-hop is NOT frontier bundling. Rules collapse known chains; TripleStore handles the rest.
8. Per-composite unbind, NOT bundle-then-unbind.
9. QPS at 10M atoms: 6-10K (bandwidth-bound with sorted-merge).
10. 55 crafted composites can hijack queries. Defense: Proportional IDF Clipping.
11. Meaning is structure, not similarity.
