# What "holographic" means here

In a hologram every part of the plate carries a degraded copy of the whole image. A
**hypervector** memory works the same way: many items are superposed into one high-dimensional
vector, no single coordinate stores any one item, and damaging the vector degrades every item a
little instead of erasing a few. That is the sense in which HMS is holographic. It is a property
of the vector-symbolic layer, and this page says exactly which parts of the engine have it.

## The vectors

An `EntangledHVec` is a sparse binary vector of `D` dimensions. By default about `D/256` bits are
active (`DEFAULT_RHO_DENOM = 256` in `src/core/entangled.rs`), stored as a sorted list of active
indices. At `D = 16,384` that is 64 indices, 256 bytes, against 64 KiB for a dense `f32` vector.
Vectors for atoms (names, relations, words) are generated deterministically from a hash seed, so
the same atom has the same vector in every process and no codebook has to be stored.

## The three operations

| Operation | On sparse index sets | What it is for |
|-----------|----------------------|----------------|
| **Bind** (`bind`) | symmetric difference (XOR) of the active sets | Tie a role to a filler. It is its own inverse: binding the result with the role recovers a noisy copy of the filler. |
| **Bundle** (`bundle`) | per-index count, keep indices above a threshold | Superpose many items into one vector that stays similar to each of them. |
| **Permute** (`permute`) | cyclic shift of every index | Encode order. `bind_ordered(a, b)` differs from `bind_ordered(b, a)`. |

Similarity is Jaccard overlap of the active sets. Bind keeps the result dissimilar to both
inputs; bundle keeps it similar to every input, which is the superposition.

`bundle_bloom` is a union rather than a thresholded count (a Bloom filter over the items). It is
what the capacity sweep in [Capacity limits](capacity.md) measures.

## Where the algebra is used

- **Meaning memory** (`MeaningConfig::enabled`): a triple `(subject, relation, object)` is stored
  as one composite vector, with the roles separated by cyclic shifts and combined by bind. A
  query binds the known parts to isolate the unknown one, and Hopfield cleanup (sparse softmax
  attention over the stored atoms) snaps the noisy residue back to the nearest stored atom.
  `multi_hop` chains this over several relations.
- **Sequences and analogies**: positional binding by permutation, and relation mapping between
  domains.
- **Resonator network**: factors a bound product back into its parts by iterating cleanup.
- **Sparse vector index**: `memorize_vector` / `query_vector` store sparse codes of dense
  embeddings. It is an experimental path and not a competitive nearest-neighbour index; see the
  [public benchmarks](../PUBLIC-BENCHMARKS.md) for the measured recall.

## Where it is not used

The **document pipeline does not rank with hypervectors**. A document is split into chunks;
each chunk keeps its stemmed term counts, its text, and optionally the dense embeddings you
supply. A query is scored by BM25 over the terms and by exact cosine over the embeddings, then
the two lists are fused (a min-max normalised blend, or reciprocal rank fusion). Each chunk also
carries a sparse hypervector from the text encoder, serialised with the chunk, but search does not
read it. To check this against the code: `src/core/documents/search.rs` has no use of
`chunk.vector`.

So `search_documents` is a conventional hybrid retriever with a metadata filter, byte offsets
and atomic replace. Choose it for that, not for holographic behaviour.

## Graceful degradation, measured

Corrupting a stored vector should degrade retrieval smoothly. The committed sweep
[`benchmark_scaling_results.json`](https://github.com/writerslogic/holographic-memory/blob/main/benchmarks/results/benchmark_scaling_results.json)
holds a codebook of 100 stored vectors, replaces 0 to 90 percent (steps of 10) of a probe's active
indices with random ones, and checks whether the original vector is the nearest neighbour
(Jaccard) or the Hopfield cleanup output. Both were correct for all 30 probes at every corruption
level tested, at each of the nine dimension and density settings in the file. That is one
codebook size (100), 30 probes, and uniform random corruption of a single stored vector. It says
nothing about a codebook of 10,000, about a vector that is itself a superposition, or about
adversarial noise; for superpositions see [Capacity limits](capacity.md).
