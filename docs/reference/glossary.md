---
description: The terms WARM uses, defined as this code uses them, with a pointer to where each is explained or measured.
---

# Glossary

Terms are defined as this code and these docs use them. Where a term is measured, the entry
points at the page and the committed results file.

## Vector-symbolic memory

Hypervector
:   A very long vector, thousands of dimensions, used as a code for a thing. Unrelated
    hypervectors are nearly orthogonal, so random ones almost never collide. WARM uses sparse
    binary hypervectors (`EntangledHVec`): `D` dimensions with about `D/256` bits set. See
    [What "holographic" means here](../concepts/holographic.md).

Atom
:   The hypervector for a name, relation or word. Generated deterministically from a hash seed,
    so the same atom is the same vector in every process and no codebook is stored.

Bind
:   Combine two hypervectors into one that resembles neither, used to tie a role to a filler.
    On sparse index sets it is the symmetric difference (XOR). Binding the result with the role
    again recovers a noisy copy of the filler.

Bundle
:   Superpose many hypervectors into one that stays similar to each of them, by counting per
    index and keeping the indices above a threshold. The bundle cannot list its members; it can
    only be asked whether a given item resembles it.

Permute
:   A cyclic shift of every index. It encodes order: `bind_ordered(a, b)` differs from
    `bind_ordered(b, a)`.

Superposition
:   Several items stored in the same vector at once. Capacity is finite; see
    [Capacity limits](../concepts/capacity.md) for the tested range.

Resonance
:   Recalling an item by similarity: unbind with a key, then find the stored vector the result
    most resembles. The interactive demo on the [home page](../index.md) shows it with
    dense binary vectors.

Cleanup
:   Snapping a noisy vector back to the nearest stored atom. WARM uses a modern Hopfield network
    (sparse softmax attention over the stored atoms) for it.

Resonator network
:   A network that factors a bound product back into its parts by iterating cleanup on each
    factor in turn.

Meaning memory
:   The optional layer that stores facts as composite vectors and answers relational and
    multi-hop queries. It is where the algebra is used; document search is not.

Triple
:   A `(subject, relation, object)` fact. The triple store keeps an exact symbolic index next to
    the composite vectors.

Superposition floor
:   The ratio of stored pairs `M` to dimensions `D` beyond which blind recovery from a random-code
    superposition fails; measured near `M/D = 0.27`. It is a property of the random code, not an
    information-theoretic wall. See [Superposition floor](../SUPERPOSITION-FLOOR.md).

## Document search

Chunk
:   A passage of a document, 256 words by default. A chunk keeps its text, stemmed term counts,
    optional embeddings you supply, and its byte range in the source.

Byte-offset hit
:   A search result that carries the passage text and the start and end byte of that passage in
    the source document, so a UI can highlight it.

BM25
:   The lexical ranking function WARM uses over stemmed terms (Porter/Snowball English).

Exact cosine
:   Dense similarity computed against every stored embedding, with no approximation.

Hybrid search
:   BM25 and exact cosine run together and fused, by a min-max normalised blend by default or by
    reciprocal rank fusion (`fusion: "rrf"`).

RRF (reciprocal rank fusion)
:   Combining ranked lists by summing the inverse of each item's rank in each list.

## Storage

Store
:   A directory holding one engine's data. It persists there and reopens with the same
    arguments.

Arena
:   The memory-mapped segmented log that holds vector and document frames. Compaction writes a
    new generation and swaps it in atomically. See
    [Production readiness](../production-readiness.md).

Tenant
:   One isolated store inside `warm-server`. A client names its tenant on every request.

## Measurement

nDCG@10
:   Normalised discounted cumulative gain over the top 10 results: a rank-aware relevance score
    from 0 to 1.

Recall@k
:   The share of the true top-k neighbours (or relevant documents) that appear in the first `k`
    results.

QPS
:   Queries per second. WARM benchmarks report single-threaded QPS, one query at a time.

ANN
:   Approximate nearest neighbour search: a faster index that may miss true neighbours.

HNSW
:   Hierarchical navigable small-world graph, the ANN index used by FAISS, hnswlib and Lucene,
    the comparison baselines.

VGraph
:   WARM's quantized graph index in `core::qgraph::vertex`: a Vamana-style graph with one
    low-bit code per vertex and a residual code for the re-rank. See the
    [benchmarks](../PUBLIC-BENCHMARKS.md).

Load gate
:   The rule that a timing run counts only if the machine's one-minute load average is below a
    set limit. Runs above it are reported but marked indicative.

Dev split
:   The frozen set of questions used for tuning. Held-out questions are not touched until a
    final single run.
