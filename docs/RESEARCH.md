# Research: Neuro-Symbolic Reasoning with Resonator Networks

This document details the mathematical foundation of the "Resonator Network" implemented in WARM, which allows the system to solve symbolic factorization problems.

## The Inverse Product Problem

Vector Symbolic Architectures (VSA) are excellent at **Composition**:
$$ C = A \odot B \odot C $$
(e.g., "Red Square North" = Red * Square * North)

However, the inverse problem is hard:
> Given $C$ (the composite) and dictionaries $D_{color}, D_{shape}, D_{loc}$, find the factors $a, b, c$.

Standard search is $O(|D|^3)$ (brute force). 

## The Resonator Solution

The Resonator Network (Kanerva, 2022) solves this iteratively in $O(Iter 	imes N)$. It works by maintaining a **superposition estimate** for each factor and refining it against the dictionaries.

### Algorithm Dynamics

1.  **Initialization**: Set initial estimates $\hat{b}(0)$ and $\hat{c}(0)$ to the superposition (sum) of all vectors in their respective domains.
    $$ \hat{b}(0) = \sum_{v \in D_B} v $$

2.  **Iterative Update**:
    To find factor $\hat{a}$, we "unbind" the current guesses of $B$ and $C$ from the target $S$:
    $$ \hat{a}(t+1) = S \odot \hat{b}(t)^{-1} \odot \hat{c}(t)^{-1} $$
    
    *Note: In Binary Spatter Code (XOR), the inverse is the vector itself ($x^{-1} = x$).*

3.  **Cleanup (Projection)**:
    The resulting $\hat{a}$ is noisy. We "clean it up" by finding the nearest neighbor in the codebook $D_A$:
    $$ \hat{a}_{clean}(t+1) = 	ext{argmax}_{v \in D_A} (	ext{Sim}(\hat{a}(t+1), v)) $$

4.  **Repeat**:
    We use the clean $\hat{a}$ to update our guess for $\hat{b}$, then $\hat{c}$, and loop until convergence.

### Convergence

The system behaves like a hopfield network or a coupled oscillator. It typically converges within 10-50 iterations. If it fails to converge (oscillates), it indicates either high noise or multiple valid factorizations (ambiguity).

## Applications

This capability transforms WARM from a passive storage system into an active solver for:
*   **Visual Scene Decomposition**: Factorizing a scene vector into (Object, Attribute, Location).
*   **Language Parsing**: Unpacking sentence structures (Subject, Verb, Object).
*   **Analogical Mapping**: Solving $A:B :: C:?$.

## SOTA Retrieval: Sparse-Native Inverted Indexing

As of version 0.2.0, WARM has transitioned its primary retrieval engine from standard ANN techniques (IVF/PQ) to a **Sparse-Native Inverted Index** optimized for the specific sparsity observed in VSA hypervectors.

### The Problem with Dense ANN (PQ/HNSW) for VSAs
Standard Approximate Nearest Neighbor (ANN) algorithms like Product Quantization (PQ) are designed for dense embeddings. When applied to ultra-sparse hypervectors ($\rho \approx 1/256$), they introduce significant quantization noise and return scores on incompatible scales, requiring expensive re-ranking.

### The Sparse-Native Solution
For hypervectors with fixed sparsity $m = D/256$, retrieval is mathematically equivalent to finding the maximum intersection of support sets in an inverted index.

1.  **Index Structure**: An inverted list mapping each dimension $d \in [0, D)$ to a sorted list of document IDs that are active in that dimension.
2.  **Epoch-Stamped Accumulator**: To avoid $O(N)$ zeroing of scores for each query, the system uses an epoch-stamped array. This allows sub-millisecond accumulation of intersection counts across relevant posting lists.
3.  **WAND / Block-Max WAND Pruning**: By sorting query terms by document frequency and maintaining upper bounds on potential intersection scores for blocks of the inverted index, the system can prune up to 90% of candidate evaluations while guaranteeing Top-K correctness.

### Impact on Factorization
The Sparse-Native engine provides a "candidate shortlist" (e.g., Top-1000) for unbinding tasks. This allows the Resonator Network to operate on a constrained hypothesis space, reducing the unbinding problem from a global search to a local ranking task, significantly improving both latency and convergence stability.

## Deterministic Quantized Substrate (validated)

The resonator dynamics above run on WARM's quantized-phase substrate (`core::PhaseHVec`), where each dimension is a discrete phase index in $\mathbb{Z}_N$ and binding is phase-addition mod $N$. The stored state is therefore an exact, replay-verifiable integer fold of its inputs, not a lossy float embedding.

A pre-registered experiment (`docs/PREREGISTRATION-binding-readout.md` §20) compared quantized and continuous-phase factorization. The reported accuracy distributions overlap down to 4 bits/dimension across the tested operating range. A 30-seed follow-up spans 4- and 3-bit resolution and $D \in \{512, 1024, 2048\}$; 2-bit shows a deficit near the capacity knee. Overlapping distributions are evidence for these workloads, not a universal equivalence result.

This pairs the published qFHRR substrate (arXiv 2604.25939) with the Frady/Kent resonator network (Frady, Kent, Olshausen & Sommer, *Neural Computation* 32(12), 2020). The stored phases are integers; the factorization uses floating-point arithmetic. Public phase vectors use `u16` per coordinate. Expanded resonator codebooks use two `f64` values per coordinate; the optional packed backend uses 4-, 8-, or 16-bit indices plus its lookup table and floating working state. Total allocated memory must include those costs.

* **Numbers, method, chance floor, and exact reproduction command:** `docs/DETERMINISTIC-RESONATOR.md`.
* **API:** `core::PhaseResonator` (reusable index over fixed factor codebooks) and `core::phase_resonator_factorize` (one-shot), over `core::PhaseHVec`. These APIs are available with default features; `PhaseGraph` requires `experimental`.
* **Reproduce the sweeps:** `cargo run --release --bin resonator-factorize` (the §20 table) and `cargo run --release --bin resonator-sweep` (the phase-bit × dimension hardening).

## Supplied-paper implementation review (2026-10-24)

The implementation work separates software algorithms from physical holographic
hardware. Document retrieval uses lexical and dense retrieval; improvements to
experimental phase operations do not establish a document-search or LongMemEval
score improvement.

| Paper | Applicable mechanism and limit |
|---|---|
| [Holographic One-Point Functions and Geodesics in SdS3](https://arxiv.org/abs/2603.21425) | A gravitational observable encodes geometric information under a specific model. It supplies no general-purpose memory compression or retrieval algorithm. |
| [Building Holographic Entanglement by Measurement](https://arxiv.org/abs/2601.21065) | Structured graph interactions and conditional boundary reduction are useful analogies. The constant-time preparation claim concerns a quantum protocol, not a CPU graph query. |
| [Holographic Banners](https://arxiv.org/abs/2604.02514) | A semiclassical construction relating boundary and interior descriptions; no directly implementable associative-memory shortcut was identified. |
| [Holographic Knowledge Manifolds](https://arxiv.org/abs/2509.10518) | Withdrawn by its author for significant errors. Its accuracy, speed, and forgetting claims are excluded from implementation evidence. |
| [WARM for Zero-Shot Compositional Reasoning in Knowledge Graphs](https://arxiv.org/abs/2606.24948) | An overloaded global bundle can lose second-hop facts even with the correct intermediate entity. This motivates explicit directionality and per-hop accuracy checks; its results do not prove a universal capacity bound. |
| [Adapting A Vector-Symbolic Memory for Lisp ACT-R](https://arxiv.org/abs/2508.15630) | A partial cue can recover an episode's associated slots. WARM applies this to exact source dependency closure, with budget limits and explicit unresolved links. A separate phase sidecar must beat exact episode indexing before adoption. |
| [Dense Holographic Associative Memories](https://arxiv.org/abs/2606.18492) | The optical architecture realizes an attention-style associative readout. WARM already has softmax/sparse cleanup; optical material efficiency does not establish CPU speed or eliminate stored codebooks. |
| [High-Capacity Angularly-Multiplexed WARM Operating at Single Photon Level](https://arxiv.org/abs/1604.06049) | Independently addressed channels reduce cross-talk while consuming physical degrees of freedom. Software partitions must likewise count their storage and routing cost. |
| [Divide-Conquer-and-Merge: Memory- and Time-Efficient Holographic Displays](https://arxiv.org/abs/2404.10777) | Partitioned computation motivates bounded working state and streamed codebook access. Its CNN hologram-synthesis measurements are not transferable software speedup estimates. |
| [Deep-learning-based data page classification for holographic memory](https://arxiv.org/abs/1707.00684) | Robustness to optical page corruption is task-specific. WARM checks corrupted-query parity; introducing a trained image classifier would not solve symbolic factorization. |

The resulting changes optimize exact phase arithmetic, offer packed resonator
codebooks, preserve directed phase-graph roles, and enrich conversation context
from canonical source records. The graph originally used the same phase code for
subject and object, making every asserted edge indistinguishable from its reverse.
Separate role domains remove that structural ambiguity; query-time cue factoring
also removes repeated string hashing and repeated known-role work.

Evidence and reproduction commands:

- [Phase graph](../benchmarks/results/holography_phase_graph_v1/README.md): baseline
  failures, directed chains, per-hop retrieval, timing, and replay checks.
- [Phase arithmetic](../benchmarks/results/holography_phase_substrate_v1/README.md):
  exact differential checks, cold lookup costs, and warm kernel timings.
- [Resonator](../benchmarks/results/holography_resonator_v1/README.md): packed and
  expanded memory, output parity, corrupted queries, and latency tradeoffs.
- [Conversation context](../benchmarks/results/holographic_context_v1/README.md):
  source closure, budget boundaries, and the exact-index versus phase-sidecar screen.
- [Capacity audit](../benchmarks/results/holography_capacity_v1/README.md): coupled,
  dense, and independently sharded Gaussian designs with full decoder accounting.

The capacity audit reproduces a useful coupled-code advantage on three synthetic
seeds, including modest additive noise. Its dense design matrix occupies about
225 MB for a 9,984-byte signal, so those recovery results do not establish a
deployable memory-compression improvement. Prior negative
[phase-routing](../benchmarks/results/holographic_router_synthetic_v1/README.md)
and [late-interaction](../benchmarks/research/hli/RUN.md) results remain in force;
renaming their phase operations does not overcome their measured budget limits.
