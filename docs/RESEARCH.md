# Research: Neuro-Symbolic Reasoning with Resonator Networks

This document details the mathematical foundation of the "Resonator Network" implemented in HMS, which allows the system to solve symbolic factorization problems.

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

This capability transforms HMS from a passive storage system into an active solver for:
*   **Visual Scene Decomposition**: Factorizing a scene vector into (Object, Attribute, Location).
*   **Language Parsing**: Unpacking sentence structures (Subject, Verb, Object).
*   **Analogical Mapping**: Solving $A:B :: C:?$.

## SOTA Retrieval: Sparse-Native Inverted Indexing

As of version 0.2.0, HMS has transitioned its primary retrieval engine from standard ANN techniques (IVF/PQ) to a **Sparse-Native Inverted Index** optimized for the specific sparsity observed in VSA hypervectors.

### The Problem with Dense ANN (PQ/HNSW) for VSAs
Standard Approximate Nearest Neighbor (ANN) algorithms like Product Quantization (PQ) are designed for dense embeddings. When applied to ultra-sparse hypervectors ($\rho \approx 1/256$), they introduce significant quantization noise and return scores on incompatible scales, requiring expensive re-ranking.

### The Sparse-Native Solution
For hypervectors with fixed sparsity $m = D/256$, retrieval is mathematically equivalent to finding the maximum intersection of support sets in an inverted index.

1.  **Index Structure**: An inverted list mapping each dimension $d \in [0, D)$ to a sorted list of document IDs that are active in that dimension.
2.  **Epoch-Stamped Accumulator**: To avoid $O(N)$ zeroing of scores for each query, the system uses an epoch-stamped array. This allows sub-millisecond accumulation of intersection counts across relevant posting lists.
3.  **WAND / Block-Max WAND Pruning**: By sorting query terms by document frequency and maintaining upper bounds on potential intersection scores for blocks of the inverted index, the system can prune up to 90% of candidate evaluations while guaranteeing Top-K correctness.

### Impact on Factorization
The Sparse-Native engine provides a "candidate shortlist" (e.g., Top-1000) for unbinding tasks. This allows the Resonator Network to operate on a constrained hypothesis space, reducing the unbinding problem from a global search to a local ranking task, significantly improving both latency and convergence stability.
