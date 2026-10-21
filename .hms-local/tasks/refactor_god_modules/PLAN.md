# Plan: Refactor HmsCore God Module

## Goal
Decompose the `HmsCore` struct into smaller, focused modules to improve maintainability and follow the single responsibility principle.

## Constraints
- Maintain existing public API (where possible, or update `lib.rs` to bridge).
- Ensure thread safety (many fields in `HmsCore` are `Arc<RwLock<...>>`).
- No changes to the underlying storage format.

## Specification

### 1. Extract Knowledge Representation
Move triplet and sequence logic to `src/core/engine/knowledge.rs`.
- `memorize_triplet`
- `memorize_sequence`
- `query_triplet`

### 2. Extract Clustering Logic
Move concept synthesis logic to `src/core/engine/concepts.rs`.
- `synthesize_concepts`
- Associated constants like `CONCEPT_SIMILARITY_THRESHOLD` and `MIN_CONCEPT_CLUSTER_SIZE`.

### 3. Consolidate Data Loading
Add a helper `load_all_vectors` to `HmsCore` to deduplicate registry scanning logic.

### 4. Extract Maintenance Operations
Move `compact` and related logic to a dedicated module or separate implementation block.

## Verification Steps
1. Run existing tests (`cargo test`).
2. Verify that `HmsCore` line count decreases significantly.
3. Ensure the Node.js API remains functional (if applicable to local testing).

## Acceptance Criteria
- `HmsCore` in `mod.rs` and `query.rs` is primarily an orchestrator.
- Algorithmic details are isolated in their respective modules.
- All tests pass.
