# warm-research-forget: exact deletion from a superposition memory, with certificates

Package: `benchmarks/research/forget/` (standalone crate `warm-research-forget`,
not a workspace member, own `[workspace]` table). Not compiled by the author
(round-2 hard rule); the mediator builds and runs.

## Claim under test

A counter-backed bundle (`CounterBundle`, `src/bundle.rs`) stores the
per-dimension integer sum of the engine's sparse hypervectors unthresholded
(`i16` or `i32`, overflow-checked). Insert is `+1`, delete is `-1` at the
item's indices, so delete is the exact group inverse of insert: after deleting
any subset, the counts are bit-for-bit the fold of the survivors. The recall
view `{ i : count[i] >= tau }` is derived on demand and at `tau = 1` is exactly
`EntangledHVec::bundle_bloom` of the survivors. Each delete can issue a
`DeletionCertificate` (`src/certificate.rs`): SHA-256 of the counts before,
after, and of the item, plus a residual bound (deleted item's overlap with the
post-delete view against the exact hypergeometric null of a random item, and
the same for the unthresholded count readout). A verifier holding the
post-delete bundle and the item recomputes `state_after` and reproduces
`state_before` by digesting `after + item` in one streaming pass, proving the
transition was exactly that subtraction.

Today's mechanism (`ThresholdedBundle`, the union bitset) has no inverse: a
delete can leave the bits (the item stays fully recognisable) or clear them
(every surviving item sharing those positions loses them).

Document-store shards are per-document, not superpositions (see "Shards");
deletion there is already exact and needs no counter.

## Dependencies

`writerslogic-warm = { path = "../../..", default-features = false }`,
`anyhow 1.0`, `clap 4.4 (derive)`, `serde 1.0 (derive)`, `serde_json 1.0
(float_roundtrip)`, `sha2 0.11` (the root crate's optional `security`
dependency, same version). No new third-party crates. `Cargo.lock` and
`target/` are git-ignored inside the package.

## Root-crate APIs used (file:line)

- `writerslogic_warm::EntangledHVec` — `src/lib.rs:29` re-export of `src/core/entangled.rs:31` (`dim`, sorted `indices: Vec<u32>`, density `1/256`, `src/core/entangled.rs:12`).
- `EntangledHVec::new_deterministic(dim, seed)` — `src/core/entangled.rs:43` (item, probe generation).
- `EntangledHVec::from_indices(Vec<u32>, dim)` — `src/core/entangled.rs:117` (`CounterBundle::view_hvec`).
- `EntangledHVec::indices()` — `src/core/entangled.rs:240` (everywhere).
- `EntangledHVec::bundle_bloom(&[V])` — `src/core/entangled.rs:422` (cross-check of the `tau = 1` view).
- `writerslogic_warm::core::entangled::hash_u64(a, b)` — `src/core/entangled.rs:556` (`pub mod core` at `src/lib.rs:15`, `pub mod entangled` at `src/core/mod.rs:57`); seeds for items, deleted subset, probes.
- Read for semantics only (not called): `bundle` `src/core/entangled.rs:356-413` (majority threshold `ceil(n/2)`, then top `dim/256` by count), `similarity` (Jaccard) `:288`, `containment_similarity` `:449`, `bind` `:300`, `permute` `:309`; `CountingBloomMemory` `src/core/counting_bloom_memory.rs:31-61` (counts, insert only, `saturating_add`, no delete, no certificate).

`sha2 0.11`: `Sha256::new() / update / finalize().into() -> [u8; 32]`, the same
calls as `src/core/provenance/types.rs:50-52`.

## Shards (deliverable 5)

`src/core/engine/shard.rs:24-35`: `Shard { vectors: RwLock<FxHashMap<String,
EntangledHVec>>, registry, slots, inverted: SparseInvertedIndex, nsg, ivf, .. }`
— one hypervector per id, no superposition. `Shard::remove`
(`src/core/engine/shard.rs:100-116`) removes the map entry, calls
`inverted.remove_doc(slot, old.indices())`, frees the slot and sets `ivf` and
`nsg` to `None`. `mutation.rs` only calls `shards.insert` / `shards.remove`
(`src/core/engine/mutation.rs:451-454,600`). Deletion in the document store is
therefore already exact (the remaining derived state, IVF/NSG, is dropped, not
patched), and no counter shard is needed there. The superposition stores that
would benefit are `bloom_memory`, `counting_bloom_memory` and
`composite_memory`, none of which has a delete today. The JSON report records
this under `shards`.

## Commands (run from the repo root)

```sh
export RUSTFLAGS="-C target-cpu=native"
M=benchmarks/research/forget/Cargo.toml
OUT=benchmarks/research/forget/out

# 0. unit tests: insert/delete inverse, rebuild equality, view == bundle_bloom at tau 1,
#    non-member refusal, i16 overflow refused atomically, capacity policy, certificate
#    verify + chain, baseline noop/clear semantics, digest hex round trip
cargo test --release --manifest-path $M

# 1. the kill test at the engine dimension (N in {100, 1k, 10k} x delete {1%, 10%}), i16 counts
cargo run --release --manifest-path $M -- --out $OUT/forget_d16384_i16_tau1.json

# 2. same with i32 counts (memory accounting column changes; exactness must not)
cargo run --release --manifest-path $M -- --counter i32 --out $OUT/forget_d16384_i32_tau1.json

# 3. a dimension where 10k items do not saturate the union view (lambda ~ 2.4),
#    so the baseline leak check is applicable on every row
cargo run --release --manifest-path $M -- --dim 262144 --out $OUT/forget_d262144_i16_tau1.json

# 4. thresholded view at the Poisson rate (tau = ceil(n k / dim)); baselines n/a
cargo run --release --manifest-path $M -- --tau auto --out $OUT/forget_d16384_i16_tauauto.json

# 5. bounded-capacity policy exercised: the 10k rows must fail with DensityLimit
#    (exit code 1, error text names the density), the 100-item rows pass
cargo run --release --manifest-path $M -- --ns 100 --max-view-density 0.5 --out $OUT/forget_d16384_policy_ok.json
cargo run --release --manifest-path $M -- --ns 10000 --max-view-density 0.5 --out $OUT/forget_d16384_policy_refused.json; echo "exit=$?"
```

Other knobs: `--ns 100,1000,10000`, `--delete-fracs 0.01,0.10`, `--probes 2000`,
`--survivor-sample 2000`, `--runs 5`, `--warmup 1`, `--seed 42`, `--sigma 3`,
`--z-threshold 3`, `--max-items N`.

Expected wall clock per full run at D=16384: well under a minute (10k inserts
are 640k increments; certified deletes cost two 32 KiB SHA-256 digests each,
about 0.2 ms, so 1,000 certified deletes are ~0.2 s per repeat). At D=262144
the digests are 16x larger: budget a few minutes.

## Pass/fail (printed per row and overall)

Per row (`verdict`):

1. `exact`: counts bit-for-bit equal to the from-scratch rebuild of the
   survivors, equal state and view digests, every certificate verified against
   the post-delete state, first certificate's `state_before` equal to the
   full-load digest. At `tau = 1`, `view_matches_bundle_bloom` must also be
   `true` (reported, included in `rebuild_exact`'s view check).
2. `chain_consistent`: each certificate's `state_after` is the next's `state_before`.
3. `counter_indistinguishable`: on both readouts (`view_overlap`,
   `count_cosine`) the deleted-vs-random z-score satisfies `|z| < 3` and at
   most 1% of deleted items individually exceed `random.mean + 3 sd`
   (expected 0.135% under the null).
4. `baseline_leaks`: the thresholded baseline with the `noop` delete scores
   deleted items above random by `z > 3`. Evaluated only when the baseline is
   informative (its survivors score above random by `z > 3`); when the union
   view is saturated (density ~ 1.0, which happens at N=10k, D=16384 where
   lambda ~ 39) the check is `null` and the row does not fail on it. Command 3
   makes the check applicable on every row.

`pass = exact && chain_consistent && counter_indistinguishable && baseline_leaks != false`.
Overall passes when every row passes. Timing (c) is reported, not thresholded:
`delete_ns_per_item` should be within a small factor of `insert_ns_per_item`
(both are `O(k)` with two passes), and `delete_with_certificate_ns_per_item`
is dominated by the two `O(dim)` digests.

Expected qualitative outcome: every row `exact = true`; counter z-scores near
0; `count_cosine.z_survivors_vs_random` large positive even at N=10k (the
unthresholded readout still recognises members past the Bloom wall); baseline
`noop` z strongly positive where informative, baseline `clear` z strongly
negative with `survivor_bits_lost_frac > 0`.

## Output JSON schema (`schema = "warm-research-forget/v1"`)

```text
{
  schema, generated_unix_s,
  config:  { dim, ns[], delete_fracs[], tau ("1" | "auto" | ...), counter ("i16"|"i32"), probes,
             survivor_sample, runs, warmup, seed, sigma, z_threshold,
             policy: { max_items, max_view_density }, threads: 1 },
  shards:  { per_document: true, deletion_already_exact: true, note, evidence[] },
  rows: [ {
    n, delete_frac, k_deleted, k_active, tau,
    counter: {
      counter, rebuild_exact, counts_equal,
      state_digest_after, rebuild_state_digest, view_digest_after, rebuild_view_digest   (64-hex),
      view_matches_bundle_bloom (bool | null when tau != 1),
      view_density_full, view_density_after, lambda_full, lambda_after,
      certificates: { issued, verified, chain_consistent, first_state_before_matches_full_state,
                      within_bound, max_abs_view_z, max_abs_count_z,
                      example: DeletionCertificate | null },
      readouts: [ { name ("view_overlap" | "count_cosine"),
                    deleted: Sample, random: Sample, survivors: Sample,
                    z_deleted_vs_random, z_survivors_vs_random,
                    frac_deleted_over_threshold, informative } ],
      memory: { dim, items, active_per_item, counter_bits, counter_bytes, ledger_entries,
                ledger_payload_bytes, view_size, view_density, view_bitset_bytes,
                view_sparse_bytes, bytes_per_item_counter, bytes_per_item_total,
                per_document_store_bytes, per_document_bytes_per_item, breakeven_items },
      timing: { insert_ns_per_item, delete_ns_per_item, delete_with_certificate_ns_per_item,
                verify_certificate_us, state_digest_us, rebuild_ms, baseline_insert_ns_per_item
                : each { median, runs[] } }
    },
    baselines: [ { mode ("noop" | "clear"), view_density_full, view_density_after, bits_cleared,
                   readout (as above, name "bit_overlap"), survivor_bits_lost_frac,
                   leaks, distinguishable } ]   (empty when tau != 1),
    verdict: { exact, chain_consistent, counter_indistinguishable, baseline_leaks (bool|null), pass }
  } ],
  overall: { pass, rows_pass, rows_total, rows_baseline_not_applicable }
}

Sample = { n, mean, sd, min, max }

DeletionCertificate (schema "warm-forget/certificate-v1") = {
  schema, dim, tau, counter_bits, item, state_before, state_after, view_after (64-hex),
  items_before, items_after,
  residual: { active, view_size, view_density, view_overlap, overlap_fraction, cosine,
              view_null_mean, view_null_sd, view_z, count_dot, count_null_mean, count_null_sd,
              count_z, sigma, within_bound }
}
```

Digest domains: item = SHA-256(`"warm-forget/item-v1"` || dim u64 LE || len u64
LE || indices u32 LE); state = SHA-256(`"warm-forget/state-v1"` || dim u64 ||
tau u32 || counter bits u32 || live items u64 || counts LE). The ledger key is
the first 8 bytes of the item digest.

## Open risks

- Not compiled by the author. Likely friction points: clap `default_value`
  with `value_delimiter` on `Vec<usize>`/`Vec<f64>`; `sha2 0.11`
  `finalize().into()` to `[u8; 32]` (mirrors the root crate's call).
- The engine's default dimension was taken from the documented D=16384 /
  k=64 operating point (`src/core/entangled.rs:9`,
  `src/core/counting_bloom_memory.rs:14`); `--dim` overrides it.
- Certificates cost `O(dim)` per delete (two full-state digests). An
  incremental homomorphic commitment (lattice hash) would make it `O(k)`; not
  prototyped.
- The residual bound certifies that the item's own contribution is gone; a
  surviving near-duplicate legitimately keeps the deleted item's score high on
  real data. On seeded random items (this test) that case does not arise.
