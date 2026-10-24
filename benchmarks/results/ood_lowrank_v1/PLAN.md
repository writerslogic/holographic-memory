# Low-rank OOD development protocol

This recovers the experiment registered in the coordination VIBE notes on
2026-10-23 before any measurements. It is not a standing or a winning claim.

- Full yi-128-ip base; coco-nomic-768 first 100,000 base rows only if yi passes.
- Train with the first 32,768 IDs in a NumPy default_rng(24301) permutation of
  official learn. Tune on the next 1,000. No official test reads before selection.
- Low-rank grid: clusters 256/1024, rank 16/32/64, nprobe
  8/12/16/24/32/48/64/96/128/192/256, rerank 100/200/400/800 (264 cells).
- Graph: degree64/build_ef500/eight-bit/query-aware training, 15 ef settings by
  three rerank settings (45 cells), as encoded in ood_ivf_ab.py.
- Exact float64 reference uses original IDs for ties. Both methods learn from
  exactly the same training rows. Query k100, one pinned query thread.
- Three rotated tuning rounds; choose each arm's highest median QPS among cells
  with recall >=.95; tie by build label then arguments. Preserve every cell.
- Write selection before reading official test. Check only selected points, once,
  with three alternating paired timing rounds. Pass requires recall>=.95 for both
  and low-rank/graph speed >=1 in every pair. A yi failure stops this experiment.
- Linux exclusive timing lock, load1<3 at each window's beginning/end; wait at most
  600s and hold at most720s. A failed gate invalidates the run; never continue loaded.
- Record immutable source/module/harness/dataset hashes, train/validation ID hashes,
  CPU affinity, build/query costs, loads, exact selected answers hash and all losses.

The abandoned harness violated its own plan by evaluating official test queries at
all264 candidate cells; this recovered harness restricts tuning to learn rows and
opens official test only after writing the selected indices. No result from that
abandoned script is used. Existing numerical tests passed before recovery; added
boundary regressions cover original-ID ties and invalid ranks. Saved formats are
unchanged; this index has no serialization and remains an experimental prototype.

The underlying regression follows the primary derivation in
[LoRANN, NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/b939da3932e88ded5e9b08026e35069d-Paper-Conference.pdf).
