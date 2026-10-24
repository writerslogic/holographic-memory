// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;

struct PayloadProbe {
    graph: Option<VGraph>,
    rows: BaseRows,
    slots: Vec<u32>,
    delta: Delta,
    manifest: Manifest,
    footprint: serde_json::Value,
}

impl PayloadProbe {
    fn compact(index: UpdatableVGraph, output: &Path) -> Result<Self, UpdateError> {
        let original_path = index.state.read().manifest.rows_path(&index.dir);
        let original_file_bytes = std::fs::metadata(original_path)?.len();
        let (base, tombstones, delta, manifest) = {
            let mut state = index.state.write();
            assert!(state.pending.is_empty(), "probe requires committed input");
            (
                Arc::clone(&state.base),
                std::mem::take(&mut state.tombstones),
                std::mem::take(&mut state.delta),
                state.manifest.clone(),
            )
        };
        drop(index);
        let Base {
            graph,
            rows: original,
            ..
        } = Arc::try_unwrap(base).unwrap_or_else(|_| panic!("probe requires sole base ownership"));
        let count = original.ids.len();
        let original_id_heap = original.ids.capacity() * size_of::<u32>();
        let original_tombstone_heap = tombstones.capacity() * size_of::<u64>();
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(count)
            .map_err(|_| UpdateError::Capacity)?;
        slots.resize(count, u32::MAX);
        let mut ids = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| UpdateError::Capacity)?;
        for (slot, &id) in original.ids.iter().enumerate() {
            if tombstones[slot / 64] & (1 << (slot % 64)) == 0 {
                slots[slot] = u32::try_from(ids.len()).map_err(|_| UpdateError::Capacity)?;
                ids.push(id);
            }
        }
        drop(tombstones);
        let mut writer = BaseRowsWriter::new(output, &manifest, &ids)?;
        for (slot, &row) in slots.iter().enumerate() {
            if row != u32::MAX {
                writer.append(original.row(slot)?)?;
            }
        }
        writer.finish()?;
        drop(ids);
        drop(original);
        let rows = BaseRows::open(output, &manifest)?;
        let compact_file_bytes = std::fs::metadata(output)?.len();
        let footprint = serde_json::json!({
            "original_base_rows":count,
            "live_base_rows":rows.ids.len(),
            "dim":manifest.dim,
            "metric":format!("{:?}",manifest.build.metric),
            "retained_graph_bytes":graph.as_ref().map_or(0, VGraph::index_bytes),
            "slot_map_bytes":slots.capacity()*size_of::<u32>(),
            "original_id_heap_bytes":original_id_heap,
            "original_tombstone_heap_bytes":original_tombstone_heap,
            "compact_id_heap_bytes":rows.ids.capacity()*size_of::<u32>(),
            "original_rows_file_bytes":original_file_bytes,
            "compact_rows_file_bytes":compact_file_bytes,
            "reclaimable_rows_file_bytes":original_file_bytes-compact_file_bytes,
            "retained_delta_slots":delta.len(),
            "retained_delta_live":delta.live.len(),
            "retained_delta_vector_bytes":delta.vectors.len()*size_of::<f32>(),
            "original_rows_mapping_dropped":true,
            "original_durable_files_retained":true
        });
        Ok(Self {
            graph,
            rows,
            slots,
            delta,
            manifest,
            footprint,
        })
    }

    fn payload_slot(&self, slot: usize) -> Result<Option<usize>, UpdateError> {
        let row = *self
            .slots
            .get(slot)
            .ok_or_else(|| UpdateError::CorruptBase("routing slot outside slot map".into()))?;
        if row == u32::MAX {
            return Ok(None);
        }
        let row = row as usize;
        if row >= self.rows.ids.len() {
            return Err(UpdateError::CorruptBase("slot map outside payload".into()));
        }
        Ok(Some(row))
    }

    fn score(&self, query: &[f32], id: u32) -> Result<u32, UpdateError> {
        let row = if let Some(&slot) = self.delta.live.get(&id) {
            self.delta.row(slot, self.manifest.dim)
        } else {
            let slot = self
                .rows
                .ids
                .binary_search(&id)
                .map_err(|_| UpdateError::UnknownId(id))?;
            self.rows.row(slot)?
        };
        Ok(distance(query, row, self.manifest.build.metric).to_bits())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: VSearchParams,
    ) -> Result<Vec<(u32, u32)>, UpdateError> {
        validate_vector(query, self.manifest.dim, self.manifest.build.metric)?;
        let k = k.min(self.rows.ids.len() + self.delta.live.len());
        let mut out = Vec::new();
        if k == 0 {
            return Ok(Vec::new());
        }
        if self.delta.len() == 0 && self.rows.ids.len() == self.slots.len() {
            if let Some(graph) = &self.graph {
                SCRATCH.with(|cache| {
                    graph.search_reusing(&mut cache.borrow_mut(), query, k, params, &mut out)
                })?;
            }
            return out
                .into_iter()
                .map(|slot| {
                    let row = self.payload_slot(slot as usize)?.expect("clean row");
                    let id = self.rows.ids[row];
                    Ok((id, self.score(query, id)?))
                })
                .collect();
        }
        if let Some(graph) = self.graph.as_ref().filter(|_| !self.rows.ids.is_empty()) {
            let live = self.rows.ids.len();
            let mut take = widened_width(params.ef.max(params.rerank).max(k), graph.len(), live);
            loop {
                let mut widened = params;
                widened.ef = take;
                widened.rerank = 0;
                widened.patience = 0;
                SCRATCH.with(|cache| {
                    graph.search_reusing(&mut cache.borrow_mut(), query, take, widened, &mut out)
                })?;
                let found = out.iter().try_fold(0, |count, &slot| {
                    Ok::<_, UpdateError>(
                        count + usize::from(self.payload_slot(slot as usize)?.is_some()),
                    )
                })?;
                if take == graph.len() || found >= k.min(live) {
                    break;
                }
                take = take.saturating_mul(2).min(graph.len());
            }
        }
        let metric = self.manifest.build.metric;
        let mut candidates = Vec::new();
        for slot in out {
            if let Some(row) = self.payload_slot(slot as usize)? {
                candidates.push((
                    distance(query, self.rows.row(row)?, metric),
                    self.rows.ids[row],
                ));
            }
        }
        if !self.delta.live.is_empty() {
            let mut ef = widened_width(
                params.ef.max(params.rerank).max(k),
                self.delta.len(),
                self.delta.live.len(),
            );
            loop {
                let found = self.delta.search(query, ef, metric);
                if ef == self.delta.len() || found.len() >= k.min(self.delta.live.len()) {
                    candidates.extend(found);
                    break;
                }
                ef = ef.saturating_mul(2).min(self.delta.len());
            }
        }
        candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        candidates.truncate(k);
        Ok(candidates
            .into_iter()
            .map(|(score, id)| (id, score.to_bits()))
            .collect())
    }
}

fn rows(count: usize) -> Vec<f32> {
    (0..count)
        .flat_map(|id| [(id % 11) as f32 - 5., (id % 7) as f32 - 3., 1.])
        .collect()
}

fn build(metric: Metric) -> BuildParams {
    BuildParams {
        metric,
        degree: 8,
        build_ef: 16,
        ..Default::default()
    }
}

fn check_compaction(index: UpdatableVGraph, output: &Path) {
    let queries = [[0.31, -0.47, 0.63], [1., 2., 3.], [-2., -1., 1.]];
    let n = index.len();
    let state = index.state.read();
    let weak = Arc::downgrade(&state.base);
    let graph_bytes = state.base.graph.as_ref().map_or(0, VGraph::index_bytes);
    let graph_count = state.base.graph.as_ref().map_or(0, VGraph::len);
    let delta_ptr = state.delta.vectors.as_ptr();
    let original_path = state.manifest.rows_path(&index.dir);
    let dirty = state.deleted > 0 || state.delta.len() > 0;
    drop(state);
    let original_crc = crc32fast::hash(&std::fs::read(&original_path).unwrap());
    let mut cases = Vec::new();
    for query in queries {
        for k in [0, 1, 9, 10, 11, n, n + 1] {
            for ef in [1, 8, 128] {
                for rerank in [0, 16] {
                    for patience in [0, 2] {
                        let params = VSearchParams {
                            ef,
                            rerank,
                            patience,
                            ..Default::default()
                        };
                        let ids = index.search(&query, k, params).unwrap();
                        let state = index.state.read();
                        let found: Vec<_> = ids
                            .into_iter()
                            .map(|id| {
                                let row = if let Some(&slot) = state.delta.live.get(&id) {
                                    state.delta.row(slot, 3)
                                } else {
                                    state.base.rows.row(state.base_slot(id).unwrap()).unwrap()
                                };
                                (
                                    id,
                                    distance(&query, row, state.manifest.build.metric).to_bits(),
                                )
                            })
                            .collect();
                        if dirty && ef == 128 {
                            let mut all: Vec<_> = state
                                .base
                                .rows
                                .ids
                                .iter()
                                .copied()
                                .filter(|&id| state.base_slot(id).is_some())
                                .chain(state.delta.live.keys().copied())
                                .map(|id| {
                                    let row = if let Some(&slot) = state.delta.live.get(&id) {
                                        state.delta.row(slot, 3)
                                    } else {
                                        state.base.rows.row(state.base_slot(id).unwrap()).unwrap()
                                    };
                                    (distance(&query, row, state.manifest.build.metric), id)
                                })
                                .collect();
                            all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                            let expected: Vec<_> = all
                                .into_iter()
                                .take(k)
                                .map(|(score, id)| (id, score.to_bits()))
                                .collect();
                            assert_eq!(found, expected);
                        }
                        cases.push((query, k, params, found));
                    }
                }
            }
        }
    }
    let probe = PayloadProbe::compact(index, output).unwrap();
    assert!(weak.upgrade().is_none());
    assert_eq!(
        probe.graph.as_ref().map_or(0, VGraph::index_bytes),
        graph_bytes
    );
    assert_eq!(probe.graph.as_ref().map_or(0, VGraph::len), graph_count);
    assert_eq!(probe.delta.vectors.as_ptr(), delta_ptr);
    assert_eq!(
        crc32fast::hash(&std::fs::read(original_path).unwrap()),
        original_crc
    );
    for (query, k, params, expected) in cases {
        assert_eq!(probe.search(&query, k, params).unwrap(), expected);
    }
    println!("PAYLOAD_PROBE {}", probe.footprint);
}

#[test]
fn stable_payload_preserves_global_queries_across_slot_boundaries() {
    for count in [63, 64, 65] {
        for metric in [Metric::Euclidean, Metric::Angular] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("index");
            let index = UpdatableVGraph::create(&path, &rows(count), 3, &build(metric)).unwrap();
            check_compaction(index, &directory.path().join("clean.bin"));
            let index = UpdatableVGraph::open(&path).unwrap();
            for id in [0, (count / 2) as u32, (count - 1) as u32] {
                index.delete(id).unwrap();
            }
            index.commit().unwrap();
            index.replace(1, &[-1., 2., 1.]).unwrap();
            index.insert(0, &[9., 0., 1.]).unwrap();
            index.insert(u32::MAX - 1, &[1., 2., 3.]).unwrap();
            index.insert((count + 1) as u32, &[1., 2., 3.]).unwrap();
            assert!(matches!(
                index.insert(u32::MAX, &[1., 2., 3.]),
                Err(UpdateError::IdOutOfRange)
            ));
            index.commit().unwrap();
            check_compaction(index, &directory.path().join("churn.bin"));
            let index = UpdatableVGraph::open(&path).unwrap();
            let state = index.state.read();
            let ids: Vec<_> = state
                .base
                .rows
                .ids
                .iter()
                .copied()
                .filter(|&id| state.base_slot(id).is_some())
                .chain(state.delta.live.keys().copied())
                .collect();
            drop(state);
            index.delete_batch(&ids).unwrap();
            index.commit().unwrap();
            index.insert(0, &[1., 2., 3.]).unwrap();
            index.commit().unwrap();
            check_compaction(index, &directory.path().join("empty-base.bin"));
        }
    }
}

#[test]
fn stable_payload_rejects_outside_slots_and_payload_rows() {
    let directory = tempfile::tempdir().unwrap();
    let index = UpdatableVGraph::create(
        &directory.path().join("index"),
        &rows(5),
        3,
        &build(Metric::Euclidean),
    )
    .unwrap();
    index.delete(1).unwrap();
    index.commit().unwrap();
    let mut probe = PayloadProbe::compact(index, &directory.path().join("rows.bin")).unwrap();
    assert_eq!(probe.payload_slot(1).unwrap(), None);
    assert_eq!(probe.payload_slot(4).unwrap(), Some(3));
    for slot in [5, 6] {
        assert!(matches!(
            probe.payload_slot(slot),
            Err(UpdateError::CorruptBase(_))
        ));
    }
    probe.slots[0] = 3;
    assert_eq!(probe.payload_slot(0).unwrap(), Some(3));
    for slot in [4, 5] {
        probe.slots[0] = slot;
        assert!(matches!(
            probe.payload_slot(0),
            Err(UpdateError::CorruptBase(_))
        ));
    }
}

#[test]
fn stable_payload_empty_graph_keeps_live_delta() {
    let directory = tempfile::tempdir().unwrap();
    let index = UpdatableVGraph::create(
        &directory.path().join("index"),
        &[],
        3,
        &build(Metric::Euclidean),
    )
    .unwrap();
    index.insert(u32::MAX - 1, &[1., 2., 3.]).unwrap();
    index.commit().unwrap();
    check_compaction(index, &directory.path().join("rows.bin"));
}

#[test]
fn stable_payload_preserves_sparse_base_ids_after_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index");
    let index = UpdatableVGraph::create(&path, &rows(17), 3, &build(Metric::Euclidean)).unwrap();
    index.delete_batch(&[0, 8, 16]).unwrap();
    index.commit().unwrap();
    index.insert(0, &[1., 2., 3.]).unwrap();
    index.insert(u32::MAX - 1, &[1., 2., 3.]).unwrap();
    index.commit().unwrap();
    index.consolidate().unwrap();
    drop(index);
    let index = UpdatableVGraph::open(&path).unwrap();
    assert!(!index.state.read().base.dense_ids);
    check_compaction(index, &directory.path().join("rows.bin"));
}
