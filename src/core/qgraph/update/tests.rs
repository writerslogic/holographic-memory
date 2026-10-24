// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use proptest::prelude::*;
use std::collections::BTreeMap;
use std::io::{Seek, SeekFrom, Write};

fn params() -> BuildParams {
    BuildParams {
        metric: Metric::Euclidean,
        degree: 32,
        build_ef: 64,
        ..Default::default()
    }
}
fn search_params() -> VSearchParams {
    VSearchParams {
        ef: 1024,
        rerank: 1024,
        ..Default::default()
    }
}
fn data(n: usize, dim: usize) -> Vec<f32> {
    let mut random = super::super::kernels::SplitMix(7654);
    (0..n * dim)
        .map(|_| (random.next_u64() >> 40) as f32 / (1u32 << 24) as f32)
        .collect()
}
fn exact(live: &BTreeMap<u32, Vec<f32>>, q: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<_> = live
        .iter()
        .map(|(&id, v)| (distance(q, v, Metric::Euclidean), id))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|x| x.1).collect()
}
fn copy_store(from: &Path, to: &Path) {
    std::fs::create_dir(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != ".lock" {
            std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
}

#[test]
fn streaming_diagnostics_preserve_search_and_measure_live_generations() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let mut index = UpdatableVGraph::create(&path, &[], 4, &params()).unwrap();
    index.diagnostics.enabled = false;
    assert!(index.streaming_diagnostics().is_none());
    let rows = data(128, 4);
    index
        .insert_batch(&(0..128).collect::<Vec<_>>(), &rows)
        .unwrap();
    index.commit().unwrap();
    index.delete(0).unwrap();
    index.replace(1, &[2., 3., 4., 5.]).unwrap();
    index.commit().unwrap();
    let query = [0.31, -0.47, 0.63, 0.19];
    let p = VSearchParams {
        ef: 16,
        rerank: 16,
        ..Default::default()
    };
    let expected = index.search(&query, 10, p).unwrap();
    index.diagnostics.enabled = true;
    assert_eq!(index.search(&query, 10, p).unwrap(), expected);
    let measured = index.streaming_diagnostics().unwrap();
    assert_eq!(measured["delta_slots"], 129.);
    assert_eq!(measured["delta_live"], 127.);
    assert_eq!(measured["delta_entry_dead"], 1.);
    assert_eq!(measured["delta_live_reachable_fraction"], 1.);
    assert_eq!(measured["queries"], 1.);
    assert!(measured["delta_visits"] > 0.);
    assert!(!expected.contains(&0));
    let state = index.state.read();
    for ef in [1, 16, 128, 129, 130] {
        assert_eq!(
            state.delta.search(&query, ef, Metric::Euclidean),
            state.delta.search_profiled(&query, ef, Metric::Euclidean).0
        );
    }
}

#[test]
fn deterministic_streaming_model_survives_generations_and_replay_twice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let mut index = UpdatableVGraph::create(&path, &[], 4, &params()).unwrap();
    let mut live = BTreeMap::new();
    let mut random = super::super::kernels::SplitMix(0x571e_a019);
    let query = [0.31, -0.47, 0.63, 0.19];
    for step in 0..120 {
        let id = (random.next_u64() % 48) as u32;
        let row: Vec<f32> = (0..4)
            .map(|_| (random.next_u64() >> 40) as f32 / (1u32 << 24) as f32)
            .collect();
        match live.entry(id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                index.insert(id, &row).unwrap();
                entry.insert(row);
            }
            std::collections::btree_map::Entry::Occupied(entry) if step % 3 == 0 => {
                index.delete(id).unwrap();
                entry.remove();
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                index.replace(id, &row).unwrap();
                entry.insert(row);
            }
        }
        index.commit().unwrap();
        if step % 17 == 0 {
            index.consolidate().unwrap();
        }
        let want = exact(&live, &query, 10);
        assert_eq!(index.search(&query, 10, search_params()).unwrap(), want);
        {
            let state = index.state.read();
            state.delta.assert_invariants(state.manifest.build.degree);
            for (&id, row) in &live {
                let actual = if let Some(&slot) = state.delta.live.get(&id) {
                    state.delta.row(slot, 4)
                } else {
                    state.base.rows.row(state.base_slot(id).unwrap()).unwrap()
                };
                assert_eq!(actual, row);
            }
        }
        for _ in 0..2 {
            drop(index);
            index = UpdatableVGraph::open(&path).unwrap();
            assert_eq!(index.search(&query, 10, search_params()).unwrap(), want);
        }
    }
}

#[test]
fn repeated_replacement_keeps_delta_vertices_reachable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let index = UpdatableVGraph::create(&path, &data(16, 4), 4, &params()).unwrap();
    index.replace(0, &[0.03, -0.08, 0.48, 0.78]).unwrap();
    index.commit().unwrap();
    index.insert(16, &[0.06, -0.59, -0.94, 0.84]).unwrap();
    index.commit().unwrap();
    index.replace(0, &[0.77, -0.83, -0.08, -0.21]).unwrap();
    index.commit().unwrap();
    let query = [0.31, -0.47, 0.63, 0.19];
    let result = index.search(&query, 17, search_params()).unwrap();
    assert_eq!(result.len(), 17);
    assert!(result.contains(&0));
    assert!(result.contains(&16));
    assert_eq!(
        index
            .state
            .read()
            .delta
            .search(&[0.77, -0.83, -0.08, -0.21], 1, Metric::Euclidean)
            .iter()
            .map(|x| x.1)
            .collect::<Vec<_>>(),
        [0]
    );
    let mut live: BTreeMap<_, _> = data(16, 4)
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(id, row)| (id as u32, row.to_vec()))
        .collect();
    live.insert(0, vec![0.77, -0.83, -0.08, -0.21]);
    live.insert(16, vec![0.06, -0.59, -0.94, 0.84]);
    assert_eq!(
        index.search(&query, 10, search_params()).unwrap(),
        exact(&live, &query, 10)
    );
    drop(index);
    assert_eq!(
        UpdatableVGraph::open(&path)
            .unwrap()
            .search(&query, 17, search_params())
            .unwrap(),
        result
    );
}

#[test]
fn buffered_search_matches_owned_results_across_updates_and_consolidation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let rows = data(64, 4);
    let index = UpdatableVGraph::create(&path, &rows, 4, &params()).unwrap();
    let mut out = Vec::with_capacity(64);
    let allocation = out.as_ptr();
    for phase in 0..3 {
        if phase == 1 {
            index.delete(0).unwrap();
            index.replace(1, &[1., 2., 3., 4.]).unwrap();
            index.insert(80, &[4., 3., 2., 1.]).unwrap();
        } else if phase == 2 {
            index.consolidate().unwrap();
        }
        for query in rows.as_chunks::<4>().0 {
            let expected = index.search(query, 10, search_params()).unwrap();
            index
                .search_into(query, 10, search_params(), &mut out)
                .unwrap();
            assert_eq!(out, expected);
            assert_eq!(out.as_ptr(), allocation);
        }
        index
            .search_into(&rows[..4], 0, search_params(), &mut out)
            .unwrap();
        assert!(out.is_empty());
    }
}

#[test]
fn close_unlocks_even_when_a_descriptor_is_inherited() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let index = UpdatableVGraph::create(&path, &[], 2, &params()).unwrap();
    let inherited = index._lock.0.try_clone().unwrap();
    assert!(matches!(
        UpdatableVGraph::open(&path),
        Err(UpdateError::Locked)
    ));
    drop(index);
    let reopened = UpdatableVGraph::open(&path).unwrap();
    drop(inherited);
    assert!(matches!(
        UpdatableVGraph::open(&path),
        Err(UpdateError::Locked)
    ));
    drop(reopened);
    UpdatableVGraph::open(&path).unwrap();
}

#[test]
fn boundaries_and_atomic_replace() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let index = UpdatableVGraph::create(&path, &[1., 2., 3., 4.], 2, &params()).unwrap();
    assert!(matches!(
        UpdatableVGraph::open(&path),
        Err(UpdateError::Locked)
    ));
    assert!(matches!(index.delete(7), Err(UpdateError::UnknownId(7))));
    assert!(matches!(
        index.insert(0, &[1., 2.]),
        Err(UpdateError::AlreadyExists(0))
    ));
    assert!(matches!(
        index.insert(7, &[1.]),
        Err(UpdateError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        index.insert(u32::MAX, &[1., 2.]),
        Err(UpdateError::IdOutOfRange)
    ));
    assert!(matches!(
        index.insert(7, &[f32::MAX, 2.]),
        Err(UpdateError::InvalidVector(_))
    ));
    assert!(matches!(
        index.search(&[f32::NAN, 1.], 1, search_params()),
        Err(UpdateError::InvalidVector(_))
    ));
    index.insert(9, &[5., 6.]).unwrap();
    assert!(matches!(
        index.insert(9, &[5., 6.]),
        Err(UpdateError::DuplicateIdInBatch(9))
    ));
    assert!(matches!(
        index.delete(9),
        Err(UpdateError::DuplicateIdInBatch(9))
    ));
    index.replace(0, &[7., 8.]).unwrap();
    index.delete(1).unwrap();
    assert_eq!(
        index
            .search(&[7., 8.], usize::MAX, search_params())
            .unwrap(),
        [0, 9]
    );
    index.commit().unwrap();
    index.commit().unwrap();
    index.delete(0).unwrap();
    index.delete(9).unwrap();
    assert!(index
        .search(&[0., 0.], usize::MAX, search_params())
        .unwrap()
        .is_empty());
    index.consolidate().unwrap();
    assert!(index.state.read().base.rows.ids.is_empty());
    drop(index);
    let index = UpdatableVGraph::open(&path).unwrap();
    assert!(index.is_empty());
    index.insert(0, &[1., 2.]).unwrap();
    index.commit().unwrap();
    index.consolidate().unwrap();
    assert_eq!(index.search(&[1., 2.], 10, search_params()).unwrap(), [0]);
}

#[test]
fn pending_batch_is_not_persisted_and_empty_base_accepts_inserts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let index = UpdatableVGraph::create(&path, &[], 2, &params()).unwrap();
    index.insert(3, &[1., 2.]).unwrap();
    drop(index);
    let index = UpdatableVGraph::open(&path).unwrap();
    assert!(index.is_empty());
    index.insert(3, &[1., 2.]).unwrap();
    index.commit().unwrap();
    drop(index);
    assert_eq!(UpdatableVGraph::open(&path).unwrap().len(), 1);
}

#[test]
fn torn_last_batch_at_every_byte_and_checksum_corruption() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source");
    let index = UpdatableVGraph::create(&source, &[1., 2., 3., 4.], 2, &params()).unwrap();
    index.insert(5, &[5., 6.]).unwrap();
    index.commit().unwrap();
    let prefix = index.state.read().log_end;
    let expected_prefix = index.search(&[2., 3.], 10, search_params()).unwrap();
    index.replace(0, &[8., 9.]).unwrap();
    index.delete(1).unwrap();
    index.commit().unwrap();
    let expected_full = index.search(&[2., 3.], 10, search_params()).unwrap();
    let end = index.state.read().log_end;
    drop(index);
    for cut in prefix..=end {
        let path = root.path().join(format!("cut-{cut}"));
        copy_store(&source, &path);
        let file = OpenOptions::new()
            .write(true)
            .open(path.join("updates.log"))
            .unwrap();
        file.set_len(cut).unwrap();
        drop(file);
        let index = UpdatableVGraph::open(&path).unwrap();
        assert_eq!(
            index.search(&[2., 3.], 10, search_params()).unwrap(),
            if cut == end {
                &expected_full
            } else {
                &expected_prefix
            }
            .clone(),
            "cut {cut}"
        );
        index.insert(8, &[4., 5.]).unwrap();
        index.commit().unwrap();
        drop(index);
        assert_eq!(
            UpdatableVGraph::open(&path).unwrap().len(),
            if cut == end { 3 } else { 4 }
        );
    }
    for target in [prefix + 20, format::LOG_HEADER + 20] {
        let path = root.path().join(format!("corrupt-{target}"));
        copy_store(&source, &path);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.join("updates.log"))
            .unwrap();
        use std::io::Read;
        file.seek(SeekFrom::Start(target)).unwrap();
        let mut byte = [0];
        file.read_exact(&mut byte).unwrap();
        byte[0] ^= 1;
        file.seek(SeekFrom::Start(target)).unwrap();
        file.write_all(&byte).unwrap();
        drop(file);
        assert!(matches!(
            UpdatableVGraph::open(&path),
            Err(UpdateError::CorruptLog { .. })
        ));
    }
}

#[test]
fn torn_batched_insert_frame_preserves_only_complete_commits() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source");
    let index = UpdatableVGraph::create(&source, &[0.; 4], 4, &params()).unwrap();
    let prefix = index.state.read().log_end;
    index.insert_batch(&[100, 101, 102], &data(3, 4)).unwrap();
    index.commit().unwrap();
    let end = index.state.read().log_end;
    drop(index);
    for cut in prefix..=end {
        let path = root.path().join(format!("insert-cut-{cut}"));
        copy_store(&source, &path);
        let file = OpenOptions::new()
            .write(true)
            .open(path.join("updates.log"))
            .unwrap();
        file.set_len(cut).unwrap();
        drop(file);
        for _ in 0..2 {
            let index = UpdatableVGraph::open(&path).unwrap();
            let found = index.search(&[1.; 4], 10, search_params()).unwrap();
            assert_eq!(found.len(), if cut == end { 4 } else { 1 });
            assert!(found.contains(&0));
            for id in 100..103 {
                assert_eq!(found.contains(&id), cut == end);
            }
        }
    }
}

#[test]
fn consolidation_interruption_recovers_committed_state() {
    for stop in 1..=3 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index");
        let index = UpdatableVGraph::create(&path, &data(64, 4), 4, &params()).unwrap();
        index.replace(0, &[1., 2., 3., 4.]).unwrap();
        index.delete(3).unwrap();
        index.insert(100, &[4., 3., 2., 1.]).unwrap();
        index.commit().unwrap();
        let before = index
            .search(&[1., 1., 1., 1.], 100, search_params())
            .unwrap();
        assert!(index
            .consolidate_inner(|step| if step == stop {
                Err(UpdateError::Io(std::io::Error::other("interruption")))
            } else {
                Ok(())
            })
            .is_err());
        if stop >= 2 {
            assert!(matches!(
                index.insert(101, &[1.; 4]),
                Err(UpdateError::RecoveryRequired)
            ));
        }
        drop(index);
        let reopened = UpdatableVGraph::open(&path).unwrap();
        assert_eq!(
            reopened.search(&[1.; 4], 100, search_params()).unwrap(),
            before
        );
        reopened.insert(101, &[2.; 4]).unwrap();
        reopened.commit().unwrap();
        reopened.consolidate().unwrap();
        drop(reopened);
        assert_eq!(UpdatableVGraph::open(&path).unwrap().len(), 65);
    }
}

#[test]
fn readers_continue_while_consolidation_performs_io() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let index = Arc::new(UpdatableVGraph::create(&path, &data(64, 4), 4, &params()).unwrap());
    index.delete(0).unwrap();
    index.commit().unwrap();
    let (arrived_tx, arrived_rx) = std::sync::mpsc::channel();
    let (resume_tx, resume_rx) = std::sync::mpsc::channel();
    let writer_index = Arc::clone(&index);
    let writer = std::thread::spawn(move || {
        writer_index.consolidate_inner(|step| {
            if step == 1 {
                arrived_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
            }
            Ok(())
        })
    });
    arrived_rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .unwrap();
    let (read_tx, read_rx) = std::sync::mpsc::channel();
    let reader_index = Arc::clone(&index);
    let reader = std::thread::spawn(move || {
        read_tx
            .send(reader_index.search(&[1.; 4], 10, search_params()).unwrap())
            .unwrap()
    });
    let result = read_rx.recv_timeout(std::time::Duration::from_secs(5));
    assert!(matches!(
        index.insert(100, &[1.; 4]),
        Err(UpdateError::WriterBusy)
    ));
    assert!(matches!(index.delete(1), Err(UpdateError::WriterBusy)));
    assert!(matches!(
        index.replace(1, &[2.; 4]),
        Err(UpdateError::WriterBusy)
    ));
    assert!(matches!(index.commit(), Err(UpdateError::WriterBusy)));
    resume_tx.send(()).unwrap();
    writer.join().unwrap().unwrap();
    reader.join().unwrap();
    assert!(!result.unwrap().contains(&0));
}

#[test]
fn streaming_crash_child() {
    let Some(path) = std::env::var_os("HMS_TEST_CRASH_DIR") else {
        return;
    };
    let path = PathBuf::from(path);
    let stop: u8 = std::env::var("HMS_TEST_CRASH_STEP")
        .unwrap()
        .parse()
        .unwrap();
    let mode = std::env::var("HMS_TEST_CRASH_MODE").unwrap();
    let index = UpdatableVGraph::open(&path).unwrap();
    index.replace(1, &[9.; 4]).unwrap();
    index.delete(2).unwrap();
    index.insert(100, &[8.; 4]).unwrap();
    let checkpoint = |point| {
        if point == stop {
            std::fs::write(path.join("kill-ready"), [point]).unwrap();
            loop {
                std::thread::park();
            }
        }
        Ok(())
    };
    if mode == "commit" {
        let writer = index.writer().unwrap();
        index.commit_with_checkpoints(&writer, checkpoint).unwrap();
    } else {
        index.commit().unwrap();
        index.consolidate_inner(checkpoint).unwrap();
    }
    panic!("crash checkpoint was not reached");
}

#[test]
fn killed_commit_and_consolidation_recover_complete_generations() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source");
    let rows = data(16, 4);
    drop(UpdatableVGraph::create(&source, &rows, 4, &params()).unwrap());
    let initial: BTreeMap<_, _> = rows
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(id, row)| (id as u32, row.to_vec()))
        .collect();
    let mut updated = initial.clone();
    updated.insert(1, vec![9.; 4]);
    updated.remove(&2);
    updated.insert(100, vec![8.; 4]);
    for (mode, points) in [("commit", 4), ("consolidate", 11)] {
        for point in 0..points {
            let path = root.path().join(format!("{mode}-{point}"));
            copy_store(&source, &path);
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "core::qgraph::update::tests::streaming_crash_child",
                    "--nocapture",
                ])
                .env("HMS_TEST_CRASH_DIR", &path)
                .env("HMS_TEST_CRASH_STEP", point.to_string())
                .env("HMS_TEST_CRASH_MODE", mode)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while !path.join("kill-ready").exists() {
                if let Some(status) = child.try_wait().unwrap() {
                    let output = child.wait_with_output().unwrap();
                    panic!(
                        "child exited {status}: {} {}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                if std::time::Instant::now() >= deadline {
                    child.kill().unwrap();
                    let output = child.wait_with_output().unwrap();
                    panic!(
                        "child did not reach {mode} point {point}: {}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            child.kill().unwrap();
            child.wait().unwrap();
            for _ in 0..2 {
                let reopened = UpdatableVGraph::open(&path).unwrap();
                let state = reopened.state.read();
                let mut actual = BTreeMap::new();
                for &id in initial.keys().chain(std::iter::once(&100)) {
                    if let Some(&slot) = state.delta.live.get(&id) {
                        actual.insert(id, state.delta.row(slot, 4).to_vec());
                    } else if let Some(slot) = state.base_slot(id) {
                        actual.insert(id, state.base.rows.row(slot).unwrap().to_vec());
                    }
                }
                state.delta.assert_invariants(state.manifest.build.degree);
                drop(state);
                if mode == "consolidate" || point >= 2 {
                    assert_eq!(actual, updated);
                } else if point == 0 {
                    assert_eq!(actual, initial);
                } else {
                    assert!(actual == initial || actual == updated);
                }
                let query = [0.31, -0.47, 0.63, 0.19];
                assert_eq!(
                    reopened.search(&query, 100, search_params()).unwrap(),
                    exact(&actual, &query, 100)
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn random_updates_match_exact_and_reopen(operations in prop::collection::vec((0u8..3, 0u32..32, prop::array::uniform4(-100i16..100)), 1..40)) {
        let dir = tempfile::tempdir().unwrap(); let path = dir.path().join("index"); let initial = data(16,4);
        let mut live: BTreeMap<_,_> = initial.as_chunks::<4>().0.iter().enumerate().map(|(id,v)| (id as u32,v.to_vec())).collect();
        let mut index = UpdatableVGraph::create(&path, &initial, 4, &params()).unwrap();
        for (tag,id,values) in operations {
            let row: Vec<_> = values.iter().map(|&v| f32::from(v) / 100.).collect();
            match tag {
                0 if !live.contains_key(&id) => { index.insert(id,&row).unwrap(); live.insert(id,row); },
                1 if live.contains_key(&id) => { index.delete(id).unwrap(); live.remove(&id); },
                2 if live.contains_key(&id) => { index.replace(id,&row).unwrap(); live.insert(id,row); },
                _ => continue,
            }
            let query = [0.31, -0.47, 0.63, 0.19];
            let expected = exact(&live, &query, 10);
            let actual = index.search(&query, 10, search_params()).unwrap();
            prop_assert!(actual.iter().all(|id| live.contains_key(id)));
            let recall = actual.iter().filter(|id| expected.contains(id)).count();
            let rows: Vec<_> = live.values().flatten().copied().collect();
            if !rows.is_empty() {
                let graph = Graph::build(&rows,4,&params()).unwrap(); let vertex = VGraph::from_graph(&graph,true).unwrap();
                let ids: Vec<_> = live.keys().copied().collect(); let mut baseline = Vec::new(); vertex.searcher().search(&query,10,search_params(),&mut baseline).unwrap();
                let base_recall = baseline.iter().filter(|&&slot| expected.contains(&ids[slot as usize])).count();
                prop_assert!(recall >= base_recall, "recall {} < baseline {}", recall, base_recall);
            }
            index.commit().unwrap(); drop(index); index = UpdatableVGraph::open(&path).unwrap();
            prop_assert_eq!(index.search(&query,10,search_params()).unwrap(), actual);
        }
    }
}

#[test]
fn consolidation_recall_on_50k_and_physical_deletion() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let rows = data(50_000, 16);
    let build = BuildParams {
        degree: 32,
        build_ef: 64,
        ..params()
    };
    let index = UpdatableVGraph::create(&path, &rows, 16, &build).unwrap();
    for id in 0..500 {
        index.delete(id).unwrap();
    }
    for id in 500..1000 {
        index
            .replace(
                id,
                &rows[(id as usize + 1000) * 16..(id as usize + 1001) * 16],
            )
            .unwrap();
    }
    index.commit().unwrap();
    let queries = data(100, 16);
    let p = VSearchParams {
        ef: 1024,
        rerank: 1024,
        ..Default::default()
    };
    let before: Vec<_> = queries
        .as_chunks::<16>()
        .0
        .iter()
        .map(|q| index.search(q, 10, p).unwrap())
        .collect();
    let live: BTreeMap<_, _> = (500..50000u32)
        .map(|id| {
            let slot = if id < 1000 {
                id as usize + 1000
            } else {
                id as usize
            };
            (id, rows[slot * 16..(slot + 1) * 16].to_vec())
        })
        .collect();
    let truth: Vec<_> = queries
        .as_chunks::<16>()
        .0
        .iter()
        .map(|q| exact(&live, q, 10))
        .collect();
    index.consolidate().unwrap();
    assert!(index.state.read().base.rows.ids.iter().all(|&id| id >= 500));
    assert_eq!(index.state.read().base.graph.as_ref().unwrap().len(), 49500);
    let after: Vec<_> = queries
        .as_chunks::<16>()
        .0
        .iter()
        .map(|q| index.search(q, 10, p).unwrap())
        .collect();
    let recall = |results: &[Vec<u32>]| {
        results
            .iter()
            .zip(&truth)
            .map(|(r, t)| r.iter().filter(|id| t.contains(id)).count())
            .sum::<usize>() as f64
            / 1000.
    };
    let before_recall = recall(&before);
    let after_recall = recall(&after);
    assert!(
        (before_recall - after_recall).abs() <= 0.002,
        "before={before_recall}, after={after_recall}"
    );
    drop(index);
    let reopened = UpdatableVGraph::open(&path).unwrap();
    for (q, result) in queries.as_chunks::<16>().0.iter().zip(after) {
        assert_eq!(reopened.search(q, 10, p).unwrap(), result);
    }
}

#[test]
fn vector_format_golden_and_drift_gate() {
    assert_eq!(crate::core::format::VECTOR_UPDATE_FORMAT_VERSION, 1);
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index");
    let index =
        UpdatableVGraph::create(&path, &[0., 0., 1., 0., 0., 1., 2., 2.], 2, &params()).unwrap();
    {
        let mut state = index.state.write();
        state.manifest.identity = [0x16; 16];
        let manifest = state.manifest.clone();
        drop(state);
        atomic_write(&path.join("CURRENT"), &manifest.encode()).unwrap();
        format::rotate(&path.join("updates.log"), &manifest).unwrap();
    }
    index.insert(8, &[4., 4.]).unwrap();
    index.replace(1, &[3., 3.]).unwrap();
    index.delete(2).unwrap();
    index.commit().unwrap();
    // `consolidated` holds a generation written with 4-byte neighbour ids, before
    // consolidation chose 3-byte ids below 2^24 - 1 rows; it must still open and search the
    // same. Consolidation now writes `consolidated-3-byte-ids`.
    for phase in ["logged", "consolidated-3-byte-ids"] {
        if phase != "logged" {
            index.consolidate().unwrap();
        }
        let golden = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/golden/vector-updates-v1")
            .join(phase);
        let state = index.state.read();
        let manifest = state.manifest.clone();
        drop(state);
        for file in [
            PathBuf::from("CURRENT"),
            PathBuf::from("updates.log"),
            manifest.rows_path(Path::new("")),
            manifest.vertex_path(Path::new("")),
        ] {
            let generated = std::fs::read(path.join(&file)).unwrap();
            if std::env::var_os("HMS_GENERATE_VECTOR_GOLDEN").is_some() {
                std::fs::create_dir_all(&golden).unwrap();
                std::fs::write(golden.join(&file), &generated).unwrap();
            }
            assert_eq!(
                std::fs::read(golden.join(&file)).unwrap(),
                generated,
                "vector format drift: {}",
                file.display()
            );
        }
        let copy = root.path().join(phase);
        copy_store(&golden, &copy);
        let reader = UpdatableVGraph::open(&copy).unwrap();
        assert_eq!(
            reader.search(&[0., 1.], 10, search_params()).unwrap(),
            [0, 3, 1, 8]
        );
    }
    let golden =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/vector-updates-v1/consolidated");
    let copy = root.path().join("consolidated-4-byte-ids");
    copy_store(&golden, &copy);
    let reader = UpdatableVGraph::open(&copy).unwrap();
    assert_eq!(reader.state.read().manifest.encoding.id_bytes, 4);
    assert_eq!(
        reader.search(&[0., 1.], 10, search_params()).unwrap(),
        [0, 3, 1, 8]
    );
}

#[test]
fn untrusted_headers_and_rows_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");
    let index = UpdatableVGraph::create(&source, &[1., 2., 3., 4.], 2, &params()).unwrap();
    let manifest = index.state.read().manifest.clone();
    drop(index);
    for name in ["CURRENT", "rows-0.bin", "updates.log"] {
        let original = std::fs::read(source.join(name)).unwrap();
        for len in [8, original.len(), original.len() + 1] {
            let copy = dir.path().join(format!("future-{name}-{len}"));
            copy_store(&source, &copy);
            let mut bytes = original.clone();
            bytes[7] = b'2';
            bytes.resize(len, 0);
            std::fs::write(copy.join(name), bytes).unwrap();
            assert!(matches!(
                UpdatableVGraph::open(&copy),
                Err(UpdateError::UnsupportedFormat(_))
            ));
        }
    }
    for (name, at) in [
        ("CURRENT", 32usize),
        ("rows-0.bin", 64),
        ("rows-0.bin", 68),
        ("updates.log", 16),
    ] {
        let copy = dir.path().join(format!("{}-{at}", name.replace('.', "-")));
        copy_store(&source, &copy);
        let file = copy.join(name);
        let mut bytes = std::fs::read(&file).unwrap();
        bytes[at] ^= 2;
        std::fs::write(&file, bytes).unwrap();
        assert!(UpdatableVGraph::open(&copy).is_err());
    }
    let file = manifest.rows_path(&source);
    let mut bytes = std::fs::read(&file).unwrap();
    bytes[128] ^= 1;
    std::fs::write(&file, bytes).unwrap();
    let index = UpdatableVGraph::open(&source).unwrap();
    index.insert(3, &[1., 1.]).unwrap();
    assert!(matches!(
        index.search(&[1., 1.], 10, search_params()),
        Err(UpdateError::CorruptBase(_))
    ));
}

#[test]
fn log_and_dimension_limit_boundaries() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source");
    let index = UpdatableVGraph::create(&source, &[], 2, &params()).unwrap();
    let manifest = index.state.read().manifest.clone();
    drop(index);
    for len in [163, 164, 165, 1u64 << 32] {
        let path = root.path().join(format!("manifest-length-{len}"));
        copy_store(&source, &path);
        OpenOptions::new()
            .write(true)
            .open(path.join("CURRENT"))
            .unwrap()
            .set_len(len)
            .unwrap();
        assert_eq!(UpdatableVGraph::open(&path).is_ok(), len == 164);
    }
    for len in [8, 164, 165] {
        let path = root.path().join(format!("future-manifest-{len}"));
        copy_store(&source, &path);
        let mut bytes = manifest.encode();
        bytes[7] = b'2';
        bytes.resize(len, 0);
        assert!(matches!(
            Manifest::decode(&bytes),
            Err(UpdateError::UnsupportedFormat(_))
        ));
        std::fs::write(path.join("CURRENT"), bytes).unwrap();
        assert!(matches!(
            UpdatableVGraph::open(&path),
            Err(UpdateError::UnsupportedFormat(_))
        ));
    }
    for dim in [MAX_DIM - 1, MAX_DIM, MAX_DIM + 1] {
        let mut manifest = manifest.clone();
        manifest.dim = dim;
        assert_eq!(Manifest::decode(&manifest.encode()).is_ok(), dim <= MAX_DIM);
    }
    for len in [MAX_PAYLOAD - 1, MAX_PAYLOAD, MAX_PAYLOAD + 1] {
        let path = root.path().join(format!("length-{len}"));
        copy_store(&source, &path);
        let mut header = Vec::new();
        header.extend_from_slice(&(len as u32).to_le_bytes());
        header.extend_from_slice(&(!(len as u32)).to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        header.extend_from_slice(&1u32.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        OpenOptions::new()
            .append(true)
            .open(path.join("updates.log"))
            .unwrap()
            .write_all(&header)
            .unwrap();
        assert_eq!(UpdatableVGraph::open(&path).is_ok(), len <= MAX_PAYLOAD);
    }
    for count in [MAX_OPS - 1, MAX_OPS, MAX_OPS + 1] {
        let operations: Vec<_> = (0..count as u32).map(Operation::Delete).collect();
        assert_eq!(format::frame(&operations, 1, 2).is_ok(), count <= MAX_OPS);
    }
}

#[test]
fn angular_updates_reopen_and_torn_unpublished_graph() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index");
    let build = BuildParams::default();
    let rows = data(64, 4);
    let index = UpdatableVGraph::create(&path, &rows, 4, &build).unwrap();
    for id in 0..8 {
        index.replace(id, &[1., 2., 3., id as f32 + 1.]).unwrap();
    }
    index.delete(9).unwrap();
    index.commit().unwrap();
    let expected = index
        .search(&[1., 2., 3., 4.], 10, search_params())
        .unwrap();
    assert!(index
        .needs_consolidation(ConsolidationPolicy {
            min_delta: 1,
            ..Default::default()
        })
        .unwrap());
    assert!(index
        .consolidate_inner(|step| if step == 1 {
            Err(UpdateError::Capacity)
        } else {
            Ok(())
        })
        .is_err());
    let mut manifest = index.state.read().manifest.clone();
    manifest.sequence = 1;
    let unpublished = manifest.vertex_path(&path);
    let length = std::fs::metadata(&unpublished).unwrap().len();
    drop(index);
    for cut in [0, 1, 399, length / 2, length - 1] {
        let copy = root.path().join(format!("torn-graph-{cut}"));
        copy_store(&path, &copy);
        OpenOptions::new()
            .write(true)
            .open(copy.join(unpublished.file_name().unwrap()))
            .unwrap()
            .set_len(cut)
            .unwrap();
        let reader = UpdatableVGraph::open(&copy).unwrap();
        assert_eq!(
            reader
                .search(&[1., 2., 3., 4.], 10, search_params())
                .unwrap(),
            expected
        );
    }
}

#[test]
fn uncertain_append_requires_reopen_and_writer_is_exclusive() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index");
    let index = UpdatableVGraph::create(&path, &[1., 2.], 2, &params()).unwrap();
    let writer = index.writer().unwrap();
    assert!(matches!(
        index.insert(2, &[3., 4.]),
        Err(UpdateError::WriterBusy)
    ));
    drop(writer);
    index.insert(2, &[3., 4.]).unwrap();
    std::fs::rename(path.join("updates.log"), path.join("saved-log")).unwrap();
    assert!(index.commit().is_err());
    assert!(matches!(index.commit(), Err(UpdateError::RecoveryRequired)));
    std::fs::rename(path.join("saved-log"), path.join("updates.log")).unwrap();
    drop(index);
    let reopened = UpdatableVGraph::open(&path).unwrap();
    assert_eq!(reopened.len(), 1);
    reopened.insert(2, &[3., 4.]).unwrap();
    reopened.commit().unwrap();
}

#[test]
fn reordered_compressed_base_preserves_external_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let rows = data(64, 8);
    let graph = Graph::build(&rows, 8, &params()).unwrap();
    let vertex = VGraph::from_graph_with(&graph, 8, 8, true, true, 3, false).unwrap();
    let index = UpdatableVGraph::create_from_graph(&path, &graph, &vertex).unwrap();
    let query = [0.7; 8];
    let mut baseline = Vec::new();
    vertex
        .searcher()
        .search(&query, 10, search_params(), &mut baseline)
        .unwrap();
    assert_eq!(index.search(&query, 10, search_params()).unwrap(), baseline);
    index.replace(5, &[0.3; 8]).unwrap();
    index.delete(4).unwrap();
    index.insert(1000, &[0.5; 8]).unwrap();
    let result = index.search(&query, 100, search_params()).unwrap();
    assert!(!result.contains(&4));
    assert_eq!(result.len(), 64);
    assert!(result.contains(&1000));
    index.commit().unwrap();
    drop(index);
    let index = UpdatableVGraph::open(&path).unwrap();
    assert_eq!(index.search(&query, 100, search_params()).unwrap(), result);
    index.consolidate().unwrap();
    assert!(!index.state.read().base.rows.ids.contains(&4));
    let all = index.search(&query, 100, search_params()).unwrap();
    assert_eq!(all.len(), 64);
    assert!(!all.contains(&4));
    assert!(all.contains(&1000));
}

#[test]
fn mixed_search_reproduces_pinned_results_and_encoded_generations() {
    let root = tempfile::tempdir().unwrap();
    let mut fingerprint = crc32fast::Hasher::new();
    for metric in [Metric::Angular, Metric::Euclidean] {
        let build = BuildParams { metric, ..params() };
        let dim = 16;
        let graph = Graph::build(&data(192, dim), dim, &build).unwrap();
        let mut encodings = vec![
            (8, 0, false, false, 4, false, 0),
            (4, 4, true, true, 3, false, 0),
            (4, 8, true, false, 0, false, 0),
            (8, 8, true, true, 3, false, 0),
        ];
        if metric == Metric::Angular {
            encodings.extend([
                (8, 8, true, true, 3, true, 0),
                (8, 8, true, true, 3, false, 4),
                (8, 8, true, false, 4, false, 2),
            ]);
        }
        for (case, (bits, residual, align, reorder, ids, screen, blocks)) in
            encodings.into_iter().enumerate()
        {
            let mut vertex =
                VGraph::from_graph_with(&graph, bits, residual, align, reorder, ids, screen)
                    .unwrap();
            if blocks != 0 {
                vertex = vertex.with_blocks(&graph, blocks).unwrap();
            }
            let path = root.path().join(format!("{metric:?}-{case}"));
            let mut index = UpdatableVGraph::create_from_graph(&path, &graph, &vertex).unwrap();
            let mut live: BTreeMap<_, _> = graph
                .vectors
                .chunks_exact(dim)
                .enumerate()
                .map(|(id, row)| (id as u32, row.to_vec()))
                .collect();
            let mut queries = data(12, dim);
            queries.extend_from_slice(&[0.5; 16]);
            for phase in 0..4 {
                if phase == 1 {
                    for id in (0..192).step_by(3) {
                        index.delete(id).unwrap();
                        live.remove(&id);
                    }
                    let tie = [0.5; 16];
                    index.replace(1, &tie).unwrap();
                    index.insert(10_000, &tie).unwrap();
                    index.insert(5_000, &tie).unwrap();
                    let mut prepared = tie.to_vec();
                    super::super::prepare_rows(&mut prepared, dim, metric);
                    for id in [1, 10_000, 5_000] {
                        live.insert(id, prepared.clone());
                    }
                    let ids: Vec<_> = (1_000..1_032).rev().collect();
                    let mut rows = data(ids.len(), dim);
                    index.insert_batch(&ids, &rows).unwrap();
                    super::super::prepare_rows(&mut rows, dim, metric);
                    live.extend(ids.into_iter().zip(rows.chunks_exact(dim).map(Vec::from)));
                    index.commit().unwrap();
                } else if phase == 2 {
                    drop(index);
                    index = UpdatableVGraph::open(&path).unwrap();
                } else if phase == 3 {
                    index.consolidate().unwrap();
                }
                if phase == 0 || phase == 3 {
                    let state = index.state.read();
                    fingerprint.update(&std::fs::read(state.manifest.vertex_path(&path)).unwrap());
                }
                for q in queries.chunks_exact(dim) {
                    for ef in [7, 16, 63, 64, 65, 256] {
                        for rerank in [0, ef * 2] {
                            let p = VSearchParams {
                                ef,
                                rerank,
                                patience: 5,
                                fuse: blocks != 0,
                                ..Default::default()
                            };
                            let found = index.search(q, 10, p).unwrap();
                            for &id in &found {
                                fingerprint.update(&id.to_le_bytes());
                            }
                            if (phase == 1 || phase == 2) && ef == 256 {
                                let mut truth: Vec<_> = live
                                    .iter()
                                    .map(|(&id, row)| (distance(q, row, metric), id))
                                    .collect();
                                truth.sort_unstable_by(|a, b| {
                                    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
                                });
                                assert_eq!(
                                    found,
                                    truth.iter().take(10).map(|x| x.1).collect::<Vec<_>>()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(fingerprint.finalize(), 1_364_000_328);
}

#[test]
fn incomplete_streamed_rows_preserve_the_previous_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let index = UpdatableVGraph::create(&path, &data(3, 4), 4, &params()).unwrap();
    let manifest = index.state.read().manifest.clone();
    let rows_path = manifest.rows_path(&path);
    let before = std::fs::read(&rows_path).unwrap();
    for count in [2, 3, 4] {
        let mut writer = BaseRowsWriter::new(&rows_path, &manifest, &[0, 1, 2]).unwrap();
        let append = writer.append(&vec![1.; count * 4]);
        if count == 4 {
            assert!(append.is_err());
        } else if count == 2 {
            append.unwrap();
            assert!(writer.finish().is_err());
        } else {
            append.unwrap();
            drop(writer);
        }
        assert_eq!(std::fs::read(&rows_path).unwrap(), before);
    }
}

#[test]
fn clustered_deletes_retry_without_global_tombstone_budget() {
    assert_eq!(widened_width(64, 1_000_000, 990_000), 65);
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index");
    let rows = data(64, 4);
    let index = UpdatableVGraph::create(&path, &rows, 4, &params()).unwrap();
    let live: BTreeMap<_, _> = rows
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(id, row)| (id as u32, row.to_vec()))
        .collect();
    let query = [0.4; 4];
    let all = exact(&live, &query, 64);
    for &id in &all[..32] {
        index.delete(id).unwrap();
    }
    let p = VSearchParams {
        ef: 4,
        rerank: 4,
        ..Default::default()
    };
    let result = index.search(&query, 5, p).unwrap();
    assert_eq!(result, all[32..37]);
    index.commit().unwrap();
    drop(index);
    assert_eq!(
        UpdatableVGraph::open(&path)
            .unwrap()
            .search(&query, 5, p)
            .unwrap(),
        result
    );
}

#[test]
fn scratch_reuse_distinguishes_dimensions_with_equal_padding() {
    let root = tempfile::tempdir().unwrap();
    let graph = Graph::build(&data(128, 31), 31, &params()).unwrap();
    let vertex = VGraph::from_graph(&graph, true).unwrap();
    let narrow =
        UpdatableVGraph::create_from_graph(&root.path().join("narrow"), &graph, &vertex).unwrap();
    let wide =
        UpdatableVGraph::create(&root.path().join("wide"), &data(128, 32), 32, &params()).unwrap();
    let p = VSearchParams {
        ef: 16,
        rerank: 16,
        ..Default::default()
    };
    let mut query = [1.; 32];
    query[31] = 1000.;
    wide.search(&query, 10, p).unwrap();
    let mut expected = Vec::new();
    vertex
        .searcher()
        .search(&[1.; 31], 10, p, &mut expected)
        .unwrap();
    assert_eq!(narrow.search(&[1.; 31], 10, p).unwrap(), expected);
}

#[test]
fn batch_insert_is_atomic_reachable_and_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let dim = 16;
    let n = 3000;
    let rows = data(n, dim);
    let index = UpdatableVGraph::create(&path, &[], dim, &params()).unwrap();
    let ids: Vec<u32> = (0..n as u32).map(|i| i * 3).collect();
    // A duplicate rejects the whole batch.
    let mut bad = ids[..10].to_vec();
    bad[9] = bad[0];
    assert!(matches!(
        index.insert_batch(&bad, &rows[..10 * dim]),
        Err(UpdateError::DuplicateIdInBatch(_))
    ));
    assert_eq!(index.len(), 0);
    for (chunk, rows) in ids.chunks(700).zip(rows.chunks(700 * dim)) {
        index.insert_batch(chunk, rows).unwrap();
        index.commit().unwrap();
    }
    for &id in ids.iter().step_by(3) {
        index.delete(id).unwrap();
    }
    index.commit().unwrap();
    let live: BTreeMap<u32, Vec<f32>> = ids
        .iter()
        .zip(rows.chunks_exact(dim))
        .enumerate()
        .filter(|(i, _)| i % 3 != 0)
        .map(|(_, (&id, row))| (id, row.to_vec()))
        .collect();
    let check = |index: &UpdatableVGraph| {
        let params = VSearchParams {
            ef: 64,
            rerank: 64,
            ..Default::default()
        };
        let mut hits = 0;
        for (&id, row) in &live {
            let found = index.search(row, 10, params).unwrap();
            assert_eq!(found[0], id, "live id {id} is not its own nearest");
            let want = exact(&live, row, 10);
            hits += found.iter().filter(|f| want.contains(f)).count();
        }
        let recall = hits as f64 / (live.len() * 10) as f64;
        assert!(recall > 0.95, "recall@10 {recall}");
    };
    check(&index);
    drop(index);
    check(&UpdatableVGraph::open(&path).unwrap());
}

#[test]
fn consolidated_base_matches_static_build_at_equal_params() {
    // The base a consolidation rebuilds is the graph the static index builds from the same rows
    // and parameters, so the streaming path serves the static index's recall at every search
    // budget instead of drifting through build defaults of its own.
    let dir = tempfile::tempdir().unwrap();
    let dim = 16;
    let n = 4000;
    let all = data(n + 200, dim);
    let (rows, queries) = all.split_at(n * dim);
    let build = params();
    let index = UpdatableVGraph::create(&dir.path().join("index"), &[], dim, &build).unwrap();
    let ids: Vec<u32> = (0..n as u32).collect();
    for (chunk, rows) in ids.chunks(1000).zip(rows.chunks(1000 * dim)) {
        index.insert_batch(chunk, rows).unwrap();
    }
    index.consolidate().unwrap();
    let graph = Graph::build(rows, dim, &build).unwrap();
    let encoding = Encoding::default();
    let fixed = VGraph::from_graph_with(
        &graph,
        encoding.bits,
        encoding.residual_bits,
        encoding.align,
        encoding.reorder,
        encoding.id_bytes,
        encoding.screen,
    )
    .unwrap();
    let mut searcher = fixed.searcher();
    let search = VSearchParams {
        ef: 48,
        rerank: 48,
        ..Default::default()
    };
    let mut out = Vec::new();
    for query in queries.chunks_exact(dim) {
        searcher.search(query, 10, search, &mut out).unwrap();
        assert_eq!(index.search(query, 10, search).unwrap(), out);
    }
}

#[test]
fn consolidated_id_width_switches_at_the_three_byte_limit() {
    let limit = (1 << 24) - 1;
    for configured in [3, 4] {
        assert_eq!(neighbour_id_bytes(configured, limit - 1), 3);
        assert_eq!(neighbour_id_bytes(configured, limit), 4);
        assert_eq!(neighbour_id_bytes(configured, limit + 1), 4);
    }
    assert_eq!(neighbour_id_bytes(0, 10), 0);
    assert_eq!(neighbour_id_bytes(0, limit + 1), 0);
}

#[test]
fn merged_consolidation_tracks_rebuild_recall_across_churn() {
    // The same delete/insert schedule consolidated by merging into the previous base and by
    // rebuilding: the merged graph must hold the rebuilt graph's recall at a fixed search
    // budget, serve only live ids, and persist.
    let dim = 16;
    let all = data(16_000, dim);
    let (initial, rest) = all.split_at(8000 * dim);
    let dir = tempfile::tempdir().unwrap();
    let merged =
        UpdatableVGraph::create(&dir.path().join("merged"), initial, dim, &params()).unwrap();
    let rebuilt =
        UpdatableVGraph::create(&dir.path().join("rebuilt"), initial, dim, &params()).unwrap();
    let mut live: BTreeMap<u32, Vec<f32>> = initial
        .chunks_exact(dim)
        .enumerate()
        .map(|(id, row)| (id as u32, row.to_vec()))
        .collect();
    let queries = data(200, dim + 1);
    let budget = VSearchParams {
        ef: 24,
        rerank: 24,
        ..Default::default()
    };
    let mut next = 8000u32;
    for round in 0..4 {
        let doomed: Vec<u32> = live.keys().copied().filter(|id| id % 7 == round).collect();
        let fresh: Vec<u32> = (next..next + 1500).collect();
        let rows = &rest[(next as usize - 8000) * dim..(next as usize - 8000 + 1500) * dim];
        next += 1500;
        for index in [&merged, &rebuilt] {
            for &id in &doomed {
                index.delete(id).unwrap();
            }
            index.commit().unwrap();
            index.insert_batch(&fresh, rows).unwrap();
            index.commit().unwrap();
        }
        for id in &doomed {
            live.remove(id);
        }
        live.extend(
            fresh
                .iter()
                .zip(rows.chunks_exact(dim))
                .map(|(&id, r)| (id, r.to_vec())),
        );
        let mode = ConsolidationMode::Merge {
            rebuild_after: f64::INFINITY,
        };
        assert_eq!(
            merged.consolidate_with(mode).unwrap(),
            Consolidation::Merged
        );
        rebuilt.consolidate().unwrap();
        assert_eq!(merged.len(), live.len());
        let mut hits = [0usize; 2];
        for q in queries.chunks_exact(dim + 1).map(|q| &q[..dim]) {
            let truth = exact(&live, q, 10);
            for (hit, index) in hits.iter_mut().zip([&merged, &rebuilt]) {
                let found = index.search(q, 10, budget).unwrap();
                assert!(found.iter().all(|id| live.contains_key(id)));
                *hit += found.iter().filter(|id| truth.contains(id)).count();
            }
        }
        let [m, r] = hits.map(|h| h as f64 / 2000.0);
        assert!(m >= r - 0.02, "round {round}: merged {m}, rebuilt {r}");
    }
    let before: Vec<_> = queries
        .chunks_exact(dim + 1)
        .map(|q| merged.search(&q[..dim], 10, budget).unwrap())
        .collect();
    drop(merged);
    let reopened = UpdatableVGraph::open(&dir.path().join("merged")).unwrap();
    for (q, found) in queries.chunks_exact(dim + 1).zip(before) {
        assert_eq!(reopened.search(&q[..dim], 10, budget).unwrap(), found);
    }
    reopened.delete(next - 1).unwrap();
    reopened.commit().unwrap();
    let mode = ConsolidationMode::Merge { rebuild_after: 0.0 };
    assert_eq!(
        reopened.consolidate_with(mode).unwrap(),
        Consolidation::Rebuilt
    );
}

#[test]
fn merged_consolidation_interruption_recovers_committed_state() {
    let dim = 4;
    let mode = ConsolidationMode::Merge {
        rebuild_after: f64::INFINITY,
    };
    for stop in 0..=10 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index");
        let index = UpdatableVGraph::create(&path, &data(600, dim), dim, &params()).unwrap();
        index.replace(0, &[1., 2., 3., 4.]).unwrap();
        for id in 1..40 {
            index.delete(id).unwrap();
        }
        index.insert(1000, &[4., 3., 2., 1.]).unwrap();
        index.commit().unwrap();
        let before = index.search(&[1.; 4], 100, search_params()).unwrap();
        assert!(index
            .consolidate_mode(mode, |step| if step == stop {
                Err(UpdateError::Io(std::io::Error::other("interruption")))
            } else {
                Ok(())
            })
            .is_err());
        drop(index);
        let reopened = UpdatableVGraph::open(&path).unwrap();
        let found = reopened.search(&[1.; 4], 100, search_params()).unwrap();
        let published = reopened.state.read().base.rows.ids.len() != 600;
        if published {
            // A published generation scores its base through codes: compare the sets.
            assert!(found.iter().filter(|id| before.contains(id)).count() >= 95);
        } else {
            assert_eq!(found, before, "stop {stop}");
        }
        let expected = if published {
            Consolidation::Unchanged
        } else {
            Consolidation::Merged
        };
        assert_eq!(
            reopened.consolidate_with(mode).unwrap(),
            expected,
            "stop {stop}"
        );
        let after = reopened.search(&[1.; 4], 100, search_params()).unwrap();
        let shared = after.iter().filter(|id| before.contains(id)).count();
        assert!(shared >= 95, "stop {stop}: {shared} of 100");
        drop(reopened);
        assert_eq!(UpdatableVGraph::open(&path).unwrap().len(), 600 - 39 + 1);
    }
}

#[test]
fn interrupted_consolidation_keeps_spilled_rows_searchable_and_writable() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("index");
    let index = UpdatableVGraph::create(&path, &data(64, 4), 4, &params()).unwrap();
    index
        .insert_batch(&(100..132).collect::<Vec<_>>(), &data(32, 4))
        .unwrap();
    let query = [0.5; 4];
    let before = index.search(&query, 100, search_params()).unwrap();
    assert!(index
        .consolidate_inner(|step| {
            if step == 6 {
                Err(UpdateError::Io(std::io::Error::other("interruption")))
            } else {
                Ok(())
            }
        })
        .is_err());
    assert!(!index.state.read().delta.vectors.is_empty());
    assert!(index.state.read().delta.vectors.spill_source().is_none());
    assert_eq!(index.search(&query, 100, search_params()).unwrap(), before);
    index.insert(200, &[1.0; 4]).unwrap();
    index.commit().unwrap();
    let after = index.search(&query, 100, search_params()).unwrap();
    drop(index);
    let reopened = UpdatableVGraph::open(&path).unwrap();
    assert_eq!(
        reopened.search(&query, 100, search_params()).unwrap(),
        after
    );
    reopened.consolidate().unwrap();
    assert_eq!(reopened.len(), 97);
}

#[test]
fn merged_deleted_entry_preserves_the_pinned_generation() {
    let dim = 16;
    let rows = data(1200, dim);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let index = UpdatableVGraph::create(&path, &rows[..1024 * dim], dim, &params()).unwrap();
    let entry = index
        .state
        .read()
        .base
        .graph
        .as_ref()
        .unwrap()
        .entry_vertex();
    let doomed: Vec<u32> = (0..1024).filter(|&id| id % 7 == 0 || id == entry).collect();
    index.delete_batch(&doomed).unwrap();
    index
        .insert_batch(&(1024..1200).collect::<Vec<_>>(), &rows[1024 * dim..])
        .unwrap();
    assert_eq!(
        index
            .consolidate_with(ConsolidationMode::Merge {
                rebuild_after: 10.0
            })
            .unwrap(),
        Consolidation::Merged
    );
    let state = index.state.read();
    let bytes = std::fs::read(state.manifest.vertex_path(&path)).unwrap();
    assert_eq!(crc32fast::hash(&bytes), 832_240_014);
}

#[test]
#[ignore = "development consolidation memory and timing measurement"]
fn consolidation_live_set_peak() {
    super::super::prof::HOOK
        .set(|phase| {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status
                    .lines()
                    .filter(|s| s.starts_with("VmRSS:") || s.starts_with("VmHWM:"))
                {
                    println!("MEMORY {phase} {line}");
                }
            }
        })
        .unwrap();
    let n: usize = std::env::var("HMS_CONSOLIDATION_ROWS")
        .map_or(Ok(100_000), |s| s.parse())
        .unwrap();
    assert!((1024..=3_000_000).contains(&n));
    let dim = 100;
    let rows = data(n + n / 5, dim);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let index = UpdatableVGraph::create(&path, &rows[..n * dim], dim, &params()).unwrap();
    let entry = index
        .state
        .read()
        .base
        .graph
        .as_ref()
        .unwrap()
        .entry_vertex();
    let doomed: Vec<u32> = (0..n as u32)
        .filter(|&id| id % 7 == 0 && id != entry)
        .collect();
    let capacity = index.batch_capacity();
    for ids in doomed.chunks(capacity) {
        index.delete_batch(ids).unwrap();
        index.commit().unwrap();
    }
    let fresh: Vec<u32> = (n as u32..(n + n / 5) as u32).collect();
    for ids in fresh.chunks(capacity) {
        let begin = ids[0] as usize * dim;
        index
            .insert_batch(ids, &rows[begin..begin + ids.len() * dim])
            .unwrap();
        index.commit().unwrap();
    }
    drop(rows);
    if cfg!(target_os = "linux") {
        std::fs::write("/proc/self/clear_refs", b"5").unwrap();
    }
    let start = std::time::Instant::now();
    assert_eq!(
        index
            .consolidate_with(ConsolidationMode::Merge {
                rebuild_after: 10.0
            })
            .unwrap(),
        Consolidation::Merged
    );
    let elapsed = start.elapsed().as_nanos();
    if cfg!(target_os = "linux") {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let peak = status
            .lines()
            .find(|line| line.starts_with("VmHWM:"))
            .unwrap();
        println!(
            "consolidation_peak_rss_kib={}",
            peak.split_whitespace().nth(1).unwrap()
        );
    }
    let state = index.state.read();
    let bytes = std::fs::read(state.manifest.vertex_path(&path)).unwrap();
    println!(
        "consolidation_ns={elapsed} vertex_crc={}",
        crc32fast::hash(&bytes)
    );
}
