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
    resume_tx.send(()).unwrap();
    writer.join().unwrap().unwrap();
    reader.join().unwrap();
    assert!(!result.unwrap().contains(&0));
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
    for phase in ["logged", "consolidated"] {
        if phase == "consolidated" {
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
}

#[test]
fn untrusted_headers_and_rows_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");
    let index = UpdatableVGraph::create(&source, &[1., 2., 3., 4.], 2, &params()).unwrap();
    let manifest = index.state.read().manifest.clone();
    drop(index);
    for (name, at) in [
        ("CURRENT", 32usize),
        ("rows-0.bin", 64),
        ("updates.log", 16),
    ] {
        let copy = dir.path().join(name.replace('.', "-"));
        copy_store(&source, &copy);
        let file = copy.join(name);
        let mut bytes = std::fs::read(&file).unwrap();
        bytes[at] ^= 0xFF;
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
