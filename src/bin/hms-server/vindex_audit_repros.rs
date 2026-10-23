// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#[test]
fn audit_repro_normalization_overflow() {
    let row = normalize(&[f64::MAX, f64::MAX]).unwrap();
    assert!((dot(&row, &row) - 1.0).abs() < 1e-6, "row={row:?}");
}

#[test]
fn legacy_snapshot_gains_row_checksum_during_maintenance() {
    let dir = tempfile::tempdir().unwrap();
    let core = core(dir.path(), 8);
    let vdir = dir.path().join("vindex");
    let index = VectorIndex::open(&vdir, 8, &core).unwrap();
    ingest(&index, &core, 0, 2, 8);
    index.maintain(1, true).unwrap();
    index.close().unwrap();
    drop(index);
    let path = vdir.join("CURRENT");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("rows_crc");
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let index = VectorIndex::open(&vdir, 8, &core).unwrap();
    assert!(index.covers(&core));
    index.maintain(1, false).unwrap();
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert!(manifest.rows_crc.is_some());
    assert_eq!(top(&index, 0, 8)[0], "d0");
}

#[test]
fn audit_repro_append_capture_publish_gap() {
    let dir = tempfile::tempdir().unwrap();
    let core = core(dir.path(), 8);
    let vdir = dir.path().join("vindex");
    let index = VectorIndex::open(&vdir, 8, &core).unwrap();
    ingest(&index, &core, 0, 2, 8);
    let row = normalize(&vector(2, 8)).unwrap();
    let token = index.next_token();
    let mut bytes = Vec::new();
    push_frame(&mut bytes, UPSERT, "d2", token, &row);
    std::thread::scope(|scope| {
        let log = index.append(&bytes).unwrap();
        let maintenance = scope.spawn(|| index.maintain(1, true));
        assert!(index.log.try_lock().is_none());
        index.state.write().upsert(Arc::from("d2"), token, &row);
        drop(log);
        maintenance.join().unwrap().unwrap();
    });
    core.memorize_documents(vec![holographic_memory::DocumentInput {
        id: "d2".into(),
        text: "_".into(),
        source_uri: None,
        version: Some(version_stamp(token)),
        metadata: None,
        chunk_words: None,
        overlap_words: None,
        store_text: Some(false),
        embeddings: Some(vec![vector(2, 8)]),
        title_weight: None,
    }])
    .unwrap();
    core.flush().unwrap();
    assert_eq!(index.live(), 3);
    index.close().unwrap();
    drop(index);
    let reopened = VectorIndex::open(&vdir, 8, &core).unwrap();
    assert_eq!(
        reopened.live(),
        3,
        "snapshot covered a frame absent from its captured state"
    );
}

#[test]
fn audit_repro_snapshot_row_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let core = core(dir.path(), 8);
    let vdir = dir.path().join("vindex");
    let index = VectorIndex::open(&vdir, 8, &core).unwrap();
    ingest(&index, &core, 0, 2, 8);
    index.maintain(1, true).unwrap();
    index.close().unwrap();
    drop(index);
    let path = vdir.join("rows-1.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[ROWS_HEADER..ROWS_HEADER + 8 * 4].fill(0);
    std::fs::write(path, bytes).unwrap();
    let reopened = VectorIndex::open(&vdir, 8, &core).unwrap();
    assert!(
        !reopened.covers(&core),
        "corrupt rows remain eligible for authoritative graph routing"
    );
}

#[test]
fn audit_repro_deleted_nearest_rows_underfill() {
    let dir = tempfile::tempdir().unwrap();
    let core = core(dir.path(), 8);
    let vdir = dir.path().join("vindex");
    let index = VectorIndex::open(&vdir, 8, &core).unwrap();
    ingest(&index, &core, 0, 20, 8);
    index.maintain(1, true).unwrap();
    let query = normalize(&vector(0, 8)).unwrap();
    let nearest = index.search(&query, 4, 64).unwrap();
    let ids: Vec<&str> = nearest.iter().map(|(id, _)| id.as_ref()).collect();
    for id in &ids {
        assert!(core.delete_document(id).unwrap());
    }
    index.remove(&ids).unwrap();
    assert_eq!(index.live(), 16);
    assert!(index.covers(&core));
    assert_eq!(
        index.search(&query, 1, 64).unwrap().len(),
        1,
        "four tombstones exhaust fixed overfetch"
    );
}
