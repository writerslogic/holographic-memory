// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! An independent reader of `docs/file-format/`: it parses the golden stores from the
//! specification alone (raw little-endian reads, LZ4 block, CRC32, Argon2id, AES-GCM) and uses
//! none of the crate's readers. If the spec were incomplete or wrong it could not decode them.
//! Also pins what an open may change and how future versions are refused.

mod format_common;

use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use format_common::*;
use holographic_memory::core::format::{verify_path, Verdict, VerifyOptions};
use holographic_memory::HmsCore;
use serde_json::Value;

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

struct Frame {
    offset: usize,
    seq: u32,
    payload: Vec<u8>,
}

/// JOURNAL.md, "Frame" and "Segment end", written from the text. Returns frames and the offset
/// of the end header and its kind.
fn read_segment(seg: &[u8]) -> Result<(Vec<Frame>, usize, &'static str)> {
    let mut frames = Vec::new();
    let mut at = 0;
    while at + 16 <= seg.len() {
        let (crc, raw, comp, seq) = (
            le32(seg, at),
            le32(seg, at + 4),
            le32(seg, at + 8),
            le32(seg, at + 12),
        );
        if raw == 0 && comp == 0 {
            if crc == 0 && seq == 0x454E_444C {
                return Ok((frames, at, "marker"));
            }
            ensure!(crc == 0 && seq == 0, "invalid end header at {at}");
            return Ok((frames, at, "zero"));
        }
        ensure!(
            (1..=52_428_800).contains(&raw) && comp >= 1 && comp <= raw,
            "lengths at {at}"
        );
        let stored = &seg[at + 16..at + 16 + comp as usize];
        let payload = if comp < raw {
            ensure!(raw as u64 <= 255 * comp as u64);
            lz4_flex::block::decompress(stored, raw as usize)?
        } else {
            stored.to_vec()
        };
        ensure!(crc32fast::hash(&payload) == crc, "crc at {at}");
        frames.push(Frame {
            offset: at,
            seq,
            payload,
        });
        at += 16 + comp as usize;
    }
    Ok((frames, at, "exhausted"))
}

#[cfg(feature = "security")]
fn open_envelope(salt: &[u8], sealed: &[u8]) -> Result<Vec<u8>> {
    use aes_gcm::aead::{Aead, KeyInit};
    use argon2::{Algorithm, Argon2, Params, Version};
    let mut key = [0u8; 32]; // codeql[rust/hard-coded-cryptographic-value] Fixed passphrase of the committed golden fixture stores
    Argon2::new(
        Algorithm::Argon2id,
        Version::V0x13,
        Params::new(19456, 2, 1, Some(32)).unwrap(),
    )
    .hash_password_into(PASSPHRASE.as_bytes(), salt, &mut key)
    .unwrap();
    let cipher = aes_gcm::Aes256Gcm::new_from_slice(&key).unwrap();
    let nonce = aes_gcm::Nonce::try_from(&sealed[..12]).unwrap();
    cipher
        .decrypt(&nonce, &sealed[12..])
        .map_err(|e| anyhow::anyhow!("{e}"))
}

#[test]
fn the_spec_alone_decodes_every_golden_journal() -> Result<()> {
    let mut decoded = 0;
    for golden in goldens()? {
        if golden.str("expectOpenError").is_some() {
            continue;
        }
        let scratch = tempfile::tempdir()?;
        let store = scratch.path().join("store");
        golden.unpack(&store)?;
        let arena = store.join("vectors_data.bin");
        // LAYOUT.md: CURRENT names the live generation.
        let live = match std::fs::read_to_string(arena.join("CURRENT")) {
            Ok(name) => {
                ensure!(name.starts_with("gen-") && !name.contains('\n'));
                arena.join(name)
            }
            Err(_) => arena.clone(),
        };
        let encrypted = golden.meta["encrypted"] == true;
        let mut total = 0usize;
        let mut last_end = 0usize;
        let mut segments = 0usize;
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        for id in 0.. {
            let path = live.join(format!("seg_{id}.bin"));
            if !path.exists() {
                break;
            }
            segments += 1;
            let seg = std::fs::read(&path)?;
            let (frames, end, how) = read_segment(&seg).with_context(|| golden.name.clone())?;
            ensure!(
                seg[end..].iter().skip(16).all(|&b| b == 0),
                "{}: bytes after the end",
                golden.name
            );
            if how == "zero" {
                ensure!(
                    golden.meta["storageFormat"].as_u64() <= Some(2),
                    "{}: zero terminator",
                    golden.name
                );
            }
            for (n, frame) in frames.iter().enumerate() {
                ensure!(
                    frame.seq as usize == total + n || id > 0,
                    "{}: seq {}",
                    golden.name,
                    frame.seq
                );
                let plain = if encrypted {
                    #[cfg(feature = "security")]
                    {
                        open_envelope(
                            &std::fs::read(store.join("encryption.salt"))?,
                            &frame.payload,
                        )?
                    }
                    #[cfg(not(feature = "security"))]
                    {
                        continue;
                    }
                } else {
                    frame.payload.clone()
                };
                // MUTATIONS.md: HMS-TXN-2 is a JSON array; HMS-TXN-3 is binary, read back as
                // the JSON array of the same mutations.
                let json = match plain.strip_prefix(b"HMS-TXN-2\0") {
                    Some(json) => std::borrow::Cow::Borrowed(json),
                    None => {
                        ensure!(plain.starts_with(b"HMS-TXN-3\0"), "journal magic");
                        holographic_memory::core::format::scan::transaction_json(&plain)
                            .context("journal magic")?
                            .map_err(anyhow::Error::msg)?
                    }
                };
                let mutations: Vec<Value> = serde_json::from_slice(&json)?;
                ensure!((1..=4096).contains(&mutations.len()));
                for m in mutations {
                    let object = m.as_object().context("mutation object")?;
                    ensure!(object.len() == 1);
                    *kinds
                        .entry(object.keys().next().unwrap().clone())
                        .or_default() += 1;
                    let _ = frame.offset;
                }
            }
            total += frames.len();
            last_end = end;
            let _ = last_end;
        }
        ensure!(total > 0 && segments > 0, "{}", golden.name);
        // LAYOUT.md: checksum sidecars are CRC32 as 8 hex digits and a newline.
        for (file, sidecar) in [
            ("store.json", "store.json.crc32"),
            (
                "vectors_data.bin/format.json",
                "vectors_data.bin/format.json.crc32",
            ),
        ] {
            if let Ok(recorded) = std::fs::read_to_string(store.join(sidecar)) {
                let actual = crc32fast::hash(&std::fs::read(store.join(file))?);
                ensure!(
                    recorded == format!("{actual:08x}\n"),
                    "{}: {sidecar}",
                    golden.name
                );
            }
        }
        // CACHES.md: the text segment header.
        if let Ok(text) = std::fs::read(store.join("documents.text")) {
            if golden.meta["textSegment"] == 2 {
                ensure!(&text[..4] == b"HMST" && text[4] == 2 && text[6..8] == [0, 0]);
                ensure!(
                    (text[5] & 1 == 1) == encrypted,
                    "{}: text flags",
                    golden.name
                );
            }
        }
        // LAYOUT.md: the schema keys, in order.
        let schema = std::fs::read_to_string(store.join("store.json"));
        if let Ok(schema) = schema {
            let keys: Vec<_> = serde_json::from_str::<serde_json::Map<String, Value>>(&schema)?
                .keys()
                .cloned()
                .collect();
            let mut sorted = keys.clone();
            sorted.sort();
            let expect = [
                "dimensions",
                "embedding_space",
                "encrypted",
                "meaning_enabled",
                "metric",
                "text_encoder",
                "vector_encoder",
                "version",
            ];
            ensure!(sorted == expect, "{}: {keys:?}", golden.name);
        }
        if golden.name == "v4-plain" {
            ensure!(
                kinds.get("Document") == Some(&8) && kinds.get("Vector") == Some(&3),
                "{kinds:?}"
            );
            // CACHES.md: the checkpoint is the generation name and the journal length.
            let indices: Value =
                serde_json::from_slice(&std::fs::read(store.join("indices.json"))?)?;
            ensure!(indices["checkpoint"] == serde_json::json!(["vectors_data.bin", last_end]));
        }
        decoded += 1;
    }
    ensure!(decoded >= 10);
    Ok(())
}

/// LAYOUT.md "Open sequence": opening changes only the lock file, `documents.text`, the end of
/// the final segment (zero header or tail to end marker), and the manifest of a format-1 store.
#[test]
fn open_changes_only_what_the_spec_lists() -> Result<()> {
    for golden in goldens()? {
        if golden.str("expectOpenError").is_some() || !golden.openable_here() {
            continue;
        }
        let scratch = tempfile::tempdir()?;
        let store = scratch.path().join("store");
        golden.unpack(&store)?;
        let before = file_fingerprints(&store)?;
        let ends: BTreeMap<String, usize> = before
            .keys()
            .filter(|k| k.contains("seg_"))
            .map(|k| {
                let bytes = std::fs::read(store.join(k)).unwrap();
                let end =
                    holographic_memory::core::format::scan::scan_segment(&bytes, true, |_, _| {})
                        .end_offset;
                (k.clone(), end)
            })
            .collect();
        let before_bytes: BTreeMap<_, _> = ends
            .keys()
            .map(|k| (k.clone(), std::fs::read(store.join(k)).unwrap()))
            .collect();
        drop(HmsCore::new(
            golden.dimensions(),
            Some(store.display().to_string()),
            Some(golden.config()?),
        )?);
        let after = file_fingerprints(&store)?;
        for (file, print) in &after {
            if before.get(file) == Some(print) {
                continue;
            }
            let allowed = file == ".hms.lock"
                || file == "documents.text"
                || file.contains("seg_")
                || (golden.meta["storageFormat"] == 1
                    && file.starts_with("vectors_data.bin/format.json"));
            ensure!(allowed, "{}: open changed {file}", golden.name);
        }
        ensure!(
            before.keys().all(|k| after.contains_key(k)),
            "{}: open removed a file",
            golden.name
        );
        for (file, old) in &before_bytes {
            let new = std::fs::read(store.join(file))?;
            let end = ends[file];
            ensure!(
                new[..end] == old[..end],
                "{}: open changed committed bytes of {file}",
                golden.name
            );
        }
    }
    Ok(())
}

fn refused(golden: &Golden, store: &std::path::Path, needle: &str) -> Result<()> {
    let error = match HmsCore::new(
        golden.dimensions(),
        Some(store.display().to_string()),
        Some(golden.config()?),
    ) {
        Ok(_) => anyhow::bail!("opened"),
        Err(e) => format!("{e:#}"),
    };
    ensure!(error.contains(needle), "{error}");
    let report = verify_path(store, &VerifyOptions::default())?;
    ensure!(
        report.verdict == Verdict::Unsupported,
        "{:?}",
        report.findings
    );
    Ok(())
}

/// STABILITY.md rule 3: every versioned artefact one step beyond this build is refused
/// cleanly, by the engine and by the verifier, before anything is modified.
#[test]
fn future_versions_are_refused_cleanly() -> Result<()> {
    let golden = goldens()?
        .into_iter()
        .find(|g| g.name == "v4-plain")
        .unwrap();
    let make = || -> Result<(tempfile::TempDir, std::path::PathBuf)> {
        let scratch = tempfile::tempdir()?;
        let store = scratch.path().join("store");
        golden.unpack(&store)?;
        Ok((scratch, store))
    };
    // Arena manifest.
    let (_k, store) = make()?;
    let manifest = store.join("vectors_data.bin/format.json");
    let text = std::fs::read_to_string(&manifest)?.replace("\"version\": 4", "\"version\": 5");
    std::fs::write(&manifest, &text)?;
    std::fs::write(
        store.join("vectors_data.bin/format.json.crc32"),
        format!("{:08x}\n", crc32fast::hash(text.as_bytes())),
    )?;
    refused(&golden, &store, "newer than supported")?;
    // Store schema.
    let (_k, store) = make()?;
    let schema = store.join("store.json");
    let text = std::fs::read_to_string(&schema)?.replace("\"version\": 2", "\"version\": 4");
    std::fs::write(&schema, &text)?;
    std::fs::write(
        store.join("store.json.crc32"),
        format!("{:08x}\n", crc32fast::hash(text.as_bytes())),
    )?;
    refused(&golden, &store, "newer than the supported")?;
    // Journal magic: a well-formed frame whose payload is HMS-TXN-4.
    let (_k, store) = make()?;
    let path = store.join("vectors_data.bin/seg_0.bin");
    let mut seg = std::fs::read(&path)?;
    let (_, end, _) = read_segment(&seg)?;
    let payload = b"HMS-TXN-4\0[]";
    let mut frame = Vec::new();
    frame.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&99u32.to_le_bytes());
    frame.extend_from_slice(payload);
    frame.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x4c, 0x44, 0x4e, 0x45]);
    seg[end..end + frame.len()].copy_from_slice(&frame);
    std::fs::write(&path, seg)?;
    refused(&golden, &store, "HMS-TXN-4 is newer")?;
    // A text segment from the future is only a cache: it is overwritten.
    let (_k, store) = make()?;
    let mut text = std::fs::read(store.join("documents.text"))?;
    text[4] = 3;
    std::fs::write(store.join("documents.text"), text)?;
    ensure!(HmsCore::new(
        golden.dimensions(),
        Some(store.display().to_string()),
        Some(golden.config()?)
    )?
    .has_document("doc-harbor"));
    Ok(())
}

/// A whole-document record may cover up to the document limit in its one chunk: a store holding
/// one over the 64 KiB chunk limit verifies clean and plans an upgrade.
#[test]
fn whole_document_record_over_the_chunk_limit_verifies() -> Result<()> {
    use holographic_memory::core::documents::{EncodeMode, IngestOptions};
    use holographic_memory::DocumentInput;
    let dir = tempfile::tempdir()?;
    let store = dir.path().join("store");
    let core = HmsCore::new(64, Some(store.display().to_string()), None)?;
    let text = "harbor lantern ".repeat(5000);
    ensure!(text.len() > 64 * 1024);
    let input: DocumentInput = serde_json::from_value(serde_json::json!({
        "id": "article", "text": text, "chunkWords": 4096, "overlapWords": 0
    }))?;
    core.memorize_documents_with(
        vec![input],
        &IngestOptions {
            encode: EncodeMode::Lexical,
            positions: true,
            whole_document: true,
            ..Default::default()
        },
    )?;
    drop(core);
    let report = verify_path(&store, &VerifyOptions::default())?;
    ensure!(
        report.verdict == Verdict::Ok,
        "whole-document store does not verify: {:?}",
        report.findings
    );
    holographic_memory::core::format::plan_upgrade(&store)?;
    Ok(())
}
