// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::testing::*;
use super::*;
use std::sync::Arc;

fn opts(threads: usize, batch: usize) -> EmbedOptions {
    EmbedOptions {
        threads,
        batch_size: batch,
        ..EmbedOptions::default()
    }
}

fn corpus() -> Vec<String> {
    let mut rng = Lcg(99);
    (0..23)
        .map(|i| {
            let n = 1 + (i * 5) % 28;
            (0..n)
                .map(|_| WORDS[((rng.next() + 1.0) * 10.0) as usize % WORDS.len()])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

#[test]
#[ignore = "writes the shared synthetic native-binding fixture; run deliberately"]
fn generate_binding_fixture() {
    assert_eq!(std::env::var("HMS_EMBED_FIXTURE_WRITE").as_deref(), Ok("1"));
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let bytes = tiny_pack_bytes(Quantization::Int8, Pooling::Mean);
    std::fs::write(root.join("tiny-embed-int8.hmsemb"), &bytes).unwrap();
    std::fs::write(root.join("tiny-embed-int8.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "copyright": "2024-2026 WritersLogic Contributors",
        "license": "Apache-2.0",
        "model": "test/tiny",
        "source": "core::embed::testing::tiny_pack_bytes(Int8, Mean); deterministic synthetic weights, no pretrained model",
        "dimensions": 32,
        "sha256": sha256_hex(&bytes)
    })).unwrap()).unwrap();
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn round_trip_and_normalization_for_every_quantization() {
    let dir = tempfile::tempdir().unwrap();
    let texts = corpus();
    let mut by_quant = Vec::new();
    for q in [Quantization::F32, Quantization::F16, Quantization::Int8] {
        let (path, sha) = write_tiny(dir.path(), q, Pooling::Mean);
        let e = Embedder::from_pack(&path, Some(&sha), &opts(2, 8)).unwrap();
        assert_eq!(e.dimensions(), 32);
        assert_eq!(e.pack_sha256(), sha);
        let v = e.embed_passages(&texts).unwrap();
        for x in &v {
            assert!((cos(x, x) - 1.0).abs() < 1e-4);
        }
        by_quant.push(v);
    }
    // Quantized packs track the f32 pack.
    for (q, v) in by_quant.iter().enumerate().skip(1) {
        let worst = v
            .iter()
            .zip(&by_quant[0])
            .map(|(a, b)| cos(a, b))
            .fold(1.0, f32::min);
        assert!(
            worst > if q == 1 { 0.999 } else { 0.97 },
            "quantization {q} drifted: min cosine {worst}"
        );
    }
    // Identical texts map to identical vectors; different texts do not.
    let mut with_dup = texts.clone();
    with_dup.push(texts[3].clone());
    let e = Embedder::from_pack(
        &write_tiny(dir.path(), Quantization::F32, Pooling::Mean).0,
        None,
        &opts(1, 4),
    )
    .unwrap();
    let v = e.embed_passages(&with_dup).unwrap();
    assert_eq!(v[3], v[v.len() - 1]);
    assert_ne!(v[3], v[4]);
}

#[test]
fn output_is_bit_identical_across_threads_and_batching() {
    let dir = tempfile::tempdir().unwrap();
    let texts = corpus();
    for q in [Quantization::F32, Quantization::Int8] {
        let (path, sha) = write_tiny(dir.path(), q, Pooling::Cls);
        let reference = Embedder::from_pack(&path, Some(&sha), &opts(1, 1))
            .unwrap()
            .embed_passages(&texts)
            .unwrap();
        for (threads, batch) in [(2, 3), (7, 32), (3, 5)] {
            let got = Embedder::from_pack(&path, Some(&sha), &opts(threads, batch)).unwrap();
            assert_eq!(
                got.embed_passages(&texts).unwrap(),
                reference,
                "threads {threads} batch {batch}"
            );
            // A text alone and the same text inside a batch give the same bits.
            assert_eq!(got.embed_passages(&texts[4..5]).unwrap()[0], reference[4]);
        }
    }
}

#[test]
fn audit_effective_options_bind_store_and_index_identity() {
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let (path, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let plain = Arc::new(Embedder::from_pack(&path, Some(&sha), &opts(1, 1)).unwrap());
    let mut index = SemanticIndex::new(plain.clone());
    index.add(&["cat"], &["the cat sat"]).unwrap();
    assert!(index.top_k(&vec![0.0; plain.dimensions()], 0).is_empty());
    let saved = dir.path().join("semantic.index");
    index.save(&saved).unwrap();
    for changed in [
        EmbedOptions {
            pooling: Some(Pooling::Cls),
            ..opts(1, 1)
        },
        EmbedOptions {
            normalize: Some(false),
            ..opts(1, 1)
        },
        EmbedOptions {
            query_prefix: Some("cat ".into()),
            ..opts(1, 1)
        },
        EmbedOptions {
            passage_prefix: Some("cat ".into()),
            ..opts(1, 1)
        },
        EmbedOptions {
            max_length: Some(8),
            ..opts(1, 1)
        },
    ] {
        let changed = Arc::new(Embedder::from_pack(&path, Some(&sha), &changed).unwrap());
        assert_ne!(plain.space(), changed.space());
        assert!(SemanticIndex::load(&saved, changed).is_err());
    }
}

#[test]
fn token_budget_splits_batches_without_changing_results() {
    let dir = tempfile::tempdir().unwrap();
    let (path, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let texts = corpus();
    let wide = Embedder::from_pack(&path, Some(&sha), &opts(2, 64))
        .unwrap()
        .embed_passages(&texts)
        .unwrap();
    let narrow = Embedder::from_pack(
        &path,
        Some(&sha),
        &EmbedOptions {
            max_batch_tokens: 1,
            threads: 2,
            ..EmbedOptions::default()
        },
    )
    .unwrap()
    .embed_passages(&texts)
    .unwrap();
    assert_eq!(wide, narrow);
}

#[test]
fn call_limits_check_boundaries_before_inference_and_preserve_identity() {
    let error = Embedder::new(&EmbedOptions {
        model: "missing-fixture.hmsemb".into(),
        limits: EmbedLimits {
            max_tokens: 0,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("embedding limits must be positive"));
    let dir = tempfile::tempdir().unwrap();
    let (path, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let original = Embedder::from_pack(&path, Some(&sha), &opts(1, 4)).unwrap();
    let limited = Embedder::from_pack(
        &path,
        Some(&sha),
        &EmbedOptions {
            limits: EmbedLimits {
                max_texts: 2,
                max_input_bytes: 8,
                max_tokens: 8,
                ..Default::default()
            },
            ..opts(1, 4)
        },
    )
    .unwrap();
    assert_eq!(limited.space(), original.space());
    for count in [1, 2, 3] {
        assert_eq!(
            limited.embed_passages(&vec!["cat"; count]).is_ok(),
            count <= 2
        );
    }
    for bytes in [7, 8, 9] {
        assert_eq!(
            limited.embed_passages(&["x".repeat(bytes)]).is_ok(),
            bytes <= 8
        );
    }
    let prefixed = Embedder::from_pack(
        &path,
        Some(&sha),
        &EmbedOptions {
            query_prefix: Some("cat ".into()),
            limits: EmbedLimits {
                max_input_bytes: 8,
                ..Default::default()
            },
            ..opts(1, 4)
        },
    )
    .unwrap();
    for bytes in [3, 4, 5] {
        assert_eq!(
            prefixed.embed_query(&["x".repeat(bytes)]).is_ok(),
            bytes <= 4
        );
    }
    for bytes in [255, 256, 257] {
        let embedder = Embedder::from_pack(
            &path,
            Some(&sha),
            &EmbedOptions {
                limits: EmbedLimits {
                    max_output_bytes: bytes,
                    ..Default::default()
                },
                ..opts(1, 4)
            },
        )
        .unwrap();
        assert_eq!(embedder.embed_query(&["cat"]).is_ok(), bytes >= 256);
    }
    let limits = EmbedLimits {
        max_texts: usize::MAX,
        max_output_bytes: usize::MAX,
        ..Default::default()
    };
    assert!(limits.output_values(usize::MAX, 2).is_err());
    assert!(limits.output_values(usize::MAX / 2, 1).is_err());
}

#[test]
fn token_limits_fail_before_writing_and_release_admission() {
    let dir = tempfile::tempdir().unwrap();
    let (path, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let embedder = Embedder::from_pack(
        &path,
        Some(&sha),
        &EmbedOptions {
            limits: EmbedLimits {
                max_tokens: 8,
                max_concurrent_calls: 1,
                ..Default::default()
            },
            ..opts(2, 4)
        },
    )
    .unwrap();
    for total in [7, 8, 9] {
        let text = vec!["cat"; total - 2].join(" ");
        let mut out = vec![123.0; embedder.dimensions()];
        let result = embedder.embed_into(&[text], EmbedKind::Passage, &mut out);
        assert_eq!(result.is_ok(), total <= 8);
        if total > 8 {
            assert!(result.unwrap_err().to_string().contains("tokens limit"));
            assert!(out.iter().all(|&value| value == 123.0));
        }
    }
    assert!(embedder
        .embed_passages(&["cat", "dog", "mat"])
        .unwrap_err()
        .to_string()
        .contains("tokens limit"));
    assert!(embedder.embed_query(&["cat"]).is_ok());
}

#[test]
fn concurrent_call_limit_is_fail_fast_and_releases_after_failure() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Barrier,
    };
    struct GatedInput<'a> {
        entered: &'a Barrier,
        release: &'a Barrier,
        first: AtomicBool,
    }
    impl AsRef<str> for GatedInput<'_> {
        fn as_ref(&self) -> &str {
            if self.first.swap(false, Ordering::SeqCst) {
                self.entered.wait();
                self.release.wait();
            }
            "cat"
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let (path, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let embedder = Embedder::from_pack(
        &path,
        Some(&sha),
        &EmbedOptions {
            limits: EmbedLimits {
                max_concurrent_calls: 1,
                ..Default::default()
            },
            ..opts(1, 4)
        },
    )
    .unwrap();
    let entered = Barrier::new(2);
    let release = Barrier::new(2);
    std::thread::scope(|scope| {
        let running = scope.spawn(|| {
            embedder.embed_passages(&[GatedInput {
                entered: &entered,
                release: &release,
                first: AtomicBool::new(true),
            }])
        });
        entered.wait();
        let rejected = embedder.embed_query(&["dog"]);
        release.wait();
        assert!(rejected
            .unwrap_err()
            .to_string()
            .contains("concurrent calls limit"));
        assert!(running.join().unwrap().is_ok());
    });
    assert!(embedder.embed_query(&["dog"]).is_ok());
    assert!(embedder
        .embed_into(&["cat"], EmbedKind::Query, &mut [])
        .is_err());
    assert!(embedder.embed_query(&["cat"]).is_ok());
}

#[test]
fn truncation_reports_preserve_input_order_and_utf8_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let (path, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let embedder = Embedder::from_pack(
        &path,
        Some(&sha),
        &EmbedOptions {
            max_length: Some(8),
            ..opts(2, 4)
        },
    )
    .unwrap();
    let texts = vec![
        "cat ".repeat(5),
        "cat ".repeat(6),
        "cat ".repeat(7),
        format!("{}🦀", "é".repeat(127)),
    ];
    let batch = embedder
        .embed_with_report(&texts, EmbedKind::Passage)
        .unwrap();
    assert_eq!(batch.embeddings, embedder.embed_passages(&texts).unwrap());
    assert_eq!(
        batch
            .inputs
            .iter()
            .map(|input| input.token_truncated)
            .collect::<Vec<_>>(),
        [false, false, true, false]
    );
    assert_eq!(
        batch
            .inputs
            .iter()
            .map(|input| input.tokens)
            .take(3)
            .collect::<Vec<_>>(),
        [7, 8, 8]
    );
    assert!(batch.inputs[3].byte_truncated);
    assert_eq!(batch.inputs[3].retained_bytes, 254);
    assert_eq!(batch.inputs[3].input_bytes, texts[3].len());
    for bytes in [255, 256, 257] {
        let report = embedder
            .embed_with_report(&["x".repeat(bytes)], EmbedKind::Query)
            .unwrap();
        assert_eq!(report.inputs[0].byte_truncated, bytes > 256);
    }
}

#[test]
fn hash_mismatch_is_rejected_before_use() {
    let dir = tempfile::tempdir().unwrap();
    let (path, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    let bad = dir.path().join("bad.hmsemb");
    std::fs::write(&bad, &bytes).unwrap();
    let err = Embedder::from_pack(&bad, Some(&sha), &opts(1, 1)).unwrap_err();
    assert!(err.downcast_ref::<Sha256Mismatch>().is_some(), "{err}");
    assert!(err.to_string().contains("sha256 mismatch"));
    // A wrong pin on an intact file is rejected the same way.
    let err = Embedder::from_pack(&path, Some(&"0".repeat(64)), &opts(1, 1))
        .unwrap_err()
        .to_string();
    assert!(err.contains("sha256 mismatch"), "{err}");
    // Truncated and foreign files fail with their own reasons, never a panic.
    let cut = dir.path().join("cut.hmsemb");
    std::fs::write(&cut, &bytes[..bytes.len() / 2]).unwrap();
    assert!(Embedder::from_pack(&cut, None, &opts(1, 1))
        .unwrap_err()
        .to_string()
        .contains("truncated"));
    let junk = dir.path().join("junk.hmsemb");
    std::fs::write(&junk, b"not a pack at all").unwrap();
    assert!(Embedder::from_pack(&junk, None, &opts(1, 1)).is_err());
}

#[test]
fn registry_model_from_directory_is_checked_against_its_pin() {
    let dir = tempfile::tempdir().unwrap();
    let spec = registry::find("default").unwrap();
    // A well-formed pack under the registry's file name but not the pinned content.
    let (made, _) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    std::fs::rename(made, dir.path().join(spec.pack_name())).unwrap();
    let err = Embedder::new(&EmbedOptions {
        model_dir: Some(dir.path().to_path_buf()),
        offline: Some(true),
        ..EmbedOptions::default()
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("sha256 mismatch"), "{err}");
}

#[test]
fn offline_mode_fails_with_actionable_error_and_never_downloads() {
    let empty = tempfile::tempdir().unwrap();
    let err = Embedder::new(&EmbedOptions {
        offline: Some(true),
        cache_dir: Some(empty.path().to_path_buf()),
        model_dir: Some(empty.path().to_path_buf()),
        ..EmbedOptions::default()
    })
    .unwrap_err()
    .to_string();
    for needle in [
        ENV_PACK,
        ENV_MODEL_DIR,
        "offline mode is on",
        "hms-embed fetch",
        "sha256",
    ] {
        assert!(err.contains(needle), "missing {needle:?} in {err}");
    }
    assert_eq!(
        std::fs::read_dir(empty.path()).unwrap().count(),
        0,
        "offline mode wrote to the cache"
    );
}

#[test]
fn path_models_and_missing_files_report_the_path() {
    let err = Embedder::new(&EmbedOptions::model("/nonexistent/dir/model.hmsemb"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("/nonexistent/dir/model.hmsemb"), "{err}");
    let dir = tempfile::tempdir().unwrap();
    let (path, sha) = write_tiny(dir.path(), Quantization::F16, Pooling::Mean);
    let e = Embedder::new(&EmbedOptions {
        sha256: Some(sha),
        ..EmbedOptions::model(path.to_string_lossy())
    })
    .unwrap();
    assert_eq!(e.model(), "test/tiny");
}

#[test]
fn prefixes_truncation_and_huge_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let (path, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let plain = Embedder::from_pack(&path, Some(&sha), &opts(2, 8)).unwrap();
    let prefixed = Embedder::from_pack(
        &path,
        Some(&sha),
        &EmbedOptions {
            query_prefix: Some("the cat ".into()),
            ..opts(2, 8)
        },
    )
    .unwrap();
    let a = plain.embed_query(&["sat on mat"]).unwrap();
    let b = prefixed.embed_query(&["sat on mat"]).unwrap();
    assert_ne!(a, b, "the query prefix must change the encoding");
    assert_eq!(
        prefixed.embed_passages(&["sat on mat"]).unwrap(),
        plain.embed_passages(&["sat on mat"]).unwrap()
    );
    // Truncation at max_length: text past the window cannot change the vector.
    let short = Embedder::from_pack(
        &path,
        Some(&sha),
        &EmbedOptions {
            max_length: Some(8),
            ..opts(2, 8)
        },
    )
    .unwrap();
    let long = "the cat sat on the mat the dog ran in the park";
    let longer = format!("{long} bird flew over tree fish swam river");
    assert_eq!(
        short.embed_passages(&[long]).unwrap(),
        short.embed_passages(&[longer.as_str()]).unwrap()
    );
    // A pathological input is capped before tokenization and still embeds quickly.
    let huge = "cat ".repeat(2_000_000);
    let started = std::time::Instant::now();
    let v = short.embed_passages(&[huge.as_str()]).unwrap();
    assert_eq!(v[0].len(), 32);
    assert!(started.elapsed().as_secs() < 5);
    // Buffer API validates its size.
    assert!(short
        .embed_into(&["cat"], EmbedKind::Passage, &mut [0.0; 5])
        .is_err());
}

#[test]
fn from_pack_rejects_corrupt_headers_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = tiny_pack_bytes(Quantization::Int8, Pooling::Mean);
    // Header length claims more than the file holds.
    let mut huge = bytes.clone();
    huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    let p = dir.path().join("huge.hmsemb");
    std::fs::write(&p, &huge).unwrap();
    assert!(Embedder::from_pack(&p, None, &opts(1, 1)).is_err());
    // Every truncation point is an error, not a panic.
    for cut in (0..bytes.len()).step_by(bytes.len() / 37 + 1) {
        let p = dir.path().join("cut.hmsemb");
        std::fs::write(&p, &bytes[..cut]).unwrap();
        assert!(
            Embedder::from_pack(&p, None, &opts(1, 1)).is_err(),
            "cut at {cut}"
        );
    }
}

#[test]
fn registry_entries_are_pinned_and_unique() {
    use std::collections::HashSet;
    assert!(!MODELS.is_empty());
    let mut ids = HashSet::new();
    for m in MODELS {
        assert!(ids.insert(m.id), "duplicate id {}", m.id);
        assert!(
            matches!(m.license, "MIT" | "Apache-2.0"),
            "{} licence {}",
            m.id,
            m.license
        );
        assert_eq!(m.revision.len(), 40, "{} revision is not a commit", m.id);
        let hex = |s: &str| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit());
        assert!(hex(m.pack_sha256), "{} pack pin", m.id);
        assert!(
            !m.files.is_empty() && m.files.iter().all(|f| hex(f.sha256) && f.size > 0),
            "{} file pins",
            m.id
        );
        assert!(m.files.iter().any(|f| f.name == "model.safetensors") && m.int8_group > 0);
        assert!(m.pack_bytes > 0 && m.max_length >= 8);
    }
    assert_eq!(
        registry::find("default").unwrap().id,
        registry::find(DEFAULT_MODEL).unwrap().id
    );
    assert!(registry::find("no-such-model")
        .unwrap_err()
        .to_string()
        .contains("known models"));
}

#[test]
fn int8_group_sets_the_scale_granularity() {
    let dir = tempfile::tempdir().unwrap();
    let texts = corpus();
    let reference = {
        let (p, s) = write_tiny(dir.path(), Quantization::F32, Pooling::Mean);
        Embedder::from_pack(&p, Some(&s), &opts(1, 8))
            .unwrap()
            .embed_passages(&texts)
            .unwrap()
    };
    let worst = |group: usize| {
        let bytes = tiny_pack_bytes_with_group(Quantization::Int8, Pooling::Mean, group);
        let p = dir.path().join(format!("g{group}.hmsemb"));
        std::fs::write(&p, &bytes).unwrap();
        let v = Embedder::from_pack(&p, None, &opts(1, 8))
            .unwrap()
            .embed_passages(&texts)
            .unwrap();
        v.iter()
            .zip(&reference)
            .map(|(a, b)| cos(a, b))
            .fold(1.0, f32::min)
    };
    // 0 and 24 (not a divisor of any row length) fall back to one scale per row.
    assert_eq!(worst(0), worst(24));
    assert!(worst(8) > 0.97 && worst(0) > 0.97);
    // The int8 pack grows with the number of scales.
    let size = |g| tiny_pack_bytes_with_group(Quantization::Int8, Pooling::Mean, g).len();
    assert!(size(8) > size(16) && size(16) > size(0));
}

#[test]
fn index_save_leaves_files_named_like_its_temp_alone() {
    let dir = tempfile::tempdir().unwrap();
    let (pack, sha) = write_tiny(dir.path(), Quantization::F32, Pooling::Mean);
    let embedder = Arc::new(Embedder::from_pack(&pack, Some(&sha), &opts(1, 4)).unwrap());
    let mut index = SemanticIndex::new(Arc::clone(&embedder));
    let texts = corpus();
    let ids: Vec<String> = (0..texts.len()).map(|i| format!("d{i}")).collect();
    index.add(&ids, &texts).unwrap();
    // A file with the temp name a save would derive, and an index whose own name is that.
    let sibling = dir.path().join("a.tmp");
    std::fs::write(&sibling, b"someone else's file").unwrap();
    index.save(&dir.path().join("a.idx")).unwrap();
    assert_eq!(std::fs::read(&sibling).unwrap(), b"someone else's file");
    let loaded = SemanticIndex::load(&dir.path().join("a.idx"), Arc::clone(&embedder)).unwrap();
    assert_eq!(loaded.len(), index.len());
    index.save(&sibling).unwrap();
    assert_eq!(
        SemanticIndex::load(&sibling, embedder).unwrap().len(),
        index.len()
    );
}

#[test]
fn safetensors_offsets_and_shapes_that_overflow_are_errors() {
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, header: &str| {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    };
    let ok = write(
        "ok",
        r#"{"t":{"dtype":"F32","shape":[0,5],"data_offsets":[0,0]}}"#,
    );
    assert!(pack::read_safetensors(&ok).is_ok());
    let end = write(
        "end",
        r#"{"t":{"dtype":"F32","shape":[0],"data_offsets":[0,18446744073709551615]}}"#,
    );
    assert!(pack::read_safetensors(&end).is_err());
    let shape = write(
        "shape",
        r#"{"t":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#,
    );
    assert!(pack::read_safetensors(&shape).is_err());
}
