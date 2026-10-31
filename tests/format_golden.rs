// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Golden stores: small committed stores for every storage format that has existed, opened by
//! the current code. See `tests/golden/README.md`.
//!
//! A change that breaks reading any golden store fails here. The only way to land it is the
//! way `docs/STABILITY.md` describes: bump the version constant, keep reading (or migrate) the
//! old format, and commit a new golden store for the new one.

mod format_common;
#[path = "golden/generator/ops.rs"]
mod ops;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use format_common::*;
use serde_json::{json, Value};
use writerslogic_warm::core::admin::migrate_store;
use writerslogic_warm::core::format::{self, verify_path, Verdict, VerifyOptions};
use writerslogic_warm::core::memory_api::{Clock, Memory, RecallOptions, RememberInput};
use writerslogic_warm::core::WarmConfig;
use writerslogic_warm::{DocumentInput, EntangledHVec, WarmCore};

const DIM: u32 = 1024;

// ---------------------------------------------------------------------------------------
// The golden stores and how each is built. `provenance` says what produced the bytes:
//   released-code      the code of a tagged release, built and run unmodified
//   current-code       the code in this tree
//   synthesized        current code, steered into an older layout (see `description`)
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Text,
    Embed,
    Memory,
    Governance,
}

struct Spec {
    name: &'static str,
    kind: Kind,
    description: &'static str,
    provenance: &'static str,
    encrypted: bool,
    encoder: &'static str,
    /// Arena manifest version to seed before the first open (the writer honours it).
    manifest: Option<u32>,
    /// `store.json` schema the store is created with: 2 writes `HMS-TXN-2` frames, 3
    /// `HMS-TXN-3` (the writer keeps the schema it finds).
    schema: u32,
    compact: bool,
    snapshot: bool,
    /// Train and persist the IVF and NSG caches.
    caches: bool,
    shard: bool,
    audit: bool,
    /// Replace documents.text with the headerless version-1 layout.
    text_v1: bool,
    /// Delete documents.text (formats written before it existed).
    no_text: bool,
    /// Lexical-only documents and nothing else, closed into a journal checkpoint
    /// (`documents.ckpt`) instead of the frames before it.
    checkpoint: bool,
}

const BASE: Spec = Spec {
    name: "",
    kind: Kind::Text,
    description: "",
    provenance: "current-code",
    encrypted: false,
    encoder: "multiscale-words-v1",
    manifest: None,
    schema: 2,
    compact: false,
    snapshot: false,
    caches: false,
    shard: false,
    audit: false,
    text_v1: false,
    no_text: false,
    checkpoint: false,
};

fn specs() -> Vec<Spec> {
    vec![
        Spec {
            name: "v6-audit-effects",
            description: "Schema 6 audit intents and acknowledgements with compaction and post-compaction writes.",
            schema: 6,
            audit: true,
            compact: true,
            ..BASE
        },
        Spec {
            name: "v4-plain",
            description: "Current format, unencrypted: documents (one replaced, one deleted, \
                          one multi-chunk, one non-ASCII), vectors, a relation, deletes.",
            ..BASE
        },
        Spec {
            name: "v4-encrypted",
            description: "Current format with AES-256-GCM frames, text segment and (when \
                          trained) index caches; passphrase golden-passphrase-1.",
            encrypted: true,
            ..BASE
        },
        Spec {
            name: "v4-embeddings",
            kind: Kind::Embed,
            description: "Current format with a client embedding space (3 dimensions): chunk \
                          embeddings are stored and searched.",
            ..BASE
        },
        Spec {
            name: "v4-memory",
            kind: Kind::Memory,
            description: "Facts written by the memory API: a mem1 version stamp, a supersession \
                          link, entities, and one fact whose stamp overflowed into the _mem \
                          metadata key.",
            ..BASE
        },
        Spec {
            name: "v4-capped-union",
            description: "Current format written with the multiscale-capped-union-v1 text \
                          encoder (experimental).",
            encoder: "multiscale-capped-union-v1",
            ..BASE
        },
        Spec {
            name: "v5-words-v2",
            description: "Schema 5 with multiscale-words-v2: majority ties use frequency descending, coordinate ascending.",
            encoder: "multiscale-words-v2",
            schema: 5,
            ..BASE
        },
        Spec {
            name: "v4-compacted",
            description: "After compaction: CURRENT points at a gen-* arena generation, and \
                          journal frames written after the compaction follow it.",
            compact: true,
            ..BASE
        },
        Spec {
            name: "v4-snapshot",
            description: "The output of `warm-admin migrate` (a verified point-in-time copy) of \
                          a compacted store.",
            compact: true,
            snapshot: true,
            ..BASE
        },
        Spec {
            name: "v4-indices",
            description: "Trained IVF and NSG index caches with their checksums in indices.json, \
                          bound to the journal end.",
            caches: true,
            ..BASE
        },
        Spec {
            name: "v4-sharded",
            description: "A store that auto-sharded: shard_meta.json and per-shard cache files.",
            caches: true,
            shard: true,
            ..BASE
        },
        Spec {
            name: "v4-audit",
            description: "Audit log enabled: audit.bin with unsigned entries.",
            audit: true,
            ..BASE
        },
        Spec {
            name: "v4-txn3-plain",
            description: "Schema 3: binary HMS-TXN-3 journal frames with a per-frame term \
                          dictionary; the v4-plain corpus and operations.",
            schema: 3,
            ..BASE
        },
        Spec {
            name: "v4-txn3-encrypted",
            description: "Schema 3 with AES-256-GCM frames and text segment; passphrase \
                          golden-passphrase-1.",
            schema: 3,
            encrypted: true,
            ..BASE
        },
        Spec {
            name: "v4-txn3-embeddings",
            kind: Kind::Embed,
            description: "Schema 3 with a client embedding space (3 dimensions): chunk \
                          embeddings in the binary frames.",
            schema: 3,
            ..BASE
        },
        Spec {
            name: "v4-txn3-memory",
            kind: Kind::Memory,
            description: "Schema 3: facts written by the memory API (version stamps, \
                          supersession, entities) in binary frames.",
            schema: 3,
            ..BASE
        },
        Spec {
            name: "v5-governance",
            kind: Kind::Governance,
            description: "Schema 5: namespace-scoped atom deletion and durable index refresh preserve a raw-vector alias and a live owned triple.",
            schema: 5,
            ..BASE
        },
        Spec {
            name: "v4-schema4-plain",
            description: "Schema 4: binary HMS-TXN-3 frames in a store that may hold document \
                          checkpoints; the v4-plain corpus and operations (vectors and \
                          relations, so no checkpoint is written).",
            schema: 4,
            ..BASE
        },
        Spec {
            name: "v4-schema4-checkpoint",
            description: "Schema 4 with a journal checkpoint: the v4-plain documents stored \
                          lexical-only (one deleted), no vectors or relations, closed into \
                          documents.ckpt in the journal generation in place of the frames.",
            schema: 4,
            checkpoint: true,
            ..BASE
        },
        Spec {
            name: "v4-txn3-compacted",
            description: "Schema 3 after compaction: one binary frame per document in the new \
                          generation, then the journal tail.",
            schema: 3,
            compact: true,
            ..BASE
        },
        Spec {
            name: "v3-fixed64",
            description: "Storage format 3: 64 MiB fixed segments, headerless version-1 text \
                          segment. Written by current code with a format-3 manifest seeded \
                          before the first open (format 3 was never released).",
            provenance: "synthesized",
            manifest: Some(3),
            text_v1: true,
            ..BASE
        },
        Spec {
            name: "v2-marker",
            description: "Storage format 2 as written after end markers and checksum sidecars \
                          were added: 1 GiB segments, no text segment. Current code with a \
                          format-2 manifest seeded before the first open.",
            provenance: "synthesized",
            manifest: Some(2),
            no_text: true,
            ..BASE
        },
    ]
}

fn seed_manifest(store: &Path, version: u32) -> Result<()> {
    let arena = store.join("vectors_data.bin");
    std::fs::create_dir_all(&arena)?;
    let segment_size: u64 = if version == 3 { 64 << 20 } else { 1 << 30 };
    let manifest = json!({"magic": "HMS_ARENA", "version": version, "segment_size": segment_size});
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    std::fs::write(arena.join("format.json"), &bytes)?;
    std::fs::write(
        arena.join("format.json.crc32"),
        format!("{:08x}\n", crc32fast::hash(&bytes)),
    )?;
    Ok(())
}

/// Rewrites the `store.json` schema version of a store with an empty journal (and its
/// checksum sidecar), so the engine writes that schema's journal format from then on.
fn seed_schema(store: &Path, version: u32) -> Result<()> {
    let path = store.join("store.json");
    // A textual edit keeps the field order the engine writes.
    let current = format!("\"version\": {},", format::STORE_SCHEMA_VERSION);
    let text = std::fs::read_to_string(&path)?;
    ensure!(text.contains(&current), "store.json without {current}");
    let bytes = text
        .replacen(&current, &format!("\"version\": {version},"), 1)
        .into_bytes();
    std::fs::write(&path, &bytes)?;
    std::fs::write(
        store.join("store.json.crc32"),
        format!("{:08x}\n", crc32fast::hash(&bytes)),
    )?;
    Ok(())
}

fn embedding_space() -> writerslogic_warm::EmbeddingSpace {
    writerslogic_warm::EmbeddingSpace {
        model: "golden-model".into(),
        revision: "1".into(),
        dimensions: 3,
        normalization: "l2".into(),
        metric: "cosine".into(),
    }
}

fn config_for(spec: &Spec) -> Result<WarmConfig> {
    let mut config = WarmConfig {
        text_encoder: spec.encoder.parse()?,
        ..Default::default()
    };
    if spec.kind == Kind::Embed {
        config.embedding_space = Some(embedding_space());
    }
    if spec.encrypted {
        config.security.encryption_enabled = true;
        config.security.encryption_passphrase = Some(PASSPHRASE.into());
    }
    config.security.audit_enabled = spec.audit;
    if spec.kind == Kind::Governance {
        config.meaning.enabled = true;
        config.cognition.governor_forget_unreferenced = true;
    }
    if spec.checkpoint {
        config.document_checkpoint.min_journal_bytes = 0;
    }
    if spec.caches {
        config.ivf.n_clusters = 4;
        config.ivf.n_landmarks = 16;
        config.ivf.d_reduced = 8;
        config.ivf.n_probe = 2;
        config.nsg.max_degree = 8;
        config.nsg.ef_construction = 16;
    }
    if spec.shard {
        config.shard.enabled = true;
        config.shard.shard_count = 0;
        config.shard.auto_threshold = 64;
        config.shard.target_shard_size = 40;
    }
    Ok(config)
}

struct Tick(AtomicI64);

impl Clock for Tick {
    fn now_millis(&self) -> i64 {
        self.0.fetch_add(1000, Ordering::SeqCst)
    }
}

fn memory_facts(core: Arc<WarmCore>) -> Result<Value> {
    let memory = Memory::new(core).with_clock(Arc::new(Tick(AtomicI64::new(1_760_000_000_000))));
    let fact = |id: &str, text: &str| RememberInput {
        id: Some(id.into()),
        text: text.into(),
        ..Default::default()
    };
    memory.remember(RememberInput {
        subject: Some("alice/city".into()),
        valid_from: Some(1_577_836_800_000),
        ..fact("m-paris", "Alice lives in Paris")
    })?;
    memory.remember(RememberInput {
        subject: Some("alice/city".into()),
        valid_from: Some(1_735_689_600_000),
        supersedes: Some(vec!["m-paris".into()]),
        ..fact("m-berlin", "Alice lives in Berlin")
    })?;
    memory.remember(RememberInput {
        entities: Some(vec!["Bob".into(), "Rex".into()]),
        event_time: Some(1_700_000_000_000),
        metadata: Some(json!({"source": "chat", "topic": "pets"})),
        ..fact("m-bob", "Bob adopted a dog named Rex")
    })?;
    memory.remember(RememberInput {
        entities: Some(
            (0..80)
                .map(|n| format!("entity-{n:04}-padding-padding"))
                .collect(),
        ),
        ..fact("m-wide", "A fact that mentions very many entities at once")
    })?;
    Ok(json!({
        "facts": [
            {"id": "m-bob", "text": "Bob adopted a dog named Rex", "stampStyle": "version"},
            {"id": "m-wide", "text": "A fact that mentions very many entities at once", "stampStyle": "metadata"},
        ],
        "recalls": [
            {"query": "where does Alice live", "asOf": 1_767_225_600_000i64, "ids": ["m-berlin"]},
            {"query": "where does Alice live", "asOf": 1_622_505_600_000i64, "ids": ["m-paris"]},
            {"query": "dog named Rex", "asOf": 1_767_225_600_000i64, "top": "m-bob"},
        ],
    }))
}

fn corpus(name: &str) -> Result<Vec<DocumentInput>> {
    ops::load_corpus(&golden_root().join("corpus").join(name))
}

/// What the corpus leaves live: later entries replace earlier ones with the same id.
fn expected_documents(name: &str, deleted: &[&str]) -> Result<Vec<Value>> {
    let raw: Vec<Value> =
        serde_json::from_slice(&std::fs::read(golden_root().join("corpus").join(name))?)?;
    let mut live: BTreeMap<String, Value> = BTreeMap::new();
    for entry in raw {
        let id = entry["id"].as_str().unwrap().to_owned();
        live.insert(id, entry);
    }
    Ok(live
        .into_iter()
        .filter(|(id, _)| !deleted.contains(&id.as_str()))
        .map(|(id, e)| {
            json!({
                "id": id, "text": e["text"], "version": e["version"],
                "sourceUri": e["sourceUri"], "metadata": e["metadata"],
            })
        })
        .collect())
}

/// Builds the store of `spec` in `store` with the current code and returns its `expect`.
fn build(spec: &Spec, store: &Path) -> Result<Value> {
    if spec.schema != format::STORE_SCHEMA_VERSION {
        drop(WarmCore::new(
            DIM,
            Some(store.display().to_string()),
            Some(config_for(spec)?),
        )?);
        seed_schema(store, spec.schema)?;
    }
    if let Some(version) = spec.manifest {
        // The engine refuses an arena without a store.json, so let it write the schema first,
        // then swap the empty current-format arena for one with an older manifest.
        drop(WarmCore::new(
            DIM,
            Some(store.display().to_string()),
            Some(config_for(spec)?),
        )?);
        std::fs::remove_dir_all(store.join("vectors_data.bin"))?;
        for stale in ["indices.json", "indices.json.crc32", "documents.text"] {
            let _ = std::fs::remove_file(store.join(stale));
        }
        seed_manifest(store, version)?;
    }
    let core = Arc::new(WarmCore::new(
        DIM,
        Some(store.display().to_string()),
        Some(config_for(spec)?),
    )?);
    let dim = DIM as usize;
    let mut expect = match spec.kind {
        Kind::Text if spec.checkpoint => {
            let lexical = writerslogic_warm::core::documents::IngestOptions {
                encode: writerslogic_warm::core::documents::EncodeMode::Lexical,
                ..Default::default()
            };
            for document in corpus("text.json")? {
                core.memorize_document_with(document, &lexical)?;
            }
            core.delete_document("doc-obsolete")?;
            json!({
                "documents": expected_documents("text.json", &["doc-obsolete"])?,
                "absentDocuments": ["doc-obsolete"],
                "vectors": [],
                "relations": 0,
                "searches": [
                    {"query": "harbor master ship dawn", "top": "doc-harbor"},
                    {"query": "quarterly ledger invoice bank statement", "top": "doc-ledger"},
                    {"query": "crème brûlée café 東京", "top": "doc-unicode"},
                ],
            })
        }
        Kind::Text => {
            ops::populate(&core, corpus("text.json")?, dim, true)?;
            let mut documents = expected_documents("text.json", &["doc-obsolete"])?;
            if spec.compact {
                core.compact()?;
                core.memorize_document(serde_json::from_value(json!({
                    "id": "doc-after-compact", "text": "Written after the compaction, so a journal tail follows the new generation.",
                    "version": "1", "sourceUri": "notes/tail.md", "metadata": {"phase": "tail"}
                }))?)?;
                documents.push(json!({
                    "id": "doc-after-compact",
                    "text": "Written after the compaction, so a journal tail follows the new generation.",
                    "version": "1", "sourceUri": "notes/tail.md", "metadata": {"phase": "tail"}
                }));
            }
            if spec.caches || spec.shard {
                for n in 0..120u64 {
                    core.memorize(
                        format!("bulk-{n:03}"),
                        EntangledHVec::new_deterministic(dim, 1000 + n),
                    )?;
                }
                if spec.shard {
                    core.maintain_indices()?;
                }
                core.train_ivf()?;
                core.train_nsg()?;
            }
            json!({
                "documents": documents,
                "absentDocuments": ["doc-obsolete"],
                "vectors": [{"id": "vec-a", "seed": 7}, {"id": "vec-b", "seed": 8}],
                "relations": 1,
                "searches": [
                    {"query": "harbor master ship dawn", "top": "doc-harbor"},
                    {"query": "quarterly ledger invoice bank statement", "top": "doc-ledger"},
                    {"query": "intermediate representation optimization register allocator", "top": "doc-compiler"},
                    {"query": "thirty daily snapshots twelve monthly archives", "top": "doc-backup"},
                    {"query": "crème brûlée café 東京", "top": "doc-unicode"},
                ],
            })
        }
        Kind::Embed => {
            ops::populate(&core, corpus("embed.json")?, dim, false)?;
            json!({
                "documents": expected_documents("embed.json", &[])?,
                "searches": [
                    {"query": "", "embedding": [1.0, 0.0, 0.0], "top": "emb-north"},
                    {"query": "", "embedding": [0.0, 1.0, 0.0], "top": "emb-east"},
                    {"query": "", "embedding": [0.0, 0.0, 1.0], "top": "emb-up"},
                ],
            })
        }
        Kind::Governance => {
            core.memorize("alias".into(), EntangledHVec::new_deterministic(dim, 7))?;
            core.memorize_triplet("owned".into(), "alice".into(), "knows".into(), "bob".into())?;
            core.govern_memory()?;
            json!({
                "vectors": [{"id": "alias", "seed": 7}],
                "meaning": {"atoms": 3, "composites": 1, "triples": 1},
            })
        }
        Kind::Memory => {
            let facts = memory_facts(core.clone())?;
            core.flush()?;
            facts
        }
    };
    core.flush()?;
    if spec.kind != Kind::Memory {
        expect["vectorCount"] = json!(core.vector_count());
    }
    drop(core);
    if spec.text_v1 {
        // Version 1 of documents.text had no header and stored plaintext chunk text back to
        // back; the engine truncates it on open, so only its shape matters.
        let mut bytes = Vec::new();
        for entry in expected_documents("text.json", &["doc-obsolete"])? {
            bytes.extend_from_slice(entry["text"].as_str().unwrap().as_bytes());
        }
        std::fs::write(store.join("documents.text"), bytes)?;
    }
    if spec.no_text {
        std::fs::remove_file(store.join("documents.text"))?;
    }
    if spec.checkpoint {
        let generations = std::fs::read_dir(store.join("vectors_data.bin"))?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().join("documents.ckpt").is_file())
            .count();
        ensure!(generations == 1, "closing wrote no document checkpoint");
    }
    Ok(expect)
}

fn meta_for(spec: &Spec, expect: Value) -> Value {
    let format_version = spec.manifest.unwrap_or(format::STORAGE_FORMAT_CURRENT);
    let mut features = Vec::new();
    if spec.encrypted {
        features.push("security");
    }
    if spec.encoder == "multiscale-capped-union-v1" {
        features.push("experimental");
    }
    let mut meta = json!({
        "name": spec.name,
        "description": spec.description,
        "provenance": spec.provenance,
        "storageFormat": format_version,
        "schema": spec.schema,
        "transaction": if spec.schema >= 3 { "HMS-TXN-3" } else { "HMS-TXN-2" },
        "textSegment": if spec.no_text { Value::Null } else if spec.text_v1 { json!(1) } else { json!(format::TEXT_SEGMENT_VERSION) },
        "dimensions": DIM,
        "textEncoder": spec.encoder,
        "encrypted": spec.encrypted,
        "audit": spec.audit,
        "embeddingSpace": if spec.kind == Kind::Embed { serde_json::to_value(embedding_space()).unwrap() } else { Value::Null },
        "features": features,
        "verifyVerdict": "ok",
        "expect": expect,
    });
    if spec.kind == Kind::Governance {
        meta["meaningEnabled"] = true.into();
    }
    meta
}

/// Writes every golden store that does not exist yet. Existing goldens are never rewritten:
/// they are the record of what old code wrote.
fn generate(only: Option<&str>) -> Result<Vec<String>> {
    let mut written = Vec::new();
    for spec in specs() {
        if only.is_some_and(|o| o != spec.name) {
            continue;
        }
        let destination = golden_root().join(spec.name);
        if destination.exists() {
            continue;
        }
        if (spec.encrypted && !cfg!(feature = "security"))
            || (spec.encoder.contains("capped") && !cfg!(feature = "experimental"))
        {
            anyhow::bail!(
                "golden {} needs the security and experimental features",
                spec.name
            );
        }
        let scratch = tempfile::tempdir()?;
        let store = scratch.path().join("store");
        let expect = build(&spec, &store)?;
        let source = if spec.snapshot {
            let snapshot = scratch.path().join("snapshot");
            migrate_store(&store, &snapshot)?;
            snapshot
        } else {
            store
        };
        pack(&source, &destination, meta_for(&spec, expect))?;
        written.push(spec.name.to_owned());
    }
    write_manifest()?;
    Ok(written)
}

/// Imports the stores written by the v0.6.1 release build (`gen_v061`), see the README.
fn import_released(root: &Path) -> Result<Vec<String>> {
    let mut written = Vec::new();
    for (name, encrypted, description) in [
        ("v061-plain", false, "Written by the v0.6.1 release (tag v0.6.1, commit 28c4f94): storage format 2 with 1 GiB preallocated segments, a zero terminator instead of end markers, no checksum sidecars, no text segment."),
        ("v061-encrypted", true, "Written by the v0.6.1 release with the security feature: every frame is [nonce:12][AES-256-GCM ciphertext+tag]."),
    ] {
        let destination = golden_root().join(name);
        if destination.exists() {
            continue;
        }
        let mut documents = expected_documents("text.json", &["doc-obsolete"])?;
        documents.iter_mut().for_each(|d| {
            if d["sourceUri"].is_null() {
                d["sourceUri"] = Value::Null;
            }
        });
        let expect = json!({
            "documents": documents,
            "absentDocuments": ["doc-obsolete"],
            "vectors": [{"id": "vec-a", "seed": 7}, {"id": "vec-b", "seed": 8}],
            "relations": 1,
            "searches": [
                {"query": "harbor master ship dawn", "top": "doc-harbor"},
                {"query": "quarterly ledger invoice bank statement", "top": "doc-ledger"},
                {"query": "intermediate representation optimization register allocator", "top": "doc-compiler"},
                {"query": "thirty daily snapshots twelve monthly archives", "top": "doc-backup"},
                {"query": "crème brûlée café 東京", "top": "doc-unicode"},
            ],
        });
        let meta = json!({
            "name": name,
            "description": description,
            "provenance": "released-code",
            "producedBy": "git archive v0.6.1; tests/golden/generator/gen_v061.rs built as an example (cargo build --release --features security --example gen_v061); run as `gen_v061 <out> tests/golden/corpus/text.json 1024 [passphrase]`",
            "storageFormat": 2,
            "schema": 2,
            "transaction": "HMS-TXN-2",
            "textSegment": Value::Null,
            "dimensions": DIM,
            "textEncoder": "multiscale-words-v1",
            "encrypted": encrypted,
            "audit": false,
            "embeddingSpace": Value::Null,
            "features": if encrypted { json!(["security"]) } else { json!([]) },
            "verifyVerdict": "ok",
            "expect": expect,
        });
        pack(&root.join(name), &destination, meta)?;
        written.push(name.to_owned());
    }
    write_manifest()?;
    Ok(written)
}

/// Goldens derived from others by removing files, which reproduces a store state an older
/// release could have left.
fn derive() -> Result<Vec<String>> {
    let mut written = Vec::new();
    let derived: [(&str, &str, &str, &[&str], Value); 2] = [
        (
            "v1-premanifest",
            "v061-plain",
            "A store with no arena manifest (storage format 1): the v0.6.1 store minus vectors_data.bin/format.json. The engine reads it as format 2 and writes the manifest on open.",
            &["vectors_data.bin/format.json"],
            json!({"storageFormat": 1}),
        ),
        (
            "rejected-pre-schema",
            "v061-plain",
            "A store with no store.json (written before stores recorded their embedding schema). The engine must refuse it with the reencode message and leave it untouched.",
            &["store.json"],
            json!({"verifyVerdict": "unsupported", "expectOpenError": "legacy store has no embedding schema"}),
        ),
    ];
    for (name, from, description, remove, overrides) in derived {
        let destination = golden_root().join(name);
        if destination.exists() {
            continue;
        }
        let source = golden_root().join(from);
        if !source.exists() {
            continue;
        }
        let scratch = tempfile::tempdir()?;
        let store = scratch.path().join("store");
        let mut original: Value =
            serde_json::from_slice(&std::fs::read(source.join("golden.json"))?)?;
        copy_tree(&source, &store, &["golden.json"])?;
        for relative in remove {
            std::fs::remove_file(store.join(relative))?;
        }
        if let Value::Object(map) = overrides {
            for (key, value) in map {
                original[&key] = value;
            }
        }
        original["name"] = json!(name);
        original["description"] = json!(description);
        original["provenance"] = json!("derived");
        original["derivedFrom"] = json!(from);
        // Segments are already trimmed and padTo is inherited: copy the files as they are.
        copy_tree(&store, &destination, &[])?;
        std::fs::write(destination.join("golden.json"), json_file(&original)?)?;
        written.push(name.to_owned());
    }
    write_manifest()?;
    Ok(written)
}

fn manifest_value() -> Result<Value> {
    let mut goldens = serde_json::Map::new();
    for golden in goldens_list()? {
        let files: serde_json::Map<String, Value> = file_fingerprints(&golden.dir)?
            .into_iter()
            .map(|(k, (len, crc))| (k, json!([len, format!("{crc:08x}")])))
            .collect();
        goldens.insert(golden.name, Value::Object(files));
    }
    let corpus: serde_json::Map<String, Value> = file_fingerprints(&golden_root().join("corpus"))?
        .into_iter()
        .map(|(k, (len, crc))| (k, json!([len, format!("{crc:08x}")])))
        .collect();
    Ok(json!({
        "comment": "Length and CRC32 of every golden file. Golden stores are append-only: CI rejects a change to an existing entry (scripts/format-gate.sh).",
        "corpus": corpus,
        "goldens": goldens,
    }))
}

fn goldens_list() -> Result<Vec<Golden>> {
    goldens()
}

fn write_manifest() -> Result<()> {
    std::fs::write(
        golden_root().join("MANIFEST.json"),
        json_file(&manifest_value()?)?,
    )?;
    Ok(())
}

/// `WARM_GOLDEN_WRITE=1 cargo test --features security,experimental --test format_golden
/// generate_goldens -- --ignored --nocapture`; `WARM_V061_DIR` points at the directory holding
/// the stores `gen_v061` wrote.
#[test]
#[ignore = "writes tests/golden; run deliberately"]
fn generate_goldens() -> Result<()> {
    ensure!(
        std::env::var_os("WARM_GOLDEN_WRITE").is_some(),
        "set WARM_GOLDEN_WRITE=1"
    );
    let mut written = Vec::new();
    if let Some(root) = std::env::var_os("WARM_V061_DIR") {
        written.extend(import_released(Path::new(&root))?);
    }
    written.extend(derive()?);
    written.extend(generate(std::env::var("WARM_GOLDEN_ONLY").ok().as_deref())?);
    println!("wrote goldens: {written:?}");
    Ok(())
}

// ---------------------------------------------------------------------------------------
// The tests CI runs.
// ---------------------------------------------------------------------------------------

#[test]
fn golden_files_match_their_manifest() -> Result<()> {
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(golden_root().join("MANIFEST.json"))?)?;
    assert_eq!(
        manifest,
        manifest_value()?,
        "tests/golden changed without the manifest (and golden stores are append-only: an \
         existing golden file must never change; see tests/golden/README.md)"
    );
    Ok(())
}

/// Every version number this build writes has a golden store, and no golden claims a version
/// newer than this build writes. Moving a constant fails this until a golden for the new
/// value is committed.
#[test]
fn every_current_version_has_a_golden() -> Result<()> {
    let all = goldens()?;
    ensure!(!all.is_empty(), "no golden stores");
    let current = format::CURRENT;
    let has = |pred: &dyn Fn(&Golden) -> bool| all.iter().any(pred);
    assert!(
        has(&|g| g.meta["storageFormat"] == current.storage_format_current),
        "no golden store has storage format {}",
        current.storage_format_current
    );
    for version in current.storage_format_oldest..=current.storage_format_current {
        assert!(
            has(&|g| g.meta["storageFormat"] == version),
            "no golden store has storage format {version}"
        );
    }
    assert!(has(&|g| g.meta["storageFormat"] == 1), "no format-1 golden");
    assert!(has(&|g| g.meta["schema"] == current.store_schema));
    assert!(has(&|g| g.meta["transaction"] == current.transaction_magic));
    assert!(has(&|g| g.meta["textSegment"] == current.text_segment));
    assert!(
        has(&|g| g.meta["textSegment"] == 1),
        "no golden with a version-1 text segment"
    );
    assert!(has(&|g| g.meta["encrypted"] == true));
    assert!(has(
        &|g| g.meta["textEncoder"] == "multiscale-capped-union-v1"
    ));
    assert!(has(&|g| !g.meta["embeddingSpace"].is_null()));
    for golden in &all {
        assert!(
            golden.meta["storageFormat"].as_u64().unwrap()
                <= u64::from(current.storage_format_current)
        );
        assert!(golden.meta["schema"].as_u64().unwrap() <= u64::from(current.store_schema));
    }
    Ok(())
}

fn expect_open_error(golden: &Golden, store: &Path, needle: &str) -> Result<()> {
    let before = file_fingerprints(store)?;
    let config = golden.config()?;
    let error = match WarmCore::new(
        golden.dimensions(),
        Some(store.display().to_string()),
        Some(config),
    ) {
        Ok(_) => anyhow::bail!("{} opened but must be refused", golden.name),
        Err(error) => format!("{error:#}"),
    };
    ensure!(error.contains(needle), "{}: {error}", golden.name);
    let after = file_fingerprints(store)?;
    ensure!(
        before
            .keys()
            .all(|k| k.ends_with(".hms.lock") || after.get(k) == before.get(k)),
        "{}: a refused open modified the store",
        golden.name
    );
    Ok(())
}

fn check_memory(core: &Arc<WarmCore>, expect: &Value) -> Result<()> {
    let memory = Memory::new(core.clone());
    for recall in expect["recalls"].as_array().into_iter().flatten() {
        let hits = memory.recall(
            recall["query"].as_str().unwrap(),
            &RecallOptions {
                as_of: recall["asOf"].as_i64(),
                k: Some(5),
                ..Default::default()
            },
        )?;
        if let Some(ids) = recall["ids"].as_array() {
            let got: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
            let want: Vec<&str> = ids.iter().filter_map(Value::as_str).collect();
            ensure!(
                got == want,
                "recall {:?}: {got:?}, expected {want:?}",
                recall["query"]
            );
        }
        if let Some(top) = recall["top"].as_str() {
            ensure!(
                hits.first().map(|h| h.id.as_str()) == Some(top),
                "recall top {top}"
            );
        }
    }
    for document in expect["facts"].as_array().into_iter().flatten() {
        let id = document["id"].as_str().unwrap();
        let hits = core.search_documents(
            document["text"].as_str().unwrap(),
            &writerslogic_warm::SearchOptions {
                document_ids: Some(vec![id.into()]),
                k: Some(1),
                candidate_limit: Some(10),
                ..Default::default()
            },
        )?;
        let hit = hits
            .first()
            .with_context(|| format!("fact {id} not found"))?;
        let by_version = hit.version.starts_with("mem1:");
        let by_metadata = hit.metadata.get("_mem").is_some();
        match document["stampStyle"].as_str() {
            Some("version") => ensure!(by_version && !by_metadata, "{id}: stamp is not in version"),
            Some("metadata") => {
                ensure!(by_metadata && !by_version, "{id}: stamp is not in metadata")
            }
            _ => {}
        }
    }
    Ok(())
}

fn open_and_check(golden: &Golden, store: &Path) -> Result<()> {
    let core = Arc::new(WarmCore::new(
        golden.dimensions(),
        Some(store.display().to_string()),
        Some(golden.config()?),
    )?);
    if golden.meta["expect"]["recalls"].is_array() {
        check_memory(&core, &golden.meta["expect"])?;
    } else {
        check_expectations(&core, &golden.meta["expect"], golden.dimensions())?;
    }
    if golden.name == "v6-audit-effects" {
        let entries = core.audit_since(0)?;
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.op as u8)
                .collect::<Vec<_>>(),
            [1, 1, 1, 2, 3]
        );
        assert!(entries.iter().all(|entry| entry.effect_id.is_some()));
        assert!(core.durable_audit_delivery());
        assert_eq!(core.pending_audit_effects(), 0);
    }
    Ok(())
}

/// The golden test proper: the current code opens every golden store, and what it finds is
/// what the golden says, before and after the open has repaired and rewritten the store.
#[test]
fn every_golden_opens_with_current_code() -> Result<()> {
    let mut opened = 0;
    for golden in goldens()? {
        let scratch = tempfile::tempdir()?;
        let store = scratch.path().join("store");
        golden.unpack(&store)?;
        if let Some(needle) = golden.str("expectOpenError") {
            expect_open_error(&golden, &store, needle)?;
            continue;
        }
        if !golden.openable_here() {
            // Without the feature the store cannot be created, so it must be refused cleanly.
            let result = WarmCore::new(
                golden.dimensions(),
                Some(store.display().to_string()),
                Some(golden.config().unwrap_or_default()),
            );
            ensure!(
                result.is_err(),
                "{} opened without its feature",
                golden.name
            );
            continue;
        }
        open_and_check(&golden, &store).with_context(|| format!("golden {}", golden.name))?;
        // A second open of what the first open wrote.
        open_and_check(&golden, &store)
            .with_context(|| format!("golden {} (reopened)", golden.name))?;
        opened += 1;
    }
    ensure!(opened > 0, "no golden opened");
    Ok(())
}

/// `warm-format verify` agrees with what each golden declares about itself, and a deep check
/// (a private copy opened by the engine) agrees with the journal replay.
#[test]
fn verify_agrees_with_every_golden() -> Result<()> {
    for golden in goldens()? {
        let scratch = tempfile::tempdir()?;
        let store = scratch.path().join("store");
        golden.unpack(&store)?;
        let options = VerifyOptions {
            passphrase_env: None,
            deep: false,
        };
        let report = verify_path(&store, &options)?;
        let verdict = match golden.str("verifyVerdict").unwrap_or("ok") {
            "ok" => Verdict::Ok,
            "unsupported" => Verdict::Unsupported,
            other => anyhow::bail!("unknown verdict {other}"),
        };
        ensure!(
            report.verdict == verdict,
            "{}: {:?}\n{:#?}",
            golden.name,
            report.verdict,
            report.findings
        );
        if verdict != Verdict::Ok {
            continue;
        }
        let v = &report.versions;
        ensure!(
            u64::from(v.storage_format.unwrap_or(0))
                == golden.meta["storageFormat"].as_u64().unwrap(),
            "{}: storage format {:?}",
            golden.name,
            v.storage_format
        );
        ensure!(
            v.store_schema.map(u64::from) == golden.meta["schema"].as_u64(),
            "{}: schema {:?}",
            golden.name,
            v.store_schema
        );
        ensure!(v.encrypted == Some(golden.meta["encrypted"] == true));
        ensure!(v.text_encoder.as_deref() == golden.str("textEncoder"));
        if golden.meta["encrypted"] == true {
            ensure!(!report.payloads_authenticated, "no passphrase was given");
        }
        // Authenticated payloads and the engine cross-check need the security feature when
        // the store is encrypted.
        if golden.meta["encrypted"] == true && !cfg!(feature = "security") {
            continue;
        }
        if golden
            .str("textEncoder")
            .is_some_and(|e| e.contains("capped"))
            && !cfg!(feature = "experimental")
        {
            continue;
        }
        std::env::set_var("WARM_GOLDEN_PASSPHRASE", PASSPHRASE);
        let deep = verify_path(
            &store,
            &VerifyOptions {
                passphrase_env: Some("WARM_GOLDEN_PASSPHRASE".into()),
                deep: true,
            },
        )?;
        ensure!(
            deep.verdict == Verdict::Ok,
            "{}: deep {:?}",
            golden.name,
            deep.findings
        );
        ensure!(deep.payloads_authenticated);
        ensure!(
            deep.checks.contains(&"engine open"),
            "{}: {:?}",
            golden.name,
            deep.checks
        );
        if let Some(count) = golden.meta["expect"]["vectorCount"]
            .as_u64()
            .filter(|&c| c > 0)
        {
            ensure!(
                deep.live_vectors == Some(count as usize),
                "{}: live vectors {:?}",
                golden.name,
                deep.live_vectors
            );
        }
        if let Some(documents) = golden.meta["expect"]["documents"].as_array() {
            ensure!(
                deep.live_documents == Some(documents.len()),
                "{}",
                golden.name
            );
        }
    }
    Ok(())
}

/// What the current code writes for the canonical corpus equals the newest golden, frame for
/// frame, once decoded. A change to the journal encoding, the encoders, tokenization,
/// chunking or any serialized field fails this, and so cannot land without a new golden.
#[test]
fn canonical_corpus_regenerates_the_current_goldens() -> Result<()> {
    for spec in specs() {
        // A checkpoint binds the random generation name.
        let deterministic =
            !spec.encrypted && !spec.compact && !spec.caches && !spec.shard && !spec.checkpoint;
        if !deterministic || spec.manifest.is_some() {
            continue;
        }
        if spec.encoder.contains("capped") && !cfg!(feature = "experimental") {
            continue;
        }
        let golden = goldens()?
            .into_iter()
            .find(|g| g.name == spec.name)
            .with_context(|| format!("no golden {}", spec.name))?;
        let scratch = tempfile::tempdir()?;
        let fresh = scratch.path().join("fresh");
        build(&spec, &fresh)?;
        let committed = scratch.path().join("committed");
        golden.unpack(&committed)?;
        ensure!(
            decoded_frames(&fresh)? == decoded_frames(&committed)?,
            "{}: the journal the current code writes for the canonical corpus differs from the \
             committed golden. If the change is intended it is a format change: bump the \
             version, keep reading the old golden, and commit a new golden (see \
             docs/STABILITY.md).",
            spec.name
        );
    }
    Ok(())
}

/// The decompressed payload of every journal frame, in order.
fn decoded_frames(store: &Path) -> Result<Vec<Vec<u8>>> {
    let mut payloads = Vec::new();
    for id in 0.. {
        let path = store.join("vectors_data.bin").join(format!("seg_{id}.bin"));
        if !path.exists() {
            break;
        }
        let bytes = std::fs::read(&path)?;
        format::scan::scan_segment(&bytes, true, |_, payload| payloads.push(payload.to_vec()));
    }
    Ok(payloads)
}

/// Writing to an old store keeps its format and the result still opens: the engine never
/// silently converts a format.
#[test]
fn old_formats_stay_writable_in_place() -> Result<()> {
    for golden in goldens()? {
        if golden.str("expectOpenError").is_some()
            || !golden.openable_here()
            || golden.meta["expect"]["documents"].is_null()
            || golden.meta["expect"]["recalls"].is_array()
            || golden.meta["embeddingSpace"] != Value::Null
        {
            continue;
        }
        let scratch = tempfile::tempdir()?;
        let store = scratch.path().join("store");
        golden.unpack(&store)?;
        {
            let core = WarmCore::new(
                golden.dimensions(),
                Some(store.display().to_string()),
                Some(golden.config()?),
            )?;
            core.memorize_document(serde_json::from_value(json!({
                "id": "added-later", "text": "zephyrous quokka added by a newer build", "version": "9",
                "metadata": {"era": "later"}
            }))?)?;
            core.flush()?;
        }
        let core = WarmCore::new(
            golden.dimensions(),
            Some(store.display().to_string()),
            Some(golden.config()?),
        )?;
        ensure!(core.has_document("added-later"), "{}", golden.name);
        let mut expect = golden.meta["expect"].clone();
        expect["vectorCount"] = Value::Null;
        check_expectations(&core, &expect, golden.dimensions())
            .map_err(|e| e.context(golden.name.clone()))?;
        drop(core);
        let report = verify_path(&store, &VerifyOptions::default())?;
        ensure!(
            report.verdict == Verdict::Ok,
            "{}: {:?}",
            golden.name,
            report.findings
        );
        ensure!(
            u64::from(report.versions.storage_format.unwrap())
                == golden.meta["storageFormat"].as_u64().unwrap().max(2),
            "{}: the engine changed the storage format ({:?})",
            golden.name,
            report.versions.storage_format
        );
    }
    Ok(())
}
