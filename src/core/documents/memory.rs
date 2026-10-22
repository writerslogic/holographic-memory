// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! A self-contained, in-memory document store: the same chunking, stemmed BM25, metadata
//! filters and optional dense retrieval as the engine's document path, without the journal,
//! shards, locks or files. It is what the WebAssembly binding runs, and its serialized form
//! (`export` / `import`) is the unit that persistence adapters store.

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::Arc;

use super::{
    build_dense_index, capture_dense_index, capture_search, execute_search, prepare,
    prepare_search, publish_dense_index, DocumentInput, DocumentResult, DocumentStore,
    SearchOptions, StoredDocument,
};
use crate::core::encoding::TextEncoder;

const MAGIC: &[u8; 4] = b"HMSD";
const FORMAT_VERSION: u8 = 1;
/// Fixed header: magic, version, dimensions, embedding dimensions, document count,
/// uncompressed length, crc32 of the uncompressed payload.
const HEADER_BYTES: usize = 4 + 1 + 4 + 4 + 4 + 8 + 4;
/// Upper bound on an uncompressed snapshot, so a forged length cannot request an unbounded
/// allocation (a wasm32 linear memory tops out at 4 GiB).
const MAX_SNAPSHOT_BYTES: usize = 1 << 30;
const MAX_DIMENSIONS: usize = 1 << 20;
const MAX_BATCH: usize = 4096;
/// A chunk is at most 64 KiB of text, so no honest chunk counts more words than this.
const MAX_CHUNK_WORDS: u64 = 1 << 16;
/// Keeps word totals inside a 32-bit `usize`.
const MAX_STORE_WORDS: u64 = (i32::MAX / 2) as u64;

#[derive(Serialize, Deserialize)]
struct Snapshot {
    encoder: String,
    documents: Vec<StoredDocument>,
}

#[derive(Serialize)]
struct SnapshotRef<'a> {
    encoder: &'a str,
    documents: Vec<&'a StoredDocument>,
}

/// Counts describing a [`MemoryDocumentStore`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreStats {
    pub documents: usize,
    pub chunks: usize,
    pub embedded_chunks: usize,
    pub dimensions: usize,
    pub embedding_dimensions: Option<usize>,
    pub index_trained: bool,
}

pub struct MemoryDocumentStore {
    dimensions: usize,
    embedding_dimensions: Option<usize>,
    encoder: TextEncoder,
    store: DocumentStore,
}

impl MemoryDocumentStore {
    /// `dimensions` is the sparse text-vector dimension; `embedding_dimensions` enables
    /// caller-supplied dense embeddings for semantic and hybrid search.
    pub fn new(dimensions: usize, embedding_dimensions: Option<usize>) -> Result<Self> {
        ensure!(
            (1..=MAX_DIMENSIONS).contains(&dimensions),
            "dimensions must be in 1..={MAX_DIMENSIONS}"
        );
        ensure!(
            embedding_dimensions.is_none_or(|d| (1..=MAX_DIMENSIONS).contains(&d)),
            "embedding dimensions must be in 1..={MAX_DIMENSIONS}"
        );
        Ok(Self {
            dimensions,
            embedding_dimensions,
            encoder: TextEncoder::default(),
            store: DocumentStore::default(),
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn embedding_dimensions(&self) -> Option<usize> {
        self.embedding_dimensions
    }

    /// Stores or replaces a document and returns its chunk count.
    pub fn memorize(&mut self, input: DocumentInput) -> Result<u32> {
        let document = self.prepare(input)?;
        let count = document.chunks.len() as u32;
        self.store.insert(document.id.clone(), document);
        Ok(count)
    }

    /// Stores or replaces several documents; either all are stored or none is.
    pub fn memorize_many(&mut self, inputs: Vec<DocumentInput>) -> Result<Vec<u32>> {
        ensure!(
            inputs.len() <= MAX_BATCH,
            "a batch holds at most {MAX_BATCH} documents"
        );
        let mut seen = BTreeSet::new();
        for input in &inputs {
            ensure!(
                seen.insert(input.id.as_str()),
                "document id {:?} appears twice in the batch",
                input.id
            );
        }
        let documents = inputs
            .into_iter()
            .map(|input| self.prepare(input))
            .collect::<Result<Vec<_>>>()?;
        let counts = documents.iter().map(|d| d.chunks.len() as u32).collect();
        for document in documents {
            self.store.insert(document.id.clone(), document);
        }
        Ok(counts)
    }

    fn prepare(&self, input: DocumentInput) -> Result<StoredDocument> {
        prepare(
            input,
            self.dimensions,
            self.embedding_dimensions,
            self.encoder,
        )
    }

    pub fn search(&self, text: &str, options: &SearchOptions) -> Result<Vec<DocumentResult>> {
        let prepared = prepare_search(text, options, self.embedding_dimensions)?;
        let captured = capture_search(&self.store, vec![prepared]);
        Ok(execute_search(&captured, 0))
    }

    /// Removes a document; returns whether it existed.
    pub fn delete(&mut self, id: &str) -> bool {
        self.store.remove(id).is_some()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.store.contains_key(id)
    }

    /// Builds the dense index over the stored embeddings. A store with no embedding space or
    /// too few embedded chunks is left as it is; returns whether an index is now current.
    pub fn train_index(&mut self) -> Result<bool> {
        let Some(dimensions) = self.embedding_dimensions else {
            return Ok(false);
        };
        let Some(snapshot) = capture_dense_index(&self.store, dimensions)? else {
            return Ok(self.store.dense_index_trained());
        };
        let Some(trained) = build_dense_index(snapshot)? else {
            return Ok(self.store.dense_index_trained());
        };
        ensure!(
            publish_dense_index(&mut self.store, &trained),
            "store changed during index training"
        );
        Ok(true)
    }

    pub fn index_trained(&self) -> bool {
        self.store.dense_index_trained()
    }

    pub fn stats(&self) -> MemoryStoreStats {
        let documents = self.store.shared_values().count();
        let chunks = self.store.shared_values().map(|d| d.chunks.len()).sum();
        MemoryStoreStats {
            documents,
            chunks,
            embedded_chunks: self.store.embedded_count(),
            dimensions: self.dimensions,
            embedding_dimensions: self.embedding_dimensions,
            index_trained: self.store.dense_index_trained(),
        }
    }

    /// Ids of every stored document, sorted.
    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.store.shared_values().map(|d| d.id.clone()).collect();
        ids.sort_unstable();
        ids
    }

    /// Serializes the store: a checksummed header and an LZ4-compressed JSON payload.
    pub fn export(&self) -> Result<Vec<u8>> {
        let held: Vec<Arc<StoredDocument>> = self.store.shared_values().collect();
        let mut documents: Vec<&StoredDocument> = held.iter().map(Arc::as_ref).collect();
        documents.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        let count = documents.len();
        let snapshot = SnapshotRef {
            encoder: self.encoder.name(),
            documents,
        };
        let payload = serde_json::to_vec(&snapshot).context("serializing the store")?;
        ensure!(
            payload.len() <= MAX_SNAPSHOT_BYTES,
            "the store exceeds the {MAX_SNAPSHOT_BYTES}-byte snapshot limit"
        );
        let compressed = lz4_flex::block::compress(&payload);
        let mut out = Vec::with_capacity(HEADER_BYTES + compressed.len());
        out.extend_from_slice(MAGIC);
        out.push(FORMAT_VERSION);
        out.extend_from_slice(&(self.dimensions as u32).to_le_bytes());
        out.extend_from_slice(&(self.embedding_dimensions.unwrap_or(0) as u32).to_le_bytes());
        out.extend_from_slice(&(count as u32).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        out.extend_from_slice(&compressed);
        Ok(out)
    }

    /// Restores a store written by [`MemoryDocumentStore::export`]. The bytes are validated
    /// (magic, version, bounded length, checksum, per-document shape) before any is used.
    pub fn import(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= HEADER_BYTES, "snapshot is truncated");
        ensure!(&bytes[..4] == MAGIC, "not a holographic-memory snapshot");
        ensure!(
            bytes[4] == FORMAT_VERSION,
            "unsupported snapshot version {}",
            bytes[4]
        );
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let dimensions = word(5) as usize;
        let embedding_dimensions = Some(word(9) as usize).filter(|&d| d != 0);
        let expected_documents = word(13) as usize;
        let length = u64::from_le_bytes(bytes[17..25].try_into().unwrap());
        let checksum = word(25);
        let length = usize::try_from(length)
            .ok()
            .filter(|&n| n <= MAX_SNAPSHOT_BYTES)
            .context("snapshot payload length is out of range")?;
        let payload = lz4_flex::block::decompress(&bytes[HEADER_BYTES..], length)
            .map_err(|e| anyhow::anyhow!("snapshot payload is corrupt: {e}"))?;
        ensure!(
            payload.len() == length && crc32fast::hash(&payload) == checksum,
            "snapshot checksum mismatch"
        );
        let snapshot: Snapshot =
            serde_json::from_slice(&payload).context("snapshot payload is malformed")?;
        let encoder: TextEncoder = snapshot.encoder.parse()?;
        ensure!(
            snapshot.documents.len() == expected_documents,
            "snapshot document count mismatch"
        );
        let mut store = Self::new(dimensions, embedding_dimensions)?;
        store.encoder = encoder;
        let mut seen = BTreeSet::new();
        let mut total_words = 0u64;
        for document in snapshot.documents {
            store.validate_imported(&document, &mut total_words)?;
            if !seen.insert(document.id.clone()) {
                bail!("snapshot holds document {:?} twice", document.id);
            }
            store.store.insert(document.id.clone(), document);
        }
        Ok(store)
    }

    /// Every field the store's own invariants assert on is checked here, because a forged
    /// snapshot can carry a valid checksum and an assertion failure would abort the module.
    fn validate_imported(&self, document: &StoredDocument, total_words: &mut u64) -> Result<()> {
        ensure!(
            !document.id.is_empty() && document.id.len() <= 4096 && !document.id.contains('\0'),
            "snapshot holds an invalid document id"
        );
        ensure!(
            !document.chunks.is_empty() && document.chunks.len() <= 4096,
            "snapshot document {:?} has an invalid chunk count",
            document.id
        );
        for (index, chunk) in document.chunks.iter().enumerate() {
            ensure!(
                chunk.id
                    == format!(
                        "{}{}:{}:{}",
                        super::CHUNK_PREFIX,
                        document.id.len(),
                        document.id,
                        index
                    ),
                "snapshot chunk id does not match its document and position"
            );
            ensure!(
                chunk.vector.dim == self.dimensions
                    && chunk.vector.indices.windows(2).all(|w| w[0] < w[1])
                    && chunk
                        .vector
                        .indices
                        .last()
                        .is_none_or(|&i| (i as usize) < self.dimensions),
                "snapshot chunk vector is malformed"
            );
            ensure!(
                chunk.embedding.as_ref().is_none_or(|e| {
                    Some(e.len()) == self.embedding_dimensions && e.iter().all(|x| x.is_finite())
                }),
                "snapshot embedding does not match the store's embedding space"
            );
            ensure!(
                chunk.start_byte <= chunk.end_byte,
                "snapshot chunk has inverted byte offsets"
            );
            ensure!(
                !chunk.terms.is_whole_document()
                    || (chunk.vector_state == super::VectorState::Lexical
                        && document.chunks.len() == 1
                        && chunk.start_byte == 0
                        && chunk.end_byte as usize <= super::MAX_DOCUMENT_BYTES),
                "snapshot whole-document postings require one bounded lexical chunk"
            );
            let mut words = 0u64;
            for (_, &frequency) in &chunk.terms {
                ensure!(frequency > 0, "snapshot term has a zero frequency");
                words += u64::from(frequency);
            }
            ensure!(
                words == chunk.word_count as u64
                    && words
                        <= if chunk.terms.is_whole_document() {
                            super::MAX_DOCUMENT_BYTES as u64
                        } else {
                            MAX_CHUNK_WORDS
                        },
                "snapshot chunk word count disagrees with its terms"
            );
            *total_words += words;
            ensure!(
                *total_words <= MAX_STORE_WORDS,
                "snapshot exceeds the store word limit"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(id: &str, text: &str) -> DocumentInput {
        DocumentInput {
            id: id.into(),
            text: text.into(),
            source_uri: Some(format!("file:///{id}")),
            version: None,
            metadata: Some(serde_json::json!({"kind": "note", "n": 3})),
            chunk_words: Some(8),
            overlap_words: Some(2),
            store_text: None,
            embeddings: None,
            title_weight: None,
        }
    }

    fn corpus(store: &mut MemoryDocumentStore) {
        let texts = [
            ("a", "Rust ownership rules prevent data races at compile time without a garbage collector and keep memory use predictable across threads"),
            ("b", "The holographic reduced representation binds vectors with circular convolution and superposes many bindings into one trace"),
            ("c", "Okapi BM25 ranks passages by term frequency saturation and length normalization over an inverse document frequency weight"),
            ("d", "Gardening tips for spring include pruning roses and planting tomatoes after the last frost has passed in your region"),
        ];
        for (id, text) in texts {
            store.memorize(input(id, text)).unwrap();
        }
    }

    fn rendered(results: &[DocumentResult]) -> String {
        serde_json::to_string(results).unwrap()
    }

    #[test]
    fn search_ranks_filters_and_deletes() {
        let mut store = MemoryDocumentStore::new(1024, None).unwrap();
        corpus(&mut store);
        let hits = store
            .search("BM25 term frequency", &SearchOptions::default())
            .unwrap();
        assert_eq!(hits[0].document_id, "c");
        let only_d = store
            .search(
                "the",
                &SearchOptions {
                    document_ids: Some(vec!["d".into()]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(only_d.iter().all(|h| h.document_id == "d"));
        assert!(store.delete("c"));
        assert!(!store.delete("c"));
        let after = store
            .search("BM25 term frequency", &SearchOptions::default())
            .unwrap();
        assert!(after.iter().all(|h| h.document_id != "c"));
        assert_eq!(store.stats().documents, 3);
    }

    #[test]
    fn bm25_options_and_title_weight_change_scores() {
        let mut store = MemoryDocumentStore::new(1024, None).unwrap();
        corpus(&mut store);
        let base = store.search("memory", &SearchOptions::default()).unwrap();
        let tuned = store
            .search(
                "memory",
                &SearchOptions {
                    bm25_k1: Some(2.0),
                    bm25_b: Some(0.2),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_ne!(rendered(&base), rendered(&tuned));
        assert!(store
            .search(
                "x",
                &SearchOptions {
                    bm25_b: Some(1.5),
                    ..Default::default()
                }
            )
            .is_err());

        let build = |title_weight: Option<u32>| {
            let mut store = MemoryDocumentStore::new(1024, None).unwrap();
            let filler = "alpha beta gamma delta epsilon theta iota kappa lambda sigma";
            store
                .memorize(DocumentInput {
                    title_weight,
                    ..input("titled", &format!("Zebra\n{filler}"))
                })
                .unwrap();
            store
                .memorize(input("body", &format!("{filler} zebra")))
                .unwrap();
            store.memorize(input("other", filler)).unwrap();
            store.search("zebra", &SearchOptions::default()).unwrap()
        };
        let plain = build(None);
        let weighted = build(Some(8));
        let score = |hits: &[DocumentResult], id: &str| {
            hits.iter().find(|h| h.document_id == id).unwrap().score
        };
        assert_eq!(weighted[0].document_id, "titled");
        assert!(score(&weighted, "titled") > score(&plain, "titled"));
    }

    #[test]
    fn export_import_round_trips_search_results_exactly() {
        let mut store = MemoryDocumentStore::new(1024, Some(3)).unwrap();
        corpus(&mut store);
        let mut with_embedding = input("e", "dense vector document about retrieval");
        with_embedding.embeddings = Some(vec![vec![0.2, 0.5, 0.1]]);
        store.memorize(with_embedding).unwrap();
        let bytes = store.export().unwrap();
        let restored = MemoryDocumentStore::import(&bytes).unwrap();
        assert_eq!(restored.stats(), store.stats());
        for query in ["holographic binding", "frost tomatoes", "BM25"] {
            let options = SearchOptions {
                embedding: Some(vec![0.2, 0.5, 0.1]),
                ..Default::default()
            };
            assert_eq!(
                rendered(&store.search(query, &options).unwrap()),
                rendered(&restored.search(query, &options).unwrap())
            );
        }
        assert_eq!(restored.export().unwrap(), bytes);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn matches_the_engine_document_path() {
        let dir = tempfile::tempdir().unwrap();
        let engine =
            crate::core::HmsCore::new(1024, Some(dir.path().display().to_string()), None).unwrap();
        let mut store = MemoryDocumentStore::new(1024, None).unwrap();
        corpus(&mut store);
        for (id, text) in [
            ("a", "Rust ownership rules prevent data races at compile time without a garbage collector and keep memory use predictable across threads"),
            ("b", "The holographic reduced representation binds vectors with circular convolution and superposes many bindings into one trace"),
            ("c", "Okapi BM25 ranks passages by term frequency saturation and length normalization over an inverse document frequency weight"),
            ("d", "Gardening tips for spring include pruning roses and planting tomatoes after the last frost has passed in your region"),
        ] {
            engine.memorize_document(input(id, text)).unwrap();
        }
        for query in [
            "BM25 term frequency",
            "holographic trace",
            "frost",
            "memory threads",
        ] {
            let options = SearchOptions::default();
            assert_eq!(
                rendered(&store.search(query, &options).unwrap()),
                rendered(&engine.search_documents(query, &options).unwrap()),
                "query {query:?}"
            );
        }
    }

    #[test]
    fn import_rejects_tampered_and_truncated_snapshots() {
        let mut store = MemoryDocumentStore::new(1024, None).unwrap();
        corpus(&mut store);
        let bytes = store.export().unwrap();
        assert!(MemoryDocumentStore::import(&bytes[..bytes.len() - 1]).is_err());
        assert!(MemoryDocumentStore::import(&bytes[..HEADER_BYTES - 1]).is_err());
        let mut flipped = bytes.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 0x55;
        assert!(MemoryDocumentStore::import(&flipped).is_err());
        let mut huge = bytes.clone();
        huge[17..25].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(MemoryDocumentStore::import(&huge).is_err());
        let mut wrong_magic = bytes;
        wrong_magic[0] = b'X';
        assert!(MemoryDocumentStore::import(&wrong_magic).is_err());
    }

    fn reseal(original: &[u8], edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let length = u64::from_le_bytes(original[17..25].try_into().unwrap()) as usize;
        let payload = lz4_flex::block::decompress(&original[HEADER_BYTES..], length).unwrap();
        let mut json: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        edit(&mut json);
        let payload = serde_json::to_vec(&json).unwrap();
        let mut out = original[..17].to_vec();
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        out.extend_from_slice(&lz4_flex::block::compress(&payload));
        out
    }

    fn first_term(
        json: &mut serde_json::Value,
    ) -> (&mut serde_json::Map<String, serde_json::Value>, String) {
        let terms = json["documents"][0]["chunks"][0]["terms"]
            .as_object_mut()
            .unwrap();
        let first = terms.keys().next().unwrap().clone();
        (terms, first)
    }

    #[test]
    fn forged_snapshots_with_valid_checksums_are_rejected_not_asserted() {
        let mut store = MemoryDocumentStore::new(1024, Some(3)).unwrap();
        corpus(&mut store);
        let mut embedded = input("e", "dense vector document");
        embedded.embeddings = Some(vec![vec![0.2, 0.5, 0.1]]);
        store.memorize(embedded).unwrap();
        let good = store.export().unwrap();
        assert!(MemoryDocumentStore::import(&reseal(&good, |_| {})).is_ok());
        type Edit = fn(&mut serde_json::Value);
        let cases: [(&str, Edit); 9] = [
            ("huge word count", |j| {
                j["documents"][0]["chunks"][0]["word_count"] = 4_000_000_000u64.into()
            }),
            ("word count mismatch", |j| {
                j["documents"][0]["chunks"][0]["word_count"] = 1.into()
            }),
            ("zero term frequency", |j| {
                let (terms, first) = first_term(j);
                terms.insert(first, 0.into());
            }),
            ("term frequency over u32", |j| {
                let (terms, first) = first_term(j);
                terms.insert(first, 4_000_000_000u64.into());
            }),
            ("foreign chunk id", |j| {
                j["documents"][0]["chunks"][0]["id"] = "other".into()
            }),
            ("vector index past dimension", |j| {
                j["documents"][0]["chunks"][0]["vector"]["indices"] = serde_json::json!([5000]);
            }),
            ("unsorted vector", |j| {
                j["documents"][0]["chunks"][0]["vector"]["indices"] = serde_json::json!([9, 3]);
            }),
            ("duplicate document", |j| {
                let first = j["documents"][0].clone();
                j["documents"].as_array_mut().unwrap().push(first);
            }),
            ("wrong embedding length", |j| {
                let docs = j["documents"].as_array_mut().unwrap();
                let last = docs.last_mut().unwrap();
                last["chunks"][0]["embedding"] = serde_json::json!([1.0]);
            }),
        ];
        for (name, edit) in cases {
            let forged = reseal(&good, edit);
            let outcome = std::panic::catch_unwind(|| MemoryDocumentStore::import(&forged));
            match outcome {
                Ok(result) => assert!(result.is_err(), "{name}: accepted"),
                Err(_) => panic!("{name}: import panicked"),
            }
        }
    }

    #[test]
    fn a_failing_batch_stores_nothing() {
        let mut store = MemoryDocumentStore::new(1024, None).unwrap();
        let bad = input("", "no id");
        assert!(store
            .memorize_many(vec![input("ok", "fine text"), bad])
            .is_err());
        assert_eq!(store.stats().documents, 0);
        assert!(store
            .memorize_many(vec![input("same", "one"), input("same", "two")])
            .is_err());
    }
}
