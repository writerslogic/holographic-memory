// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::core;
use napi_derive::napi;

#[napi]
// codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
pub struct HolographicMemorySystem {
    core: Arc<HmsCore>,
    memory: crate::core::sync::RwLock<Arc<crate::core::memory_api::Memory>>,
}

#[napi]
// codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
impl HolographicMemorySystem {
    #[napi(constructor)]
    pub fn new(
        dimensions: u32,
        storage_path: Option<String>,
        config: Option<HmsConfigJs>,
    ) -> Result<Self> {
        let cfg = config.map(|c| c.into_config());
        let core = HmsCore::new(dimensions, storage_path, cfg).map_err(napi_err)?;
        let core = Arc::new(core);
        Ok(Self {
            memory: crate::core::sync::RwLock::new(Arc::new(crate::core::memory_api::Memory::new(
                core.clone(),
            ))),
            core,
        })
    }

    /// Replaces the memory API's configuration (policies and defaults; see
    /// docs/memory-api). Pass nothing to restore the defaults.
    #[napi]
    pub fn configure_memory(&self, config: Option<serde_json::Value>) -> Result<()> {
        let config: crate::core::memory_api::MemoryConfig = match config {
            Some(value) => serde_json::from_value(value)
                .map_err(|e| napi::Error::from_reason(format!("invalid memory config: {e}")))?,
            None => Default::default(),
        };
        let memory = crate::core::memory_api::Memory::with_config(self.core.clone(), config)
            .map_err(napi_err)?;
        *self.memory.write() = Arc::new(memory);
        Ok(())
    }

    /// One memory API operation as JSON in and JSON out: `remember`, `recall`, `explain`,
    /// `forget`, `get`, `history`, `applyForgetting` or `consolidate`. The typed wrapper is
    /// `Memory` in `memory.js`.
    #[napi]
    pub async fn memory_call(
        &self,
        op: String,
        request: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let memory = self.memory.read().clone();
        tokio::task::spawn_blocking(move || memory.call(&op, request))
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?
            .map_err(memory_err)
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn memorize_document(&self, input: DocumentInput) -> Result<u32> {
        let core = self.core.clone();
        run_async(move || core.memorize_document(input)).await
    }

    #[napi]
    pub async fn memorize_documents(&self, inputs: Vec<DocumentInput>) -> Result<Vec<u32>> {
        let core = self.core.clone();
        run_async(move || core.memorize_documents(inputs)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn chunk_document(&self, input: DocumentInput) -> Result<Vec<DocumentChunk>> {
        run_async(move || chunk_document(&input)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn search_documents(
        &self,
        text: String,
        options: Option<SearchOptions>,
    ) -> Result<Vec<DocumentResult>> {
        let core = self.core.clone();
        run_async(move || core.search_documents(&text, &options.unwrap_or_default())).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn delete_document(&self, id: String) -> Result<bool> {
        let core = self.core.clone();
        run_async(move || core.delete_document(&id)).await
    }

    /// One page of stored documents in id order, without their text. Pass the last id of the
    /// previous page as `after` to continue; `limit` is 1..=1000 (default 100).
    #[napi]
    pub async fn list_documents(
        &self,
        after: Option<String>,
        limit: Option<f64>,
    ) -> Result<Vec<ListedDocument>> {
        // An f64 so that a value outside u32 or a fraction is rejected, not wrapped or truncated.
        let limit = limit.unwrap_or(100.0);
        let core = self.core.clone();
        run_async(move || {
            anyhow::ensure!(
                limit.fract() == 0.0 && (1.0..=1000.0).contains(&limit),
                "limit must be an integer in 1..=1000"
            );
            let limit = limit as usize;
            // Document ids cannot contain NUL, so `id + NUL` is the first key after `id`.
            let from = after.map(|id| format!("{id}\0"));
            Ok(core
                .list_documents(from.as_deref(), limit)
                .into_iter()
                .map(ListedDocument::from)
                .collect())
        })
        .await
    }

    #[napi]
    pub async fn get_document_info(&self, id: String) -> Result<Option<ListedDocument>> {
        let core = self.core.clone();
        run_async(move || Ok(core.get_document(&id).map(ListedDocument::from))).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn security_status(&self) -> serde_json::Value {
        self.core.security_status()
    }

    #[napi(getter)]
    pub fn vector_count(&self) -> u32 {
        self.core.vector_count() as u32
    }

    #[napi(getter)]
    pub fn nsg_trained(&self) -> bool {
        self.core.nsg_trained()
    }

    #[napi(getter)]
    pub fn ivf_trained(&self) -> bool {
        self.core.ivf_trained()
    }

    #[napi(getter)]
    pub fn dimensions(&self) -> u32 {
        self.core.dimensions() as u32
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn index_status(&self) -> IndexStatus {
        self.core.index_status()
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn storage_health(&self) -> StorageHealth {
        self.core.storage_health()
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn hardware_capabilities(&self) -> HardwareCapabilities {
        crate::core::hardware::capabilities()
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn explain_query(&self, text: String, k: u32) -> QueryExplanation {
        let query = self.core.encode_text(&text);
        self.core.explain_query(&query, k)
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn maintain_indices(&self) -> Result<IndexStatus> {
        let core = self.core.clone();
        run_async(move || core.maintain_indices()).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn flush(&self) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.flush()).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn analyze_text(&self, text: String) -> Result<TextMetrics> {
        let core = self.core.clone();
        run_async(move || Ok(core.analyze_text(&text))).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn calculate_readability(&self, metrics: TextMetrics) -> Result<f64> {
        let core = self.core.clone();
        run_async(move || Ok(core.calculate_readability(&metrics))).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn memorize_text(
        &self,
        id: String,
        text: String,
        trace_id: Option<String>,
    ) -> Result<()> {
        let core = self.core.clone();
        run_async(move || {
            let _span =
                info_span!("memorize_text", id = %id, trace_id = trace_id.as_deref().unwrap_or(""))
                    .entered();
            let vec = core.encode_text(&text);
            core.memorize(id, vec)
        })
        .await
    }

    /// Zero-copy text ingestion from a Node.js Buffer. Avoids the UTF-8 copy
    /// that occurs with String parameters by reading bytes in-place.
    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn memorize_text_buffer(
        &self,
        id: String,
        text: Buffer,
        trace_id: Option<String>,
    ) -> Result<()> {
        let core = self.core.clone();
        let text_str = std::str::from_utf8(&text)
            .map_err(|e| napi::Error::from_reason(format!("Invalid UTF-8: {}", e)))?
            .to_owned();
        run_async(move || {
            let _span = info_span!("memorize_text_buffer", id = %id, trace_id = trace_id.as_deref().unwrap_or("")).entered();
            let vec = core.encode_text(&text_str);
            core.memorize(id, vec)
        })
        .await
    }

    /// Batch memorize multiple id/text pairs in a single native call.
    /// Uses rayon for parallel encoding, then inserts sequentially.
    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn memorize_batch(
        &self,
        items: Vec<MemorizeBatchItem>,
        trace_id: Option<String>,
    ) -> Result<()> {
        let core = self.core.clone();
        run_async(move || {
            let _span = info_span!(
                "memorize_batch",
                count = items.len(),
                trace_id = trace_id.as_deref().unwrap_or("")
            )
            .entered();
            core.memorize_batch(&items)
        })
        .await
    }

    /// Ingest a UTF-8 file (up to 8 MiB) as a chunked document with source offsets.
    /// Retrieve passages with searchDocuments and remove them with deleteDocument.
    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn memorize_file(&self, id: String, file_path: String) -> Result<()> {
        let core = self.core.clone();
        run_async(move || {
            use std::io::Read;
            let file = std::fs::File::open(&file_path)
                .map_err(|e| anyhow::anyhow!("Failed to open {}: {}", file_path, e))?;
            anyhow::ensure!(
                file.metadata()?.is_file(),
                "filePath must name a regular file"
            );
            let mut bytes = Vec::new();
            file.take(core::documents::MAX_DOCUMENT_BYTES as u64 + 1)
                .read_to_end(&mut bytes)?;
            anyhow::ensure!(
                bytes.len() <= core::documents::MAX_DOCUMENT_BYTES,
                "file exceeds 8 MiB"
            );
            let text = String::from_utf8(bytes)?;
            core.memorize_document(DocumentInput {
                id,
                text,
                source_uri: Some(file_path),
                version: None,
                metadata: None,
                chunk_words: None,
                overlap_words: None,
                store_text: None,
                title_weight: None,
                embeddings: None,
            })?;
            Ok(())
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn memorize_vector(&self, id: String, vector: Float32Array) -> Result<()> {
        let core = self.core.clone();
        let dense: Vec<f32> = vector.to_vec();
        run_async(move || core.memorize_vector(id, &dense)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn memorize_scalar(&self, id: String, value: f64, min: f64, max: f64) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.memorize_scalar(id, value, min, max)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn query(
        &self,
        text: String,
        k: u32,
        trace_id: Option<String>,
    ) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        run_async(move || {
            let _span =
                info_span!("query", k = k, trace_id = trace_id.as_deref().unwrap_or("")).entered();
            let q_vec = core.encode_text(&text);
            let results = core.query(&q_vec, k);
            Ok(results)
        })
        .await
    }

    /// Converts float32→sparse EntangledHVec on JS thread, then queries on background.
    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn query_vector(&self, vector: Float32Array, k: u32) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        let dense = vector.to_vec();
        run_async(move || core.query_vector(&dense, k)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn query_scalar(
        &self,
        value: f64,
        min: f64,
        max: f64,
        k: u32,
    ) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        run_async(move || {
            let q_vec = EntangledHVec::from_scalar(value, min, max, core.dimensions());
            let results = core.query(&q_vec, k);
            Ok(results)
        })
        .await
    }

    /// Process multiple text queries in parallel, returning results for each.
    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn query_batch(
        &self,
        texts: Vec<String>,
        k: u32,
    ) -> Result<Vec<Vec<RetrievalResult>>> {
        let core = self.core.clone();
        run_async(move || {
            let queries: Vec<EntangledHVec> = texts.iter().map(|t| core.encode_text(t)).collect();
            Ok(core.query_batch(&queries, k))
        })
        .await
    }

    /// Process multiple float32 vector queries in parallel.
    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn query_vector_batch(
        &self,
        vectors: Vec<Float32Array>,
        k: u32,
    ) -> Result<Vec<Vec<RetrievalResult>>> {
        let core = self.core.clone();
        let dense: Vec<Vec<f32>> = vectors.iter().map(|v| v.to_vec()).collect();
        run_async(move || {
            anyhow::ensure!(dense.len() <= 4096, "query batch exceeds 4096 vectors");
            use rayon::prelude::*;
            dense.par_iter().map(|v| core.query_vector(v, k)).collect()
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn analyze_components(&self, text: String) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        run_async(move || {
            let vec = core.encode_text(&text);
            let results = core.analyze_components(&vec);
            Ok(results)
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn factorize_diffusion(
        &self,
        product_text: String,
        domains: Vec<Vec<String>>,
        max_iter: u32,
    ) -> Result<Vec<Option<String>>> {
        let core = self.core.clone();
        run_async(move || {
            let vec = core.encode_text(&product_text);
            let domain_vecs: Vec<Vec<EntangledHVec>> = domains
                .iter()
                .map(|d| d.iter().map(|s| core.encode_text(s)).collect())
                .collect();
            let results = core.factorize_diffusion(&vec, &domain_vecs, max_iter as usize);

            // Map EntangledHVec results back to IDs from the domain strings
            let mapped = results
                .into_iter()
                .enumerate()
                .map(|(i, opt_vec)| {
                    opt_vec.and_then(|evec| {
                        domains[i]
                            .iter()
                            .zip(domain_vecs[i].iter())
                            .min_by_key(|(_, enc)| evec.hamming(enc))
                            .map(|(s, _)| s.clone())
                    })
                })
                .collect();
            Ok(mapped)
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn memorize_triplet(
        &self,
        id: String,
        head: String,
        relation: String,
        tail: String,
    ) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.memorize_triplet(id, head, relation, tail)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn query_triplet(
        &self,
        head: String,
        relation: String,
        k: u32,
    ) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        run_async(move || core.query_triplet(head, relation, k)).await
    }

    /// Finds an analogy: A is to B as C is to ?.
    ///
    /// NOTE: Currently uses character trigram encoding, which has limited semantic
    /// understanding. Complex semantic analogies may require higher-level word
    /// embeddings (slated for future upgrade).
    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn find_analogy(
        &self,
        a: String,
        b: String,
        c: String,
        k: Option<u32>,
        trace_id: Option<String>,
    ) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        run_async(move || {
            let _span =
                info_span!("find_analogy", trace_id = trace_id.as_deref().unwrap_or("")).entered();
            let results = core.find_analogy(&a, &b, &c, k.unwrap_or(5));
            Ok(results)
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn synthesize_concepts(&self) -> Result<Vec<ConceptCandidate>> {
        let core = self.core.clone();
        run_async(move || {
            core.synthesize_concepts_with_limits(crate::core::query_limits::QueryLimits::default())
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn memorize_sequence(&self, id: String, sequence: Vec<String>) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.memorize_sequence(id, &sequence)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn train_nsg(&self) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.train_nsg()).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn train_ivf(&self) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.train_ivf()).await
    }

    #[napi]
    pub async fn train_document_index(&self) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.train_document_index()).await
    }

    #[napi(getter)]
    pub fn document_index_trained(&self) -> bool {
        self.core.document_index_trained()
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn query_sequence(
        &self,
        partial: Vec<String>,
        k: u32,
    ) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        run_async(move || core.query_sequence(&partial, k)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn delete(&self, id: String) -> Result<bool> {
        let core = self.core.clone();
        run_async(move || core.delete(&id)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn compact(&self) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.compact()).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn audit_since(&self, timestamp_ms: f64) -> Result<Vec<AuditEntryJs>> {
        let core = self.core.clone();
        let ts = timestamp_ms as u64;
        run_async(move || {
            let entries = core.audit_since(ts)?;
            Ok(entries
                .into_iter()
                .map(|e| AuditEntryJs {
                    timestamp_ms: e.timestamp_ms as f64,
                    op: match e.op {
                        crate::core::audit::AuditOp::Memorize => "memorize".to_string(),
                        crate::core::audit::AuditOp::Delete => "delete".to_string(),
                        crate::core::audit::AuditOp::Compact => "compact".to_string(),
                    },
                    id_hash: e.id_hash.iter().map(|b| format!("{:02x}", b)).collect(),
                    signed: e.signature != [0u8; 64],
                })
                .collect())
        })
        .await
    }

    /// Bundle multiple text items into a single hypervector.
    /// Respects the PrivacyConfig: when dp_enabled, uses epsilon-DP noise.
    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn bundle_texts(&self, texts: Vec<String>) -> Result<Vec<u32>> {
        let core = self.core.clone();
        run_async(move || {
            let vecs: Vec<EntangledHVec> = texts.iter().map(|t| core.encode_text(t)).collect();
            let bundled = core.bundle(&vecs);
            Ok(bundled.indices().to_vec())
        })
        .await
    }

    // === Meaning Memory API ===

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn memorize_meaning(&self, id: String, text: String) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.memorize_meaning(&id, &text)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn structural_query(
        &self,
        known_subjects: Vec<String>,
        known_relations: Vec<String>,
        target_role: String,
    ) -> Result<Vec<StructuralResultJs>> {
        let core = self.core.clone();
        run_async(move || {
            let known_vecs: Vec<EntangledHVec> = known_subjects
                .iter()
                .chain(known_relations.iter())
                .map(|t| core.encode_text(t))
                .collect();
            let mut bindings: Vec<(&str, &EntangledHVec)> = Vec::new();
            for (i, v) in known_vecs.iter().enumerate() {
                if i < known_subjects.len() {
                    bindings.push(("subject", v));
                } else {
                    bindings.push(("relation", v));
                }
            }
            let results = core.structural_query(&bindings, &target_role);
            Ok(results
                .into_iter()
                .map(|r| StructuralResultJs {
                    entity_id: r.entity_id,
                    confidence: r.confidence,
                    path: format!("{:?}", r.path),
                })
                .collect())
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn multi_hop_query(
        &self,
        start_entity: String,
        relations: Vec<String>,
    ) -> Result<Vec<MultiHopResultJs>> {
        let core = self.core.clone();
        run_async(move || {
            let rel_refs: Vec<&str> = relations.iter().map(|s| s.as_str()).collect();
            let results = core.multi_hop_with_limits(
                &start_entity,
                &rel_refs,
                crate::core::query_limits::QueryLimits::default(),
            )?;
            Ok(results
                .into_iter()
                .map(|r| MultiHopResultJs {
                    entity_id: r.entity_id,
                    confidence: r.confidence,
                    method: format!("{:?}", r.method),
                })
                .collect())
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn meaning_cleanup(&self, text: String) -> Result<Option<CleanupResultJs>> {
        let core = self.core.clone();
        run_async(move || {
            let vec = core.encode_text(&text);
            Ok(core
                .meaning_cleanup(&vec)
                .map(|(id, confidence)| CleanupResultJs { id, confidence }))
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn cleanup_vector(&self, vector: Float32Array) -> Result<Option<CleanupResultJs>> {
        let core = self.core.clone();
        let q_vec = EntangledHVec::from_dense(&vector, core.dimensions());
        run_async(move || {
            Ok(core
                .meaning_cleanup(&q_vec)
                .map(|(id, confidence)| CleanupResultJs { id, confidence }))
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn declare_composition_rule(
        &self,
        name: String,
        input_relations: Vec<String>,
        output_relation: String,
    ) -> Result<()> {
        self.core
            .declare_rule(&name, input_relations, output_relation)
            .map_err(napi_err)
    }

    #[napi(getter)]
    pub fn meaning_enabled(&self) -> bool {
        self.core.meaning_enabled()
    }

    // === Cognition API ===

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn start_cognition(&self) -> Result<()> {
        self.core.start_cognition().map_err(napi_err)
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn stop_cognition(&self) {
        self.core.stop_cognition();
    }

    #[napi(getter)]
    pub fn cognition_running(&self) -> bool {
        self.core.cognition_running()
    }

    #[napi(getter)]
    pub fn cognition_cycle_count(&self) -> u32 {
        self.core.cognition_cycle_count() as u32
    }

    #[napi(getter)]
    pub fn cognition_insight_count(&self) -> u32 {
        self.core.cognition_insight_count() as u32
    }

    #[napi(getter)]
    pub fn cognition_enabled(&self) -> bool {
        self.core.cognition_enabled()
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn run_cognition_once(&self) -> u32 {
        self.core.run_cognition_once().len() as u32
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn govern_memory(&self) -> GovernanceReportJs {
        let report = self.core.govern_memory();
        GovernanceReportJs {
            composites_merged: report.composites_merged as u32,
            composites_forgotten: report.composites_forgotten as u32,
            atoms_forgotten: report.atoms_forgotten as u32,
            idf_refreshed: report.idf_refreshed,
            atoms_refined: report.refinement.atoms_refined as u32,
        }
    }

    // === Agency API ===

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn add_goal(
        &self,
        name: String,
        description: String,
        relevance: f64,
        urgency: f64,
        cost: f64,
    ) -> Option<u32> {
        self.core
            .add_goal(&name, &description, relevance, urgency, cost)
            .map(|id| id as u32)
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn deactivate_goal(&self, name: String) -> bool {
        self.core.deactivate_goal(&name)
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn active_goals(&self) -> Vec<GoalJs> {
        self.core
            .active_goals()
            .into_iter()
            .map(|(name, utility)| GoalJs { name, utility })
            .collect()
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn plan_goal(
        &self,
        goal: String,
        causal_relations: Vec<String>,
        max_depth: Option<u32>,
    ) -> PlanJs {
        let rel_refs: Vec<&str> = causal_relations.iter().map(|s| s.as_str()).collect();
        let plan = self
            .core
            .plan_goal(&goal, &rel_refs, max_depth.unwrap_or(10) as usize);
        PlanJs {
            goal: plan.goal,
            actions: plan
                .actions
                .into_iter()
                .map(|a| PlannedActionJs {
                    subject: a.subject,
                    relation: a.relation,
                    object: a.object,
                    depth: a.depth as u32,
                    cost: a.cost,
                })
                .collect(),
            complete: plan.complete,
            total_cost: plan.total_cost,
        }
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn generate_questions(&self) -> Vec<QuestionJs> {
        self.core
            .generate_questions()
            .into_iter()
            .map(|q| QuestionJs {
                text: q.text,
                goal_relevance: q.goal_relevance,
            })
            .collect()
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn propose_rule(
        &self,
        name: String,
        input_relations: Vec<String>,
        output_relation: String,
        reason: String,
    ) -> Option<u32> {
        self.core
            .propose_rule(&name, input_relations, &output_relation, &reason)
            .map(|id| id as u32)
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn approve_proposal(&self, id: u32) -> bool {
        self.core.approve_proposal(id as usize)
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn reject_proposal(&self, id: u32) -> bool {
        self.core.reject_proposal(id as usize)
    }

    #[napi(getter)]
    pub fn pending_proposals(&self) -> u32 {
        self.core.pending_proposals() as u32
    }

    #[napi(getter)]
    pub fn goal_count(&self) -> u32 {
        self.core.goal_count() as u32
    }

    #[napi(getter)]
    pub fn active_goal_count(&self) -> u32 {
        self.core.active_goal_count() as u32
    }

    // === Graph API ===

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn add_relation(
        &self,
        source_id: String,
        relation_type: String,
        target_id: String,
        properties: Option<String>,
        valid_from: Option<f64>,
        valid_to: Option<f64>,
    ) -> Result<()> {
        let core = self.core.clone();
        run_async(move || {
            let rel = crate::core::types::Relation {
                source_id,
                relation_type,
                target_id,
                properties,
                valid_from: valid_from.unwrap_or(0.0),
                valid_to: valid_to.unwrap_or(0.0),
            };
            core.add_relation(&rel)
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn remove_relation(
        &self,
        source_id: String,
        relation_type: String,
        target_id: String,
    ) -> Result<bool> {
        let core = self.core.clone();
        run_async(move || core.remove_relation(&source_id, &relation_type, &target_id)).await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub fn declare_relation_type(
        &self,
        name: String,
        transitive: Option<bool>,
        symmetric: Option<bool>,
    ) -> Result<()> {
        self.core
            .declare_relation_type(crate::core::types::RelationType {
                name,
                transitive: transitive.unwrap_or(false),
                symmetric: symmetric.unwrap_or(false),
            })
            .map_err(napi_err)
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn traverse(
        &self,
        start_id: String,
        relation_type: Option<String>,
        max_depth: Option<u32>,
        at_time: Option<f64>,
    ) -> Result<Vec<crate::core::types::GraphPath>> {
        let core = self.core.clone();
        run_async(move || {
            Ok(core.traverse(
                &start_id,
                relation_type.as_deref(),
                max_depth.unwrap_or(3),
                at_time.unwrap_or(0.0),
            ))
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn outgoing_relations(
        &self,
        source_id: String,
        relation_type: Option<String>,
        at_time: Option<f64>,
    ) -> Result<Vec<crate::core::types::Relation>> {
        let core = self.core.clone();
        run_async(move || {
            Ok(core.outgoing_relations(
                &source_id,
                relation_type.as_deref(),
                at_time.unwrap_or(0.0),
            ))
        })
        .await
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn incoming_relations(
        &self,
        target_id: String,
        relation_type: Option<String>,
        at_time: Option<f64>,
    ) -> Result<Vec<crate::core::types::Relation>> {
        let core = self.core.clone();
        run_async(move || {
            Ok(core.incoming_relations(
                &target_id,
                relation_type.as_deref(),
                at_time.unwrap_or(0.0),
            ))
        })
        .await
    }

    #[napi(getter)]
    pub fn relation_count(&self) -> u32 {
        self.core.relation_count() as u32
    }

    #[napi]
    // codeql[rust/access-invalid-pointer] False positive: NAPI macro generated FFI bindings
    pub async fn federated_query(
        &self,
        peer_paths: Vec<String>,
        text: String,
        k: u32,
    ) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        run_async(move || {
            let q_vec = core.encode_text(&text);
            core.federated_query(&peer_paths, &q_vec, k)
        })
        .await
    }
}
