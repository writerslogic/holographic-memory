#![deny(clippy::all)]

#[cfg(feature = "node-api")]
use napi::bindgen_prelude::*;
#[cfg(feature = "node-api")]
use napi_derive::napi;
#[cfg(feature = "node-api")]
use std::sync::Arc;
#[cfg(feature = "node-api")]
use tracing::info_span;

pub mod core;
pub use crate::core::entangled::EntangledHVec;
pub use crate::core::error::HmsError;
pub use crate::core::types::{ConceptCandidate, MemorizeBatchItem, RetrievalResult, TextMetrics};
pub use crate::core::HmsCore;

#[cfg(feature = "node-api")]
#[napi]
pub struct HolographicMemorySystem {
    core: Arc<HmsCore>,
}

#[cfg(feature = "node-api")]
fn napi_err(e: anyhow::Error) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}

#[cfg(feature = "node-api")]
async fn run_async<T: Send + 'static, F: FnOnce() -> anyhow::Result<T> + Send + 'static>(
    f: F,
) -> Result<T> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
        .map_err(napi_err)
}

#[cfg(feature = "node-api")]
#[napi(object)]
pub struct HmsConfigJs {
    pub nsg_max_degree: Option<u32>,
    pub nsg_ef_construction: Option<u32>,
    pub nsg_auto_threshold: Option<u32>,
    pub ivf_enabled: Option<bool>,
    pub ivf_n_clusters: Option<u32>,
    pub ivf_n_landmarks: Option<u32>,
    pub ivf_d_reduced: Option<u32>,
    pub ivf_n_probe: Option<u32>,
    pub ivf_auto_threshold: Option<u32>,
    pub shard_enabled: Option<bool>,
    pub shard_count: Option<u32>,
    pub shard_auto_threshold: Option<u32>,
    pub shard_target_size: Option<u32>,
    pub component_similarity_threshold: Option<f64>,
    pub component_max_neighbors: Option<u32>,
    pub concept_similarity_threshold: Option<f64>,
    pub concept_min_cluster_size: Option<u32>,
    pub diffusion_steps: Option<u32>,
    pub diffusion_sigma_max: Option<f64>,
    pub diffusion_sigma_min: Option<f64>,
    pub diffusion_step_size: Option<f64>,
    pub diffusion_n_langevin: Option<u32>,
    pub signing_enabled: Option<bool>,
    pub signing_key_path: Option<String>,
    pub encryption_enabled: Option<bool>,
    pub encryption_passphrase: Option<String>,
    pub audit_enabled: Option<bool>,
    pub dp_enabled: Option<bool>,
    pub dp_epsilon: Option<f64>,
}

#[cfg(feature = "node-api")]
impl HmsConfigJs {
    fn into_config(self) -> crate::core::config::HmsConfig {
        use crate::core::config::*;
        let mut cfg = HmsConfig::default();
        if let Some(v) = self.nsg_max_degree { cfg.nsg.max_degree = v as usize; }
        if let Some(v) = self.nsg_ef_construction { cfg.nsg.ef_construction = v as usize; }
        if let Some(v) = self.nsg_auto_threshold { cfg.nsg.auto_threshold = v as usize; }
        if let Some(v) = self.ivf_enabled { cfg.ivf.enabled = v; }
        if let Some(v) = self.ivf_n_clusters { cfg.ivf.n_clusters = v as usize; }
        if let Some(v) = self.ivf_n_landmarks { cfg.ivf.n_landmarks = v as usize; }
        if let Some(v) = self.ivf_d_reduced { cfg.ivf.d_reduced = v as usize; }
        if let Some(v) = self.ivf_n_probe { cfg.ivf.n_probe = v as usize; }
        if let Some(v) = self.ivf_auto_threshold { cfg.ivf.auto_threshold = v as usize; }
        if let Some(v) = self.shard_enabled { cfg.shard.enabled = v; }
        if let Some(v) = self.shard_count { cfg.shard.shard_count = v as usize; }
        if let Some(v) = self.shard_auto_threshold { cfg.shard.auto_threshold = v as usize; }
        if let Some(v) = self.shard_target_size { cfg.shard.target_shard_size = v as usize; }
        if let Some(v) = self.component_similarity_threshold { cfg.query.component_similarity_threshold = v; }
        if let Some(v) = self.component_max_neighbors { cfg.query.component_max_neighbors = v; }
        if let Some(v) = self.concept_similarity_threshold { cfg.concepts.similarity_threshold = v; }
        if let Some(v) = self.concept_min_cluster_size { cfg.concepts.min_cluster_size = v as usize; }
        if let Some(v) = self.diffusion_steps { cfg.diffusion.steps = v as usize; }
        if let Some(v) = self.diffusion_sigma_max { cfg.diffusion.sigma_max = v; }
        if let Some(v) = self.diffusion_sigma_min { cfg.diffusion.sigma_min = v; }
        if let Some(v) = self.diffusion_step_size { cfg.diffusion.step_size = v; }
        if let Some(v) = self.diffusion_n_langevin { cfg.diffusion.n_langevin = v as usize; }
        if let Some(v) = self.signing_enabled { cfg.security.signing_enabled = v; }
        if let Some(v) = self.signing_key_path { cfg.security.key_path = Some(v); }
        if let Some(v) = self.encryption_enabled { cfg.security.encryption_enabled = v; }
        if let Some(v) = self.encryption_passphrase { cfg.security.encryption_passphrase = Some(v); }
        if let Some(v) = self.audit_enabled { cfg.security.audit_enabled = v; }
        if let Some(v) = self.dp_enabled { cfg.privacy.dp_enabled = v; }
        if let Some(v) = self.dp_epsilon { cfg.privacy.epsilon = v; }
        cfg
    }
}

#[cfg(feature = "node-api")]
#[napi]
impl HolographicMemorySystem {
    #[napi(constructor)]
    pub fn new(dimensions: u32, storage_path: Option<String>, config: Option<HmsConfigJs>) -> Result<Self> {
        let cfg = config.map(|c| c.into_config());
        let core = HmsCore::new(dimensions, storage_path, cfg)
            .map_err(napi_err)?;
        Ok(Self {
            core: Arc::new(core),
        })
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
    pub async fn analyze_text(&self, text: String) -> Result<TextMetrics> {
        let core = self.core.clone();
        run_async(move || Ok(core.analyze_text(&text))).await
    }

    #[napi]
    pub async fn calculate_readability(&self, metrics: TextMetrics) -> Result<f64> {
        let core = self.core.clone();
        run_async(move || Ok(core.calculate_readability(&metrics))).await
    }

    #[napi]
    pub async fn memorize_text(&self, id: String, text: String, trace_id: Option<String>) -> Result<()> {
        let core = self.core.clone();
        run_async(move || {
            let _span = info_span!("memorize_text", id = %id, trace_id = trace_id.as_deref().unwrap_or("")).entered();
            let vec = core.encode_text(&text);
            core.memorize(id, vec)
        })
        .await
    }

    /// Zero-copy text ingestion from a Node.js Buffer. Avoids the UTF-8 copy
    /// that occurs with String parameters by reading bytes in-place.
    #[napi]
    pub async fn memorize_text_buffer(&self, id: String, text: Buffer, trace_id: Option<String>) -> Result<()> {
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
    pub async fn memorize_batch(&self, items: Vec<MemorizeBatchItem>, trace_id: Option<String>) -> Result<()> {
        let core = self.core.clone();
        run_async(move || {
            let _span = info_span!("memorize_batch", count = items.len(), trace_id = trace_id.as_deref().unwrap_or("")).entered();
            use rayon::prelude::*;
            let encoded: Vec<(String, EntangledHVec)> = items
                .into_par_iter()
                .map(|item| {
                    let vec = core.encode_text(&item.text);
                    (item.id, vec)
                })
                .collect();
            for (id, vec) in encoded {
                core.memorize(id, vec)?;
            }
            Ok(())
        })
        .await
    }

    /// Read a file directly from disk via memory-mapping and memorize its content.
    /// Avoids passing file content through the JS string boundary entirely.
    #[napi]
    pub async fn memorize_file(&self, id: String, file_path: String) -> Result<()> {
        let core = self.core.clone();
        run_async(move || {
            let file = std::fs::File::open(&file_path)
                .map_err(|e| anyhow::anyhow!("Failed to open {}: {}", file_path, e))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| anyhow::anyhow!("Failed to mmap {}: {}", file_path, e))?;
            let text = std::str::from_utf8(&mmap)
                .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in {}: {}", file_path, e))?;
            let vec = core.encode_text(text);
            core.memorize(id, vec)
        })
        .await
    }

    #[napi]
    pub async fn memorize_vector(&self, id: String, vector: Float32Array) -> Result<()> {
        let core = self.core.clone();
        let dense: Vec<f32> = vector.to_vec();
        run_async(move || {
            core.memorize_vector(id, &dense)
        })
        .await
    }

    #[napi]
    pub async fn memorize_scalar(&self, id: String, value: f64, min: f64, max: f64) -> Result<()> {
        let core = self.core.clone();
        run_async(move || {
            core.memorize_scalar(id, value, min, max)
        })
        .await
    }

    #[napi]
    pub async fn query(&self, text: String, k: u32, trace_id: Option<String>) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        run_async(move || {
            let _span = info_span!("query", k = k, trace_id = trace_id.as_deref().unwrap_or("")).entered();
            let q_vec = core.encode_text(&text);
            let results = core.query(&q_vec, k);
            Ok(results)
        })
        .await
    }

    /// Converts float32→sparse EntangledHVec on JS thread, then queries on background.
    #[napi]
    pub async fn query_vector(&self, vector: Float32Array, k: u32) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        let q_vec = EntangledHVec::from_dense(&vector, core.dimensions());
        run_async(move || {
            let results = core.query(&q_vec, k);
            Ok(results)
        })
        .await
    }

    #[napi]
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
    pub async fn query_vector_batch(
        &self,
        vectors: Vec<Float32Array>,
        k: u32,
    ) -> Result<Vec<Vec<RetrievalResult>>> {
        let core = self.core.clone();
        let queries: Vec<EntangledHVec> = vectors
            .iter()
            .map(|v| EntangledHVec::from_dense(v, core.dimensions()))
            .collect();
        run_async(move || Ok(core.query_batch(&queries, k))).await
    }

    #[napi]
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
    pub async fn memorize_triplet(
        &self,
        id: String,
        head: String,
        relation: String,
        tail: String,
    ) -> Result<()> {
        let core = self.core.clone();
        run_async(move || {
            core.memorize_triplet(id, head, relation, tail)
        })
        .await
    }

    #[napi]
    pub async fn query_triplet(
        &self,
        head: String,
        relation: String,
        k: u32,
    ) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        run_async(move || {
            core.query_triplet(head, relation, k)
        })
        .await
    }

    /// Finds an analogy: A is to B as C is to ?.
    ///
    /// NOTE: Currently uses character trigram encoding, which has limited semantic
    /// understanding. Complex semantic analogies may require higher-level word
    /// embeddings (slated for future upgrade).
    #[napi]
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
            let _span = info_span!("find_analogy", trace_id = trace_id.as_deref().unwrap_or("")).entered();
            let results = core.find_analogy(&a, &b, &c, k.unwrap_or(5));
            Ok(results)
        })
        .await
    }

    #[napi]
    pub async fn synthesize_concepts(&self) -> Result<Vec<ConceptCandidate>> {
        let core = self.core.clone();
        run_async(move || {
            let results = core.synthesize_concepts();
            Ok(results)
        })
        .await
    }

    #[napi]
    pub async fn memorize_sequence(&self, id: String, sequence: Vec<String>) -> Result<()> {
        let core = self.core.clone();
        run_async(move || {
            core.memorize_sequence(id, &sequence)
        })
        .await
    }

    #[napi]
    pub async fn train_nsg(&self) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.train_nsg()).await
    }

    #[napi]
    pub async fn train_ivf(&self) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.train_ivf()).await
    }

    #[napi]
    pub async fn query_sequence(
        &self,
        partial: Vec<String>,
        k: u32,
    ) -> Result<Vec<RetrievalResult>> {
        let core = self.core.clone();
        run_async(move || core.query_sequence(&partial, k)).await
    }

    #[napi]
    pub async fn delete(&self, id: String) -> Result<bool> {
        let core = self.core.clone();
        run_async(move || core.delete(&id)).await
    }

    #[napi]
    pub async fn compact(&self) -> Result<()> {
        let core = self.core.clone();
        run_async(move || core.compact()).await
    }

    #[napi]
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
}

#[cfg(feature = "node-api")]
#[napi(object)]
pub struct AuditEntryJs {
    pub timestamp_ms: f64,
    pub op: String,
    pub id_hash: String,
    pub signed: bool,
}

#[cfg(test)]
mod tests {
    use crate::core::engine::HmsCore;
    use crate::core::entangled::EntangledHVec;

    #[test]
    fn test_determinism() {
        let hms = HmsCore::new(1000, None, None).unwrap();
        let v1 = hms.encode_text("hello world");
        let v2 = hms.encode_text("hello world");
        assert!((v1.similarity(&v2) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_from_dense_produces_sparse() {
        let dense: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let e = EntangledHVec::from_dense(&dense, 1000);
        assert_eq!(e.dim, 1000);
        // Should have ~dim/256 active indices
        let expected = 1000 / 256;
        assert_eq!(e.indices.len(), expected);
    }

    // === Batch Memorization ===

    #[test]
    fn test_batch_memorize() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        // Simulate what memorize_batch does: parallel encode, sequential insert
        let items = vec![
            ("b1", "first batch item"),
            ("b2", "second batch item"),
            ("b3", "third batch item"),
        ];
        for (id, text) in &items {
            let vec = hms.encode_text(text);
            hms.memorize(id.to_string(), vec).unwrap();
        }
        assert_eq!(hms.vector_count(), 3);

        let q = hms.encode_text("first batch item");
        let results = hms.query(&q, 3);
        assert!(!results.is_empty());
    }

    // === Custom Config ===

    #[test]
    fn test_custom_concept_config() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.concepts.similarity_threshold = 0.5;
        config.concepts.min_cluster_size = 5;

        let hms = HmsCore::new(
            10_000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        ).unwrap();

        // With high threshold and min size, clusters are harder to form
        for i in 0..10 {
            let vec = hms.encode_text(&format!("config test document {}", i));
            hms.memorize(format!("cfg_{}", i), vec).unwrap();
        }

        let concepts = hms.synthesize_concepts();
        // With strict thresholds, fewer or no concepts should form
        // (this validates the config is actually used)
        for c in &concepts {
            assert!(c.member_count >= 5, "Min cluster size should be 5, got {}", c.member_count);
        }
    }

    // === Diffusion factorizer ===

    #[test]
    fn test_factorize_returns_factors() {
        let dir = tempfile::tempdir().unwrap();
        let hms =
            HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        // Memorize some domain items
        let colors = vec!["red", "blue", "green"];
        let shapes = vec!["circle", "square", "triangle"];
        for c in &colors {
            let v = hms.encode_text(c);
            hms.memorize(c.to_string(), v).unwrap();
        }
        for s in &shapes {
            let v = hms.encode_text(s);
            hms.memorize(s.to_string(), v).unwrap();
        }

        // Create a composite: red * circle
        let red_vec = hms.encode_text("red");
        let circle_vec = hms.encode_text("circle");
        let product = red_vec.bind(&circle_vec);

        let domains = vec![
            colors
                .iter()
                .map(|s| hms.encode_text(s))
                .collect::<Vec<EntangledHVec>>(),
            shapes
                .iter()
                .map(|s| hms.encode_text(s))
                .collect::<Vec<EntangledHVec>>(),
        ];

        let results = hms.factorize_diffusion(&product, &domains, 20);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_concept_synthesis() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(1000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        // Create 20 slightly-varied versions of a "base" concept
        // Use the same seed with small variations to create tight clusters
        let base = hms.encode_text("base concept");
        for i in 0..20 {
            // Create variants by permuting base slightly
            let variant = if i == 0 {
                base.clone()
            } else {
                // Bind with a "small" perturbation (most indices shared)
                let perturb = EntangledHVec::new_deterministic(1000, 10000 + i);
                // Bundle many copies of base with one perturbation to stay close
                EntangledHVec::bundle(&[
                    base.clone(),
                    base.clone(),
                    base.clone(),
                    base.clone(),
                    perturb,
                ])
            };
            hms.memorize(format!("var_{}", i), variant).unwrap();
        }

        let concepts = hms.synthesize_concepts();
        // We expect at least one synthesized concept representing the cluster
        assert!(
            !concepts.is_empty(),
            "Should synthesize at least one concept"
        );
        assert!(
            concepts[0].coherence > 0.7,
            "Synthesized concept should have high coherence, got {}",
            concepts[0].coherence
        );
    }

    // === NSG Integration Tests ===

    #[test]
    fn test_nsg_train_and_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.nsg.max_degree = 8;

        config.nsg.ef_construction = 16;

        let hms = HmsCore::new(
            1000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        )
        .unwrap();

        for i in 0..50 {
            let vec = hms.encode_text(&format!("nsg document {}", i));
            hms.memorize(format!("nsg_{}", i), vec).unwrap();
        }

        assert!(!hms.nsg_trained());
        hms.train_nsg().unwrap();
        assert!(hms.nsg_trained());

        let q = hms.encode_text("nsg document 0");
        let results = hms.query(&q, 5);
        assert!(!results.is_empty(), "NSG query should return results");
    }

    #[test]
    fn test_nsg_auto_train_at_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.nsg.auto_threshold = 30;
        config.nsg.max_degree = 8;

        config.nsg.ef_construction = 16;

        let hms = HmsCore::new(
            1000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        )
        .unwrap();

        for i in 0..29 {
            let vec = hms.encode_text(&format!("auto nsg {}", i));
            hms.memorize(format!("ansg_{}", i), vec).unwrap();
        }
        assert!(!hms.nsg_trained());

        let vec = hms.encode_text("auto nsg 29");
        hms.memorize("ansg_29".to_string(), vec).unwrap();
        assert!(hms.nsg_trained(), "NSG should auto-train at threshold");
    }

    // === Phase 4 Integration Tests ===

    #[test]
    fn test_adaptive_routing_e2e() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.nsg.max_degree = 8;

        config.nsg.ef_construction = 16;

        let hms = HmsCore::new(
            1000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        )
        .unwrap();

        for i in 0..50 {
            let vec = hms.encode_text(&format!("routing item {}", i));
            hms.memorize(format!("rt_{}", i), vec).unwrap();
        }

        // Train NSG — queries should now route through NSG
        hms.train_nsg().unwrap();
        assert!(hms.nsg_trained());

        let q = hms.encode_text("routing item 0");
        let results = hms.query(&q, 5);
        assert!(
            !results.is_empty(),
            "Adaptive routing should return results via NSG"
        );
    }

    // === Knowledge Graph Tests ===

    #[test]
    fn test_triplet_memorize_and_query() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        hms.memorize_triplet("paris_capital".to_string(), "Paris".to_string(), "is_capital_of".to_string(), "France".to_string()).unwrap();
        hms.memorize_triplet("berlin_capital".to_string(), "Berlin".to_string(), "is_capital_of".to_string(), "Germany".to_string()).unwrap();
        hms.memorize_triplet("tokyo_capital".to_string(), "Tokyo".to_string(), "is_capital_of".to_string(), "Japan".to_string()).unwrap();

        let results = hms.query_triplet("Paris".to_string(), "is_capital_of".to_string(), 3).unwrap();
        assert!(!results.is_empty(), "Triplet query should return results");
        assert!(
            results.iter().any(|r| r.id == "paris_capital"),
            "Paris triplet should appear in top-3 results"
        );
    }

    // === Sequence Tests ===

    #[test]
    fn test_sequence_memorize_and_query() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        hms.memorize_sequence("recipe_1".to_string(), &[
            "preheat oven".to_string(),
            "mix ingredients".to_string(),
            "pour into pan".to_string(),
            "bake for thirty minutes".to_string(),
        ]).unwrap();
        hms.memorize_sequence("recipe_2".to_string(), &[
            "boil water".to_string(),
            "add pasta".to_string(),
            "drain and serve".to_string(),
        ]).unwrap();

        // Query with a partial sequence match
        let q = hms.encode_text("preheat oven").permute(0);
        let results = hms.query(&q, 2);
        assert!(!results.is_empty(), "Sequence query should return results");
    }

    // === Scalar Query Tests ===

    #[test]
    fn test_scalar_query_ordering() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        for i in 0..20 {
            let val = i as f64 * 5.0;
            hms.memorize_scalar(format!("temp_{}", i), val, 0.0, 100.0).unwrap();
        }

        // Query near value 50 — should return items closest to 50
        let q = EntangledHVec::from_scalar(50.0, 0.0, 100.0, 10_000);
        let results = hms.query(&q, 5);
        assert!(!results.is_empty(), "Scalar query should return results");

        // Top results should cluster around value 50 (idx 10)
        let top_idx: usize = results[0].id.strip_prefix("temp_").unwrap().parse().unwrap();
        assert!(
            (5..=15).contains(&top_idx),
            "Top scalar result should be near value 50 (idx 10), got idx {}",
            top_idx
        );
    }

    // === Component Analysis Tests ===

    #[test]
    fn test_analyze_components() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        for i in 0..30 {
            let vec = hms.encode_text(&format!("component analysis document {}", i));
            hms.memorize(format!("ca_{}", i), vec).unwrap();
        }

        let vec = hms.encode_text("component analysis document 0");
        let results = hms.analyze_components(&vec);
        assert!(!results.is_empty(), "analyze_components should return results");
        assert!(
            results.iter().all(|r| r.similarity > 0.05),
            "All results should exceed similarity threshold"
        );
    }

    // === Delete Tests ===

    #[test]
    fn test_delete_existing() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        let vec = hms.encode_text("hello");
        hms.memorize("hello".to_string(), vec).unwrap();
        assert_eq!(hms.vector_count(), 1);

        assert!(hms.delete("hello").unwrap());
        assert_eq!(hms.vector_count(), 0);

        let q = hms.encode_text("hello");
        let results = hms.query(&q, 5);
        assert!(results.is_empty(), "Deleted vector should not appear in results");
    }

    #[test]
    fn test_delete_nonexistent() {
        let hms = HmsCore::new(10_000, None, None).unwrap();
        assert!(!hms.delete("no_such_id").unwrap());
    }

    #[test]
    fn test_basic_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        {
            let hms = HmsCore::new(10_000, Some(path.clone()), None).unwrap();
            let v = hms.encode_text("persist me");
            hms.memorize("p1".to_string(), v).unwrap();
            assert_eq!(hms.vector_count(), 1);
        }

        let hms = HmsCore::new(10_000, Some(path), None).unwrap();
        assert_eq!(hms.vector_count(), 1);
    }

    #[test]
    fn test_delete_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        {
            let hms = HmsCore::new(10_000, Some(path.clone()), None).unwrap();
            let v1 = hms.encode_text("keep me");
            let v2 = hms.encode_text("delete me");
            hms.memorize("keep".to_string(), v1).unwrap();
            hms.memorize("del".to_string(), v2).unwrap();
            assert_eq!(hms.vector_count(), 2);
            hms.delete("del").unwrap();
            assert_eq!(hms.vector_count(), 1);
        }

        let hms = HmsCore::new(10_000, Some(path), None).unwrap();
        assert_eq!(hms.vector_count(), 1);
        let q = hms.encode_text("keep me");
        let results = hms.query(&q, 5);
        assert!(!results.is_empty(), "Kept vector should survive restart");
    }

    #[test]
    fn test_delete_and_rememorize() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        let v1 = hms.encode_text("version 1");
        hms.memorize("doc".to_string(), v1).unwrap();
        hms.delete("doc").unwrap();

        let v2 = hms.encode_text("version 2");
        hms.memorize("doc".to_string(), v2.clone()).unwrap();
        assert_eq!(hms.vector_count(), 1);

        let q = hms.encode_text("version 2");
        let results = hms.query(&q, 1);
        assert_eq!(results[0].id, "doc");
    }

    // === Compact Tests ===

    #[test]
    fn test_compact_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        let hms = HmsCore::new(10_000, Some(path.clone()), None).unwrap();
        for i in 0..50 {
            let vec = hms.encode_text(&format!("item {}", i));
            hms.memorize(format!("id_{}", i), vec).unwrap();
        }
        // Delete half
        for i in 0..25 {
            hms.delete(&format!("id_{}", i)).unwrap();
        }
        assert_eq!(hms.vector_count(), 25);

        hms.compact().unwrap();

        // Verify all live vectors still queryable
        let q = hms.encode_text("item 30");
        let results = hms.query(&q, 5);
        assert!(!results.is_empty(), "Should find results after compaction");
        assert_eq!(hms.vector_count(), 25);
    }

    #[test]
    fn test_compact_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        {
            let hms = HmsCore::new(10_000, Some(path.clone()), None).unwrap();
            for i in 0..20 {
                let vec = hms.encode_text(&format!("doc {}", i));
                hms.memorize(format!("d_{}", i), vec).unwrap();
            }
            for i in 0..10 {
                hms.delete(&format!("d_{}", i)).unwrap();
            }
            hms.compact().unwrap();
        }

        // Re-open from compacted arena
        let hms = HmsCore::new(10_000, Some(path), None).unwrap();
        assert_eq!(hms.vector_count(), 10);
    }

    // === Shard Tests ===

    #[test]
    fn test_multi_shard_insert_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.shard.enabled = true;
        config.shard.shard_count = 4;

        let hms = HmsCore::new(
            10_000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        )
        .unwrap();

        for i in 0..100 {
            let vec = hms.encode_text(&format!("shard document {}", i));
            hms.memorize(format!("sd_{}", i), vec).unwrap();
        }

        assert_eq!(hms.vector_count(), 100);

        let q = hms.encode_text("shard document 0");
        let results = hms.query(&q, 5);
        assert!(!results.is_empty(), "Multi-shard query should return results");
    }

    #[test]
    fn test_multi_shard_delete() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.shard.enabled = true;
        config.shard.shard_count = 4;

        let hms = HmsCore::new(
            10_000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        )
        .unwrap();

        for i in 0..20 {
            let vec = hms.encode_text(&format!("item {}", i));
            hms.memorize(format!("m_{}", i), vec).unwrap();
        }
        assert_eq!(hms.vector_count(), 20);

        for i in 0..10 {
            assert!(hms.delete(&format!("m_{}", i)).unwrap());
        }
        assert_eq!(hms.vector_count(), 10);
    }

    #[test]
    fn test_auto_shard_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.shard.enabled = true;
        config.shard.shard_count = 0; // auto
        config.shard.auto_threshold = 50;
        config.shard.target_shard_size = 25;

        let hms = HmsCore::new(
            10_000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        )
        .unwrap();

        for i in 0..50 {
            let vec = hms.encode_text(&format!("auto shard {}", i));
            hms.memorize(format!("as_{}", i), vec).unwrap();
        }

        assert_eq!(hms.vector_count(), 50);

        // Verify queries still work after auto-sharding
        let q = hms.encode_text("auto shard 0");
        let results = hms.query(&q, 5);
        assert!(!results.is_empty(), "Should find results after auto-shard");
    }

    // === Audit Integration Tests ===

    #[test]
    fn test_audit_records_operations() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.security.audit_enabled = true;

        let hms = HmsCore::new(
            10_000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        )
        .unwrap();

        let v = hms.encode_text("audit test");
        hms.memorize("aud_1".to_string(), v).unwrap();
        hms.delete("aud_1").unwrap();

        let entries = hms.audit_since(0).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].op, crate::core::audit::AuditOp::Memorize);
        assert_eq!(entries[1].op, crate::core::audit::AuditOp::Delete);
    }

    #[test]
    fn test_audit_disabled_returns_empty() {
        let hms = HmsCore::new(10_000, None, None).unwrap();
        let entries = hms.audit_since(0).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_audit_compact_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.security.audit_enabled = true;

        let hms = HmsCore::new(
            10_000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        )
        .unwrap();

        let v = hms.encode_text("compact audit");
        hms.memorize("ca_1".to_string(), v).unwrap();
        hms.compact().unwrap();

        let entries = hms.audit_since(0).unwrap();
        assert!(entries.iter().any(|e| e.op == crate::core::audit::AuditOp::Compact));
    }

    // === Analogy Tests ===

    #[test]
    fn test_find_analogy() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        for word in &["walking", "talking", "running", "walked", "talked", "runner"] {
            let v = hms.encode_text(word);
            hms.memorize(word.to_string(), v).unwrap();
        }

        let results = hms.find_analogy("walking", "walked", "talking", 5);
        assert!(!results.is_empty(), "Analogy should return results");
    }

    // === IVF Persistence ===

    #[test]
    fn test_ivf_index_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        {
            let mut config = crate::core::config::HmsConfig::default();
            config.ivf.n_clusters = 8;
            config.ivf.n_landmarks = 64;
            config.ivf.d_reduced = 16;
            config.ivf.n_probe = 8;

            let hms = HmsCore::new(10_000, Some(path.clone()), Some(config)).unwrap();
            for i in 0..200 {
                let v = hms.encode_text(&format!("ivf persist {}", i));
                hms.memorize(format!("ip_{}", i), v).unwrap();
            }
            hms.train_ivf().unwrap();
            assert!(hms.ivf_trained());
        }

        let mut config = crate::core::config::HmsConfig::default();
        config.ivf.n_clusters = 8;
        config.ivf.n_landmarks = 64;
        config.ivf.d_reduced = 16;
        config.ivf.n_probe = 8;
        let hms = HmsCore::new(10_000, Some(path), Some(config)).unwrap();
        assert!(hms.ivf_trained(), "IVF should be loaded from disk on re-open");
        let q = hms.encode_text("ivf persist 0");
        let results = hms.query(&q, 5);
        assert!(!results.is_empty(), "IVF query should work after reload");
    }

    // === Query Batch Tests ===

    #[test]
    fn test_query_batch() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        for i in 0..20 {
            let v = hms.encode_text(&format!("batch query doc {}", i));
            hms.memorize(format!("bq_{}", i), v).unwrap();
        }

        let queries: Vec<EntangledHVec> = (0..3)
            .map(|i| hms.encode_text(&format!("batch query doc {}", i)))
            .collect();
        let batch_results = hms.query_batch(&queries, 3);
        assert_eq!(batch_results.len(), 3, "Should return one result set per query");
        for results in &batch_results {
            assert!(!results.is_empty(), "Each query should return results");
        }
    }

    // === Memorize Vector Tests ===

    #[test]
    fn test_memorize_vector_through_core() {
        let dir = tempfile::tempdir().unwrap();
        let hms = HmsCore::new(10_000, Some(dir.path().to_string_lossy().to_string()), None).unwrap();

        let dense: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        hms.memorize_vector("dense_1".to_string(), &dense).unwrap();
        assert_eq!(hms.vector_count(), 1);

        let q = EntangledHVec::from_dense(&dense, 10_000);
        let results = hms.query(&q, 1);
        assert_eq!(results[0].id, "dense_1");
    }

    // === Readability Integration ===

    #[test]
    fn test_readability_through_core() {
        let hms = HmsCore::new(10_000, None, None).unwrap();
        let metrics = hms.analyze_text("The cat sat on the mat.");
        assert!(metrics.word_count > 0);
        let score = hms.calculate_readability(&metrics);
        assert!(score > 50.0, "Simple sentence should be highly readable (got {:.1})", score);
    }

    // === Index Persistence Tests ===

    #[test]
    fn test_nsg_index_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        {
            let mut config = crate::core::config::HmsConfig::default();
            config.nsg.max_degree = 8;
            config.nsg.ef_construction = 16;

            let hms = HmsCore::new(10_000, Some(path.clone()), Some(config)).unwrap();
            for i in 0..50 {
                let v = hms.encode_text(&format!("persist nsg {}", i));
                hms.memorize(format!("pn_{}", i), v).unwrap();
            }
            hms.train_nsg().unwrap();
            assert!(hms.nsg_trained());
        }

        // Re-open and verify NSG was reloaded
        let mut config = crate::core::config::HmsConfig::default();
        config.nsg.max_degree = 8;
        config.nsg.ef_construction = 16;
        let hms = HmsCore::new(10_000, Some(path), Some(config)).unwrap();
        assert!(hms.nsg_trained(), "NSG should be loaded from disk on re-open");
        let q = hms.encode_text("persist nsg 0");
        let results = hms.query(&q, 5);
        assert!(!results.is_empty(), "NSG query should work after reload");
    }

    // === NSG Accuracy ===

    #[test]
    fn test_nsg_accuracy_matches_brute_force() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.nsg.max_degree = 8;
        config.nsg.ef_construction = 16;

        let hms = HmsCore::new(
            10_000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        ).unwrap();

        for i in 0..50 {
            let v = hms.encode_text(&format!("accuracy test {}", i));
            hms.memorize(format!("acc_{}", i), v).unwrap();
        }

        // Query before training (brute force)
        let q = hms.encode_text("accuracy test 0");
        let brute_results = hms.query(&q, 1);

        hms.train_nsg().unwrap();

        // Query after training (NSG)
        let nsg_results = hms.query(&q, 1);

        assert_eq!(
            brute_results[0].id, nsg_results[0].id,
            "NSG top-1 should match brute-force top-1"
        );
    }

    // === Multi-Shard Merge Verification ===

    #[test]
    fn test_multi_shard_results_from_multiple_shards() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.shard.enabled = true;
        config.shard.shard_count = 4;

        let hms = HmsCore::new(
            10_000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        ).unwrap();

        for i in 0..100 {
            let v = hms.encode_text(&format!("multi verify {}", i));
            hms.memorize(format!("mv_{}", i), v).unwrap();
        }

        let q = hms.encode_text("multi verify");
        let results = hms.query(&q, 20);
        assert!(results.len() >= 10, "Should return many results from across shards");

        // Verify results come from at least 2 different shards
        use fxhash::FxHasher;
        use std::hash::Hasher;
        let mut shard_ids: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for r in &results {
            let mut hasher = FxHasher::default();
            hasher.write(r.id.as_bytes());
            shard_ids.insert((hasher.finish() as usize) % 4);
        }
        assert!(
            shard_ids.len() >= 2,
            "Results should come from multiple shards (got {} distinct shards)",
            shard_ids.len()
        );
    }

    // === Compact Multi-Shard ===

    #[test]
    fn test_compact_multi_shard() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.shard.enabled = true;
        config.shard.shard_count = 2;

        let hms = HmsCore::new(
            10_000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        ).unwrap();

        for i in 0..30 {
            let v = hms.encode_text(&format!("compact shard {}", i));
            hms.memorize(format!("cs_{}", i), v).unwrap();
        }
        for i in 0..15 {
            hms.delete(&format!("cs_{}", i)).unwrap();
        }
        assert_eq!(hms.vector_count(), 15);

        hms.compact().unwrap();
        assert_eq!(hms.vector_count(), 15);

        let q = hms.encode_text("compact shard 20");
        let results = hms.query(&q, 5);
        assert!(!results.is_empty(), "Should find results after multi-shard compact");
    }

    // === IVF Integration Tests ===

    #[test]
    fn test_manual_ivf_train_and_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.ivf.n_clusters = 8;
        config.ivf.n_landmarks = 64;
        config.ivf.d_reduced = 16;
        config.ivf.n_probe = 8;

        let hms = HmsCore::new(
            10000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        )
        .unwrap();

        for i in 0..200 {
            let vec = hms.encode_text(&format!("document number {}", i));
            hms.memorize(format!("doc_{}", i), vec).unwrap();
        }

        assert!(!hms.ivf_trained());
        hms.train_ivf().unwrap();
        assert!(hms.ivf_trained());

        let q = hms.encode_text("document number 0");
        let results = hms.query(&q, 10);
        assert!(!results.is_empty(), "IVF query should return results");
    }

    #[test]
    fn test_auto_train_ivf_at_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::core::config::HmsConfig::default();
        config.ivf.enabled = true;
        config.ivf.auto_threshold = 50;
        config.ivf.n_clusters = 8;
        config.ivf.n_landmarks = 64;
        config.ivf.d_reduced = 16;
        config.ivf.n_probe = 8;

        let hms = HmsCore::new(
            1000,
            Some(dir.path().to_string_lossy().to_string()),
            Some(config),
        )
        .unwrap();

        for i in 0..49 {
            let vec = hms.encode_text(&format!("auto item {}", i));
            hms.memorize(format!("auto_{}", i), vec).unwrap();
        }
        assert!(!hms.ivf_trained(), "Should not be trained before threshold");

        // This 50th insert should trigger auto-training
        let vec = hms.encode_text("auto item 49");
        hms.memorize("auto_49".to_string(), vec).unwrap();
        assert!(
            hms.ivf_trained(),
            "Should be trained after reaching threshold"
        );
    }
}

#[cfg(test)]
mod lib_proptest;

#[cfg(test)]
mod ts_export_tests {
    use ts_rs::TS;

    #[test]
    fn export_ts_bindings() {
        crate::RetrievalResult::export_all().unwrap();
        crate::ConceptCandidate::export_all().unwrap();
        crate::TextMetrics::export_all().unwrap();
        crate::MemorizeBatchItem::export_all().unwrap();
        crate::HmsError::export_all().unwrap();
    }

    #[test]
    fn export_json_schemas() {
        use schemars::schema_for;
        let dir = std::path::Path::new("schemas");
        std::fs::create_dir_all(dir).unwrap();

        let schemas: Vec<(&str, schemars::schema::RootSchema)> = vec![
            ("RetrievalResult", schema_for!(crate::RetrievalResult)),
            ("ConceptCandidate", schema_for!(crate::ConceptCandidate)),
            ("TextMetrics", schema_for!(crate::TextMetrics)),
            ("MemorizeBatchItem", schema_for!(crate::MemorizeBatchItem)),
            ("HmsError", schema_for!(crate::HmsError)),
        ];

        for (name, schema) in schemas {
            let json = serde_json::to_string_pretty(&schema).unwrap();
            std::fs::write(dir.join(format!("{}.json", name)), json).unwrap();
        }
    }
}
