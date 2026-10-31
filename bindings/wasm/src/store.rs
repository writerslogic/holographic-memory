// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The document store exported to JavaScript: ingest, search, delete, stats and a
//! checksummed export/import, over the engine's in-memory document store.

use writerslogic_warm::core::documents::{DocumentInput, MemoryDocumentStore, SearchOptions};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::Serializer;
use wasm_bindgen::prelude::*;

const DEFAULT_DIMENSIONS: usize = 16384;

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct StoreOptions {
    dimensions: Option<usize>,
    embedding_dimensions: Option<usize>,
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    // json_compatible: objects as plain objects (not Map) and undefined-free output.
    value
        .serialize(&Serializer::json_compatible().serialize_missing_as_null(true))
        .map_err(|e| JsError::new(&e.to_string()))
}

fn from_js<T: for<'de> Deserialize<'de>>(value: JsValue, what: &str) -> Result<T, JsError> {
    serde_wasm_bindgen::from_value(value).map_err(|e| JsError::new(&format!("invalid {what}: {e}")))
}

fn err(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

/// A local document store. Mirrors the Node binding's document surface:
/// `memorizeDocument(s)`, `searchDocuments`, `deleteDocument`, `chunkDocument`.
#[wasm_bindgen]
pub struct WarmStore {
    inner: MemoryDocumentStore,
}

#[wasm_bindgen]
impl WarmStore {
    /// `options`: `{ dimensions?: number, embeddingDimensions?: number }` (default 16384, none).
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> Result<WarmStore, JsError> {
        let options: StoreOptions = if options.is_undefined() || options.is_null() {
            StoreOptions::default()
        } else {
            from_js(options, "store options")?
        };
        let inner = MemoryDocumentStore::new(
            options.dimensions.unwrap_or(DEFAULT_DIMENSIONS),
            options.embedding_dimensions,
        )
        .map_err(err)?;
        Ok(WarmStore { inner })
    }

    /// Restores a store from bytes produced by `exportBytes`.
    #[wasm_bindgen(js_name = importBytes)]
    pub fn import_bytes(bytes: &[u8]) -> Result<WarmStore, JsError> {
        Ok(WarmStore {
            inner: MemoryDocumentStore::import(bytes).map_err(err)?,
        })
    }

    /// Serializes the whole store to one checksummed, compressed byte string.
    #[wasm_bindgen(js_name = exportBytes)]
    pub fn export_bytes(&self) -> Result<Vec<u8>, JsError> {
        self.inner.export().map_err(err)
    }

    /// Stores or replaces one document; resolves to its chunk count.
    #[wasm_bindgen(js_name = memorizeDocument)]
    pub fn memorize_document(&mut self, input: JsValue) -> Result<u32, JsError> {
        let input: DocumentInput = from_js(input, "document")?;
        self.inner.memorize(input).map_err(err)
    }

    /// Stores or replaces several documents atomically; returns each one's chunk count.
    #[wasm_bindgen(js_name = memorizeDocuments)]
    pub fn memorize_documents(&mut self, inputs: JsValue) -> Result<Vec<u32>, JsError> {
        let inputs: Vec<DocumentInput> = from_js(inputs, "documents")?;
        self.inner.memorize_many(inputs).map_err(err)
    }

    /// Splits a document into chunks without storing it.
    #[wasm_bindgen(js_name = chunkDocument)]
    pub fn chunk_document(&self, input: JsValue) -> Result<JsValue, JsError> {
        let input: DocumentInput = from_js(input, "document")?;
        to_js(&writerslogic_warm::chunk_document(&input).map_err(err)?)
    }

    /// BM25 (optionally fused with dense embeddings) over the stored passages.
    #[wasm_bindgen(js_name = searchDocuments)]
    pub fn search_documents(&self, text: &str, options: JsValue) -> Result<JsValue, JsError> {
        let options: SearchOptions = if options.is_undefined() || options.is_null() {
            SearchOptions::default()
        } else {
            from_js(options, "search options")?
        };
        to_js(&self.inner.search(text, &options).map_err(err)?)
    }

    /// Removes a document; resolves to whether it existed.
    #[wasm_bindgen(js_name = deleteDocument)]
    pub fn delete_document(&mut self, id: &str) -> bool {
        self.inner.delete(id)
    }

    #[wasm_bindgen(js_name = hasDocument)]
    pub fn has_document(&self, id: &str) -> bool {
        self.inner.contains(id)
    }

    /// Sorted ids of every stored document.
    #[wasm_bindgen(js_name = documentIds)]
    pub fn document_ids(&self) -> Vec<String> {
        self.inner.ids()
    }

    /// `{ documents, chunks, embeddedChunks, dimensions, embeddingDimensions, indexTrained }`.
    pub fn stats(&self) -> Result<JsValue, JsError> {
        to_js(&self.inner.stats())
    }

    /// Builds the dense index over stored embeddings; resolves to whether one is current.
    #[wasm_bindgen(js_name = trainDocumentIndex)]
    pub fn train_document_index(&mut self) -> Result<bool, JsError> {
        self.inner.train_index().map_err(err)
    }

    #[wasm_bindgen(getter, js_name = documentIndexTrained)]
    pub fn document_index_trained(&self) -> bool {
        self.inner.index_trained()
    }

    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
}
