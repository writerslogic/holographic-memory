// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Runs inside a real browser (headless Chrome or Firefox) through wasm-bindgen-test.

#![cfg(target_arch = "wasm32")]

use hms_wasm::{argon2id_key, HmsStore};
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn object(fields: &[(&str, JsValue)]) -> JsValue {
    let o = Object::new();
    for (k, v) in fields {
        Reflect::set(&o, &JsValue::from_str(k), v).unwrap();
    }
    o.into()
}

fn doc(id: &str, text: &str) -> JsValue {
    object(&[("id", id.into()), ("text", text.into())])
}

fn get(value: &JsValue, key: &str) -> JsValue {
    Reflect::get(value, &JsValue::from_str(key)).unwrap()
}

#[wasm_bindgen_test]
fn runs_in_a_browser_window() {
    let global = js_sys::global();
    assert!(
        Reflect::has(&global, &JsValue::from_str("document")).unwrap(),
        "not running in a browser window"
    );
    let agent = get(&get(&global, "navigator"), "userAgent")
        .as_string()
        .unwrap();
    assert!(
        agent.contains("Chrome") || agent.contains("Firefox"),
        "{agent}"
    );
}

#[wasm_bindgen_test]
fn ingest_search_delete_and_round_trip() {
    let mut store = HmsStore::new(object(&[("dimensions", 2048.into())])).unwrap();
    store
        .memorize_document(doc(
            "a",
            "Okapi BM25 ranks passages by term frequency saturation",
        ))
        .unwrap();
    let batch = Array::of2(
        &doc("b", "holographic binding superposes vectors into one trace"),
        &doc("c", "tomatoes after the last frost in spring"),
    );
    assert_eq!(store.memorize_documents(batch.into()).unwrap(), vec![1, 1]);

    let hits = Array::from(
        &store
            .search_documents("bm25 frequency", JsValue::UNDEFINED)
            .unwrap(),
    );
    assert_eq!(hits.length(), 1);
    assert_eq!(get(&hits.get(0), "documentId").as_string().unwrap(), "a");

    let opts = object(&[("bm25K1", 2.0.into()), ("bm25B", 0.1.into())]);
    assert!(store.search_documents("frost", opts).is_ok());
    let bad = object(&[("bm25B", 7.0.into())]);
    assert!(store.search_documents("frost", bad).is_err());

    let bytes = store.export_bytes().unwrap();
    let restored = HmsStore::import_bytes(&bytes).unwrap();
    let again = Array::from(
        &restored
            .search_documents("holographic", JsValue::UNDEFINED)
            .unwrap(),
    );
    assert_eq!(get(&again.get(0), "documentId").as_string().unwrap(), "b");
    assert_eq!(restored.document_ids(), vec!["a", "b", "c"]);

    let mut tampered = bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert!(HmsStore::import_bytes(&tampered).is_err());

    assert!(store.delete_document("a"));
    assert!(!store.delete_document("a"));
    assert_eq!(
        get(&store.stats().unwrap(), "documents").as_f64().unwrap(),
        2.0
    );
}

#[wasm_bindgen_test]
fn argon2id_is_deterministic_and_validates_parameters() {
    let salt = [7u8; 16];
    let a = argon2id_key(b"correct horse", &salt, 64, 1, 1).unwrap();
    let b = argon2id_key(b"correct horse", &salt, 64, 1, 1).unwrap();
    let c = argon2id_key(b"battery staple", &salt, 64, 1, 1).unwrap();
    assert_eq!(a.len(), 32);
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(argon2id_key(b"x", &[0u8; 8], 64, 1, 1).is_err());
    assert!(argon2id_key(b"", &salt, 64, 1, 1).is_err());
    assert!(argon2id_key(b"x", &salt, u32::MAX, 1, 1).is_err());
}
