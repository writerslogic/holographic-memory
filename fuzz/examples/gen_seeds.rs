// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Regenerates the binary seed corpus under `fuzz/seeds/`: valid inputs for the targets whose
//! formats a fuzzer would take a long time to discover (IVF index, JUMBF boxes, signed COSE
//! envelopes, the connection-graph event log, a VC). Text seeds (JSON bodies, DIDs, questions)
//! are committed by hand.
//!
//! cargo run --manifest-path fuzz/Cargo.toml --example gen_seeds \
//!     --features provenance,experimental -- fuzz/seeds

use std::path::PathBuf;

use ed25519_dalek::SigningKey;
use holographic_memory::core::connection_graph::{ConnectionGraph, GraphConfig};
use holographic_memory::core::provenance::jumbf::JumbfBox;
use holographic_memory::core::provenance::{cose, did, vc};
use holographic_memory::core::qgraph::ivf::QuantizedIvf;

fn put(root: &std::path::Path, target: &str, name: &str, bytes: &[u8]) {
    let dir = root.join(target);
    std::fs::create_dir_all(&dir).expect("seed dir");
    std::fs::write(dir.join(name), bytes).expect("seed file");
}

fn unit(dim: usize, hot: usize) -> Vec<f32> {
    let mut v = vec![0.0; dim];
    v[hot % dim] = 1.0;
    v
}

fn main() {
    let root = PathBuf::from(std::env::args().nth(1).expect("seed root directory"));

    let dim = 8;
    let rows: Vec<f32> = (0..4).flat_map(|i| unit(dim, i)).collect();
    let centers: Vec<f32> = (0..2).flat_map(|i| unit(dim, i)).collect();
    for (name, bits, residual) in [("ivf8", 8, 0), ("ivf4_residual", 4, 4)] {
        let ivf =
            QuantizedIvf::build(&rows, dim, &centers, &[0, 0, 1, 1], bits, residual).expect("ivf");
        put(&root, "ivf_decode", name, &ivf.to_bytes());
    }

    let mut manifest = JumbfBox::superbox("c2pa", [0x63; 16]);
    manifest.add_child(JumbfBox::json_box(
        "c2pa.claim",
        serde_json::json!({"alg": "sha256"}),
    ));
    manifest.add_child(JumbfBox::cbor_box("c2pa.signature", vec![0xa1, 0x01, 0x02]));
    put(
        &root,
        "jumbf_decode",
        "manifest",
        &manifest.encode().expect("encode"),
    );

    let key = SigningKey::from_bytes(&[7u8; 32]);
    let signed = cose::sign_payload(&key, br#"{"claim":"seed"}"#).expect("sign");
    put(&root, "cose_envelopes", "signed", &signed);
    put(&root, "cawg_identity", "cose_as_cbor", &signed);

    let public = key.verifying_key().to_bytes();
    put(
        &root,
        "did_parse",
        "did_key",
        did::did_key_from_ed25519(&public).as_bytes(),
    );
    put(
        &root,
        "did_parse",
        "did_jwk",
        did::did_jwk_from_ed25519(&public).as_bytes(),
    );
    put(
        &root,
        "did_parse",
        "did_document",
        did::create_did_document("did:web:example.com", &public)
            .to_string()
            .as_bytes(),
    );

    let credential = vc::create_fact_credential(
        &did::did_key_from_ed25519(&public),
        "fact-1",
        &[9u8; 32],
        4096,
        Some("notes.md"),
        Some(("paris", "capital_of", "france")),
        1,
    );
    let credential = vc::sign_credential(&key, credential).expect("sign vc");
    put(
        &root,
        "provenance_json",
        "credential",
        serde_json::to_string(&credential).expect("json").as_bytes(),
    );

    let log = std::env::temp_dir().join("hms-seed-graph.log");
    let _ = std::fs::remove_file(&log);
    {
        let mut graph = ConnectionGraph::open(64, GraphConfig::default(), &log).expect("graph");
        graph.store("paris", "capital_of", "france");
        graph.store("john", "father", "mark");
        graph.query("paris", "capital_of", "france");
    }
    put(
        &root,
        "connection_graph_log",
        "events",
        &std::fs::read(&log).expect("log"),
    );
    let _ = std::fs::remove_file(&log);
}
