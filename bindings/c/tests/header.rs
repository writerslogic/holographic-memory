// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The committed header, compiled by the system C compiler, must agree with the Rust
//! definitions on every struct size and field offset, and its unversioned aliases must name
//! the versioned symbols. Skipped (loudly) when no C compiler is installed.

use std::mem::{offset_of, size_of};
use std::process::Command;

use writerslogic_warm_c::*;

macro_rules! layout {
    ($($ty:ident { $($field:ident),* })*) => {{
        let mut expected = String::new();
        $(
            expected += &format!("{} {}\n", stringify!($ty), size_of::<$ty>());
            $(expected += &format!("{}.{} {}\n", stringify!($ty), stringify!($field), offset_of!($ty, $field));)*
        )*
        expected
    }};
}

const PROGRAM: &str = r#"
#include <stddef.h>
#include <stdio.h>
#include "writerslogic_warm.h"
#include "holographic_memory.h"
_Static_assert(sizeof(hms_document_t) == sizeof(warm_document_t), "legacy document layout");
_Static_assert(HMS_ABI_VERSION == WARM_ABI_VERSION, "legacy ABI version");
#define T(t) printf("%s %zu\n", #t, sizeof(t))
#define STR(x) #x
#define XSTR(x) STR(x)
#define F(t, f) printf("%s.%s %zu\n", #t, #f, offsetof(t, f))
int main(void) {
  T(warm_str_t); F(warm_str_t, ptr); F(warm_str_t, len);
  T(warm_open_options_t); F(warm_open_options_t, struct_size); F(warm_open_options_t, flags);
  F(warm_open_options_t, encryption_key); F(warm_open_options_t, signing); F(warm_open_options_t, audit);
  F(warm_open_options_t, embedding_model); F(warm_open_options_t, embedding_revision);
  F(warm_open_options_t, embedding_dimensions);
  T(warm_document_t); F(warm_document_t, id); F(warm_document_t, text); F(warm_document_t, source_uri);
  F(warm_document_t, version); F(warm_document_t, metadata_json); F(warm_document_t, chunk_words);
  F(warm_document_t, overlap_words); F(warm_document_t, title_weight); F(warm_document_t, store_text);
  F(warm_document_t, embeddings); F(warm_document_t, embedding_rows); F(warm_document_t, embedding_dim);
  T(warm_search_options_t); F(warm_search_options_t, struct_size); F(warm_search_options_t, flags);
  F(warm_search_options_t, k); F(warm_search_options_t, candidate_limit);
  F(warm_search_options_t, filter_json); F(warm_search_options_t, source_uri);
  F(warm_search_options_t, document_ids); F(warm_search_options_t, document_id_count);
  F(warm_search_options_t, embedding); F(warm_search_options_t, embedding_len);
  F(warm_search_options_t, lexical_weight); F(warm_search_options_t, semantic_weight);
  F(warm_search_options_t, min_semantic_score); F(warm_search_options_t, bm25_k1);
  F(warm_search_options_t, bm25_b); F(warm_search_options_t, fusion);
  T(warm_result_t); F(warm_result_t, id); F(warm_result_t, document_id); F(warm_result_t, source_uri);
  F(warm_result_t, version); F(warm_result_t, metadata_json); F(warm_result_t, text);
  F(warm_result_t, start_byte); F(warm_result_t, end_byte); F(warm_result_t, score);
  F(warm_result_t, lexical_score); F(warm_result_t, semantic_score);
  T(warm_chunk_t); F(warm_chunk_t, id); F(warm_chunk_t, text); F(warm_chunk_t, start_byte);
  F(warm_chunk_t, end_byte);
  T(warm_stats_t); F(warm_stats_t, struct_size); F(warm_stats_t, dimensions); F(warm_stats_t, vector_count);
  F(warm_stats_t, nsg_trained); F(warm_stats_t, ivf_trained); F(warm_stats_t, nsg_training_recommended);
  F(warm_stats_t, ivf_training_recommended); F(warm_stats_t, document_index_trained);
  F(warm_stats_t, storage_format_version); F(warm_stats_t, storage_segment_count);
  F(warm_stats_t, document_embedding_count); F(warm_stats_t, storage_used_bytes);
  F(warm_stats_t, storage_capacity_bytes);
  T(warm_security_status_t); F(warm_security_status_t, struct_size);
  F(warm_security_status_t, security_compiled); F(warm_security_status_t, encryption_active);
  F(warm_security_status_t, signing_active); F(warm_security_status_t, audit_active);
  F(warm_security_status_t, differential_privacy_active); F(warm_security_status_t, epsilon);
  /* The unversioned aliases must expand to the versioned symbols. */
  printf("alias %s\n", XSTR(warm_store_open));
  printf("alias %s\n", XSTR(warm_chunks_free));
  printf("alias legacy %s\n", XSTR(hms_store_open));
  printf("abi %d\n", WARM_ABI_VERSION);
  return 0;
}
"#;

#[test]
fn header_layout_matches_rust() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("layout.c");
    std::fs::write(&source, PROGRAM).unwrap();
    let binary = dir.path().join("layout");
    let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
    let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let compiled = Command::new(&compiler)
        .args(["-std=c11", "-Wall", "-Werror", "-I", include])
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output();
    let compiled = match compiled {
        Ok(output) => output,
        Err(error) => {
            eprintln!("SKIPPED: no C compiler ({error})");
            return;
        }
    };
    assert!(
        compiled.status.success(),
        "C compile failed:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&binary).output().unwrap();
    assert!(
        run.status.success(),
        "layout program failed: {:?}",
        run.status
    );
    let reported = String::from_utf8(run.stdout).unwrap();
    let expected = layout! {
        warm_str_t { ptr, len }
        warm_open_options_t { struct_size, flags, encryption_key, signing, audit, embedding_model,
            embedding_revision, embedding_dimensions }
        warm_document_t { id, text, source_uri, version, metadata_json, chunk_words, overlap_words,
            title_weight, store_text, embeddings, embedding_rows, embedding_dim }
        warm_search_options_t { struct_size, flags, k, candidate_limit, filter_json, source_uri,
            document_ids, document_id_count, embedding, embedding_len, lexical_weight,
            semantic_weight, min_semantic_score, bm25_k1, bm25_b, fusion }
        warm_result_t { id, document_id, source_uri, version, metadata_json, text, start_byte,
            end_byte, score, lexical_score, semantic_score }
        warm_chunk_t { id, text, start_byte, end_byte }
        warm_stats_t { struct_size, dimensions, vector_count, nsg_trained, ivf_trained,
            nsg_training_recommended, ivf_training_recommended, document_index_trained,
            storage_format_version, storage_segment_count, document_embedding_count,
            storage_used_bytes, storage_capacity_bytes }
        warm_security_status_t { struct_size, security_compiled, encryption_active, signing_active,
            audit_active, differential_privacy_active, epsilon }
    };
    let layout_lines: String = reported
        .lines()
        .filter(|line| !line.starts_with("abi ") && !line.starts_with("alias "))
        .map(|line| format!("{line}\n"))
        .collect();
    assert_eq!(layout_lines, expected);
    assert!(reported.contains(&format!("abi {}", WARM_ABI_VERSION)));
    assert!(reported.lines().any(|l| l == "alias hms1_store_open"));
    assert!(reported.lines().any(|l| l == "alias hms1_chunks_free"));
    assert!(reported
        .lines()
        .any(|l| l == "alias legacy hms1_store_open"));
}
