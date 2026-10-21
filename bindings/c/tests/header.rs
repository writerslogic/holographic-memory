// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The committed header, compiled by the system C compiler, must agree with the Rust
//! definitions on every struct size and field offset, and its unversioned aliases must name
//! the versioned symbols. Skipped (loudly) when no C compiler is installed.

use std::mem::{offset_of, size_of};
use std::process::Command;

use holographic_memory_c::*;

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
#include "holographic_memory.h"
#define T(t) printf("%s %zu\n", #t, sizeof(t))
#define STR(x) #x
#define XSTR(x) STR(x)
#define F(t, f) printf("%s.%s %zu\n", #t, #f, offsetof(t, f))
int main(void) {
  T(hms_str_t); F(hms_str_t, ptr); F(hms_str_t, len);
  T(hms_open_options_t); F(hms_open_options_t, struct_size); F(hms_open_options_t, flags);
  F(hms_open_options_t, encryption_key); F(hms_open_options_t, signing); F(hms_open_options_t, audit);
  F(hms_open_options_t, embedding_model); F(hms_open_options_t, embedding_revision);
  F(hms_open_options_t, embedding_dimensions);
  T(hms_document_t); F(hms_document_t, id); F(hms_document_t, text); F(hms_document_t, source_uri);
  F(hms_document_t, version); F(hms_document_t, metadata_json); F(hms_document_t, chunk_words);
  F(hms_document_t, overlap_words); F(hms_document_t, title_weight); F(hms_document_t, store_text);
  F(hms_document_t, embeddings); F(hms_document_t, embedding_rows); F(hms_document_t, embedding_dim);
  T(hms_search_options_t); F(hms_search_options_t, struct_size); F(hms_search_options_t, flags);
  F(hms_search_options_t, k); F(hms_search_options_t, candidate_limit);
  F(hms_search_options_t, filter_json); F(hms_search_options_t, source_uri);
  F(hms_search_options_t, document_ids); F(hms_search_options_t, document_id_count);
  F(hms_search_options_t, embedding); F(hms_search_options_t, embedding_len);
  F(hms_search_options_t, lexical_weight); F(hms_search_options_t, semantic_weight);
  F(hms_search_options_t, min_semantic_score); F(hms_search_options_t, bm25_k1);
  F(hms_search_options_t, bm25_b); F(hms_search_options_t, fusion);
  T(hms_result_t); F(hms_result_t, id); F(hms_result_t, document_id); F(hms_result_t, source_uri);
  F(hms_result_t, version); F(hms_result_t, metadata_json); F(hms_result_t, text);
  F(hms_result_t, start_byte); F(hms_result_t, end_byte); F(hms_result_t, score);
  F(hms_result_t, lexical_score); F(hms_result_t, semantic_score);
  T(hms_chunk_t); F(hms_chunk_t, id); F(hms_chunk_t, text); F(hms_chunk_t, start_byte);
  F(hms_chunk_t, end_byte);
  T(hms_stats_t); F(hms_stats_t, struct_size); F(hms_stats_t, dimensions); F(hms_stats_t, vector_count);
  F(hms_stats_t, nsg_trained); F(hms_stats_t, ivf_trained); F(hms_stats_t, nsg_training_recommended);
  F(hms_stats_t, ivf_training_recommended); F(hms_stats_t, document_index_trained);
  F(hms_stats_t, storage_format_version); F(hms_stats_t, storage_segment_count);
  F(hms_stats_t, document_embedding_count); F(hms_stats_t, storage_used_bytes);
  F(hms_stats_t, storage_capacity_bytes);
  T(hms_security_status_t); F(hms_security_status_t, struct_size);
  F(hms_security_status_t, security_compiled); F(hms_security_status_t, encryption_active);
  F(hms_security_status_t, signing_active); F(hms_security_status_t, audit_active);
  F(hms_security_status_t, differential_privacy_active); F(hms_security_status_t, epsilon);
  /* The unversioned aliases must expand to the versioned symbols. */
  printf("alias %s\n", XSTR(hms_store_open));
  printf("alias %s\n", XSTR(hms_chunks_free));
  printf("abi %d\n", HMS_ABI_VERSION);
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
        hms_str_t { ptr, len }
        hms_open_options_t { struct_size, flags, encryption_key, signing, audit, embedding_model,
            embedding_revision, embedding_dimensions }
        hms_document_t { id, text, source_uri, version, metadata_json, chunk_words, overlap_words,
            title_weight, store_text, embeddings, embedding_rows, embedding_dim }
        hms_search_options_t { struct_size, flags, k, candidate_limit, filter_json, source_uri,
            document_ids, document_id_count, embedding, embedding_len, lexical_weight,
            semantic_weight, min_semantic_score, bm25_k1, bm25_b, fusion }
        hms_result_t { id, document_id, source_uri, version, metadata_json, text, start_byte,
            end_byte, score, lexical_score, semantic_score }
        hms_chunk_t { id, text, start_byte, end_byte }
        hms_stats_t { struct_size, dimensions, vector_count, nsg_trained, ivf_trained,
            nsg_training_recommended, ivf_training_recommended, document_index_trained,
            storage_format_version, storage_segment_count, document_embedding_count,
            storage_used_bytes, storage_capacity_bytes }
        hms_security_status_t { struct_size, security_compiled, encryption_active, signing_active,
            audit_active, differential_privacy_active, epsilon }
    };
    let layout_lines: String = reported
        .lines()
        .filter(|line| !line.starts_with("abi ") && !line.starts_with("alias "))
        .map(|line| format!("{line}\n"))
        .collect();
    assert_eq!(layout_lines, expected);
    assert!(reported.contains(&format!("abi {}", HMS_ABI_VERSION)));
    assert!(reported.lines().any(|l| l == "alias hms1_store_open"));
    assert!(reported.lines().any(|l| l == "alias hms1_chunks_free"));
}
