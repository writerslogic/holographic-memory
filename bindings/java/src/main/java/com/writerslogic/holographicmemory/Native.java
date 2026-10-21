// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.holographicmemory;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.LinkedHashMap;
import java.util.Map;

/** Raw bindings of the C ABI: symbol handles and struct layouts. Package private. */
final class Native {
    private Native() {}

    /** The ABI major version this binding was written against (the {@code hms1_} symbol prefix). */
    static final int ABI_MAJOR = 1;

    static final int STATUS_OK = 0;

    /** {@code hms_str_t}: a pointer and a length. */
    static final StructLayout STR = MemoryLayout.structLayout(
            ValueLayout.ADDRESS.withName("ptr"), ValueLayout.JAVA_LONG.withName("len"));

    /** A C struct described field by field with natural alignment, as the C compiler lays it out. */
    static final class Struct {
        private final Map<String, Long> offsets = new LinkedHashMap<>();
        private long size;
        private long align = 1;

        private Struct field(String name, long fieldSize, long fieldAlign) {
            size = (size + fieldAlign - 1) / fieldAlign * fieldAlign;
            offsets.put(name, size);
            size += fieldSize;
            align = Math.max(align, fieldAlign);
            return this;
        }

        Struct u32(String name) {
            return field(name, 4, 4);
        }

        Struct u64(String name) {
            return field(name, 8, 8);
        }

        Struct f64(String name) {
            return field(name, 8, 8);
        }

        Struct ptr(String name) {
            return field(name, 8, 8);
        }

        Struct usize(String name) {
            return field(name, 8, 8);
        }

        Struct str(String name) {
            return field(name, 16, 8);
        }

        long offset(String name) {
            Long o = offsets.get(name);
            if (o == null) {
                throw new IllegalArgumentException("no field " + name);
            }
            return o;
        }

        /** {@code sizeof} including tail padding. */
        long size() {
            return (size + align - 1) / align * align;
        }
    }

    static final Struct OPEN_OPTIONS = new Struct().u32("struct_size").u32("flags").str("encryption_key")
            .u32("signing").u32("audit").str("embedding_model").str("embedding_revision")
            .u32("embedding_dimensions");

    static final Struct DOCUMENT = new Struct().str("id").str("text").str("source_uri").str("version")
            .str("metadata_json").u32("chunk_words").u32("overlap_words").u32("title_weight")
            .u32("store_text").ptr("embeddings").usize("embedding_rows").usize("embedding_dim");

    static final Struct SEARCH_OPTIONS = new Struct().u32("struct_size").u32("flags").u32("k")
            .u32("candidate_limit").str("filter_json").str("source_uri").ptr("document_ids")
            .usize("document_id_count").ptr("embedding").usize("embedding_len").f64("lexical_weight")
            .f64("semantic_weight").f64("min_semantic_score").f64("bm25_k1").f64("bm25_b").str("fusion");

    static final Struct RESULT = new Struct().str("id").str("document_id").str("source_uri")
            .str("version").str("metadata_json").str("text").u32("start_byte").u32("end_byte")
            .f64("score").f64("lexical_score").f64("semantic_score");

    static final Struct CHUNK = new Struct().str("id").str("text").u32("start_byte").u32("end_byte");

    static final Struct STATS = new Struct().u32("struct_size").u32("dimensions").u64("vector_count")
            .u32("nsg_trained").u32("ivf_trained").u32("nsg_training_recommended")
            .u32("ivf_training_recommended").u32("document_index_trained").u32("storage_format_version")
            .u32("storage_segment_count").u64("document_embedding_count").u64("storage_used_bytes")
            .u64("storage_capacity_bytes");

    static final Struct SECURITY = new Struct().u32("struct_size").u32("security_compiled")
            .u32("encryption_active").u32("signing_active").u32("audit_active")
            .u32("differential_privacy_active").f64("epsilon");

    static final int SEARCH_SET_LEXICAL_WEIGHT = 1;
    static final int SEARCH_SET_SEMANTIC_WEIGHT = 1 << 1;
    static final int SEARCH_SET_MIN_SEMANTIC_SCORE = 1 << 2;
    static final int SEARCH_SET_BM25_K1 = 1 << 3;
    static final int SEARCH_SET_BM25_B = 1 << 4;

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP = NativeLibrary.load(Arena.global());

    private static MethodHandle handle(String name, FunctionDescriptor descriptor) {
        MemorySegment symbol = LOOKUP.find(name).orElseThrow(
                () -> new IllegalStateException("native library lacks symbol " + name + "; ABI mismatch?"));
        return LINKER.downcallHandle(symbol, descriptor);
    }

    private static final ValueLayout.OfInt I32 = ValueLayout.JAVA_INT;
    private static final ValueLayout.OfLong I64 = ValueLayout.JAVA_LONG;
    private static final AddressLayout PTR = ValueLayout.ADDRESS;

    static final MethodHandle ABI_VERSION = handle("hms1_abi_version", FunctionDescriptor.of(I32));
    static final MethodHandle VERSION = handle("hms1_version", FunctionDescriptor.of(PTR));
    static final MethodHandle LAST_ERROR_CODE = handle("hms1_last_error_code", FunctionDescriptor.of(I32));
    static final MethodHandle LAST_ERROR_MESSAGE = handle("hms1_last_error_message", FunctionDescriptor.of(PTR));
    static final MethodHandle OPEN_OPTIONS_INIT = handle("hms1_open_options_init", FunctionDescriptor.ofVoid(PTR));
    static final MethodHandle SEARCH_OPTIONS_INIT = handle("hms1_search_options_init", FunctionDescriptor.ofVoid(PTR));
    static final MethodHandle STORE_OPEN = handle("hms1_store_open", FunctionDescriptor.of(I32, STR, I32, PTR, PTR));
    static final MethodHandle STORE_CLOSE = handle("hms1_store_close", FunctionDescriptor.of(I32, PTR));
    static final MethodHandle STORE_INGEST = handle("hms1_store_ingest", FunctionDescriptor.of(I32, PTR, PTR, I64, PTR));
    static final MethodHandle STORE_SEARCH = handle("hms1_store_search", FunctionDescriptor.of(I32, PTR, STR, PTR, PTR));
    static final MethodHandle STORE_DELETE = handle("hms1_store_delete", FunctionDescriptor.of(I32, PTR, STR, PTR));
    static final MethodHandle STORE_STATS = handle("hms1_store_stats", FunctionDescriptor.of(I32, PTR, PTR));
    static final MethodHandle STORE_SECURITY = handle("hms1_store_security_status", FunctionDescriptor.of(I32, PTR, PTR));
    static final MethodHandle STORE_FLUSH = handle("hms1_store_flush", FunctionDescriptor.of(I32, PTR));
    static final MethodHandle STORE_COMPACT = handle("hms1_store_compact", FunctionDescriptor.of(I32, PTR));
    static final MethodHandle STORE_MAINTAIN = handle("hms1_store_maintain", FunctionDescriptor.of(I32, PTR));
    static final MethodHandle STORE_SNAPSHOT = handle("hms1_store_snapshot", FunctionDescriptor.of(I32, PTR, STR));
    static final MethodHandle CHUNK_DOCUMENT = handle("hms1_chunk_document", FunctionDescriptor.of(I32, PTR, PTR));
    static final MethodHandle RESULTS_COUNT = handle("hms1_results_count", FunctionDescriptor.of(I64, PTR));
    static final MethodHandle RESULTS_GET = handle("hms1_results_get", FunctionDescriptor.of(I32, PTR, I64, PTR));
    static final MethodHandle RESULTS_FREE = handle("hms1_results_free", FunctionDescriptor.ofVoid(PTR));
    static final MethodHandle CHUNKS_COUNT = handle("hms1_chunks_count", FunctionDescriptor.of(I64, PTR));
    static final MethodHandle CHUNKS_GET = handle("hms1_chunks_get", FunctionDescriptor.of(I32, PTR, I64, PTR));
    static final MethodHandle CHUNKS_FREE = handle("hms1_chunks_free", FunctionDescriptor.ofVoid(PTR));
}
