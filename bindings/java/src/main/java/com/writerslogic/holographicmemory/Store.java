// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.holographicmemory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * An open document store. Thread safe: any number of threads may search, ingest and delete
 * concurrently; {@link #close()} waits for calls in flight.
 *
 * <p>Requires Java 22 or newer and {@code --enable-native-access=ALL-UNNAMED}.
 */
public final class Store implements AutoCloseable {
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private final MemorySegment handle;
    private boolean closed;

    private Store(MemorySegment handle) {
        this.handle = handle;
    }

    /** The loaded native library's ABI version as {@code {major, minor, patch}}. */
    public static int[] abiVersion() {
        int v = (int) invoke(Native.ABI_VERSION);
        return new int[] {v >>> 16, (v >>> 8) & 0xff, v & 0xff};
    }

    /** The native library's version string. */
    public static String nativeVersion() {
        MemorySegment p = (MemorySegment) invoke(Native.VERSION);
        return p.reinterpret(Long.MAX_VALUE).getString(0);
    }

    private static void checkAbi() {
        int major = abiVersion()[0];
        if (major != Native.ABI_MAJOR) {
            throw new IllegalStateException(
                    "native library ABI major " + major + ", this binding needs " + Native.ABI_MAJOR);
        }
    }

    /**
     * Opens or creates the store in directory {@code path} with vectors of {@code dimensions}. A
     * directory is open once at a time, across processes ({@link HmsException.Code#LOCKED}
     * otherwise).
     */
    public static Store open(Path path, int dimensions, OpenOptions options) {
        Objects.requireNonNull(path, "path");
        checkAbi();
        OpenOptions o = options == null ? OpenOptions.defaults() : options;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment opts = arena.allocate(Native.OPEN_OPTIONS.size(), 8);
            invoke(Native.OPEN_OPTIONS_INIT, opts);
            if (o.encryptionKey() != null) {
                putStr(arena, opts, Native.OPEN_OPTIONS.offset("encryption_key"), o.encryptionKey());
            }
            if (o.signing() != null) {
                opts.set(ValueLayout.JAVA_INT, Native.OPEN_OPTIONS.offset("signing"), o.signing() ? 1 : -1);
            }
            if (o.audit() != null) {
                opts.set(ValueLayout.JAVA_INT, Native.OPEN_OPTIONS.offset("audit"), o.audit() ? 1 : -1);
            }
            if (o.embeddingModel() != null) {
                putStr(arena, opts, Native.OPEN_OPTIONS.offset("embedding_model"), o.embeddingModel());
                putStr(arena, opts, Native.OPEN_OPTIONS.offset("embedding_revision"), o.embeddingRevision());
                opts.set(ValueLayout.JAVA_INT, Native.OPEN_OPTIONS.offset("embedding_dimensions"),
                        o.embeddingDimensions());
            }
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            check((int) invoke(Native.STORE_OPEN, strArg(arena, path.toString()), dimensions, opts, out));
            return new Store(out.get(ValueLayout.ADDRESS, 0));
        }
    }

    /** Opens a store with default options. */
    public static Store open(Path path, int dimensions) {
        return open(path, dimensions, null);
    }

    /** Flushes and releases the store, waiting for calls in flight. Idempotent. */
    @Override
    public void close() {
        lock.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            check((int) invoke(Native.STORE_CLOSE, handle));
        } finally {
            lock.writeLock().unlock();
        }
    }

    private interface Op<T> {
        T run(Arena arena);
    }

    private <T> T use(Op<T> op) {
        lock.readLock().lock();
        try {
            if (closed) {
                throw new IllegalStateException("store is closed");
            }
            try (Arena arena = Arena.ofConfined()) {
                return op.run(arena);
            }
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Stores the documents as one atomic batch and returns each one's chunk count. Either all
     * documents become searchable or none do.
     */
    public int[] ingest(List<Document> documents) {
        Objects.requireNonNull(documents, "documents");
        if (documents.isEmpty()) {
            return new int[0];
        }
        return use(arena -> {
            int n = documents.size();
            long stride = Native.DOCUMENT.size();
            MemorySegment docs = arena.allocate(stride * n, 8);
            for (int i = 0; i < n; i++) {
                fillDocument(arena, docs.asSlice(stride * i, stride), documents.get(i));
            }
            MemorySegment counts = arena.allocate(4L * n, 4);
            check((int) invoke(Native.STORE_INGEST, handle, docs, (long) n, counts));
            int[] out = new int[n];
            MemorySegment.copy(counts, ValueLayout.JAVA_INT, 0, out, 0, n);
            return out;
        });
    }

    /** Stores one document and returns its chunk count. */
    public int ingest(Document document) {
        return ingest(List.of(document))[0];
    }

    private static void fillDocument(Arena arena, MemorySegment d, Document doc) {
        putStr(arena, d, Native.DOCUMENT.offset("id"), doc.id());
        putStr(arena, d, Native.DOCUMENT.offset("text"), doc.text());
        if (doc.sourceUri() != null) {
            putStr(arena, d, Native.DOCUMENT.offset("source_uri"), doc.sourceUri());
        }
        if (doc.version() != null) {
            putStr(arena, d, Native.DOCUMENT.offset("version"), doc.version());
        }
        if (doc.metadataJson() != null) {
            putStr(arena, d, Native.DOCUMENT.offset("metadata_json"), doc.metadataJson());
        }
        d.set(ValueLayout.JAVA_INT, Native.DOCUMENT.offset("chunk_words"), doc.chunkWords());
        d.set(ValueLayout.JAVA_INT, Native.DOCUMENT.offset("overlap_words"), doc.overlapWords());
        d.set(ValueLayout.JAVA_INT, Native.DOCUMENT.offset("title_weight"), doc.titleWeight());
        d.set(ValueLayout.JAVA_INT, Native.DOCUMENT.offset("store_text"), doc.discardText() ? -1 : 0);
        double[][] rows = doc.embeddings();
        if (rows != null && rows.length > 0) {
            int dim = rows[0].length;
            MemorySegment flat = arena.allocate(8L * rows.length * dim, 8);
            for (int r = 0; r < rows.length; r++) {
                if (rows[r].length != dim || dim == 0) {
                    throw new IllegalArgumentException("document " + doc.id() + " has ragged or empty embeddings");
                }
                MemorySegment.copy(rows[r], 0, flat, ValueLayout.JAVA_DOUBLE, 8L * r * dim, dim);
            }
            d.set(ValueLayout.ADDRESS, Native.DOCUMENT.offset("embeddings"), flat);
            d.set(ValueLayout.JAVA_LONG, Native.DOCUMENT.offset("embedding_rows"), rows.length);
            d.set(ValueLayout.JAVA_LONG, Native.DOCUMENT.offset("embedding_dim"), dim);
        }
    }

    /** Hybrid lexical (BM25) and, with an embedding, semantic search with default options. */
    public List<Result> search(String query) {
        return search(query, SearchOptions.defaults());
    }

    /** Hybrid lexical (BM25) and, with an embedding, semantic search. */
    public List<Result> search(String query, SearchOptions options) {
        Objects.requireNonNull(query, "query");
        SearchOptions o = options == null ? SearchOptions.defaults() : options;
        return use(arena -> {
            MemorySegment opts = arena.allocate(Native.SEARCH_OPTIONS.size(), 8);
            invoke(Native.SEARCH_OPTIONS_INIT, opts);
            fillSearch(arena, opts, o);
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            check((int) invoke(Native.STORE_SEARCH, handle, strArg(arena, query), opts, out));
            MemorySegment results = out.get(ValueLayout.ADDRESS, 0);
            try {
                long n = (long) invoke(Native.RESULTS_COUNT, results);
                List<Result> hits = new ArrayList<>((int) n);
                MemorySegment hit = arena.allocate(Native.RESULT.size(), 8);
                for (long i = 0; i < n; i++) {
                    check((int) invoke(Native.RESULTS_GET, results, i, hit));
                    hits.add(readResult(hit));
                }
                return List.copyOf(hits);
            } finally {
                invoke(Native.RESULTS_FREE, results);
            }
        });
    }

    private static void fillSearch(Arena arena, MemorySegment s, SearchOptions o) {
        var layout = Native.SEARCH_OPTIONS;
        s.set(ValueLayout.JAVA_INT, layout.offset("k"), o.k());
        s.set(ValueLayout.JAVA_INT, layout.offset("candidate_limit"), o.candidateLimit());
        if (o.filterJson() != null) {
            putStr(arena, s, layout.offset("filter_json"), o.filterJson());
        }
        if (o.sourceUri() != null) {
            putStr(arena, s, layout.offset("source_uri"), o.sourceUri());
        }
        if (o.documentIds() != null && !o.documentIds().isEmpty()) {
            int n = o.documentIds().size();
            MemorySegment ids = arena.allocate(16L * n, 8);
            for (int i = 0; i < n; i++) {
                putStr(arena, ids, 16L * i, o.documentIds().get(i));
            }
            s.set(ValueLayout.ADDRESS, layout.offset("document_ids"), ids);
            s.set(ValueLayout.JAVA_LONG, layout.offset("document_id_count"), n);
        }
        if (o.embedding() != null && o.embedding().length > 0) {
            MemorySegment e = arena.allocate(8L * o.embedding().length, 8);
            MemorySegment.copy(o.embedding(), 0, e, ValueLayout.JAVA_DOUBLE, 0, o.embedding().length);
            s.set(ValueLayout.ADDRESS, layout.offset("embedding"), e);
            s.set(ValueLayout.JAVA_LONG, layout.offset("embedding_len"), o.embedding().length);
        }
        if (o.fusion() != null) {
            putStr(arena, s, layout.offset("fusion"), o.fusion());
        }
        int flags = 0;
        flags |= setDouble(s, layout.offset("lexical_weight"), o.lexicalWeight(), Native.SEARCH_SET_LEXICAL_WEIGHT);
        flags |= setDouble(s, layout.offset("semantic_weight"), o.semanticWeight(), Native.SEARCH_SET_SEMANTIC_WEIGHT);
        flags |= setDouble(s, layout.offset("min_semantic_score"), o.minSemanticScore(), Native.SEARCH_SET_MIN_SEMANTIC_SCORE);
        flags |= setDouble(s, layout.offset("bm25_k1"), o.bm25K1(), Native.SEARCH_SET_BM25_K1);
        flags |= setDouble(s, layout.offset("bm25_b"), o.bm25B(), Native.SEARCH_SET_BM25_B);
        s.set(ValueLayout.JAVA_INT, layout.offset("flags"), flags);
    }

    private static int setDouble(MemorySegment s, long offset, Double value, int bit) {
        if (value == null) {
            return 0;
        }
        s.set(ValueLayout.JAVA_DOUBLE, offset, value);
        return bit;
    }

    private static Result readResult(MemorySegment h) {
        var r = Native.RESULT;
        double semantic = h.get(ValueLayout.JAVA_DOUBLE, r.offset("semantic_score"));
        return new Result(
                getStr(h, r.offset("id")),
                getStr(h, r.offset("document_id")),
                getStr(h, r.offset("source_uri")),
                getStr(h, r.offset("version")),
                getStr(h, r.offset("metadata_json")),
                getStr(h, r.offset("text")),
                Integer.toUnsignedLong(h.get(ValueLayout.JAVA_INT, r.offset("start_byte"))),
                Integer.toUnsignedLong(h.get(ValueLayout.JAVA_INT, r.offset("end_byte"))),
                h.get(ValueLayout.JAVA_DOUBLE, r.offset("score")),
                h.get(ValueLayout.JAVA_DOUBLE, r.offset("lexical_score")),
                Double.isNaN(semantic) ? null : semantic);
    }

    /** Deletes a document and its chunks; returns whether it existed. */
    public boolean delete(String id) {
        Objects.requireNonNull(id, "id");
        return use(arena -> {
            MemorySegment deleted = arena.allocate(ValueLayout.JAVA_INT);
            check((int) invoke(Native.STORE_DELETE, handle, strArg(arena, id), deleted));
            return deleted.get(ValueLayout.JAVA_INT, 0) == 1;
        });
    }

    /** Current statistics. */
    public Stats stats() {
        return use(arena -> {
            var l = Native.STATS;
            MemorySegment s = arena.allocate(l.size(), 8);
            s.set(ValueLayout.JAVA_INT, l.offset("struct_size"), (int) l.size());
            check((int) invoke(Native.STORE_STATS, handle, s));
            return new Stats(
                    Integer.toUnsignedLong(s.get(ValueLayout.JAVA_INT, l.offset("dimensions"))),
                    s.get(ValueLayout.JAVA_LONG, l.offset("vector_count")),
                    s.get(ValueLayout.JAVA_INT, l.offset("nsg_trained")) != 0,
                    s.get(ValueLayout.JAVA_INT, l.offset("ivf_trained")) != 0,
                    s.get(ValueLayout.JAVA_INT, l.offset("nsg_training_recommended")) != 0,
                    s.get(ValueLayout.JAVA_INT, l.offset("ivf_training_recommended")) != 0,
                    s.get(ValueLayout.JAVA_INT, l.offset("document_index_trained")) != 0,
                    s.get(ValueLayout.JAVA_LONG, l.offset("document_embedding_count")),
                    Integer.toUnsignedLong(s.get(ValueLayout.JAVA_INT, l.offset("storage_format_version"))),
                    Integer.toUnsignedLong(s.get(ValueLayout.JAVA_INT, l.offset("storage_segment_count"))),
                    s.get(ValueLayout.JAVA_LONG, l.offset("storage_used_bytes")),
                    s.get(ValueLayout.JAVA_LONG, l.offset("storage_capacity_bytes")));
        });
    }

    /** Whether encryption, signing and auditing are active. */
    public SecurityStatus securityStatus() {
        return use(arena -> {
            var l = Native.SECURITY;
            MemorySegment s = arena.allocate(l.size(), 8);
            s.set(ValueLayout.JAVA_INT, l.offset("struct_size"), (int) l.size());
            check((int) invoke(Native.STORE_SECURITY, handle, s));
            return new SecurityStatus(
                    s.get(ValueLayout.JAVA_INT, l.offset("security_compiled")) != 0,
                    s.get(ValueLayout.JAVA_INT, l.offset("encryption_active")) != 0,
                    s.get(ValueLayout.JAVA_INT, l.offset("signing_active")) != 0,
                    s.get(ValueLayout.JAVA_INT, l.offset("audit_active")) != 0,
                    s.get(ValueLayout.JAVA_INT, l.offset("differential_privacy_active")) != 0,
                    s.get(ValueLayout.JAVA_DOUBLE, l.offset("epsilon")));
        });
    }

    /** Writes all state to durable storage. */
    public void flush() {
        simple(Native.STORE_FLUSH);
    }

    /** Rewrites the transaction log without superseded and deleted records. */
    public void compact() {
        simple(Native.STORE_COMPACT);
    }

    /** Trains any index whose configured threshold has been reached. */
    public void maintain() {
        simple(Native.STORE_MAINTAIN);
    }

    private void simple(MethodHandle op) {
        use(arena -> {
            check((int) invoke(op, handle));
            return null;
        });
    }

    /**
     * Writes a verified point-in-time copy of the store to the new directory {@code destination}.
     * Ingests and deletes wait while the copy is taken; searches do not.
     */
    public void snapshot(Path destination) {
        Objects.requireNonNull(destination, "destination");
        use(arena -> {
            check((int) invoke(Native.STORE_SNAPSHOT, handle, strArg(arena, destination.toString())));
            return null;
        });
    }

    /** Splits a document the way ingest would, without a store. */
    public static List<Chunk> chunkDocument(Document document) {
        Objects.requireNonNull(document, "document");
        checkAbi();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment d = arena.allocate(Native.DOCUMENT.size(), 8);
            putStr(arena, d, Native.DOCUMENT.offset("id"), document.id());
            putStr(arena, d, Native.DOCUMENT.offset("text"), document.text());
            d.set(ValueLayout.JAVA_INT, Native.DOCUMENT.offset("chunk_words"), document.chunkWords());
            d.set(ValueLayout.JAVA_INT, Native.DOCUMENT.offset("overlap_words"), document.overlapWords());
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            check((int) invoke(Native.CHUNK_DOCUMENT, d, out));
            MemorySegment chunks = out.get(ValueLayout.ADDRESS, 0);
            try {
                long n = (long) invoke(Native.CHUNKS_COUNT, chunks);
                List<Chunk> result = new ArrayList<>((int) n);
                MemorySegment c = arena.allocate(Native.CHUNK.size(), 8);
                for (long i = 0; i < n; i++) {
                    check((int) invoke(Native.CHUNKS_GET, chunks, i, c));
                    result.add(new Chunk(
                            getStr(c, Native.CHUNK.offset("id")),
                            getStr(c, Native.CHUNK.offset("text")),
                            Integer.toUnsignedLong(c.get(ValueLayout.JAVA_INT, Native.CHUNK.offset("start_byte"))),
                            Integer.toUnsignedLong(c.get(ValueLayout.JAVA_INT, Native.CHUNK.offset("end_byte")))));
                }
                return List.copyOf(result);
            } finally {
                invoke(Native.CHUNKS_FREE, chunks);
            }
        }
    }

    // ---- marshalling -------------------------------------------------------------------------

    private static MemorySegment utf8(Arena arena, String s) {
        ByteBuffer bytes;
        try {
            bytes = StandardCharsets.UTF_8.newEncoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .encode(java.nio.CharBuffer.wrap(s));
        } catch (CharacterCodingException e) {
            throw new IllegalArgumentException("string is not valid Unicode (unpaired surrogate)", e);
        }
        // At least one byte so an empty string still has a non-NULL pointer ("absent" is NULL).
        MemorySegment seg = arena.allocate(Math.max(1, bytes.remaining()), 1);
        MemorySegment.copy(MemorySegment.ofBuffer(bytes), 0, seg, 0, bytes.remaining());
        return seg.asSlice(0, bytes.remaining());
    }

    /** An {@code hms_str_t} passed by value. */
    private static MemorySegment strArg(Arena arena, String s) {
        MemorySegment seg = arena.allocate(Native.STR);
        putStrAt(arena, seg, 0, s);
        return seg;
    }

    private static void putStr(Arena arena, MemorySegment struct, long offset, String s) {
        putStrAt(arena, struct, offset, s);
    }

    private static void putStrAt(Arena arena, MemorySegment struct, long offset, String s) {
        MemorySegment bytes = utf8(arena, s);
        struct.set(ValueLayout.ADDRESS, offset, bytes);
        struct.set(ValueLayout.JAVA_LONG, offset + 8, bytes.byteSize());
    }

    private static String getStr(MemorySegment struct, long offset) {
        MemorySegment ptr = struct.get(ValueLayout.ADDRESS, offset);
        if (ptr.equals(MemorySegment.NULL)) {
            return null;
        }
        long len = struct.get(ValueLayout.JAVA_LONG, offset + 8);
        if (len == 0) {
            return "";
        }
        return new String(ptr.reinterpret(len).toArray(ValueLayout.JAVA_BYTE), StandardCharsets.UTF_8);
    }

    // ---- invocation --------------------------------------------------------------------------

    private static Object invoke(MethodHandle handle, Object... args) {
        try {
            return handle.invokeWithArguments(args);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new IllegalStateException("native call failed", t);
        }
    }

    /**
     * Raises the native error of the call that just returned {@code status}. The error slot is
     * thread local, so this must run on the same thread right after the call, with no blocking
     * operation between them.
     */
    private static void check(int status) {
        if (status == Native.STATUS_OK) {
            return;
        }
        MemorySegment message = (MemorySegment) invoke(Native.LAST_ERROR_MESSAGE);
        throw new HmsException(status, message.reinterpret(Long.MAX_VALUE).getString(0));
    }
}
