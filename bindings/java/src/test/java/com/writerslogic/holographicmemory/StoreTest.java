// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.holographicmemory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.extension.TestWatcher;
import org.junit.jupiter.api.io.TempDir;

@Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
class StoreTest {
    @RegisterExtension
    static final TestWatcher diagnostics = new TestWatcher() {
        @Override
        public void testFailed(ExtensionContext context, Throwable cause) {
            System.err.println("Failed: " + context.getDisplayName());
            Thread.getAllStackTraces().forEach((thread, stack) -> {
                System.err.println(thread + " " + thread.getState());
                for (StackTraceElement frame : stack) {
                    System.err.println("  " + frame);
                }
            });
        }
    };

    @TempDir
    Path tmp;

    @BeforeEach
    void reportStart(TestInfo test) {
        System.out.println("Starting: " + test.getTestMethod().orElseThrow().getName()
                + " / " + test.getDisplayName());
    }

    private Document fox() {
        return Document.of("fox", "The quick brown fox jumps over the lazy dog near the river bank.")
                .withSourceUri("file:///fox.txt")
                .withMetadataJson("{\"topic\":\"animals\"}");
    }

    private void seed(Store store) {
        int[] counts = store.ingest(List.of(
                fox(),
                Document.of("rust", "Rust is a systems programming language focused on memory safety and speed.")
                        .withMetadataJson("{\"topic\":\"code\"}").withTitleWeight(3),
                Document.of("rivers", "Rivers carry water from mountains to the sea and shape the valleys they cross.")
                        .withMetadataJson("{\"topic\":\"geo\"}")));
        assertEquals(3, counts.length);
        for (int c : counts) {
            assertTrue(c >= 1);
        }
    }

    @Test
    void layoutsMatchTheCCompiler() {
        // sizeof values printed by a C compiler against include/holographic_memory.h.
        assertEquals(16, Native.STR.byteSize());
        assertEquals(72, Native.OPEN_OPTIONS.size());
        assertEquals(120, Native.DOCUMENT.size());
        assertEquals(136, Native.SEARCH_OPTIONS.size());
        assertEquals(128, Native.RESULT.size());
        assertEquals(40, Native.CHUNK.size());
        assertEquals(72, Native.STATS.size());
        assertEquals(32, Native.SECURITY.size());
    }

    @Test
    void versions() {
        assertEquals(Native.ABI_MAJOR, Store.abiVersion()[0]);
        assertFalse(Store.nativeVersion().isEmpty());
    }

    @Test
    void searchRoundTrip() {
        try (Store store = Store.open(tmp.resolve("s"), 2048)) {
            seed(store);
            List<Result> hits = store.search("memory safety programming language");
            assertFalse(hits.isEmpty());
            Result top = hits.get(0);
            assertEquals("rust", top.documentId());
            assertNotNull(top.text());
            assertTrue(top.score() > 0);
            assertNull(top.semanticScore());
            assertNull(top.sourceUri());
            assertTrue(top.metadataJson().contains("code"));

            List<Result> bm25 = store.search("river water mountains",
                    SearchOptions.builder().k(1).bm25K1(2.0).bm25B(0.3).build());
            assertEquals(1, bm25.size());
            assertEquals("rivers", bm25.get(0).documentId());

            HmsException bad = assertThrows(HmsException.class,
                    () -> store.search("river", SearchOptions.builder().bm25B(1.5).build()));
            assertEquals(HmsException.Code.INVALID_ARGUMENT, bad.code());

            List<Result> bySource = store.search("fox river",
                    SearchOptions.builder().sourceUri("file:///fox.txt").build());
            assertEquals("fox", bySource.get(0).documentId());
            assertEquals("file:///fox.txt", bySource.get(0).sourceUri());

            for (Result r : store.search("river language",
                    SearchOptions.builder().documentIds(List.of("rust")).fusion("rrf").build())) {
                assertEquals("rust", r.documentId());
            }

            assertTrue(store.delete("fox"));
            assertFalse(store.delete("fox"));

            HmsException rejected = assertThrows(HmsException.class,
                    () -> store.ingest(List.of(Document.of("ok", "fine text here"), Document.of("", "no id"))));
            assertEquals(HmsException.Code.INVALID_ARGUMENT, rejected.code());
            for (Result r : store.search("fine text")) {
                assertFalse(r.documentId().equals("ok"), "rejected batch must not be partially applied");
            }
        }
    }

    @Test
    void unicodeAndEmptyStrings() {
        try (Store store = Store.open(tmp.resolve("u"), 2048)) {
            store.ingest(Document.of("uni", "naïve café 数据 memory").withMetadataJson("{\"k\":\"ü\"}"));
            List<Result> hits = store.search("café");
            assertFalse(hits.isEmpty());
            assertTrue(hits.get(0).text().contains("数据"));
            assertThrows(IllegalArgumentException.class,
                    () -> store.ingest(Document.of("bad", "lone surrogate \ud800 here")));
        }
    }

    @Test
    void statsSecurityAndMaintenance() {
        try (Store store = Store.open(tmp.resolve("m"), 2048)) {
            seed(store);
            Stats stats = store.stats();
            assertEquals(2048, stats.dimensions());
            assertTrue(stats.vectorCount() >= 3);
            SecurityStatus security = store.securityStatus();
            assertTrue(security.securityCompiled());
            assertFalse(security.encryptionActive());
            store.flush();
            store.compact();
            store.maintain();
        }
    }

    @Test
    void lockingAndClose() {
        Path dir = tmp.resolve("l");
        Store store = Store.open(dir, 2048);
        HmsException locked = assertThrows(HmsException.class, () -> Store.open(dir, 2048));
        assertEquals(HmsException.Code.LOCKED, locked.code());
        store.close();
        store.close();
        assertThrows(IllegalStateException.class, () -> store.search("x"));
        Store.open(dir, 2048).close();
    }

    @Test
    void encryption() throws IOException {
        Path dir = tmp.resolve("enc");
        OpenOptions keyed = OpenOptions.builder().encryptionKey("correct horse battery staple").build();
        try (Store store = Store.open(dir, 2048, keyed)) {
            assertTrue(store.securityStatus().encryptionActive());
            store.ingest(Document.of("secret", "the launch code is swordfish unicorn"));
        }
        try (Stream<Path> files = Files.walk(dir)) {
            for (Path f : (Iterable<Path>) files.filter(Files::isRegularFile)::iterator) {
                if (f.getFileName().toString().equals("documents.text")) {
                    continue; // engine gap, see docs/runtimes/c
                }
                assertFalse(new String(Files.readAllBytes(f), StandardCharsets.ISO_8859_1).contains("swordfish"),
                        "plaintext in " + f);
            }
        }
        HmsException wrong = assertThrows(HmsException.class,
                () -> Store.open(dir, 2048, OpenOptions.builder().encryptionKey("wrong").build()));
        assertTrue(wrong.code() == HmsException.Code.ENCRYPTION || wrong.code() == HmsException.Code.CORRUPT,
                "got " + wrong.code());
        try (Store store = Store.open(dir, 2048, keyed)) {
            assertEquals(1, store.search("launch code swordfish").size());
        }
    }

    @Test
    void snapshot() {
        Path dir = tmp.resolve("src");
        Path snap = tmp.resolve("snap");
        try (Store store = Store.open(dir, 2048)) {
            seed(store);
            store.snapshot(snap);
            assertEquals(HmsException.Code.INVALID_ARGUMENT,
                    assertThrows(HmsException.class, () -> store.snapshot(snap)).code());
            assertEquals(HmsException.Code.INVALID_ARGUMENT,
                    assertThrows(HmsException.class, () -> store.snapshot(dir.resolve("inside"))).code());
        }
        try (Store copy = Store.open(snap, 2048)) {
            assertEquals("rust", copy.search("memory safety programming language").get(0).documentId());
        }
    }

    @Test
    void chunking() {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < 400; i++) {
            text.append("word").append(i).append(' ');
        }
        List<Chunk> chunks = Store.chunkDocument(Document.of("long", text.toString()).withChunking(100, 10));
        assertTrue(chunks.size() >= 4);
        assertTrue(chunks.get(0).endByte() > chunks.get(0).startByte());
        assertEquals(HmsException.Code.INVALID_ARGUMENT,
                assertThrows(HmsException.class, () -> Store.chunkDocument(Document.of("", "x"))).code());
    }

    @Test
    void embeddings() {
        OpenOptions space = OpenOptions.builder().embeddingSpace("test-model", "r1", 4).build();
        try (Store store = Store.open(tmp.resolve("emb"), 2048, space)) {
            store.ingest(List.of(
                    Document.of("a", "alpha text").withEmbeddings(new double[][] {{1, 0, 0, 0}}),
                    Document.of("b", "beta text").withEmbeddings(new double[][] {{0, 1, 0, 0}})));
            List<Result> hits = store.search("", SearchOptions.builder()
                    .embedding(new double[] {0, 1, 0, 0}).semanticWeight(1.0).build());
            assertEquals("b", hits.get(0).documentId());
            assertNotNull(hits.get(0).semanticScore());
        }
    }

    @RepeatedTest(20)
    void concurrentReadersAndWriters() throws Exception {
        Store store = Store.open(tmp.resolve("mt"), 2048);
        seed(store);
        ExecutorService pool = Executors.newFixedThreadPool(8);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < 8; t++) {
                final int id = t;
                futures.add(pool.submit(() -> {
                    for (int i = 0; i < 25; i++) {
                        store.search("memory language river");
                        if (i % 10 == 0) {
                            store.ingest(Document.of("extra-" + id + "-" + i, "extra memory text"));
                        }
                    }
                }));
            }
            for (Future<?> f : futures) {
                f.get();
            }
        } finally {
            pool.shutdown();
            store.close();
        }
    }
}
