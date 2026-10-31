// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import com.writerslogic.warm.DocumentInput
import com.writerslogic.warm.WarmException
import com.writerslogic.warm.WarmStore
import com.writerslogic.warm.SearchOptions
import com.writerslogic.warm.StoreOptions
import com.writerslogic.warm.chunkDocument
import java.io.File
import java.nio.file.Files
import kotlin.io.path.absolutePathString
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking

class WarmStoreTest {
    private val root = Files.createTempDirectory("warm-kt").toFile()
    private val small = StoreOptions(dimensions = 1024u)

    @AfterTest
    fun cleanup() {
        root.deleteRecursively()
    }

    private fun path(name: String) = File(root, name).absolutePath

    private fun open(name: String = "store", options: StoreOptions = small) =
        WarmStore.openBlocking(path(name), options)

    private fun WarmStore.ids(query: String, options: SearchOptions? = null) =
        searchBlocking(query, options).map { it.documentId }

    @Test
    fun ingestSearchDeleteAndReopen() {
        val store = open()
        assertEquals(1u, store.ingestBlocking(DocumentInput(id = "a", text = "the quick brown fox")))
        store.ingestBlocking(DocumentInput(id = "b", text = "a slow green turtle"))
        assertEquals(listOf("b"), store.ids("turtle"))
        assertEquals("the quick brown fox", store.searchBlocking("fox")[0].text)
        assertEquals(2uL, store.stats().documentCount)
        assertFalse(store.stats().encrypted)
        assertTrue(store.deleteDocumentBlocking("a"))
        assertFalse(store.deleteDocumentBlocking("a"))
        store.shutdown()

        val reopened = open()
        assertEquals(1uL, reopened.stats().documentCount)
        assertEquals(listOf("b"), reopened.ids("turtle"))
        assertTrue(reopened.ids("fox").isEmpty())
        reopened.shutdown()
    }

    @Test
    fun filterBm25AndTitleWeight() {
        val store = open()
        for ((id, group) in listOf("a" to "x", "b" to "y", "c" to "x")) {
            store.ingestBlocking(DocumentInput(id = id, text = "shared term", metadataJson = """{"group":"$group"}"""))
        }
        assertEquals(setOf("a", "c"), store.ids("shared", SearchOptions(filterJson = """{"group":"x"}""")).toSet())

        val body = "filler words about nothing ".repeat(30)
        store.ingestBlocking(DocumentInput(id = "titled", text = "zebra\n$body", titleWeight = 8u))
        store.ingestBlocking(DocumentInput(id = "plain", text = "zebra\n$body"))
        val hits = store.searchBlocking("zebra")
        assertEquals("titled", hits[0].documentId)
        assertTrue(hits[0].score > hits[1].score)
        store.ingestBlocking(DocumentInput(id = "short", text = "needle"))
        store.ingestBlocking(DocumentInput(id = "long", text = "needle needle needle " + "filler ".repeat(200)))
        fun longScore(options: SearchOptions?) = store.searchBlocking("needle", options).first { it.documentId == "long" }.score
        assertNotEquals(longScore(null), longScore(SearchOptions(bm25B = 0.0)))
        assertFailsWith<WarmException.InvalidInput> { store.searchBlocking("zebra", SearchOptions(bm25B = 1.5)) }
        store.shutdown()
    }

    @Test
    fun typedErrors() {
        val store = open()
        assertFailsWith<WarmException.InvalidInput> { store.ingestBlocking(DocumentInput(id = "", text = "x")) }
        assertFailsWith<WarmException.StoreLocked> { WarmStore.openBlocking(path("store")) }
        store.shutdown()
        assertFailsWith<WarmException.Closed> { store.stats() }
        assertFailsWith<WarmException.NotFound> {
            WarmStore.restoreBlocking(path("missing"), path("target"))
        }
    }

    @Test
    fun encryptionRequiresTheKeyAndLeavesNoPlaintext() {
        val keyed = StoreOptions(dimensions = 1024u, encryptionKey = "passphrase-1")
        open("secure", keyed).let { store ->
            store.ingestBlocking(DocumentInput(id = "a", text = "uniquemarkerphrase secret"))
            assertTrue(store.stats().encrypted)
            store.compactBlocking()
            store.shutdown()
        }
        val marker = "uniquemarkerphrase".toByteArray()
        val leaked = File(path("secure")).walkTopDown().filter { it.isFile }.filter { file ->
            val bytes = file.readBytes()
            (0..bytes.size - marker.size).any { i -> marker.indices.all { bytes[i + it] == marker[it] } }
        }.toList()
        assertTrue(leaked.isEmpty(), "plaintext found in $leaked")
        assertFailsWith<WarmException> { open("secure", small) }
        assertFailsWith<WarmException> { open("secure", StoreOptions(dimensions = 1024u, encryptionKey = "wrong")) }
        open("secure", keyed).let { assertEquals(listOf("a"), it.ids("uniquemarkerphrase")); it.shutdown() }
    }

    @Test
    fun snapshotRestoreAndTamperDetection() {
        val store = open()
        for (i in 0 until 30) store.ingestBlocking(DocumentInput(id = "d$i", text = "document number$i topic${i % 5}"))
        val before = store.ids("topic3")
        val info = store.snapshotBlocking(path("snap"))
        assertEquals(30uL, info.documentCount)
        store.ingestBlocking(DocumentInput(id = "late", text = "after snapshot"))

        val restored = WarmStore.restoreBlocking(path("snap"), path("restored"), small)
        assertEquals(30uL, restored.stats().documentCount)
        assertFalse(restored.hasDocument("late"))
        assertContentEquals(before, restored.ids("topic3"))
        restored.shutdown()

        val manifest = File(path("snap"), "hms-snapshot.json")
        manifest.writeText(manifest.readText().replaceFirst("\"sha256\": \"", "\"sha256\": \"00"))
        assertFailsWith<WarmException.Corrupt> { WarmStore.restoreBlocking(path("snap"), path("bad"), small) }
        assertFalse(File(path("bad")).exists())
        store.shutdown()
    }

    @Test
    fun suspendFormsRunConcurrently() = runBlocking {
        val store = WarmStore.open(path("async"), small)
        val counts = store.ingestBatch((0 until 20).map { DocumentInput(id = "d$it", text = "async item$it shared") })
        assertEquals(20, counts.size)
        val searches = (0 until 8).map { i -> async { store.search("item$i").size } }
        val writes = (0 until 8).map { i -> async { store.ingest(DocumentInput(id = "x$i", text = "concurrent doc $i")); 0 } }
        assertEquals(8, searches.awaitAll().sum())
        writes.awaitAll()
        assertEquals(28uL, store.stats().documentCount)
        assertEquals(28uL, store.snapshot(path("asnap")).documentCount)
        store.shutdown()
    }

    @Test
    fun chunking() {
        val chunks = chunkDocument(DocumentInput(id = "a", text = "word ".repeat(500), chunkWords = 100u))
        assertTrue(chunks.size >= 5)
    }
}
