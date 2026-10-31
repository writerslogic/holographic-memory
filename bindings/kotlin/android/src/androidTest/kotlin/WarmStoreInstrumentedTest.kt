// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.writerslogic.warm.DocumentInput
import com.writerslogic.warm.WarmException
import com.writerslogic.warm.WarmStore
import com.writerslogic.warm.StoreOptions
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class WarmStoreInstrumentedTest {
    private val root = File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir, "warm-test")
    private val small = StoreOptions(dimensions = 1024u)

    @After
    fun cleanup() {
        root.deleteRecursively()
    }

    private fun path(name: String) = File(root, name).absolutePath

    @Test
    fun ingestSearchReopen() = runBlocking {
        val store = WarmStore.open(path("s"), small)
        store.ingest(DocumentInput(id = "a", text = "the quick brown fox"))
        store.ingest(DocumentInput(id = "b", text = "a slow green turtle"))
        assertEquals(listOf("b"), store.search("turtle").map { it.documentId })
        store.shutdown()
        val reopened = WarmStore.open(path("s"), small)
        assertEquals(2uL, reopened.stats().documentCount)
        reopened.shutdown()
    }

    @Test
    fun encryptionSnapshotAndTypedErrors() = runBlocking {
        val keyed = StoreOptions(dimensions = 1024u, encryptionKey = "passphrase-1")
        val store = WarmStore.open(path("secure"), keyed)
        store.ingest(DocumentInput(id = "a", text = "uniquemarkerphrase secret"))
        assertTrue(store.stats().encrypted)
        assertThrows(WarmException.StoreLocked::class.java) { WarmStore.openBlocking(path("secure"), keyed) }
        store.snapshot(path("snap"))
        store.shutdown()
        assertThrows(WarmException::class.java) { WarmStore.openBlocking(path("secure"), small) }
        val restored = WarmStore.restore(path("snap"), path("restored"), keyed)
        assertEquals(listOf("a"), restored.search("uniquemarkerphrase").map { it.documentId })
        assertFalse(restored.hasDocument("zzz"))
        restored.shutdown()
    }
}
