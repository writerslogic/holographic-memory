// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.holographicmemory.sample

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.writerslogic.holographicmemory.DocumentInput
import com.writerslogic.holographicmemory.HmsStore
import com.writerslogic.holographicmemory.SearchHit
import com.writerslogic.holographicmemory.SearchOptions
import com.writerslogic.holographicmemory.StoreOptions
import java.io.File
import java.util.UUID
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private var store: HmsStore? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val directory = File(filesDir, "hms-notes").absolutePath
        setContent {
            MaterialTheme {
                var ready by remember { mutableStateOf<HmsStore?>(null) }
                // Opening replays the journal; the suspend form runs it off the main thread.
                // Pass encryptionKey = (a key kept in the Android Keystore) to encrypt the store.
                LaunchedEffect(Unit) {
                    ready = HmsStore.open(directory, StoreOptions(dimensions = 2048u)).also { store = it }
                }
                ready?.let { Notes(it) }
            }
        }
    }

    override fun onDestroy() {
        store?.shutdown()
        super.onDestroy()
    }
}

@Composable
private fun Notes(store: HmsStore) {
    val scope = rememberCoroutineScope()
    var draft by remember { mutableStateOf("") }
    var query by remember { mutableStateOf("") }
    var count by remember { mutableStateOf(store.stats().documentCount) }
    var hits by remember { mutableStateOf(emptyList<SearchHit>()) }

    Column(Modifier.padding(16.dp)) {
        Text("$count notes", style = MaterialTheme.typography.titleMedium)
        Row {
            OutlinedTextField(draft, { draft = it }, label = { Text("New note") })
            Button(onClick = {
                scope.launch {
                    store.ingest(DocumentInput(id = UUID.randomUUID().toString(), text = draft))
                    draft = ""
                    count = store.stats().documentCount
                }
            }) { Text("Add") }
        }
        OutlinedTextField(query, {
            query = it
            scope.launch { hits = store.search(it, SearchOptions(k = 10u)) }
        }, label = { Text("Search") })
        LazyColumn {
            items(hits) { hit ->
                Text(hit.text ?: hit.documentId)
                Text("score %.3f".format(hit.score), style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}
