// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import WarmMemory
import SwiftUI

@main
struct WarmSampleApp: App {
    @StateObject private var model = NotesModel()

    var body: some Scene {
        WindowGroup { ContentView().environmentObject(model) }
    }
}

/// Owns the store. Opening replays the journal, so it runs off the main actor.
@MainActor
final class NotesModel: ObservableObject {
    @Published var hits: [SearchHit] = []
    @Published var documentCount: UInt64 = 0
    @Published var error: String?
    private var store: WarmStore?

    func start() async {
        do {
            let directory = try FileManager.default
                .url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                .appendingPathComponent("warm-notes")
            // Pass encryptionKey: (a key kept in the Keychain) to encrypt the store.
            store = try await WarmStore.open(path: directory.path, options: StoreOptions(dimensions: 2048))
            try await refresh()
        } catch { self.error = "\(error)" }
    }

    func add(_ text: String) async {
        guard let store, !text.isEmpty else { return }
        do {
            _ = try await store.ingest(input: DocumentInput(id: UUID().uuidString, text: text))
            try await refresh()
        } catch { self.error = "\(error)" }
    }

    func search(_ query: String) async {
        guard let store else { return }
        do { hits = try await store.search(query: query, options: SearchOptions(k: 10)) } catch { self.error = "\(error)" }
    }

    private func refresh() async throws { documentCount = try store?.stats().documentCount ?? 0 }
}

struct ContentView: View {
    @EnvironmentObject var model: NotesModel
    @State private var draft = ""
    @State private var query = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("\(model.documentCount) notes").font(.headline)
            HStack {
                TextField("New note", text: $draft)
                Button("Add") { Task { await model.add(draft); draft = "" } }
            }
            TextField("Search", text: $query)
                .onChange(of: query) { _, text in Task { await model.search(text) } }
            List(model.hits, id: \.id) { hit in
                VStack(alignment: .leading) {
                    Text(hit.text ?? hit.documentId)
                    Text(String(format: "score %.3f", hit.score)).font(.caption).foregroundStyle(.secondary)
                }
            }
            if let error = model.error { Text(error).foregroundStyle(.red) }
        }
        .padding()
        .task { await model.start() }
    }
}
