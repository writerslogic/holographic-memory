// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import XCTest
@testable import WarmMemory

final class WarmMemoryTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("warm-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func path(_ name: String) -> String { root.appendingPathComponent(name).path }

    private let small = StoreOptions(dimensions: 1024)

    private func open(_ name: String = "store", _ options: StoreOptions? = nil) throws -> WarmStore {
        try WarmStore.openBlocking(path: path(name), options: options ?? small)
    }

    private func ids(_ store: WarmStore, _ query: String, _ options: SearchOptions? = nil) throws -> [String] {
        try store.searchBlocking(query: query, options: options).map(\.documentId)
    }

    func testIngestSearchDeleteAndReopen() throws {
        var store: WarmStore? = try open()
        XCTAssertEqual(try store!.ingestBlocking(input: DocumentInput(id: "a", text: "the quick brown fox")), 1)
        _ = try store!.ingestBlocking(input: DocumentInput(id: "b", text: "a slow green turtle"))
        XCTAssertEqual(try ids(store!, "turtle"), ["b"])
        let hit = try store!.searchBlocking(query: "fox")[0]
        XCTAssertEqual(hit.text, "the quick brown fox")
        let stats = try store!.stats()
        XCTAssertEqual(stats.documentCount, 2)
        XCTAssertFalse(stats.encrypted)
        XCTAssertTrue(try store!.deleteDocumentBlocking(id: "a"))
        XCTAssertFalse(try store!.deleteDocumentBlocking(id: "a"))
        store!.shutdown()
        store = nil

        let reopened = try open()
        XCTAssertEqual(try reopened.stats().documentCount, 1)
        XCTAssertEqual(try ids(reopened, "turtle"), ["b"])
    }

    func testFilterBm25AndTitleWeight() throws {
        let store = try open()
        for (id, group) in [("a", "x"), ("b", "y"), ("c", "x")] {
            _ = try store.ingestBlocking(input: DocumentInput(
                id: id, text: "shared term", metadataJson: "{\"group\":\"\(group)\"}"))
        }
        let filtered = try ids(store, "shared", SearchOptions(filterJson: "{\"group\":\"x\"}"))
        XCTAssertEqual(Set(filtered), ["a", "c"])

        let body = String(repeating: "filler words about nothing ", count: 30)
        _ = try store.ingestBlocking(input: DocumentInput(id: "titled", text: "zebra\n" + body, titleWeight: 8))
        _ = try store.ingestBlocking(input: DocumentInput(id: "plain", text: "zebra\n" + body))
        let hits = try store.searchBlocking(query: "zebra")
        XCTAssertEqual(hits[0].documentId, "titled")
        XCTAssertGreaterThan(hits[0].score, hits[1].score)

        _ = try store.ingestBlocking(input: DocumentInput(id: "short", text: "needle"))
        _ = try store.ingestBlocking(input: DocumentInput(
            id: "long", text: "needle needle needle " + String(repeating: "filler ", count: 200)))
        func longScore(_ options: SearchOptions?) throws -> Double {
            try store.searchBlocking(query: "needle", options: options).first { $0.documentId == "long" }!.score
        }
        XCTAssertNotEqual(try longScore(nil), try longScore(SearchOptions(bm25B: 0.0)))
        XCTAssertThrowsError(try store.searchBlocking(query: "zebra", options: SearchOptions(bm25B: 1.5))) {
            guard case WarmError.InvalidInput = $0 else { return XCTFail("wrong error \($0)") }
        }
    }

    func testTypedErrors() throws {
        let store = try open()
        XCTAssertThrowsError(try store.ingestBlocking(input: DocumentInput(id: "", text: "x"))) {
            guard case WarmError.InvalidInput = $0 else { return XCTFail("wrong error \($0)") }
        }
        XCTAssertThrowsError(try WarmStore.openBlocking(path: path("store"), options: nil)) {
            guard case WarmError.StoreLocked = $0 else { return XCTFail("wrong error \($0)") }
        }
        store.shutdown()
        XCTAssertThrowsError(try store.stats()) {
            guard case WarmError.Closed = $0 else { return XCTFail("wrong error \($0)") }
        }
    }

    func testEncryptionRequiresKey() throws {
        let keyed = StoreOptions(dimensions: 1024, encryptionKey: "passphrase-1")
        do {
            let store = try open("secure", keyed)
            _ = try store.ingestBlocking(input: DocumentInput(id: "a", text: "uniquemarkerphrase secret"))
            XCTAssertTrue(try store.stats().encrypted)
            store.shutdown()
        }
        XCTAssertThrowsError(try open("secure", small))
        XCTAssertThrowsError(try open("secure", StoreOptions(dimensions: 1024, encryptionKey: "wrong")))
        let store = try open("secure", keyed)
        XCTAssertEqual(try ids(store, "uniquemarkerphrase"), ["a"])
    }

    func testSnapshotRestoreAndTamperDetection() throws {
        let store = try open()
        for i in 0..<30 {
            _ = try store.ingestBlocking(input: DocumentInput(id: "d\(i)", text: "document number\(i) topic\(i % 5)"))
        }
        let before = try ids(store, "topic3")
        let info = try store.snapshotBlocking(destination: path("snap"))
        XCTAssertEqual(info.documentCount, 30)
        _ = try store.ingestBlocking(input: DocumentInput(id: "late", text: "after snapshot"))

        let restored = try WarmStore.restoreBlocking(snapshotPath: path("snap"), path: path("restored"), options: small)
        XCTAssertEqual(try restored.stats().documentCount, 30)
        XCTAssertFalse(try restored.hasDocument(id: "late"))
        XCTAssertEqual(try ids(restored, "topic3"), before)
        restored.shutdown()

        let manifest = path("snap") + "/hms-snapshot.json"
        var text = try String(contentsOfFile: manifest, encoding: .utf8)
        text = text.replacingOccurrences(of: "\"sha256\": \"", with: "\"sha256\": \"00", options: [], range: text.range(of: "\"sha256\": \""))
        try text.write(toFile: manifest, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try WarmStore.restoreBlocking(snapshotPath: path("snap"), path: path("bad"), options: small)) {
            guard case WarmError.Corrupt = $0 else { return XCTFail("wrong error \($0)") }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: path("bad")))
    }

    func testAsyncAndConcurrentUse() async throws {
        let store = try await WarmStore.open(path: path("async"), options: small)
        let counts = try await store.ingestBatch(inputs: (0..<20).map {
            DocumentInput(id: "d\($0)", text: "async item\($0) shared")
        })
        XCTAssertEqual(counts.count, 20)
        try await withThrowingTaskGroup(of: Int.self) { group in
            for i in 0..<8 {
                group.addTask { try await store.search(query: "item\(i)").count }
                group.addTask {
                    _ = try await store.ingest(input: DocumentInput(id: "x\(i)", text: "concurrent doc \(i)"))
                    return 0
                }
            }
            var total = 0
            for try await n in group { total += n }
            XCTAssertEqual(total, 8)
        }
        XCTAssertEqual(try store.stats().documentCount, 28)
        let info = try await store.snapshot(destination: path("asnap"))
        XCTAssertEqual(info.documentCount, 28)
    }

    func testChunkDocument() throws {
        let chunks = try chunkDocument(input: DocumentInput(
            id: "a", text: String(repeating: "word ", count: 500), chunkWords: 100))
        XCTAssertGreaterThanOrEqual(chunks.count, 5)
    }
}
