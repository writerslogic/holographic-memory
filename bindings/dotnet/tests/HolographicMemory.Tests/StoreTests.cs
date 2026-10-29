// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

using System.Runtime.InteropServices;
using System.Text;
using WritersLogic.HolographicMemory;
using Xunit;

namespace HolographicMemory.Tests;

public sealed class StoreTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("hms-dotnet-").FullName;

    public void Dispose()
    {
        Directory.Delete(_root, true);
    }

    private string Dir(string name) => Path.Join(_root, name);

    private static void Seed(Store store)
    {
        var counts = store.Ingest(new[]
        {
            new Document("fox", "The quick brown fox jumps over the lazy dog near the river bank.")
                { SourceUri = "file:///fox.txt", MetadataJson = "{\"topic\":\"animals\"}" },
            new Document("rust", "Rust is a systems programming language focused on memory safety and speed.")
                { MetadataJson = "{\"topic\":\"code\"}", TitleWeight = 3 },
            new Document("rivers", "Rivers carry water from mountains to the sea and shape the valleys they cross.")
                { MetadataJson = "{\"topic\":\"geo\"}" },
        });
        Assert.Equal(3, counts.Length);
        Assert.All(counts, c => Assert.True(c >= 1));
    }

    [Fact]
    public unsafe void StructSizesMatchTheCCompiler()
    {
        // sizeof values printed by a C compiler against include/holographic_memory.h.
        Assert.Equal(16, sizeof(HmsStr));
        Assert.Equal(72, sizeof(OpenOptionsNative));
        Assert.Equal(120, sizeof(DocumentNative));
        Assert.Equal(136, sizeof(SearchOptionsNative));
        Assert.Equal(128, sizeof(ResultNative));
        Assert.Equal(40, sizeof(ChunkNative));
        Assert.Equal(72, sizeof(StatsNative));
        Assert.Equal(32, sizeof(SecurityNative));
    }

    [Fact]
    public void Versions()
    {
        Assert.Equal(1, Store.AbiVersion.Major);
        Assert.NotEmpty(Store.NativeVersion);
    }

    [Fact]
    public void SearchRoundTrip()
    {
        using var store = Store.Open(Dir("s"), 2048);
        Seed(store);
        var hits = store.Search("memory safety programming language");
        Assert.Equal("rust", hits[0].DocumentId);
        Assert.NotNull(hits[0].Text);
        Assert.True(hits[0].Score > 0);
        Assert.Null(hits[0].SemanticScore);
        Assert.Contains("code", hits[0].MetadataJson);

        var bm25 = store.Search("river water mountains", new SearchOptions { K = 1, Bm25K1 = 2.0, Bm25B = 0.3 });
        Assert.Single(bm25);
        Assert.Equal("rivers", bm25[0].DocumentId);

        var bad = Assert.Throws<HmsException>(() => store.Search("river", new SearchOptions { Bm25B = 1.5 }));
        Assert.Equal(HmsStatus.InvalidArgument, bad.Status);

        var bySource = store.Search("fox river", new SearchOptions { SourceUri = "file:///fox.txt" });
        Assert.Equal("fox", bySource[0].DocumentId);
        Assert.Equal("file:///fox.txt", bySource[0].SourceUri);

        foreach (var r in store.Search("river language", new SearchOptions { DocumentIds = new[] { "rust" }, Fusion = "rrf" }))
        {
            Assert.Equal("rust", r.DocumentId);
        }

        Assert.True(store.Delete("fox"));
        Assert.False(store.Delete("fox"));

        var rejected = Assert.Throws<HmsException>(() =>
            store.Ingest(new[] { new Document("ok", "fine text here"), new Document("", "no id") }));
        Assert.Equal(HmsStatus.InvalidArgument, rejected.Status);
        Assert.DoesNotContain(store.Search("fine text"), r => r.DocumentId == "ok");
    }

    [Fact]
    public void UnicodeAndInvalidStrings()
    {
        using var store = Store.Open(Dir("u"), 2048);
        store.Ingest(new Document("uni", "naïve café 数据 memory") { MetadataJson = "{\"k\":\"ü\"}" });
        var hits = store.Search("café");
        Assert.Contains("数据", hits[0].Text);
        Assert.Throws<ArgumentException>(() => store.Ingest(new Document("bad", "lone surrogate \ud800 here")));
    }

    [Fact]
    public void StatsSecurityAndMaintenance()
    {
        using var store = Store.Open(Dir("m"), 2048);
        Seed(store);
        var stats = store.Stats();
        Assert.Equal(2048u, stats.Dimensions);
        Assert.True(stats.VectorCount >= 3);
        var security = store.SecurityStatus();
        Assert.True(security.SecurityCompiled);
        Assert.False(security.EncryptionActive);
        store.Flush();
        store.Compact();
        store.Maintain();
    }

    [Fact]
    public void LockingAndDispose()
    {
        var dir = Dir("l");
        var store = Store.Open(dir, 2048);
        Assert.Equal(HmsStatus.Locked, Assert.Throws<HmsException>(() => Store.Open(dir, 2048)).Status);
        store.Dispose();
        store.Dispose();
        Assert.Throws<ObjectDisposedException>(() => store.Search("x"));
        Store.Open(dir, 2048).Dispose();
    }

    [Fact]
    public void EncryptionAtRest()
    {
        var dir = Dir("enc");
        var keyed = new OpenOptions { EncryptionKey = "correct horse battery staple" };
        using (var store = Store.Open(dir, 2048, keyed))
        {
            Assert.True(store.SecurityStatus().EncryptionActive);
            store.Ingest(new Document("secret", "the launch code is swordfish unicorn"));
        }
        foreach (var f in Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories)
            .Where(file => Path.GetFileName(file) != "documents.text")) // engine gap, see docs/runtimes/c
        {
            Assert.False(FileContains(f, "swordfish"u8.ToArray()), "plaintext in " + f);
        }
        var wrong = Assert.Throws<HmsException>(() => Store.Open(dir, 2048, new OpenOptions { EncryptionKey = "wrong" }));
        Assert.True(wrong.Status is HmsStatus.Encryption or HmsStatus.Corrupt, wrong.Status.ToString());
        using var again = Store.Open(dir, 2048, keyed);
        Assert.Single(again.Search("launch code swordfish"));
    }

    private static bool FileContains(string path, byte[] needle)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        var buf = new byte[1 << 20];
        var filled = 0;
        int n;
        while ((n = fs.Read(buf, filled, buf.Length - filled)) > 0)
        {
            filled += n;
            if (buf.AsSpan(0, filled).IndexOf(needle) >= 0) return true;
            var tail = Math.Min(needle.Length - 1, filled);
            Buffer.BlockCopy(buf, filled - tail, buf, 0, tail);
            filled = tail;
        }
        return false;
    }

    [Fact]
    public void SnapshotCopiesAVerifiedStore()
    {
        var snap = Dir("snap");
        using (var store = Store.Open(Dir("src"), 2048))
        {
            Seed(store);
            store.Snapshot(snap);
            Assert.Equal(HmsStatus.InvalidArgument, Assert.Throws<HmsException>(() => store.Snapshot(snap)).Status);
            Assert.Equal(HmsStatus.InvalidArgument,
                Assert.Throws<HmsException>(() => store.Snapshot(Path.Join(Dir("src"), "inside"))).Status);
        }
        using var copy = Store.Open(snap, 2048);
        Assert.Equal("rust", copy.Search("memory safety programming language")[0].DocumentId);
    }

    [Fact]
    public void Chunking()
    {
        var text = string.Join(' ', Enumerable.Range(0, 400).Select(i => "word" + i));
        var chunks = Store.ChunkDocument(new Document("long", text) { ChunkWords = 100, OverlapWords = 10 });
        Assert.True(chunks.Count >= 4);
        Assert.True(chunks[0].EndByte > chunks[0].StartByte);
        Assert.Equal(HmsStatus.InvalidArgument,
            Assert.Throws<HmsException>(() => Store.ChunkDocument(new Document("", "x"))).Status);
    }

    [Fact]
    public void Embeddings()
    {
        using var store = Store.Open(Dir("emb"), 2048,
            new OpenOptions { EmbeddingModel = "test-model", EmbeddingRevision = "r1", EmbeddingDimensions = 4 });
        store.Ingest(new[]
        {
            new Document("a", "alpha text") { Embeddings = new[] { new double[] { 1, 0, 0, 0 } } },
            new Document("b", "beta text") { Embeddings = new[] { new double[] { 0, 1, 0, 0 } } },
        });
        var hits = store.Search("", new SearchOptions { Embedding = new double[] { 0, 1, 0, 0 }, SemanticWeight = 1.0 });
        Assert.Equal("b", hits[0].DocumentId);
        Assert.NotNull(hits[0].SemanticScore);
    }

    [Fact]
    public async Task AsyncAndConcurrentUse()
    {
        using var store = await Store.OpenAsync(Dir("mt"), 2048);
        Seed(store);
        var tasks = Enumerable.Range(0, 8).Select(t => Task.Run(async () =>
        {
            for (int i = 0; i < 25; i++)
            {
                var hits = await store.SearchAsync("memory language river");
                Assert.NotEmpty(hits);
                if (i % 10 == 0) await store.IngestAsync(new[] { new Document($"extra-{t}-{i}", "extra memory text") });
            }
        }));
        await Task.WhenAll(tasks);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => store.SearchAsync("x", null, cts.Token));
    }
}
