// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

using System.Runtime.InteropServices;
using System.Text;

namespace WritersLogic.HolographicMemory;

/// <summary>
/// An open document store. Thread safe: any number of threads may search, ingest and delete
/// concurrently. Disposing waits for native calls in flight. The <c>*Async</c> methods run the
/// native call on the thread pool; the native call cannot be interrupted, so cancellation is
/// honoured before it starts and otherwise the call finishes (after a cancelled mutation the
/// outcome is unknown: re-read).
/// </summary>
public sealed class Store : IDisposable
{
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);
    private readonly StoreHandle _handle;

    private Store(StoreHandle handle) => _handle = handle;

    /// <summary>The loaded native library's ABI version.</summary>
    public static Version AbiVersion
    {
        get
        {
            var v = Native.AbiVersion();
            return new Version((int)(v >> 16), (int)((v >> 8) & 0xff), (int)(v & 0xff));
        }
    }

    /// <summary>The native library's version string.</summary>
    public static unsafe string NativeVersion => Marshal.PtrToStringUTF8((IntPtr)Native.Version())!;

    private static void CheckAbi()
    {
        if (AbiVersion.Major != Native.AbiMajor)
        {
            throw new InvalidOperationException(
                $"native library ABI major {AbiVersion.Major}, this package needs {Native.AbiMajor}");
        }
    }

    /// <summary>
    /// Opens or creates the store in <paramref name="path"/> with vectors of <paramref name="dimensions"/>.
    /// A directory is open once at a time, across processes (<see cref="HmsStatus.Locked"/> otherwise).
    /// </summary>
    public static unsafe Store Open(string path, uint dimensions, OpenOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        CheckAbi();
        using var arena = new Arena();
        OpenOptionsNative native;
        Native.OpenOptionsInit(&native);
        if (options is not null)
        {
            if (options.EncryptionKey is not null) native.EncryptionKey = arena.Str(options.EncryptionKey);
            if (options.Signing is { } sign) native.Signing = sign ? 1 : -1;
            if (options.Audit is { } audit) native.Audit = audit ? 1 : -1;
            if (options.EmbeddingModel is not null)
            {
                native.EmbeddingModel = arena.Str(options.EmbeddingModel);
                native.EmbeddingRevision = arena.Str(options.EmbeddingRevision ?? "");
                native.EmbeddingDimensions = options.EmbeddingDimensions;
            }
        }
        var status = Native.StoreOpen(arena.Str(path), dimensions, &native, out var raw);
        Check(status);
        var handle = new StoreHandle();
        handle.Set(raw);
        return new Store(handle);
    }

    /// <summary>Opens a store asynchronously (it can take long for a large store).</summary>
    public static Task<Store> OpenAsync(string path, uint dimensions, OpenOptions? options = null,
        CancellationToken cancellationToken = default) =>
        Run(() => Open(path, dimensions, options), cancellationToken);

    /// <summary>Flushes and releases the store, waiting for calls in flight.</summary>
    public void Dispose() => _handle.Dispose();

    private static Task<T> Run<T>(Func<T> work, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        return Task.Run(work, token);
    }

    private static Task Run(Action work, CancellationToken token) =>
        Run<object?>(() => { work(); return null; }, token);

    /// <summary>Stores the documents as one atomic batch and returns each one's chunk count.</summary>
    public unsafe uint[] Ingest(IReadOnlyList<Document> documents)
    {
        ArgumentNullException.ThrowIfNull(documents);
        var count = documents.Count;
        var size = NativeArrayBytes(count, (nuint)sizeof(DocumentNative), nameof(documents));
        if (count == 0) return Array.Empty<uint>();
        using var arena = new Arena();
        var native = (DocumentNative*)arena.Alloc(size);
        for (int i = 0; i < count; i++)
        {
            native[i] = arena.Document(documents[i]);
        }
        var counts = new uint[count];
        fixed (uint* countsPtr = counts)
        {
            Check(Native.StoreIngest(_handle, native, (nuint)count, countsPtr));
        }
        return counts;
    }

    /// <summary>Stores one document and returns its chunk count.</summary>
    public uint Ingest(Document document) => Ingest(new[] { document })[0];

    /// <inheritdoc cref="Ingest(IReadOnlyList{Document})"/>
    public Task<uint[]> IngestAsync(IReadOnlyList<Document> documents, CancellationToken cancellationToken = default) =>
        Run(() => Ingest(documents), cancellationToken);

    /// <summary>Hybrid lexical (BM25) and, with an embedding, semantic search.</summary>
    public unsafe IReadOnlyList<SearchResult> Search(string query, SearchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(query);
        using var arena = new Arena();
        SearchOptionsNative native;
        Native.SearchOptionsInit(&native);
        if (options is not null) arena.Fill(ref native, options);
        IntPtr results = IntPtr.Zero;
        Check(Native.StoreSearch(_handle, arena.Str(query), &native, out results));
        try
        {
            var count = (int)Native.ResultsCount(results);
            var hits = new List<SearchResult>(count);
            for (int i = 0; i < count; i++)
            {
                ResultNative hit;
                Check(Native.ResultsGet(results, (nuint)i, &hit));
                hits.Add(new SearchResult(
                    Text(hit.Id)!, Text(hit.DocumentId)!, Text(hit.SourceUri), Text(hit.Version)!,
                    Text(hit.MetadataJson)!, Text(hit.Text), hit.StartByte, hit.EndByte, hit.Score,
                    hit.LexicalScore, double.IsNaN(hit.SemanticScore) ? null : hit.SemanticScore));
            }
            return hits;
        }
        finally
        {
            Native.ResultsFree(results);
        }
    }

    /// <inheritdoc cref="Search"/>
    public Task<IReadOnlyList<SearchResult>> SearchAsync(string query, SearchOptions? options = null,
        CancellationToken cancellationToken = default) =>
        Run(() => Search(query, options), cancellationToken);

    /// <summary>Deletes a document and its chunks; returns whether it existed.</summary>
    public bool Delete(string id)
    {
        ArgumentNullException.ThrowIfNull(id);
        using var arena = new Arena();
        var deleted = 0;
        Check(Native.StoreDelete(_handle, arena.Str(id), out deleted));
        return deleted == 1;
    }

    /// <inheritdoc cref="Delete"/>
    public Task<bool> DeleteAsync(string id, CancellationToken cancellationToken = default) =>
        Run(() => Delete(id), cancellationToken);

    /// <summary>Current statistics.</summary>
    public unsafe StoreStats Stats()
    {
        StatsNative s = default;
        s.StructSize = (uint)sizeof(StatsNative);
        Check(Native.StoreStats(_handle, &s));
        return new StoreStats(s.Dimensions, s.VectorCount, s.NsgTrained != 0, s.IvfTrained != 0,
            s.NsgTrainingRecommended != 0, s.IvfTrainingRecommended != 0, s.DocumentIndexTrained != 0,
            s.DocumentEmbeddingCount, s.StorageFormatVersion, s.StorageSegmentCount, s.StorageUsedBytes,
            s.StorageCapacityBytes);
    }

    /// <summary>Whether encryption, signing and auditing are active.</summary>
    public unsafe SecurityStatus SecurityStatus()
    {
        SecurityNative s = default;
        s.StructSize = (uint)sizeof(SecurityNative);
        Check(Native.StoreSecurityStatus(_handle, &s));
        return new SecurityStatus(s.SecurityCompiled != 0, s.EncryptionActive != 0, s.SigningActive != 0,
            s.AuditActive != 0, s.DifferentialPrivacyActive != 0, s.Epsilon);
    }

    /// <summary>Writes all state to durable storage.</summary>
    public void Flush() => Check(Native.StoreFlush(_handle));

    /// <inheritdoc cref="Flush"/>
    public Task FlushAsync(CancellationToken cancellationToken = default) => Run(Flush, cancellationToken);

    /// <summary>Rewrites the transaction log without superseded and deleted records.</summary>
    public void Compact() => Check(Native.StoreCompact(_handle));

    /// <inheritdoc cref="Compact"/>
    public Task CompactAsync(CancellationToken cancellationToken = default) => Run(Compact, cancellationToken);

    /// <summary>Trains any index whose configured threshold has been reached.</summary>
    public void Maintain() => Check(Native.StoreMaintain(_handle));

    /// <inheritdoc cref="Maintain"/>
    public Task MaintainAsync(CancellationToken cancellationToken = default) => Run(Maintain, cancellationToken);

    /// <summary>
    /// Writes a verified point-in-time copy of the store to the new directory
    /// <paramref name="destination"/>. Ingests and deletes wait while the copy is taken; searches do not.
    /// </summary>
    public void Snapshot(string destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        using var arena = new Arena();
        Check(Native.StoreSnapshot(_handle, arena.Str(destination)));
    }

    /// <inheritdoc cref="Snapshot"/>
    public Task SnapshotAsync(string destination, CancellationToken cancellationToken = default) =>
        Run(() => Snapshot(destination), cancellationToken);

    /// <summary>Splits a document the way ingest would, without a store.</summary>
    public static unsafe IReadOnlyList<DocumentChunk> ChunkDocument(Document document)
    {
        ArgumentNullException.ThrowIfNull(document);
        CheckAbi();
        using var arena = new Arena();
        var native = arena.Document(document with { Embeddings = null });
        Check(Native.ChunkDocument(&native, out var chunks));
        try
        {
            var count = (int)Native.ChunksCount(chunks);
            var result = new List<DocumentChunk>(count);
            for (int i = 0; i < count; i++)
            {
                ChunkNative c;
                Check(Native.ChunksGet(chunks, (nuint)i, &c));
                result.Add(new DocumentChunk(Text(c.Id)!, Text(c.Text)!, c.StartByte, c.EndByte));
            }
            return result;
        }
        finally
        {
            Native.ChunksFree(chunks);
        }
    }

    private static unsafe string? Text(HmsStr s) =>
        s.Ptr == null ? null : s.Len == 0 ? "" : StrictUtf8.GetString(s.Ptr, checked((int)s.Len));

    /// <summary>
    /// Raises the native error of the call that just returned <paramref name="status"/>. The
    /// native error slot is per thread, so this runs on the same thread right after the call with
    /// no await between them.
    /// </summary>
    private static unsafe void Check(int status)
    {
        if (status == 0) return;
        var message = Marshal.PtrToStringUTF8((IntPtr)Native.LastErrorMessage()) ?? "";
        throw new HmsException(status, message);
    }

    internal const int MaxNativeArrayLength = 1 << 24;
    internal const int MaxNativeBufferBytes = 1 << 30;

    internal static nuint NativeArrayBytes(int count, nuint elementSize, string parameter)
    {
        if (count < 0 || count > MaxNativeArrayLength)
        {
            throw new ArgumentOutOfRangeException(parameter, $"array length must be 0..{MaxNativeArrayLength}");
        }
        return checked((nuint)count * elementSize);
    }

    internal static nuint NativeMatrixBytes(int rows, int columns, nuint elementSize, string parameter)
    {
        NativeArrayBytes(rows, 1, parameter);
        NativeArrayBytes(columns, 1, parameter);
        var count = checked((nuint)rows * (nuint)columns);
        if (count > MaxNativeArrayLength)
        {
            throw new ArgumentOutOfRangeException(parameter, $"matrix must contain at most {MaxNativeArrayLength} values");
        }
        return checked(count * elementSize);
    }

    /// <summary>Native memory for the arguments of one call, freed together.</summary>
    private sealed unsafe class Arena : IDisposable
    {
        private readonly List<IntPtr> _blocks = new();

        public void* Alloc(nuint size)
        {
            var p = NativeMemory.AllocZeroed(size == 0 ? 1 : size);
            if (p == null) throw new OutOfMemoryException();
            _blocks.Add((IntPtr)p);
            return p;
        }

        public HmsStr Str(string s)
        {
            int n;
            try { n = StrictUtf8.GetByteCount(s); }
            catch (EncoderFallbackException e) { throw new ArgumentException("string is not valid Unicode", e); }
            if (n > MaxNativeBufferBytes) throw new ArgumentOutOfRangeException(nameof(s), "UTF-8 string exceeds the native buffer limit");
            var p = (byte*)Alloc((nuint)n);
            StrictUtf8.GetBytes(s, new Span<byte>(p, n));
            return new HmsStr { Ptr = p, Len = (nuint)n };
        }

        public DocumentNative Document(Document d)
        {
            ArgumentNullException.ThrowIfNull(d);
            var n = new DocumentNative
            {
                Id = Str(d.Id),
                Text = Str(d.Text),
                ChunkWords = d.ChunkWords,
                OverlapWords = d.OverlapWords,
                TitleWeight = d.TitleWeight,
                StoreText = d.DiscardText ? -1 : 0,
            };
            if (d.SourceUri is not null) n.SourceUri = Str(d.SourceUri);
            if (d.Version is not null) n.Version = Str(d.Version);
            if (d.MetadataJson is not null) n.MetadataJson = Str(d.MetadataJson);
            if (d.Embeddings is { } rows)
            {
                var count = rows.Count;
                NativeArrayBytes(count, (nuint)sizeof(double), nameof(d.Embeddings));
                if (count == 0) return n;
                var dim = rows[0].Length;
                if (dim == 0) throw new ArgumentException($"document {d.Id} has empty embeddings");
                var flat = (double*)Alloc(NativeMatrixBytes(count, dim, (nuint)sizeof(double), nameof(d.Embeddings)));
                for (int r = 0; r < count; r++)
                {
                    var row = rows[r];
                    if (row.Length != dim) throw new ArgumentException($"document {d.Id} has ragged embeddings");
                    row.CopyTo(new Span<double>(flat + checked((long)r * dim), dim));
                }
                n.Embeddings = flat;
                n.EmbeddingRows = (nuint)count;
                n.EmbeddingDim = (nuint)dim;
            }
            return n;
        }

        public void Fill(ref SearchOptionsNative n, SearchOptions o)
        {
            n.K = o.K;
            n.CandidateLimit = o.CandidateLimit;
            if (o.FilterJson is not null) n.FilterJson = Str(o.FilterJson);
            if (o.SourceUri is not null) n.SourceUri = Str(o.SourceUri);
            if (o.DocumentIds is { } ids)
            {
                var count = ids.Count;
                var arr = (HmsStr*)Alloc(NativeArrayBytes(count, (nuint)sizeof(HmsStr), nameof(o.DocumentIds)));
                for (int i = 0; i < count; i++) arr[i] = Str(ids[i]);
                n.DocumentIds = arr;
                n.DocumentIdCount = (nuint)count;
            }
            if (o.Embedding is { Length: > 0 } e)
            {
                var p = (double*)Alloc(NativeArrayBytes(e.Length, (nuint)sizeof(double), nameof(o.Embedding)));
                e.CopyTo(new Span<double>(p, e.Length));
                n.Embedding = p;
                n.EmbeddingLen = (nuint)e.Length;
            }
            if (o.Fusion is not null) n.Fusion = Str(o.Fusion);
            uint flags = 0;
            if (o.LexicalWeight is { } lw) { n.LexicalWeight = lw; flags |= 1; }
            if (o.SemanticWeight is { } sw) { n.SemanticWeight = sw; flags |= 2; }
            if (o.MinSemanticScore is { } ms) { n.MinSemanticScore = ms; flags |= 4; }
            if (o.Bm25K1 is { } k1) { n.Bm25K1 = k1; flags |= 8; }
            if (o.Bm25B is { } b) { n.Bm25B = b; flags |= 16; }
            n.Flags = flags;
        }

        public void Dispose()
        {
            foreach (var p in _blocks) NativeMemory.Free((void*)p);
            _blocks.Clear();
        }
    }
}
