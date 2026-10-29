// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

using System.Reflection;
using System.Runtime.InteropServices;

namespace WritersLogic.HolographicMemory;

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct HmsStr
{
    public byte* Ptr;
    public nuint Len;
}

[StructLayout(LayoutKind.Sequential)]
internal struct OpenOptionsNative
{
    public uint StructSize;
    public uint Flags;
    public HmsStr EncryptionKey;
    public int Signing;
    public int Audit;
    public HmsStr EmbeddingModel;
    public HmsStr EmbeddingRevision;
    public uint EmbeddingDimensions;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct DocumentNative
{
    public HmsStr Id;
    public HmsStr Text;
    public HmsStr SourceUri;
    public HmsStr Version;
    public HmsStr MetadataJson;
    public uint ChunkWords;
    public uint OverlapWords;
    public uint TitleWeight;
    public int StoreText;
    public double* Embeddings;
    public nuint EmbeddingRows;
    public nuint EmbeddingDim;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct SearchOptionsNative
{
    public uint StructSize;
    public uint Flags;
    public uint K;
    public uint CandidateLimit;
    public HmsStr FilterJson;
    public HmsStr SourceUri;
    public HmsStr* DocumentIds;
    public nuint DocumentIdCount;
    public double* Embedding;
    public nuint EmbeddingLen;
    public double LexicalWeight;
    public double SemanticWeight;
    public double MinSemanticScore;
    public double Bm25K1;
    public double Bm25B;
    public HmsStr Fusion;
}

[StructLayout(LayoutKind.Sequential)]
internal struct ResultNative
{
    public HmsStr Id;
    public HmsStr DocumentId;
    public HmsStr SourceUri;
    public HmsStr Version;
    public HmsStr MetadataJson;
    public HmsStr Text;
    public uint StartByte;
    public uint EndByte;
    public double Score;
    public double LexicalScore;
    public double SemanticScore;
}

[StructLayout(LayoutKind.Sequential)]
internal struct ChunkNative
{
    public HmsStr Id;
    public HmsStr Text;
    public uint StartByte;
    public uint EndByte;
}

[StructLayout(LayoutKind.Sequential)]
internal struct StatsNative
{
    public uint StructSize;
    public uint Dimensions;
    public ulong VectorCount;
    public int NsgTrained;
    public int IvfTrained;
    public int NsgTrainingRecommended;
    public int IvfTrainingRecommended;
    public int DocumentIndexTrained;
    public uint StorageFormatVersion;
    public uint StorageSegmentCount;
    public ulong DocumentEmbeddingCount;
    public ulong StorageUsedBytes;
    public ulong StorageCapacityBytes;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SecurityNative
{
    public uint StructSize;
    public int SecurityCompiled;
    public int EncryptionActive;
    public int SigningActive;
    public int AuditActive;
    public int DifferentialPrivacyActive;
    public double Epsilon;
}

/// <summary>The C ABI entry points. Symbols carry the ABI major version (<c>hms1_</c>).</summary>
internal static unsafe partial class Native
{
    internal const string Library = "holographic_memory_c";
    internal const int AbiMajor = 1;

    static Native()
    {
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);
    }

    /// <summary>The platform file name, e.g. <c>libholographic_memory_c.dylib</c>.</summary>
    internal static string FileName =>
        OperatingSystem.IsWindows() ? "holographic_memory_c.dll"
        : OperatingSystem.IsMacOS() ? "libholographic_memory_c.dylib"
        : "libholographic_memory_c.so";

    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? path)
    {
        if (name != Library)
        {
            return IntPtr.Zero;
        }
        var env = Environment.GetEnvironmentVariable("HMS_LIBRARY_PATH");
        if (!string.IsNullOrWhiteSpace(env))
        {
            var file = Directory.Exists(env) ? System.IO.Path.Join(env, FileName) : env;
            return NativeLibrary.Load(file);
        }
        var rid = RuntimeInformation.RuntimeIdentifier;
        foreach (var candidate in new[]
        {
            System.IO.Path.Join(AppContext.BaseDirectory, FileName),
            System.IO.Path.Join(AppContext.BaseDirectory, "runtimes", PortableRid(), "native", FileName),
            System.IO.Path.Join(AppContext.BaseDirectory, "runtimes", rid, "native", FileName),
        })
        {
            if (File.Exists(candidate))
            {
                return NativeLibrary.Load(candidate);
            }
        }
        return NativeLibrary.TryLoad(FileName, assembly, path, out var handle) ? handle : IntPtr.Zero;
    }

    /// <summary>The portable runtime identifier used under <c>runtimes/</c>: osx-arm64, linux-x64, win-x64, ...</summary>
    internal static string PortableRid()
    {
        var os = OperatingSystem.IsWindows() ? "win" : OperatingSystem.IsMacOS() ? "osx" : "linux";
        var arch = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.Arm64 => "arm64",
            Architecture.X64 => "x64",
            var other => throw new PlatformNotSupportedException($"unsupported CPU architecture {other}"),
        };
        return $"{os}-{arch}";
    }

    [LibraryImport(Library, EntryPoint = "hms1_abi_version")]
    internal static partial uint AbiVersion();

    [LibraryImport(Library, EntryPoint = "hms1_version")]
    internal static partial byte* Version();

    [LibraryImport(Library, EntryPoint = "hms1_last_error_message")]
    internal static partial byte* LastErrorMessage();

    [LibraryImport(Library, EntryPoint = "hms1_open_options_init")]
    internal static partial void OpenOptionsInit(OpenOptionsNative* options);

    [LibraryImport(Library, EntryPoint = "hms1_search_options_init")]
    internal static partial void SearchOptionsInit(SearchOptionsNative* options);

    [LibraryImport(Library, EntryPoint = "hms1_store_open")]
    internal static partial int StoreOpen(HmsStr path, uint dimensions, OpenOptionsNative* options, out IntPtr store);

    [LibraryImport(Library, EntryPoint = "hms1_store_close")]
    internal static partial int StoreClose(IntPtr store);

    [LibraryImport(Library, EntryPoint = "hms1_store_ingest")]
    internal static partial int StoreIngest(StoreHandle store, DocumentNative* documents, nuint count, uint* chunkCounts);

    [LibraryImport(Library, EntryPoint = "hms1_store_search")]
    internal static partial int StoreSearch(StoreHandle store, HmsStr query, SearchOptionsNative* options, out IntPtr results);

    [LibraryImport(Library, EntryPoint = "hms1_store_delete")]
    internal static partial int StoreDelete(StoreHandle store, HmsStr id, out int deleted);

    [LibraryImport(Library, EntryPoint = "hms1_store_stats")]
    internal static partial int StoreStats(StoreHandle store, StatsNative* stats);

    [LibraryImport(Library, EntryPoint = "hms1_store_security_status")]
    internal static partial int StoreSecurityStatus(StoreHandle store, SecurityNative* status);

    [LibraryImport(Library, EntryPoint = "hms1_store_flush")]
    internal static partial int StoreFlush(StoreHandle store);

    [LibraryImport(Library, EntryPoint = "hms1_store_compact")]
    internal static partial int StoreCompact(StoreHandle store);

    [LibraryImport(Library, EntryPoint = "hms1_store_maintain")]
    internal static partial int StoreMaintain(StoreHandle store);

    [LibraryImport(Library, EntryPoint = "hms1_store_snapshot")]
    internal static partial int StoreSnapshot(StoreHandle store, HmsStr destination);

    [LibraryImport(Library, EntryPoint = "hms1_chunk_document")]
    internal static partial int ChunkDocument(DocumentNative* document, out IntPtr chunks);

    [LibraryImport(Library, EntryPoint = "hms1_results_count")]
    internal static partial nuint ResultsCount(IntPtr results);

    [LibraryImport(Library, EntryPoint = "hms1_results_get")]
    internal static partial int ResultsGet(IntPtr results, nuint index, ResultNative* hit);

    [LibraryImport(Library, EntryPoint = "hms1_results_free")]
    internal static partial void ResultsFree(IntPtr results);

    [LibraryImport(Library, EntryPoint = "hms1_chunks_count")]
    internal static partial nuint ChunksCount(IntPtr chunks);

    [LibraryImport(Library, EntryPoint = "hms1_chunks_get")]
    internal static partial int ChunksGet(IntPtr chunks, nuint index, ChunkNative* chunk);

    [LibraryImport(Library, EntryPoint = "hms1_chunks_free")]
    internal static partial void ChunksFree(IntPtr chunks);
}

/// <summary>Owns a native store handle; the native call holds a reference while it runs, so
/// <see cref="SafeHandle.Dispose()"/> during a call is deferred until the call returns.</summary>
internal sealed class StoreHandle : SafeHandle
{
    public StoreHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle() => Native.StoreClose(handle) == 0;

    internal void Set(IntPtr value) => SetHandle(value);
}
