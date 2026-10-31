// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

namespace WritersLogic.WarmMemory;

/// <summary>Native status codes, mirroring the <c>WARM_ERR_*</c> constants.</summary>
public enum WarmStatus
{
    NullPointer = 1,
    InvalidArgument = 2,
    InvalidUtf8 = 3,
    InvalidHandle = 4,
    IO = 5,
    Locked = 6,
    Encryption = 7,
    Corrupt = 8,
    LimitExceeded = 9,
    AbiMismatch = 10,
    Panic = 98,
    Internal = 99,
}

/// <summary>A failure reported by the native library.</summary>
public sealed class WarmException : Exception
{
    internal WarmException(int status, string message) : base(message) => Status = (WarmStatus)status;

    /// <summary>The native status code.</summary>
    public WarmStatus Status { get; }
}

/// <summary>One document to ingest.</summary>
public sealed record Document(string Id, string Text)
{
    /// <summary>Optional source URI.</summary>
    public string? SourceUri { get; init; }

    /// <summary>Optional version string.</summary>
    public string? Version { get; init; }

    /// <summary>Optional JSON value stored with the document and returned with hits.</summary>
    public string? MetadataJson { get; init; }

    /// <summary>Words per chunk; 0 is the engine default.</summary>
    public uint ChunkWords { get; init; }

    /// <summary>Overlap words between chunks; 0 is the engine default.</summary>
    public uint OverlapWords { get; init; }

    /// <summary>Counts the first line's terms this many times in the first chunk (a BM25 title boost); 0 or 1 changes nothing.</summary>
    public uint TitleWeight { get; init; }

    /// <summary>Index the document without keeping its text for retrieval.</summary>
    public bool DiscardText { get; init; }

    /// <summary>Optional caller-supplied vectors, one row per chunk, of the store's embedding dimension.</summary>
    public IReadOnlyList<double[]>? Embeddings { get; init; }
}

/// <summary>Search tuning. Every member defaults to the engine default; nullable numbers distinguish unset from zero.</summary>
public sealed record SearchOptions
{
    /// <summary>Hits to return; 0 is the engine default.</summary>
    public uint K { get; init; }

    /// <summary>Candidates gathered per list before fusion; 0 is the engine default.</summary>
    public uint CandidateLimit { get; init; }

    /// <summary>JSON metadata filter.</summary>
    public string? FilterJson { get; init; }

    /// <summary>Restrict to one source URI.</summary>
    public string? SourceUri { get; init; }

    /// <summary>Restrict to these document ids.</summary>
    public IReadOnlyList<string>? DocumentIds { get; init; }

    /// <summary>Query vector for semantic or hybrid search.</summary>
    public double[]? Embedding { get; init; }

    public double? LexicalWeight { get; init; }

    public double? SemanticWeight { get; init; }

    public double? MinSemanticScore { get; init; }

    /// <summary><c>"blend"</c> (default) or <c>"rrf"</c>.</summary>
    public string? Fusion { get; init; }

    /// <summary>BM25 term-frequency saturation (default 1.2, finite and nonnegative).</summary>
    public double? Bm25K1 { get; init; }

    /// <summary>BM25 length normalization (default 0.75, in [0, 1]).</summary>
    public double? Bm25B { get; init; }
}

/// <summary>Store open options.</summary>
public sealed record OpenOptions
{
    /// <summary>Passphrase for at-rest encryption; the same one is required on every later open.</summary>
    public string? EncryptionKey { get; init; }

    public bool? Signing { get; init; }

    public bool? Audit { get; init; }

    public string? EmbeddingModel { get; init; }

    public string? EmbeddingRevision { get; init; }

    /// <summary>Embedding dimension (1..65536); 0 when no embedding space is declared.</summary>
    public uint EmbeddingDimensions { get; init; }
}

/// <summary>One search hit.</summary>
public sealed record SearchResult(
    string ChunkId,
    string DocumentId,
    string? SourceUri,
    string Version,
    string MetadataJson,
    string? Text,
    uint StartByte,
    uint EndByte,
    double Score,
    double LexicalScore,
    double? SemanticScore);

/// <summary>A slice of a document as ingest would store it.</summary>
public sealed record DocumentChunk(string Id, string Text, uint StartByte, uint EndByte);

/// <summary>Store statistics.</summary>
public sealed record StoreStats(
    uint Dimensions,
    ulong VectorCount,
    bool NsgTrained,
    bool IvfTrained,
    bool NsgTrainingRecommended,
    bool IvfTrainingRecommended,
    bool DocumentIndexTrained,
    ulong DocumentEmbeddingCount,
    uint StorageFormatVersion,
    uint StorageSegmentCount,
    ulong StorageUsedBytes,
    ulong StorageCapacityBytes);

/// <summary>The store's protection state.</summary>
public sealed record SecurityStatus(
    bool SecurityCompiled,
    bool EncryptionActive,
    bool SigningActive,
    bool AuditActive,
    bool DifferentialPrivacyActive,
    double Epsilon);
