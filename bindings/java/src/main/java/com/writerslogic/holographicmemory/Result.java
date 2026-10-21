// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.holographicmemory;

/**
 * One search hit.
 *
 * @param chunkId id of the matching chunk
 * @param documentId id of its document
 * @param sourceUri the document's source URI, or null
 * @param version the document version
 * @param metadataJson the document metadata as JSON text
 * @param text the chunk text, or null when the document was stored without text
 * @param startByte start of the chunk in the document's UTF-8 bytes
 * @param endByte end of the chunk (exclusive)
 * @param score fused score
 * @param lexicalScore BM25 component
 * @param semanticScore semantic component, or null when the hit has none
 */
public record Result(
        String chunkId,
        String documentId,
        String sourceUri,
        String version,
        String metadataJson,
        String text,
        long startByte,
        long endByte,
        double score,
        double lexicalScore,
        Double semanticScore) {}
