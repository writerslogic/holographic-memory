// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.holographicmemory;

import java.util.Objects;

/**
 * One document to ingest.
 *
 * @param id unique non-empty identifier of at most 4096 bytes; re-ingesting an id replaces it
 * @param text the UTF-8 body, at most 8 MiB
 * @param sourceUri optional source URI, or null
 * @param version optional version string, or null
 * @param metadataJson optional JSON value stored with the document and returned with hits, or null
 * @param chunkWords words per chunk, 0 for the engine default
 * @param overlapWords overlap words between chunks, 0 for the engine default
 * @param titleWeight counts the first line's terms this many times in the first chunk (a BM25
 *     title boost); 0 or 1 leaves the document unchanged
 * @param discardText index the document without keeping its text for retrieval
 * @param embeddings optional caller-supplied vectors, one row per chunk, all of the dimension the
 *     store was opened with via {@link OpenOptions.Builder#embeddingSpace}; or null
 */
public record Document(
        String id,
        String text,
        String sourceUri,
        String version,
        String metadataJson,
        int chunkWords,
        int overlapWords,
        int titleWeight,
        boolean discardText,
        double[][] embeddings) {

    public Document {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(text, "text");
        if (chunkWords < 0 || overlapWords < 0 || titleWeight < 0) {
            throw new IllegalArgumentException("chunkWords, overlapWords and titleWeight must be >= 0");
        }
    }

    /** A document with only an id and text. */
    public static Document of(String id, String text) {
        return new Document(id, text, null, null, null, 0, 0, 0, false, null);
    }

    public Document withSourceUri(String value) {
        return new Document(id, text, value, version, metadataJson, chunkWords, overlapWords, titleWeight, discardText, embeddings);
    }

    public Document withVersion(String value) {
        return new Document(id, text, sourceUri, value, metadataJson, chunkWords, overlapWords, titleWeight, discardText, embeddings);
    }

    public Document withMetadataJson(String value) {
        return new Document(id, text, sourceUri, version, value, chunkWords, overlapWords, titleWeight, discardText, embeddings);
    }

    public Document withChunking(int words, int overlap) {
        return new Document(id, text, sourceUri, version, metadataJson, words, overlap, titleWeight, discardText, embeddings);
    }

    public Document withTitleWeight(int value) {
        return new Document(id, text, sourceUri, version, metadataJson, chunkWords, overlapWords, value, discardText, embeddings);
    }

    public Document withDiscardText(boolean value) {
        return new Document(id, text, sourceUri, version, metadataJson, chunkWords, overlapWords, titleWeight, value, embeddings);
    }

    public Document withEmbeddings(double[][] value) {
        return new Document(id, text, sourceUri, version, metadataJson, chunkWords, overlapWords, titleWeight, discardText, value);
    }
}
