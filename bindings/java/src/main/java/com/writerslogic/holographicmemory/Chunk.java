// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.holographicmemory;

/**
 * A slice of a document as ingest would store it.
 *
 * @param id chunk id
 * @param text chunk text
 * @param startByte start offset in the document's UTF-8 bytes
 * @param endByte end offset (exclusive)
 */
public record Chunk(String id, String text, long startByte, long endByte) {}
