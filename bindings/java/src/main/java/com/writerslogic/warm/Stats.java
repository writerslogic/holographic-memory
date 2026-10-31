// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.warm;

/**
 * Store statistics.
 *
 * @param dimensions vector dimension
 * @param vectorCount vectors in the store (document chunks count as vectors)
 * @param nsgTrained whether the NSG index is trained
 * @param ivfTrained whether the IVF index is trained
 * @param nsgTrainingRecommended whether NSG training is due
 * @param ivfTrainingRecommended whether IVF training is due
 * @param documentIndexTrained whether the dense document index is trained
 * @param documentEmbeddingCount embedded chunks
 * @param storageFormatVersion on-disk format version
 * @param storageSegmentCount storage segments
 * @param storageUsedBytes logical bytes used
 * @param storageCapacityBytes allocated bytes
 */
public record Stats(
        long dimensions,
        long vectorCount,
        boolean nsgTrained,
        boolean ivfTrained,
        boolean nsgTrainingRecommended,
        boolean ivfTrainingRecommended,
        boolean documentIndexTrained,
        long documentEmbeddingCount,
        long storageFormatVersion,
        long storageSegmentCount,
        long storageUsedBytes,
        long storageCapacityBytes) {}
