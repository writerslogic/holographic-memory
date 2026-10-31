// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.warm;

import java.util.List;

/**
 * Search tuning. Every field defaults to "engine default"; boxed numbers distinguish unset from
 * zero.
 *
 * @param k hits to return, 0 for the engine default
 * @param candidateLimit candidates gathered per list before fusion, 0 for the default
 * @param filterJson JSON metadata filter, or null
 * @param sourceUri restrict to one source URI, or null
 * @param documentIds restrict to these document ids, or null
 * @param embedding query vector for semantic or hybrid search, or null
 * @param lexicalWeight weight of the lexical list, or null
 * @param semanticWeight weight of the semantic list, or null
 * @param minSemanticScore minimum semantic score, or null
 * @param fusion {@code "blend"} (default) or {@code "rrf"}, or null
 * @param bm25K1 BM25 term-frequency saturation (default 1.2, finite and nonnegative), or null
 * @param bm25B BM25 length normalization (default 0.75, in [0, 1]), or null
 */
public record SearchOptions(
        int k,
        int candidateLimit,
        String filterJson,
        String sourceUri,
        List<String> documentIds,
        double[] embedding,
        Double lexicalWeight,
        Double semanticWeight,
        Double minSemanticScore,
        String fusion,
        Double bm25K1,
        Double bm25B) {

    public static Builder builder() {
        return new Builder();
    }

    /** Options that select every engine default. */
    public static SearchOptions defaults() {
        return builder().build();
    }

    /** Fluent builder. */
    public static final class Builder {
        private int k;
        private int candidateLimit;
        private String filterJson;
        private String sourceUri;
        private List<String> documentIds;
        private double[] embedding;
        private Double lexicalWeight;
        private Double semanticWeight;
        private Double minSemanticScore;
        private String fusion;
        private Double bm25K1;
        private Double bm25B;

        public Builder k(int value) {
            this.k = value;
            return this;
        }

        public Builder candidateLimit(int value) {
            this.candidateLimit = value;
            return this;
        }

        public Builder filterJson(String value) {
            this.filterJson = value;
            return this;
        }

        public Builder sourceUri(String value) {
            this.sourceUri = value;
            return this;
        }

        public Builder documentIds(List<String> value) {
            this.documentIds = value;
            return this;
        }

        public Builder embedding(double[] value) {
            this.embedding = value;
            return this;
        }

        public Builder lexicalWeight(double value) {
            this.lexicalWeight = value;
            return this;
        }

        public Builder semanticWeight(double value) {
            this.semanticWeight = value;
            return this;
        }

        public Builder minSemanticScore(double value) {
            this.minSemanticScore = value;
            return this;
        }

        public Builder fusion(String value) {
            this.fusion = value;
            return this;
        }

        public Builder bm25K1(double value) {
            this.bm25K1 = value;
            return this;
        }

        public Builder bm25B(double value) {
            this.bm25B = value;
            return this;
        }

        public SearchOptions build() {
            if (k < 0 || candidateLimit < 0) {
                throw new IllegalArgumentException("k and candidateLimit must be >= 0");
            }
            return new SearchOptions(k, candidateLimit, filterJson, sourceUri, documentIds, embedding,
                    lexicalWeight, semanticWeight, minSemanticScore, fusion, bm25K1, bm25B);
        }
    }
}
