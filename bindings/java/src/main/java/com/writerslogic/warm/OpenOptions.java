// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package com.writerslogic.warm;

/**
 * Store open options.
 *
 * @param encryptionKey passphrase for at-rest encryption, or null for none; the same passphrase is
 *     required on every later open
 * @param signing null for the default (off), or true/false
 * @param audit null for the default (off), or true/false
 * @param embeddingModel embedding model id for caller-supplied embeddings, or null
 * @param embeddingRevision embedding model revision, or null
 * @param embeddingDimensions embedding dimension (1..65536), or 0 when no embedding space is set
 */
public record OpenOptions(
        String encryptionKey,
        Boolean signing,
        Boolean audit,
        String embeddingModel,
        String embeddingRevision,
        int embeddingDimensions) {

    public static Builder builder() {
        return new Builder();
    }

    public static OpenOptions defaults() {
        return builder().build();
    }

    /** Fluent builder. */
    public static final class Builder {
        private String encryptionKey;
        private Boolean signing;
        private Boolean audit;
        private String embeddingModel;
        private String embeddingRevision;
        private int embeddingDimensions;

        public Builder encryptionKey(String passphrase) {
            this.encryptionKey = passphrase;
            return this;
        }

        public Builder signing(boolean value) {
            this.signing = value;
            return this;
        }

        public Builder audit(boolean value) {
            this.audit = value;
            return this;
        }

        public Builder embeddingSpace(String model, String revision, int dimensions) {
            this.embeddingModel = model;
            this.embeddingRevision = revision;
            this.embeddingDimensions = dimensions;
            return this;
        }

        public OpenOptions build() {
            return new OpenOptions(encryptionKey, signing, audit, embeddingModel, embeddingRevision,
                    embeddingDimensions);
        }
    }
}
