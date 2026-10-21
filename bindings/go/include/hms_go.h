// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Helpers shared by the cgo preambles of this package; not part of the C ABI.
#ifndef HMS_GO_H
#define HMS_GO_H

#include <stdlib.h>
#include <string.h>
#include "holographic_memory.h"

// A non-NULL pointer for present-but-empty strings: NULL with length 0 means "absent".
static const uint8_t hms_go_empty[1] = {0};

static inline hms_str_t hms_go_str(const uint8_t *ptr, size_t len, int present) {
	hms_str_t s;
	s.ptr = present ? (len == 0 ? hms_go_empty : ptr) : NULL;
	s.len = len;
	return s;
}

static inline hms_document_t *hms_go_docs_alloc(size_t n) {
	return (hms_document_t *)calloc(n ? n : 1, sizeof(hms_document_t));
}

static inline hms_str_t *hms_go_strs_alloc(size_t n) {
	return (hms_str_t *)calloc(n ? n : 1, sizeof(hms_str_t));
}

static inline void hms_go_doc_set_strings(hms_document_t *d, hms_str_t id, hms_str_t text,
	hms_str_t source_uri, hms_str_t version, hms_str_t metadata) {
	d->id = id;
	d->text = text;
	d->source_uri = source_uri;
	d->version = version;
	d->metadata_json = metadata;
}

static inline void hms_go_doc_set_numbers(hms_document_t *d, uint32_t chunk_words, uint32_t overlap_words,
	uint32_t title_weight, int32_t store_text, const double *embeddings, size_t rows, size_t dim) {
	d->chunk_words = chunk_words;
	d->overlap_words = overlap_words;
	d->title_weight = title_weight;
	d->store_text = store_text;
	d->embeddings = embeddings;
	d->embedding_rows = rows;
	d->embedding_dim = dim;
}

#endif
