// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package hms

/*
#include "hms_go.h"
*/
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"
)

// ABIMajor is the C ABI major version this package was written against.
const ABIMajor = 1

// ABIVersion returns the loaded native library's ABI version as major, minor, patch. The
// package refuses to open stores when the library's major differs from ABIMajor.
func ABIVersion() (major, minor, patch int) {
	v := uint32(C.hms1_abi_version())
	return int(v >> 16), int(v >> 8 & 0xff), int(v & 0xff)
}

// Version returns the native library's own version string.
func Version() string { return C.GoString(C.hms1_version()) }

func checkABI() error {
	if major, _, _ := ABIVersion(); major != ABIMajor {
		return fmt.Errorf("hms: native library ABI major %d, this package needs %d", major, ABIMajor)
	}
	return nil
}

// Code is a native status code.
type Code int

// Native status codes, mirroring the HMS_ERR_* constants.
const (
	CodeNullPointer     Code = C.HMS_ERR_NULL_POINTER
	CodeInvalidArgument Code = C.HMS_ERR_INVALID_ARGUMENT
	CodeInvalidUTF8     Code = C.HMS_ERR_INVALID_UTF8
	CodeInvalidHandle   Code = C.HMS_ERR_INVALID_HANDLE
	CodeIO              Code = C.HMS_ERR_IO
	CodeLocked          Code = C.HMS_ERR_LOCKED
	CodeEncryption      Code = C.HMS_ERR_ENCRYPTION
	CodeCorrupt         Code = C.HMS_ERR_CORRUPT
	CodeLimitExceeded   Code = C.HMS_ERR_LIMIT_EXCEEDED
	CodeABIMismatch     Code = C.HMS_ERR_ABI_MISMATCH
	CodePanic           Code = C.HMS_ERR_PANIC
	CodeInternal        Code = C.HMS_ERR_INTERNAL
)

// Error is a failure reported by the native library.
type Error struct {
	Code    Code
	Message string
}

func (e *Error) Error() string { return fmt.Sprintf("hms: %s (code %d)", e.Message, e.Code) }

// Is lets errors.Is(err, ErrLocked) match by code.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	return ok && t.Message == "" && t.Code == e.Code
}

// Sentinels for errors.Is.
var (
	ErrLocked          = &Error{Code: CodeLocked}
	ErrEncryption      = &Error{Code: CodeEncryption}
	ErrInvalidArgument = &Error{Code: CodeInvalidArgument}
	ErrInvalidHandle   = &Error{Code: CodeInvalidHandle}
	ErrCorrupt         = &Error{Code: CodeCorrupt}
	ErrIO              = &Error{Code: CodeIO}
)

// ErrClosed is returned for any use of a Store after Close.
var ErrClosed = errors.New("hms: store is closed")

// Document is one document to ingest.
type Document struct {
	ID string
	// Text is the UTF-8 body, at most 8 MiB.
	Text      string
	SourceURI *string
	Version   *string
	// Metadata is any JSON-serializable value, returned with every hit.
	Metadata any
	// ChunkWords and OverlapWords of 0 select the engine defaults.
	ChunkWords   uint32
	OverlapWords uint32
	// TitleWeight counts the first line's terms this many times in the first chunk (a BM25
	// title boost); 0 or 1 leaves the document unchanged.
	TitleWeight uint32
	// DiscardText indexes the document without keeping its text for retrieval.
	DiscardText bool
	// Embeddings are optional caller-supplied vectors, one row per chunk, all of the
	// dimension the store was opened with via WithEmbeddingSpace.
	Embeddings [][]float64
}

// Fusion selects how lexical and semantic rankings combine.
type Fusion string

// Fusion modes.
const (
	FusionDefault Fusion = ""
	FusionBlend   Fusion = "blend"
	FusionRRF     Fusion = "rrf"
)

// SearchOptions tunes a search. The zero value selects every engine default; pointer fields
// distinguish "unset" from zero.
type SearchOptions struct {
	// K is the number of hits; 0 is the engine default.
	K              uint32
	CandidateLimit uint32
	// Filter is a JSON-serializable metadata filter.
	Filter      any
	SourceURI   *string
	DocumentIDs []string
	// Embedding is a query vector for semantic or hybrid search.
	Embedding        []float64
	LexicalWeight    *float64
	SemanticWeight   *float64
	MinSemanticScore *float64
	Fusion           Fusion
	// BM25K1 (default 1.2, finite and nonnegative) and BM25B (default 0.75, in [0, 1]).
	BM25K1 *float64
	BM25B  *float64
}

// Result is one search hit.
type Result struct {
	ChunkID    string
	DocumentID string
	SourceURI  *string
	Version    string
	// Metadata is the document's metadata decoded from JSON.
	Metadata json.RawMessage
	// Text is nil when the document was stored without text.
	Text          *string
	StartByte     uint32
	EndByte       uint32
	Score         float64
	LexicalScore  float64
	SemanticScore *float64
}

// Chunk is a slice of a document as ingest would store it.
type Chunk struct {
	ID        string
	Text      string
	StartByte uint32
	EndByte   uint32
}

// Stats describes an open store.
type Stats struct {
	Dimensions             uint32
	VectorCount            uint64
	NSGTrained             bool
	IVFTrained             bool
	NSGTrainingRecommended bool
	IVFTrainingRecommended bool
	DocumentIndexTrained   bool
	DocumentEmbeddingCount uint64
	StorageFormatVersion   uint32
	StorageSegmentCount    uint32
	StorageUsedBytes       uint64
	StorageCapacityBytes   uint64
}

// SecurityStatus reports the store's protection state.
type SecurityStatus struct {
	// SecurityCompiled is false when the native library was built without encryption.
	SecurityCompiled          bool
	EncryptionActive          bool
	SigningActive             bool
	AuditActive               bool
	DifferentialPrivacyActive bool
	Epsilon                   float64
}

type openConfig struct {
	key            *string
	signing, audit int32
	model, rev     *string
	dims           uint32
}

// OpenOption configures Open.
type OpenOption func(*openConfig)

// WithEncryptionKey encrypts the store at rest under a key derived from passphrase. The same
// passphrase is required on every later open.
func WithEncryptionKey(passphrase string) OpenOption {
	return func(c *openConfig) { c.key = &passphrase }
}

// WithSigning turns mutation signing on or off.
func WithSigning(on bool) OpenOption {
	return func(c *openConfig) { c.signing = tri(on) }
}

// WithAudit turns the audit log on or off.
func WithAudit(on bool) OpenOption {
	return func(c *openConfig) { c.audit = tri(on) }
}

// WithEmbeddingSpace declares the embedding model, revision and dimension for caller-supplied
// embeddings.
func WithEmbeddingSpace(model, revision string, dimensions uint32) OpenOption {
	return func(c *openConfig) { c.model, c.rev, c.dims = &model, &revision, dimensions }
}

func tri(on bool) int32 {
	if on {
		return 1
	}
	return -1
}

// Store is an open document store. It is safe for concurrent use by multiple goroutines.
type Store struct {
	mu     sync.RWMutex
	handle *C.hms_store_t
	closed bool
	cancel runtime.Cleanup
}

// Open opens or creates the store in directory path with vectors of the given dimension. A
// directory can be open once at a time, across processes (ErrLocked otherwise). Stores that
// are never closed are closed by the garbage collector, but call Close to learn about flush
// failures.
func Open(path string, dimensions uint32, opts ...OpenOption) (*Store, error) {
	if err := checkABI(); err != nil {
		return nil, err
	}
	cfg := openConfig{}
	for _, o := range opts {
		o(&cfg)
	}
	var p pins
	defer p.release()
	var options C.hms_open_options_t
	C.hms1_open_options_init(&options)
	if cfg.key != nil {
		options.encryption_key = p.str(*cfg.key, true)
	}
	options.signing = C.int32_t(cfg.signing)
	options.audit = C.int32_t(cfg.audit)
	if cfg.model != nil {
		options.embedding_model = p.str(*cfg.model, true)
		options.embedding_revision = p.str(*cfg.rev, true)
		options.embedding_dimensions = C.uint32_t(cfg.dims)
	}
	var handle *C.hms_store_t
	// The options struct lives in Go memory and holds pointers to pinned Go memory; that is
	// allowed because every pointer it holds is pinned.
	err := call(func() C.hms_status_t {
		return C.hms1_store_open(p.str(path, true), C.uint32_t(dimensions), &options, &handle)
	})
	if err != nil {
		return nil, err
	}
	s := &Store{handle: handle}
	s.cancel = runtime.AddCleanup(s, func(h *C.hms_store_t) { C.hms1_store_close(h) }, handle)
	return s, nil
}

// call runs one native call and reads its thread-local error on the same OS thread.
func call(f func() C.hms_status_t) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	return lastError(f())
}

// run executes op on the store handle and then releases p. With a cancellable ctx the call runs
// on its own goroutine and run returns ctx.Err() as soon as ctx is done; the native call itself
// cannot be interrupted, so it finishes in the background (still holding its pinned inputs and
// handling its own results) and after a cancelled mutation the outcome is unknown: re-read.
// op must do all its work, including converting results, before returning, and must not be
// read by the caller unless run returns nil.
func (s *Store) run(ctx context.Context, p *pins, op func(h *C.hms_store_t) error) error {
	if err := ctx.Err(); err != nil {
		p.release()
		return err
	}
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		p.release()
		return ErrClosed
	}
	h := s.handle
	if ctx.Done() == nil {
		defer p.release()
		defer s.mu.RUnlock()
		return op(h)
	}
	done := make(chan error, 1)
	go func() {
		err := op(h)
		s.mu.RUnlock()
		p.release()
		done <- err
	}()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close flushes and releases the store, waiting for calls in flight. It is idempotent.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	s.cancel.Stop()
	h := s.handle
	s.handle = nil
	return call(func() C.hms_status_t { return C.hms1_store_close(h) })
}

func jsonText(v any, what string) (string, bool, error) {
	if v == nil {
		return "", false, nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return "", false, fmt.Errorf("hms: %s is not JSON-serializable: %w", what, err)
	}
	return string(b), true, nil
}

// Ingest stores the documents as one atomic batch and returns each one's chunk count. Either
// all documents become searchable or none do.
func (s *Store) Ingest(ctx context.Context, docs ...Document) ([]uint32, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	p := &pins{}
	cdocs := (*C.hms_document_t)(p.calloc(unsafe.Pointer(C.hms_go_docs_alloc(C.size_t(len(docs))))))
	array := unsafe.Slice(cdocs, len(docs))
	for i := range docs {
		d := &docs[i]
		meta, hasMeta, err := jsonText(d.Metadata, "Metadata")
		if err != nil {
			p.release()
			return nil, err
		}
		var uri, version string
		if d.SourceURI != nil {
			uri = *d.SourceURI
		}
		if d.Version != nil {
			version = *d.Version
		}
		C.hms_go_doc_set_strings(&array[i], p.str(d.ID, true), p.str(d.Text, true),
			p.str(uri, d.SourceURI != nil), p.str(version, d.Version != nil), p.str(meta, hasMeta))
		var store C.int32_t
		if d.DiscardText {
			store = -1
		}
		var flat *C.double
		var rows, dim C.size_t
		if len(d.Embeddings) > 0 {
			dim = C.size_t(len(d.Embeddings[0]))
			values := make([]float64, 0, len(d.Embeddings)*len(d.Embeddings[0]))
			for _, row := range d.Embeddings {
				if len(row) != int(dim) || dim == 0 {
					p.release()
					return nil, fmt.Errorf("hms: document %q has ragged or empty embeddings", d.ID)
				}
				values = append(values, row...)
			}
			rows = C.size_t(len(d.Embeddings))
			flat = p.floats(values)
		}
		C.hms_go_doc_set_numbers(&array[i], C.uint32_t(d.ChunkWords), C.uint32_t(d.OverlapWords),
			C.uint32_t(d.TitleWeight), store, flat, rows, dim)
	}
	counts := make([]uint32, len(docs))
	err := s.run(ctx, p, func(h *C.hms_store_t) error {
		return call(func() C.hms_status_t {
			return C.hms1_store_ingest(h, cdocs, C.size_t(len(docs)), (*C.uint32_t)(&counts[0]))
		})
	})
	if err != nil {
		return nil, err
	}
	return counts, nil
}

// Search runs a hybrid lexical and (with an embedding) semantic search.
func (s *Store) Search(ctx context.Context, query string, opts *SearchOptions) ([]Result, error) {
	p := &pins{}
	var o C.hms_search_options_t
	C.hms1_search_options_init(&o)
	if opts != nil {
		if err := fillSearch(&o, p, opts); err != nil {
			p.release()
			return nil, err
		}
	}
	var out []Result
	err := s.run(ctx, p, func(h *C.hms_store_t) error {
		var results *C.hms_results_t
		if err := call(func() C.hms_status_t {
			return C.hms1_store_search(h, p.str(query, true), &o, &results)
		}); err != nil {
			return err
		}
		defer C.hms1_results_free(results)
		hits, err := readResults(results)
		out = hits
		return err
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

func readResults(results *C.hms_results_t) ([]Result, error) {
	n := int(C.hms1_results_count(results))
	out := make([]Result, n)
	for i := 0; i < n; i++ {
		var hit C.hms_result_t
		if err := call(func() C.hms_status_t { return C.hms1_results_get(results, C.size_t(i), &hit) }); err != nil {
			return nil, err
		}
		r := Result{
			ChunkID:      goStr(hit.id),
			DocumentID:   goStr(hit.document_id),
			Version:      goStr(hit.version),
			Metadata:     json.RawMessage(goStr(hit.metadata_json)),
			StartByte:    uint32(hit.start_byte),
			EndByte:      uint32(hit.end_byte),
			Score:        float64(hit.score),
			LexicalScore: float64(hit.lexical_score),
		}
		if hit.source_uri.ptr != nil {
			v := goStr(hit.source_uri)
			r.SourceURI = &v
		}
		if hit.text.ptr != nil {
			v := goStr(hit.text)
			r.Text = &v
		}
		if sem := float64(hit.semantic_score); !math.IsNaN(sem) {
			r.SemanticScore = &sem
		}
		out[i] = r
	}
	return out, nil
}

func fillSearch(o *C.hms_search_options_t, p *pins, opts *SearchOptions) error {
	o.k = C.uint32_t(opts.K)
	o.candidate_limit = C.uint32_t(opts.CandidateLimit)
	if filter, ok, err := jsonText(opts.Filter, "Filter"); err != nil {
		return err
	} else if ok {
		o.filter_json = p.str(filter, true)
	}
	if opts.SourceURI != nil {
		o.source_uri = p.str(*opts.SourceURI, true)
	}
	if n := len(opts.DocumentIDs); n > 0 {
		ids := (*C.hms_str_t)(p.calloc(unsafe.Pointer(C.hms_go_strs_alloc(C.size_t(n)))))
		slots := unsafe.Slice(ids, n)
		for i := range slots {
			slots[i] = p.str(opts.DocumentIDs[i], true)
		}
		o.document_ids = ids
		o.document_id_count = C.size_t(n)
	}
	if len(opts.Embedding) > 0 {
		o.embedding = p.floats(opts.Embedding)
		o.embedding_len = C.size_t(len(opts.Embedding))
	}
	if opts.Fusion != FusionDefault {
		o.fusion = p.str(string(opts.Fusion), true)
	}
	set := func(v *float64, bit C.uint32_t, dst *C.double) {
		if v != nil {
			o.flags |= bit
			*dst = C.double(*v)
		}
	}
	set(opts.LexicalWeight, C.HMS_SEARCH_SET_LEXICAL_WEIGHT, &o.lexical_weight)
	set(opts.SemanticWeight, C.HMS_SEARCH_SET_SEMANTIC_WEIGHT, &o.semantic_weight)
	set(opts.MinSemanticScore, C.HMS_SEARCH_SET_MIN_SEMANTIC_SCORE, &o.min_semantic_score)
	set(opts.BM25K1, C.HMS_SEARCH_SET_BM25_K1, &o.bm25_k1)
	set(opts.BM25B, C.HMS_SEARCH_SET_BM25_B, &o.bm25_b)
	return nil
}

// Delete removes a document and its chunks, reporting whether it existed.
func (s *Store) Delete(ctx context.Context, id string) (bool, error) {
	p := &pins{}
	var deleted C.int32_t
	err := s.run(ctx, p, func(h *C.hms_store_t) error {
		return call(func() C.hms_status_t { return C.hms1_store_delete(h, p.str(id, true), &deleted) })
	})
	if err != nil {
		return false, err
	}
	return deleted == 1, nil
}

// Stats returns the store's current statistics.
func (s *Store) Stats(ctx context.Context) (Stats, error) {
	var c C.hms_stats_t
	c.struct_size = C.uint32_t(unsafe.Sizeof(c))
	err := s.run(ctx, &pins{}, func(h *C.hms_store_t) error {
		return call(func() C.hms_status_t { return C.hms1_store_stats(h, &c) })
	})
	if err != nil {
		return Stats{}, err
	}
	return Stats{
		Dimensions: uint32(c.dimensions), VectorCount: uint64(c.vector_count),
		NSGTrained: c.nsg_trained != 0, IVFTrained: c.ivf_trained != 0,
		NSGTrainingRecommended: c.nsg_training_recommended != 0,
		IVFTrainingRecommended: c.ivf_training_recommended != 0,
		DocumentIndexTrained:   c.document_index_trained != 0,
		DocumentEmbeddingCount: uint64(c.document_embedding_count),
		StorageFormatVersion:   uint32(c.storage_format_version),
		StorageSegmentCount:    uint32(c.storage_segment_count),
		StorageUsedBytes:       uint64(c.storage_used_bytes),
		StorageCapacityBytes:   uint64(c.storage_capacity_bytes),
	}, nil
}

// SecurityStatus reports whether encryption, signing and auditing are active.
func (s *Store) SecurityStatus(ctx context.Context) (SecurityStatus, error) {
	var c C.hms_security_status_t
	c.struct_size = C.uint32_t(unsafe.Sizeof(c))
	err := s.run(ctx, &pins{}, func(h *C.hms_store_t) error {
		return call(func() C.hms_status_t { return C.hms1_store_security_status(h, &c) })
	})
	if err != nil {
		return SecurityStatus{}, err
	}
	return SecurityStatus{
		SecurityCompiled: c.security_compiled != 0, EncryptionActive: c.encryption_active != 0,
		SigningActive: c.signing_active != 0, AuditActive: c.audit_active != 0,
		DifferentialPrivacyActive: c.differential_privacy_active != 0, Epsilon: float64(c.epsilon),
	}, nil
}

// Flush writes all state to durable storage.
func (s *Store) Flush(ctx context.Context) error {
	return s.run(ctx, &pins{}, func(h *C.hms_store_t) error {
		return call(func() C.hms_status_t { return C.hms1_store_flush(h) })
	})
}

// Compact rewrites the transaction log without superseded and deleted records.
func (s *Store) Compact(ctx context.Context) error {
	return s.run(ctx, &pins{}, func(h *C.hms_store_t) error {
		return call(func() C.hms_status_t { return C.hms1_store_compact(h) })
	})
}

// Maintain trains any index whose configured threshold has been reached.
func (s *Store) Maintain(ctx context.Context) error {
	return s.run(ctx, &pins{}, func(h *C.hms_store_t) error {
		return call(func() C.hms_status_t { return C.hms1_store_maintain(h) })
	})
}

// Snapshot writes a verified point-in-time copy of the store to the new directory dest.
// Ingests and deletes wait while the copy is taken; searches do not.
func (s *Store) Snapshot(ctx context.Context, dest string) error {
	p := &pins{}
	return s.run(ctx, p, func(h *C.hms_store_t) error {
		return call(func() C.hms_status_t { return C.hms1_store_snapshot(h, p.str(dest, true)) })
	})
}

// ChunkDocument splits a document the way Ingest would, without a store.
func ChunkDocument(doc Document) ([]Chunk, error) {
	if err := checkABI(); err != nil {
		return nil, err
	}
	var p pins
	defer p.release()
	cdoc := (*C.hms_document_t)(p.calloc(unsafe.Pointer(C.hms_go_docs_alloc(1))))
	C.hms_go_doc_set_strings(cdoc, p.str(doc.ID, true), p.str(doc.Text, true),
		p.str("", false), p.str("", false), p.str("", false))
	C.hms_go_doc_set_numbers(cdoc, C.uint32_t(doc.ChunkWords), C.uint32_t(doc.OverlapWords),
		0, 0, nil, 0, 0)
	var chunks *C.hms_chunks_t
	if err := call(func() C.hms_status_t { return C.hms1_chunk_document(cdoc, &chunks) }); err != nil {
		return nil, err
	}
	defer C.hms1_chunks_free(chunks)
	n := int(C.hms1_chunks_count(chunks))
	out := make([]Chunk, n)
	for i := 0; i < n; i++ {
		var c C.hms_chunk_t
		if err := call(func() C.hms_status_t { return C.hms1_chunks_get(chunks, C.size_t(i), &c) }); err != nil {
			return nil, err
		}
		out[i] = Chunk{ID: goStr(c.id), Text: goStr(c.text), StartByte: uint32(c.start_byte), EndByte: uint32(c.end_byte)}
	}
	return out, nil
}
