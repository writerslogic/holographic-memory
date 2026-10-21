// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package hms_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	hms "github.com/writerslogic/holographic-memory/bindings/go"
)

func open(t *testing.T, opts ...hms.OpenOption) (*hms.Store, string) {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "store")
	s, err := hms.Open(dir, 2048, opts...)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s, dir
}

func seed(t *testing.T, s *hms.Store) {
	t.Helper()
	uri := "file:///fox.txt"
	counts, err := s.Ingest(context.Background(),
		hms.Document{ID: "fox", Text: "The quick brown fox jumps over the lazy dog near the river bank.",
			SourceURI: &uri, Metadata: map[string]any{"topic": "animals"}},
		hms.Document{ID: "rust", Text: "Rust is a systems programming language focused on memory safety and speed.",
			Metadata: map[string]any{"topic": "code"}, TitleWeight: 3},
		hms.Document{ID: "rivers", Text: "Rivers carry water from mountains to the sea and shape the valleys they cross.",
			Metadata: map[string]any{"topic": "geo"}},
	)
	if err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	if len(counts) != 3 || counts[0] == 0 || counts[1] == 0 || counts[2] == 0 {
		t.Fatalf("chunk counts = %v", counts)
	}
}

func TestVersions(t *testing.T) {
	major, _, _ := hms.ABIVersion()
	if major != hms.ABIMajor {
		t.Fatalf("ABI major %d", major)
	}
	if hms.Version() == "" {
		t.Fatal("empty version")
	}
}

func TestSearchRoundTrip(t *testing.T) {
	s, _ := open(t)
	seed(t, s)
	ctx := context.Background()

	hits, err := s.Search(ctx, "memory safety programming language", nil)
	if err != nil || len(hits) == 0 {
		t.Fatalf("Search: %v %v", hits, err)
	}
	if hits[0].DocumentID != "rust" || hits[0].Text == nil || hits[0].Score <= 0 || hits[0].SemanticScore != nil {
		t.Fatalf("unexpected top hit %+v", hits[0])
	}
	var meta map[string]string
	if err := json.Unmarshal(hits[0].Metadata, &meta); err != nil || meta["topic"] != "code" {
		t.Fatalf("metadata %s (%v)", hits[0].Metadata, err)
	}

	k1, b := 2.0, 0.3
	hits, err = s.Search(ctx, "river water mountains", &hms.SearchOptions{K: 1, BM25K1: &k1, BM25B: &b})
	if err != nil || len(hits) != 1 || hits[0].DocumentID != "rivers" {
		t.Fatalf("bm25 search: %+v %v", hits, err)
	}
	bad := 1.5
	_, err = s.Search(ctx, "river", &hms.SearchOptions{BM25B: &bad})
	if !errors.Is(err, hms.ErrInvalidArgument) {
		t.Fatalf("bm25 b out of range: %v", err)
	}

	uri := "file:///fox.txt"
	hits, err = s.Search(ctx, "fox river", &hms.SearchOptions{SourceURI: &uri})
	if err != nil || len(hits) == 0 || hits[0].DocumentID != "fox" || hits[0].SourceURI == nil || *hits[0].SourceURI != uri {
		t.Fatalf("source filter: %+v %v", hits, err)
	}
	hits, err = s.Search(ctx, "river language", &hms.SearchOptions{DocumentIDs: []string{"rust"}, Fusion: hms.FusionRRF})
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.DocumentID != "rust" {
			t.Fatalf("document filter leaked %s", h.DocumentID)
		}
	}

	existed, err := s.Delete(ctx, "fox")
	if err != nil || !existed {
		t.Fatalf("Delete: %v %v", existed, err)
	}
	existed, err = s.Delete(ctx, "fox")
	if err != nil || existed {
		t.Fatalf("second Delete: %v %v", existed, err)
	}

	// A rejected batch changes nothing.
	_, err = s.Ingest(ctx, hms.Document{ID: "ok", Text: "fine text here"}, hms.Document{ID: "", Text: "no id"})
	if !errors.Is(err, hms.ErrInvalidArgument) {
		t.Fatalf("empty id: %v", err)
	}
	if hits, _ := s.Search(ctx, "fine text", nil); len(hits) > 0 && hits[0].DocumentID == "ok" {
		t.Fatal("rejected batch was partially applied")
	}
	if _, err := s.Ingest(ctx, hms.Document{ID: "j", Text: "t", Metadata: make(chan int)}); err == nil {
		t.Fatal("unserializable metadata accepted")
	}
}

func TestEmptyAndUnicodeText(t *testing.T) {
	s, _ := open(t)
	ctx := context.Background()
	if _, err := s.Ingest(ctx, hms.Document{ID: "uni", Text: "naïve café 数据 memory", Metadata: map[string]any{"k": "ü"}}); err != nil {
		t.Fatal(err)
	}
	hits, err := s.Search(ctx, "café", nil)
	if err != nil || len(hits) == 0 || hits[0].Text == nil || !strings.Contains(*hits[0].Text, "数据") {
		t.Fatalf("unicode round trip: %+v %v", hits, err)
	}
	if _, err := s.Search(ctx, "", nil); err != nil {
		t.Logf("empty query rejected: %v", err)
	}
}

func TestStatsAndSecurityStatus(t *testing.T) {
	s, _ := open(t)
	seed(t, s)
	st, err := s.Stats(context.Background())
	if err != nil || st.Dimensions != 2048 || st.VectorCount < 3 {
		t.Fatalf("Stats: %+v %v", st, err)
	}
	sec, err := s.SecurityStatus(context.Background())
	if err != nil || !sec.SecurityCompiled || sec.EncryptionActive {
		t.Fatalf("SecurityStatus: %+v %v", sec, err)
	}
	for _, op := range []func(context.Context) error{s.Flush, s.Compact, s.Maintain} {
		if err := op(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
}

func TestLockedAndClosed(t *testing.T) {
	s, dir := open(t)
	if _, err := hms.Open(dir, 2048); !errors.Is(err, hms.ErrLocked) {
		t.Fatalf("second open: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close is idempotent: %v", err)
	}
	if _, err := s.Search(context.Background(), "x", nil); !errors.Is(err, hms.ErrClosed) {
		t.Fatalf("after close: %v", err)
	}
}

func TestEncryption(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "enc")
	s, err := hms.Open(dir, 2048, hms.WithEncryptionKey("correct horse battery staple"))
	if err != nil {
		t.Fatal(err)
	}
	sec, err := s.SecurityStatus(context.Background())
	if err != nil || !sec.EncryptionActive {
		t.Fatalf("SecurityStatus: %+v %v", sec, err)
	}
	if _, err := s.Ingest(context.Background(), hms.Document{ID: "secret", Text: "the launch code is swordfish unicorn"}); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	// The vector/transaction log must hold no plaintext. (documents.text is an engine gap
	// tracked in docs/runtimes/c; it is excluded here and asserted by the C test.)
	entries := walk(t, dir)
	for _, f := range entries {
		if filepath.Base(f) == "documents.text" {
			continue
		}
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(data, []byte("swordfish")) {
			t.Fatalf("plaintext in %s", f)
		}
	}
	if _, err := hms.Open(dir, 2048, hms.WithEncryptionKey("wrong")); err == nil {
		t.Fatal("wrong key accepted")
	} else if !errors.Is(err, hms.ErrEncryption) && !errors.Is(err, hms.ErrCorrupt) {
		t.Fatalf("wrong key error: %v", err)
	}
	s, err = hms.Open(dir, 2048, hms.WithEncryptionKey("correct horse battery staple"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	hits, err := s.Search(context.Background(), "launch code swordfish", nil)
	if err != nil || len(hits) != 1 {
		t.Fatalf("search after reopen: %+v %v", hits, err)
	}
}

func walk(t *testing.T, dir string) []string {
	t.Helper()
	var files []string
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			files = append(files, p)
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return files
}

func TestSnapshot(t *testing.T) {
	s, dir := open(t)
	seed(t, s)
	snap := filepath.Join(t.TempDir(), "snap")
	if err := s.Snapshot(context.Background(), snap); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	if err := s.Snapshot(context.Background(), snap); !errors.Is(err, hms.ErrInvalidArgument) {
		t.Fatalf("existing destination: %v", err)
	}
	if err := s.Snapshot(context.Background(), filepath.Join(dir, "inside")); !errors.Is(err, hms.ErrInvalidArgument) {
		t.Fatalf("destination inside store: %v", err)
	}
	copy, err := hms.Open(snap, 2048)
	if err != nil {
		t.Fatalf("open snapshot: %v", err)
	}
	defer copy.Close()
	hits, err := copy.Search(context.Background(), "memory safety programming language", nil)
	if err != nil || len(hits) == 0 || hits[0].DocumentID != "rust" {
		t.Fatalf("snapshot search: %+v %v", hits, err)
	}
}

func TestChunkDocument(t *testing.T) {
	var sb strings.Builder
	for i := 0; i < 400; i++ {
		fmt.Fprintf(&sb, "word%d ", i)
	}
	chunks, err := hms.ChunkDocument(hms.Document{ID: "long", Text: sb.String(), ChunkWords: 100, OverlapWords: 10})
	if err != nil || len(chunks) < 4 {
		t.Fatalf("chunks: %d %v", len(chunks), err)
	}
	if chunks[0].EndByte <= chunks[0].StartByte || chunks[0].Text == "" {
		t.Fatalf("bad chunk %+v", chunks[0])
	}
	if _, err := hms.ChunkDocument(hms.Document{ID: "", Text: "x"}); !errors.Is(err, hms.ErrInvalidArgument) {
		t.Fatalf("empty id: %v", err)
	}
}

func TestContextCancellation(t *testing.T) {
	s, _ := open(t)
	seed(t, s)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Search(ctx, "river", nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled: %v", err)
	}
	// A deadline that fires mid-call returns promptly; the store stays usable afterwards.
	ctx, cancel = context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	time.Sleep(time.Millisecond)
	if _, err := s.Search(ctx, "river", nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired deadline: %v", err)
	}
	live, cancelLive := context.WithTimeout(context.Background(), time.Minute)
	defer cancelLive()
	if hits, err := s.Search(live, "river water", nil); err != nil || len(hits) == 0 {
		t.Fatalf("cancellable context search: %v %v", hits, err)
	}
}

func TestConcurrentUse(t *testing.T) {
	s, _ := open(t)
	seed(t, s)
	var wg sync.WaitGroup
	errs := make(chan error, 64)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < 25; i++ {
				if _, err := s.Search(context.Background(), "memory language river", nil); err != nil {
					errs <- err
					return
				}
				if i%10 == 0 {
					id := fmt.Sprintf("extra-%d-%d", g, i)
					if _, err := s.Ingest(context.Background(), hms.Document{ID: id, Text: "extra memory text"}); err != nil {
						errs <- err
						return
					}
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestEmbeddings(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "emb")
	s, err := hms.Open(dir, 2048, hms.WithEmbeddingSpace("test-model", "r1", 4))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	ctx := context.Background()
	_, err = s.Ingest(ctx,
		hms.Document{ID: "a", Text: "alpha text", Embeddings: [][]float64{{1, 0, 0, 0}}},
		hms.Document{ID: "b", Text: "beta text", Embeddings: [][]float64{{0, 1, 0, 0}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	sem := 1.0
	hits, err := s.Search(ctx, "", &hms.SearchOptions{Embedding: []float64{0, 1, 0, 0}, SemanticWeight: &sem})
	if err != nil || len(hits) == 0 || hits[0].DocumentID != "b" || hits[0].SemanticScore == nil {
		t.Fatalf("semantic search: %+v %v", hits, err)
	}
}
