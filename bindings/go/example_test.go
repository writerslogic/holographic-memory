// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package hms_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	hms "github.com/writerslogic/holographic-memory/bindings/go"
)

func Example() {
	dir, err := os.MkdirTemp("", "hms-example-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	store, err := hms.Open(filepath.Join(dir, "store"), 2048)
	if err != nil {
		log.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	if _, err := store.Ingest(ctx,
		hms.Document{ID: "rust", Text: "Rust is a systems programming language focused on memory safety."},
		hms.Document{ID: "rivers", Text: "Rivers carry water from mountains to the sea."},
	); err != nil {
		log.Fatal(err)
	}

	k1, b := 1.2, 0.75
	hits, err := store.Search(ctx, "memory safety language", &hms.SearchOptions{K: 1, BM25K1: &k1, BM25B: &b})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(hits[0].DocumentID)
	// Output: rust
}
