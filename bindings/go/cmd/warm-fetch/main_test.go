// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func archive(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for name, body := range files {
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(body)), Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func serve(t *testing.T, name string, data []byte, sum string) string {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/"+name, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(data) })
	mux.HandleFunc("/"+name+".sha256", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(sum + "  " + name + "\n")) })
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv.URL
}

func TestFetchInstallsOnlyTheStaticLibrary(t *testing.T) {
	name := "writerslogic-warm-c-9.9.9-linux-x64.tar.gz"
	data := archive(t, map[string]string{
		"writerslogic-warm-c-9.9.9-linux-x64/lib/libwriterslogic_warm_c.a":          "STATIC",
		"writerslogic-warm-c-9.9.9-linux-x64/lib/libwriterslogic_warm_c.so":         "SHARED",
		"writerslogic-warm-c-9.9.9-linux-x64/../../escape/libwriterslogic_warm_c.a": "EVIL",
	})
	sum := sha256.Sum256(data)
	base := serve(t, name, data, hex.EncodeToString(sum[:]))
	dest := t.TempDir()
	if err := run("9.9.9", base, dest, "linux", "amd64"); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(dest, "lib", "libwriterslogic_warm_c.a"))
	if err != nil || string(got) != "STATIC" {
		t.Fatalf("static library: %q %v", got, err)
	}
	if _, err := os.Stat(filepath.Join(dest, "lib", "libwriterslogic_warm_c.so")); err == nil {
		t.Fatal("shared library must not be installed")
	}
}

func TestFetchRejectsChecksumMismatch(t *testing.T) {
	name := "writerslogic-warm-c-9.9.9-osx-arm64.tar.gz"
	data := archive(t, map[string]string{"x/lib/libwriterslogic_warm_c.a": "STATIC"})
	base := serve(t, name, data, strings.Repeat("0", 64))
	err := run("9.9.9", base, t.TempDir(), "darwin", "arm64")
	if err == nil || !strings.Contains(err.Error(), "SHA-256 mismatch") {
		t.Fatalf("want checksum failure, got %v", err)
	}
}

func TestUnsupportedPlatform(t *testing.T) {
	if _, err := rid("plan9", "mips"); err == nil {
		t.Fatal("expected error")
	}
}
