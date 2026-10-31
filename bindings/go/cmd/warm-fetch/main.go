// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Command warm-fetch downloads the prebuilt static library for this platform, verifies its
// SHA-256, and installs it where the cgo build can find it.
//
//	go run github.com/writerslogic/warm/bindings/go/cmd/warm-fetch@latest
//	export CGO_LDFLAGS="-L$HOME/.cache/writerslogic-warm/0.1.0/osx-arm64/lib"
//
// The module's own #cgo directives name the library, so only the search path is needed.
package main

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// defaultVersion is the native library release this package version was written against.
const defaultVersion = "0.1.0"

const defaultBase = "https://github.com/writerslogic/warm/releases/download/c-abi-v%s"

const maxArchive = 512 << 20

func rid(goos, goarch string) (string, error) {
	switch goos + "/" + goarch {
	case "darwin/arm64":
		return "osx-arm64", nil
	case "darwin/amd64":
		return "osx-x64", nil
	case "linux/arm64":
		return "linux-arm64", nil
	case "linux/amd64":
		return "linux-x64", nil
	case "windows/amd64":
		return "win-x64-gnu", nil
	}
	return "", fmt.Errorf("no prebuilt library for %s/%s; build from source (see docs/runtimes/go)", goos, goarch)
}

func fetch(base, name string) ([]byte, error) {
	if strings.HasPrefix(base, "file://") {
		u, err := url.Parse(base)
		if err != nil {
			return nil, err
		}
		return os.ReadFile(filepath.Join(u.Path, name))
	}
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Get(strings.TrimRight(base, "/") + "/" + name)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: %s", name, resp.Status)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxArchive+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxArchive {
		return nil, fmt.Errorf("%s exceeds %d bytes", name, maxArchive)
	}
	return data, nil
}

func main() {
	version := flag.String("version", defaultVersion, "native library version")
	base := flag.String("base-url", "", "release asset URL prefix (http(s):// or file://)")
	dest := flag.String("dest", "", "install directory (default: user cache)")
	goos := flag.String("goos", runtime.GOOS, "target GOOS")
	goarch := flag.String("goarch", runtime.GOARCH, "target GOARCH")
	flag.Parse()
	if err := run(*version, *base, *dest, *goos, *goarch); err != nil {
		fmt.Fprintln(os.Stderr, "warm-fetch:", err)
		os.Exit(1)
	}
}

func run(version, base, dest, goos, goarch string) error {
	id, err := rid(goos, goarch)
	if err != nil {
		return err
	}
	if base == "" {
		base = fmt.Sprintf(defaultBase, version)
	}
	if dest == "" {
		cache, err := os.UserCacheDir()
		if err != nil {
			return err
		}
		dest = filepath.Join(cache, "writerslogic-warm", version, id)
	}
	name := fmt.Sprintf("writerslogic-warm-c-%s-%s.tar.gz", version, id)
	archive, err := fetch(base, name)
	if err != nil {
		return err
	}
	sumFile, err := fetch(base, name+".sha256")
	if err != nil {
		return err
	}
	fields := strings.Fields(string(sumFile))
	if len(fields) == 0 {
		return fmt.Errorf("%s.sha256 is empty", name)
	}
	sum := sha256.Sum256(archive)
	if !strings.EqualFold(fields[0], hex.EncodeToString(sum[:])) {
		return fmt.Errorf("SHA-256 mismatch for %s: expected %s, got %s", name, fields[0], hex.EncodeToString(sum[:]))
	}
	libDir := filepath.Join(dest, "lib")
	if err := os.MkdirAll(libDir, 0o755); err != nil {
		return err
	}
	if err := extractStatic(archive, libDir); err != nil {
		return err
	}
	fmt.Printf("installed %s\nexport CGO_LDFLAGS=\"-L%s\"\n", filepath.Join(libDir, "libwriterslogic_warm_c.a"), libDir)
	return nil
}

// extractStatic writes only the static library, so the linker cannot prefer a shared one.
func extractStatic(archive []byte, libDir string) error {
	gz, err := gzip.NewReader(strings.NewReader(string(archive)))
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return fmt.Errorf("archive has no libwriterslogic_warm_c.a")
		}
		if err != nil {
			return err
		}
		if hdr.Typeflag != tar.TypeReg || path.Base(hdr.Name) != "libwriterslogic_warm_c.a" || path.Base(path.Dir(hdr.Name)) != "lib" {
			continue
		}
		tmp, err := os.CreateTemp(libDir, ".fetch-*")
		if err != nil {
			return err
		}
		if _, err := io.Copy(tmp, io.LimitReader(tr, maxArchive)); err != nil {
			tmp.Close()
			os.Remove(tmp.Name())
			return err
		}
		if err := tmp.Close(); err != nil {
			os.Remove(tmp.Name())
			return err
		}
		return os.Rename(tmp.Name(), filepath.Join(libDir, "libwriterslogic_warm_c.a"))
	}
}
