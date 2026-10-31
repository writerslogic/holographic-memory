# Go bindings (build from source)

Module `github.com/writerslogic/warm/bindings/go`, package `warm` (cgo, Go 1.24+). Links the static C ABI library, so binaries need no runtime library path.

The binding is in this repository. No prebuilt native library or tagged Go module release is available yet; build the C ABI library from source before using it.

```go
store, _ := warm.Open("/data/store", 2048, warm.WithEncryptionKey(key))
defer store.Close()
store.Ingest(ctx, warm.Document{ID: "a", Text: "...", TitleWeight: 3})
k1, b := 1.2, 0.75
hits, _ := store.Search(ctx, "query", &warm.SearchOptions{K: 5, BM25K1: &k1, BM25B: &b})
```
`Ingest`, `Search`, `Delete`, `Stats`, `SecurityStatus`, `Flush`, `Compact`, `Maintain`, `Snapshot`, `ChunkDocument`; errors are `*warm.Error` (`errors.Is(err, warm.ErrLocked)` etc.). `Store` is goroutine safe; `Close` waits for calls in flight and is idempotent; unclosed stores are closed by a GC cleanup.

Context: checked before the call; with a cancellable context the call runs on its own goroutine and returns `ctx.Err()` at once. The native call cannot be interrupted, so it completes in the background (inputs stay pinned); after a cancelled mutation the outcome is unknown.

## Build the native library
From a clean clone of the repository (needs Rust 1.89+ and a C compiler):
```sh
bindings/go/scripts/build-native.sh     # installs bindings/go/lib/<GOOS>_<GOARCH>/
cd bindings/go && go test ./...
```
The `warm-fetch` helper is prepared for future checksummed release assets, but its default download URL cannot be used until those assets are published. Planned platforms: darwin arm64/amd64, linux arm64/amd64, windows amd64 (MinGW). Verified here: darwin/arm64 (build from source, warm-fetch via `file://`, `go test -race`). The helper's HTTP path and checksum rejection are unit tested.
