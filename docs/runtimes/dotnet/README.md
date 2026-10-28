# .NET

The `WritersLogic.HolographicMemory` binding (net8.0) is available as source in this repository; it is not published on NuGet yet. It uses P/Invoke via `LibraryImport`, a `SafeHandle` store handle, and `*Async` methods. Build the C ABI native library locally before testing or packing it.

From the repository root, run `cargo build --release --manifest-path bindings/c/Cargo.toml`. The library is written to `bindings/c/target/release/` unless `CARGO_TARGET_DIR` is set.

```csharp
using var store = Store.Open("/data/store", 2048, new OpenOptions { EncryptionKey = key });
store.Ingest(new Document("a", "text") { TitleWeight = 3 });
var hits = await store.SearchAsync("query", new SearchOptions { K = 5, Bm25K1 = 1.2, Bm25B = 0.75 });
```
`Store` is thread safe. The `SafeHandle` is passed to every native call, so `Dispose` during a call waits for it. Async methods run the sync call on the thread pool (the native call cannot be interrupted: cancellation is honoured before it starts). Errors: `HmsException.Status`.

## Native library resolution
`HMS_LIBRARY_PATH`, then next to the assembly, then `runtimes/<rid>/native/` (the NuGet layout), then the OS path. To pack, place libraries at `bindings/dotnet/src/HolographicMemory/natives/<rid>/native/<lib>` and run `dotnet pack -c Release` (rids `osx-arm64 osx-x64 linux-x64 linux-arm64 win-x64 win-arm64`).

## Test
```sh
HMS_LIBRARY_PATH="$PWD/bindings/c/target/release" dotnet test bindings/dotnet/tests/HolographicMemory.Tests
```
Verified here: .NET SDK 8 on macOS arm64, 11 xunit tests; `dotnet pack` produces `runtimes/osx-arm64/native/`. A consumer restoring the package from a feed, Linux and Windows are not verified locally.
