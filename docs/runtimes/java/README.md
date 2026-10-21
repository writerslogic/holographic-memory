# Java

Maven artifact `com.writerslogic:holographic-memory`, Java 22+, Foreign Function & Memory API, no JNI. Run with `--enable-native-access=ALL-UNNAMED` (the jar manifest also sets `Enable-Native-Access`).

```java
try (Store store = Store.open(Path.of("/data/store"), 2048,
        OpenOptions.builder().encryptionKey(key).build())) {
    store.ingest(List.of(Document.of("a", "text").withTitleWeight(3)));
    List<Result> hits = store.search("query",
        SearchOptions.builder().k(5).bm25K1(1.2).bm25B(0.75).build());
}
```
Metadata and filters are JSON strings (no JSON dependency). `Store` is thread safe; `close` waits for calls in flight. Errors are `HmsException` with a `Code`.

## Why no JNI fallback
FFM is final in JDK 22; JDK 21 has it only as a preview (needs `--enable-preview` and classes compiled with the same release), JDK 17 has an incubator API with a different shape. A JNI shim would add a second native artifact per platform plus a C toolchain in every release for JDK 17/21, which are already past their free-update windows or close to it (today is late 2026; 25 is the current LTS). Decision: Java 22+ only; older runtimes use the Go, .NET or C bindings, or the HTTP server.

## Native library resolution
`-Dhms.library.path` or `HMS_LIBRARY_PATH` (file or directory), then `META-INF/native/<rid>/<lib>` inside the jar (`osx-arm64`, `linux-x64`, `win-x64`, ...), then the OS library path. To bundle: put libraries at `bindings/java/natives/<rid>/<lib>` before `mvn package`.

## Test
```sh
cd bindings/java && mvn -Dhms.native.dir=/path/to/cargo/release test
```
Verified here: JDK 25 on macOS arm64, 11 JUnit tests, `--release 22` compile. JDK 22-24 runtime and Linux/Windows are not verified locally.
