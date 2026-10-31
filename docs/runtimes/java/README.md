# Java

The `com.writerslogic:writerslogic-warm` binding is available as source in this repository; it is not published on Maven Central yet. It targets Java 22+ and uses the Foreign Function & Memory API, without JNI. Build the C ABI native library locally before testing or packaging it. Run with `--enable-native-access=ALL-UNNAMED` (the jar manifest also sets `Enable-Native-Access`).

From the repository root, run `cargo build --release --manifest-path bindings/c/Cargo.toml`. The library is written to `bindings/c/target/release/` unless `CARGO_TARGET_DIR` is set.

```java
try (Store store = Store.open(Path.of("/data/store"), 2048,
        OpenOptions.builder().encryptionKey(key).build())) {
    store.ingest(List.of(Document.of("a", "text").withTitleWeight(3)));
    List<Result> hits = store.search("query",
        SearchOptions.builder().k(5).bm25K1(1.2).bm25B(0.75).build());
}
```
Metadata and filters are JSON strings (no JSON dependency). `Store` is thread safe; `close` waits for calls in flight. Errors are `WarmException` with a `Code`.

## Why no JNI fallback
FFM is final in JDK 22; JDK 21 has it only as a preview (needs `--enable-preview` and classes compiled with the same release), and JDK 17 has an incubator API with a different shape. A JNI shim would add a second native artifact per platform plus a C toolchain to the build. This binding targets Java 22+; applications on older Java runtimes can call the HTTP server or supply their own bridge to the C ABI.

## Native library resolution
`-Dhms.library.path` or `WARM_LIBRARY_PATH` (file or directory), then `META-INF/native/<rid>/<lib>` inside the jar (`osx-arm64`, `linux-x64`, `win-x64`, ...), then the OS library path. To bundle: put libraries at `bindings/java/natives/<rid>/<lib>` before `mvn package`.

## Test
```sh
cd bindings/java
mvn -Dhms.native.dir="$PWD/../c/target/release" test
```
Verified here: JDK 25 on macOS arm64, 11 JUnit tests, `--release 22` compile. JDK 22-24 runtime and Linux/Windows are not verified locally.
