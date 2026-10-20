# Kotlin (JVM and Android)

`bindings/kotlin` is a Gradle build with two library modules over the UniFFI crate in
`bindings/uniffi`, plus a Compose sample:

| Module | What | Native library |
| --- | --- | --- |
| `:jvm` | Kotlin/JVM library; JNA loads the library from classpath resources (`<os>-<arch>/`) | host, via `scripts/build-jvm-native.sh` |
| `:android` | Android AAR (minSdk 24), `jniLibs` for `arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86` | `scripts/build-android.sh` (cargo-ndk) |
| `:sample-compose` | Minimal Jetpack Compose note-search app | from `:android` |

The generated bindings (`jvm/src/main/kotlin`, `android/src/main/kotlin`) are committed; the two
flavors differ only in the Android-safe cleaner. Regenerate with `scripts/generate.sh`. The API
has no Android or JVM types, so the sources are a ready starting point for a Kotlin Multiplatform
`expect/actual` wrapper.

## Requirements

- JDK 17, Gradle (the wrapper pins 8.14.6; AGP 8.13.2, Kotlin 2.2.21).
- Rust stable 1.89 or later and `rustup`; the scripts add the Rust targets (user-level).
- Android: SDK platform 35, build-tools 35, **NDK r27.3** (`27.3.13750724`; r26 or later should
  work), `cargo install cargo-ndk --locked`. Put the SDK path in `bindings/kotlin/local.properties`
  (`sdk.dir=...`) or `ANDROID_HOME`; the NDK is found under `$ANDROID_HOME/ndk` or `ANDROID_NDK_HOME`.

## Build and test

```sh
cd bindings/kotlin
scripts/build-jvm-native.sh        # host library into jvm/build/natives
./gradlew :jvm:test                # JUnit on the JVM
scripts/build-android.sh           # four ABIs into android/src/main/jniLibs
./gradlew :android:assembleRelease :sample-compose:assembleDebug
./gradlew :android:connectedAndroidTest   # needs a device or emulator
```

## Use

```kotlin
val store = HmsStore.open(
    File(filesDir, "hms").absolutePath,
    StoreOptions(dimensions = 4096u, encryptionKey = keyFromKeystore))

store.ingest(DocumentInput(id = "note-1", text = "Title\nBody ...", titleWeight = 3u))
val hits = store.search("title", SearchOptions(k = 10u, bm25K1 = 1.2, bm25B = 0.75))
store.snapshot(backupDir.absolutePath)
val restored = HmsStore.restore(backupDir.absolutePath, otherDir.absolutePath, sameOptions)
```

Disk-touching calls are `suspend` functions that run on a worker thread, with `…Blocking`
variants. A store is thread-safe. A directory opens once per process (`HmsException.StoreLocked`);
`shutdown()` releases it at once, `close()` (AutoCloseable) drops the handle. Errors are the
sealed `HmsException` (`InvalidInput`, `StoreLocked`, `Encryption`, `IncompatibleStore`,
`NotFound`, `AlreadyExists`, `Corrupt`, `Io`, `Closed`, `Internal`; each carries `details`).
Metadata and filters are JSON object strings (`metadataJson`, `filterJson`).

Proguard/R8: the AAR ships `consumer-rules.pro` keeping JNA and the binding package.

## Verified

- `./gradlew :jvm:test` on macOS arm64 (JDK 17): 7 JUnit cases pass (typed errors, encryption with
  a plaintext scan of the store, snapshot/restore with tamper detection, suspend forms under
  concurrency, BM25 options, `titleWeight`, filters).
- `./gradlew :android:connectedDebugAndroidTest` on an arm64-v8a emulator (Android 15, API 35,
  running on the Apple silicon host): 2 instrumented cases pass.
- `:android:assembleRelease` produces the AAR; `:sample-compose:assembleDebug` builds the sample,
  which installed on that emulator, launched, and created its store.
- All four ABIs build with cargo-ndk (NDK r27.3, API 24).

Not verified: execution of `armeabi-v7a`, `x86` and `x86_64` libraries (built and packaged, not
run), a physical device, and Android versions other than 15. The macOS host library is built with
`strip = debuginfo` because `strip = true` produced a dylib that macOS 27's dyld refused to load;
Linux hosts and the Android libraries use fully stripped builds.

Sizes and timings: `benchmarks/results/mobile_v1.json`.
