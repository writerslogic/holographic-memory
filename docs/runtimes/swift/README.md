# Swift (iOS, macOS, Mac Catalyst, visionOS)

`bindings/swift` is a Swift Package over the UniFFI crate in `bindings/uniffi`. The engine runs
on the device: no service calls, no network.

The package is available as source in this repository; no prebuilt XCFramework release is published yet. Build the XCFramework locally before adding the package by path.

## Requirements

- macOS with Xcode 15 or later (the iOS and Catalyst slices need the full Xcode, not only the
  command line tools). Verified with Xcode 27.1 on Apple silicon.
- `rustup` with a stable toolchain, 1.89 or later. The build script runs `rustup target add` for
  the slices it builds (user-level, no root).
- visionOS is opt-in (`VISIONOS=1`): its Rust targets are tier 3, so the script builds them with
  nightly `-Zbuild-std`. It was not run in this repository's verification, see "Verified".

## Build

```sh
bindings/swift/scripts/build-xcframework.sh
```

Writes `bindings/swift/Frameworks/warm_mobileFFI.xcframework` (iOS device, iOS simulator
arm64+x86_64, macOS arm64+x86_64, Mac Catalyst arm64+x86_64) and regenerates
`Sources/WarmMemory/WarmMobile.swift` from the library's UniFFI metadata. `SLICES="ios
ios-sim"` builds a subset; `IPHONEOS_DEPLOYMENT_TARGET` (15.0) and `MACOSX_DEPLOYMENT_TARGET`
(12.0) set the deployment floor. Add the package by path, or publish the xcframework as a
release asset and switch `Package.swift` to a `url` + `checksum` binary target.

## Use

```swift
import WarmMemory

let store = try await WarmStore.open(
    path: directory.path,
    options: StoreOptions(dimensions: 4096, encryptionKey: keyFromKeychain))

_ = try await store.ingest(input: DocumentInput(id: "note-1", text: "Title\nBody ...", titleWeight: 3))
let hits = try await store.search(
    query: "title", options: SearchOptions(k: 10, bm25K1: 1.2, bm25B: 0.75))
let info = try await store.snapshot(destination: backupURL.path)
let restored = try await WarmStore.restore(
    snapshotPath: backupURL.path, path: otherDir.path, options: sameOptions)
```

Every disk-touching call has an `async` form and a `…Blocking` form. A store is thread-safe:
share one handle across tasks. A directory opens once per process (`WarmError.StoreLocked`);
call `shutdown()` to release it before reopening. Errors are the typed `WarmError` enum:
`InvalidInput`, `StoreLocked`, `Encryption`, `IncompatibleStore`, `NotFound`, `AlreadyExists`,
`Corrupt`, `Io`, `Closed`, `Internal`.

Metadata and filters are JSON strings (`metadataJson`, `filterJson`) holding an object; result
metadata comes back as `metadataJson`.

## Sample

`bindings/swift/Examples/WarmSample` is a SwiftUI note-search app (`swift run` on macOS; the same
package builds for the iOS simulator with `xcodebuild -scheme WarmSample -destination
'generic/platform=iOS Simulator' build`).

## Verified

Run against the xcframework built by the script:

- `swift test` on macOS arm64: 7 XCTest cases pass.
- `xcodebuild test -scheme WarmMemory -destination 'platform=iOS Simulator,name=iPhone 17'`
  (iOS 26.5 simulator): the same 7 cases pass.
- The sample builds with `swift build` (macOS) and `xcodebuild` (iOS simulator).

Not verified: a physical iOS device (the `ios-arm64` slice links but was not run), visionOS,
Mac Catalyst execution (the slice is built and packaged), x86_64 slices (built, not run).

## Storage and encryption notes

- `encryptionKey` encrypts the journal and the chunk-text cache with AES-256-GCM under an
  Argon2id-derived key. Use a random secret from the Keychain. Combine it with the platform's
  file protection (`NSFileProtectionComplete`) for at-rest protection of the other metadata
  files; those (schema, index checksums) hold no document text.
- The log segment is preallocated sparse. `StoreStats.diskBytes` is its logical length and
  `diskAllocatedBytes` what the file system actually allocated.
- Sizes and timings: `benchmarks/results/mobile_v1.json`.
