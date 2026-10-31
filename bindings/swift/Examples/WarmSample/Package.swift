// swift-tools-version:5.9
// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import PackageDescription

// A minimal SwiftUI search app. Build the xcframework in ../.. first.
// macOS: `swift run` ; iOS: add this folder's sources to an app target, or
// `xcodebuild -scheme WarmSample -destination 'generic/platform=iOS Simulator' build`.
let package = Package(
    name: "WarmSample",
    platforms: [.iOS(.v17), .macOS(.v14)],
    dependencies: [.package(path: "../..")],
    targets: [
        .executableTarget(
            name: "WarmSample",
            dependencies: [.product(name: "WarmMemory", package: "swift")],
            path: "Sources/WarmSample"
        ),
    ]
)
