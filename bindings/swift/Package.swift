// swift-tools-version:5.9
// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import PackageDescription

// Build Frameworks/warm_mobileFFI.xcframework first: scripts/build-xcframework.sh
let package = Package(
    name: "WarmMemory",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .macCatalyst(.v15),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "WarmMemory", targets: ["WarmMemory"]),
    ],
    targets: [
        .binaryTarget(
            name: "warm_mobileFFI",
            path: "Frameworks/warm_mobileFFI.xcframework"
        ),
        .target(
            name: "WarmMemory",
            dependencies: ["warm_mobileFFI"],
            path: "Sources/WarmMemory"
        ),
        .testTarget(
            name: "WarmMemoryTests",
            dependencies: ["WarmMemory"],
            path: "Tests/WarmMemoryTests"
        ),
    ]
)
