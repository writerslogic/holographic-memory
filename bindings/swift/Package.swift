// swift-tools-version:5.9
// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import PackageDescription

// Build Frameworks/hms_mobileFFI.xcframework first: scripts/build-xcframework.sh
let package = Package(
    name: "HolographicMemory",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .macCatalyst(.v15),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "HolographicMemory", targets: ["HolographicMemory"]),
    ],
    targets: [
        .binaryTarget(
            name: "hms_mobileFFI",
            path: "Frameworks/hms_mobileFFI.xcframework"
        ),
        .target(
            name: "HolographicMemory",
            dependencies: ["hms_mobileFFI"],
            path: "Sources/HolographicMemory"
        ),
        .testTarget(
            name: "HolographicMemoryTests",
            dependencies: ["HolographicMemory"],
            path: "Tests/HolographicMemoryTests"
        ),
    ]
)
