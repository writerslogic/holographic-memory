// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::types::HardwareCapabilities;

/// Detect CPU features without executing unsupported instructions.
pub fn capabilities() -> HardwareCapabilities {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let (avx2, avx512f) = (
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("avx512f"),
    );
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let (avx2, avx512f) = (false, false);

    #[cfg(target_arch = "aarch64")]
    let neon = std::arch::is_aarch64_feature_detected!("neon");
    #[cfg(target_arch = "arm")]
    let neon = std::arch::is_arm_feature_detected!("neon");
    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
    let neon = false;

    HardwareCapabilities {
        architecture: std::env::consts::ARCH.to_string(),
        avx2,
        avx512f,
        neon,
        // Sparse sorted u32 indices favor adaptive merge/galloping over dense SIMD.
        sparse_intersection_kernel: "adaptive_merge_galloping".to_string(),
    }
}
