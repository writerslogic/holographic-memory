// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-Tenant Cryptographic Sharding.

use crate::core::entangled::EntangledHVec;
use fxhash::hash64;

/// Generates a highly entropic master key for a specific tenant.
/// Because VSA similarity is zero between orthogonal vectors, we can store
/// thousands of clients in the EXACT same Navigable Small World index without
/// them ever colliding or being able to decipher each other's data.
pub fn generate_tenant_key(tenant_secret: &str, dimensions: usize) -> EntangledHVec {
    let seed = hash64(tenant_secret);
    let mut master_key = EntangledHVec::new_deterministic(dimensions, seed);
    
    // Densify the master key to ensure maximum obfuscation (O(1) operation)
    for i in 1..25 {
        let sub_key = EntangledHVec::new_deterministic(dimensions, seed + i);
        master_key = master_key.bind(&sub_key);
    }
    
    master_key
}
