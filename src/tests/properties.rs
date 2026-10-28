// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::entangled::EntangledHVec;
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_bind_commutativity(seed1: u64, seed2: u64) {
        let dim = 1000;
        let v1 = EntangledHVec::new_deterministic(dim, seed1);
        let v2 = EntangledHVec::new_deterministic(dim, seed2);

        let res1 = v1.bind(&v2);
        let res2 = v2.bind(&v1);

        // In XOR, exact equality is expected
        prop_assert!((res1.similarity(&res2) - 1.0).abs() < f64::EPSILON);
    }
}
